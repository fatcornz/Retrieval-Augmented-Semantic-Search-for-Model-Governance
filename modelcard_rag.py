import hashlib
import json
import logging
import os
import re
import sys
import threading
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if os.getenv("RAG_OFFLINE", "0") == "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import fitz
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer

DATA_DIR = os.getenv("RAG_DATA_DIR", "./pdfs")
CHUNK_MAX_WORDS, CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_WORDS", "180")), int(os.getenv("RAG_CHUNK_OVERLAP", "35"))
TOP_K_CANDIDATES, TOP_K_FINAL = int(os.getenv("RAG_CANDIDATES", "80")), int(os.getenv("RAG_TOP_K", "6"))
BM25_WEIGHT, RERANK_POOL, RRF_K = 0.45, 48, 60
EMBEDDING_PROFILES = {
    "fast": "sentence-transformers/all-MiniLM-L6-v2",
    "quality": "sentence-transformers/all-mpnet-base-v2",
}
EMBEDDER_MODEL = os.getenv("RAG_EMBED_MODEL", EMBEDDING_PROFILES.get(os.getenv("RAG_PROFILE", "quality"), EMBEDDING_PROFILES["quality"]))
RERANKER_MODEL = os.getenv("RAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
SUMMARIZER_MODEL = os.getenv("RAG_GENERATOR_MODEL", "google/flan-t5-large")
MAX_QUERY_VARIANTS = int(os.getenv("RAG_QUERY_VARIANTS", "6"))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("transformers").setLevel(logging.ERROR)

STOPWORDS = set(
    "a an and are as at be by did do does for from how in into is of on or the to was were "
    "what when where which who with according compare describe explain tell me about".split()
)
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
QUOTE_RE = re.compile(r"'([^']+)'|\"([^\"]+)\"")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")
POLICY_VOCAB_MAPPING = {
    "ai rmf": ["govern", "map", "measure", "manage", "trustworthy ai", "tevv"],
    "rmf": ["risk management framework", "govern", "map", "measure", "manage"],
    "tevv": ["test", "evaluation", "verification", "validation"],
    "explainability": ["interpretability", "shap", "lime", "feature importance"],
    "fairness": ["bias", "demographic parity", "equal opportunity", "disparate impact"],
    "robustness": ["adversarial attack", "data drift", "out-of-distribution"],
    "risk": ["harm", "safety", "hazard", "failure mode", "mitigation"],
    "regulation": ["compliance", "governance", "oversight", "audit"],
    "mitigation": ["safeguard", "guardrail", "monitoring", "human-in-the-loop"],
    "red teaming": ["adversarial testing", "safety evaluation", "external experts", "methodology", "process", "participants"],
    "reward model": ["rlhf", "preference model", "human feedback", "fine-tuning"],
    "rlhf": ["reinforcement learning from human feedback", "reward model", "preference"],
    "ghost attention": ["gatt", "multi-turn dialogue", "instruction retention"],
    "dual-use": ["weapon proliferation", "misuse", "capability risk"],
    "social engineering": ["phishing", "websites", "identifying individuals"],
    "training data": ["corpus", "publicly available data", "data mix", "meta user data"],
}
SUBJECT_MARKERS = [("gpt-4", "GPT-4"), ("gpt4", "GPT-4"), ("llama", "Llama 2"), ("nist", "NIST AI RMF")]


def dedupe(items):
    seen, out = set(), []
    for item in items:
        value = str(item).strip()
        if value and value.lower() not in seen:
            seen.add(value.lower())
            out.append(value)
    return out


def minmax(values):
    values = np.asarray(values, dtype="float32")
    if not values.size:
        return values
    lo, hi = float(values.min()), float(values.max())
    return (values - lo) / (hi - lo) if hi > lo else np.zeros_like(values)


def l2_normalize(values, axis=1):
    values = np.asarray(values, dtype="float32")
    return values / (np.linalg.norm(values, axis=axis, keepdims=True) + 1e-12)


def torch_device():
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def clean_text(raw_text, preserve_blocks=False):
    text = unicodedata.normalize("NFKC", raw_text or "")
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    text = re.sub(r"(?im)^\s*(?:-+\s*)?page\s+\d+\s*(?:-+)?\s*$", "", text)
    if not preserve_blocks:
        return re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n(?:\s*\n)+", "\n\n", text).strip()


def tokenize(text):
    tokens = []
    for token in TOKEN_RE.findall(text.lower()):
        token = token.replace("'", "")
        tokens += [token, *token.split("-")] if "-" in token else [token]
    return [token for token in tokens if len(token) > 1 and token not in STOPWORDS]


def _is_heading(block):
    words = block.split()
    if not 1 <= len(words) <= 14 or len(block) > 140 or block.endswith((".", "?", "!", ";")):
        return False
    titled = sum(word[:1].isupper() or word.isupper() for word in words)
    return bool(re.match(r"^(?:\d+(?:\.\d+)*[.)]?|[A-Z][A-Z\s-]+)", block)) or titled / len(words) >= 0.6


def chunk_records(text, max_words=CHUNK_MAX_WORDS, overlap=CHUNK_OVERLAP):
    blocks = [clean_text(block) for block in re.split(r"\n\s*\n+", clean_text(text, True))]
    groups, section = [], ""
    for block in filter(None, blocks):
        if _is_heading(block):
            section = block
            continue
        units = [unit.strip() for unit in SENTENCE_RE.split(block) if unit.strip()]
        units = [" ".join(words[i : i + max_words]) for unit in units for words in [unit.split()] for i in range(0, len(words), max_words)]
        if units:
            if not groups or groups[-1][0] != section:
                groups.append((section, []))
            groups[-1][1].extend(units)

    chunks = []
    max_words, overlap = max(40, max_words), min(max(overlap, 0), max_words // 2)
    for section, units in groups:
        start = 0
        while start < len(units):
            end, size = start, 0
            while end < len(units) and (size + len(units[end].split()) <= max_words or end == start):
                size += len(units[end].split())
                end += 1
            body = " ".join(units[start:end])
            chunks.append({"text": f"{section}\n{body}" if section else body, "section": section})
            if end == len(units):
                break
            next_start, carried = end, 0
            while next_start > start and carried + len(units[next_start - 1].split()) <= overlap:
                next_start -= 1
                carried += len(units[next_start].split())
            start = max(start + 1, next_start)

    merged, limit = [], max_words + 14
    for chunk in chunks:
        size = len(chunk["text"].split())
        if merged and size < 20 and len(merged[-1]["text"].split()) + size <= limit:
            merged[-1]["text"] += f"\n{chunk['text']}"
        else:
            merged.append(chunk)
    if len(merged) > 1 and len(merged[0]["text"].split()) < 20:
        first, second = merged[:2]
        if len(first["text"].split()) + len(second["text"].split()) <= limit:
            second["text"] = f"{first['text']}\n{second['text']}"
            merged.pop(0)
    return [chunk for chunk in merged if len(chunk["text"].split()) >= 5]


def chunk_text(text, max_words=CHUNK_MAX_WORDS, overlap=CHUNK_OVERLAP):
    return [record["text"] for record in chunk_records(text, max_words, overlap)]


def infer_policy_subject_and_intent(query):
    q = query.lower()
    subjects = dedupe(label for marker, label in SUBJECT_MARKERS if marker in q)
    compare = any(term in f" {q} " for term in ["compare", " vs ", "versus", "difference", "differ", "between", "both"])
    risk = any(term in q for term in ["risk", "danger", "harm", "threat", "safety", "hazard", "misuse"])
    summary = any(term in q for term in ["summary", "summarize", "overview", "briefing"])
    intent = "Comparative Analysis" if compare or len(subjects) > 1 else "Risk Assessment" if risk else "Executive Synthesis" if summary else "General Inquiry"
    return " / ".join(subjects) or "General AI Context", intent


def expand_query(original_query, generated=None):
    q, variants, related = original_query.lower(), [original_query], []
    for trigger, terms in POLICY_VOCAB_MAPPING.items():
        if trigger in q:
            related.extend(terms)
            variants.append(f"{trigger} {' '.join(terms)}")
    keywords = " ".join(dedupe(tokenize(original_query)))
    if keywords and keywords.lower() != q:
        variants.append(keywords)
    if related:
        variants.append(f"{original_query} {' '.join(dedupe(related)[:6])}")
    if any(term in f" {q} " for term in ["compare", " versus ", " vs ", "differ", "difference"]):
        focus = " ".join(token for token in tokenize(original_query) if token not in {"gpt", "gpt4", "gpt-4", "llama", "nist", "meta"})
        for marker, subject in SUBJECT_MARKERS:
            if marker in q:
                variants.append(f"{subject} {focus} methodology similarities differences")
    return dedupe([*variants, *(generated or [])])[:MAX_QUERY_VARIANTS]


def query_phrases(query):
    quoted = [left or right for left, right in QUOTE_RE.findall(query)]
    triggered = [key for key in POLICY_VOCAB_MAPPING if " " in key and key in query.lower()]
    return dedupe(phrase.lower() for phrase in quoted + triggered)


def needs_source_diversity(query):
    q = f" {query.lower()} "
    return any(term in q for term in ["compare", " versus ", " vs ", "differ", "difference", "between", "both"]) or sum(marker in q for marker, _ in SUBJECT_MARKERS) > 1


def _json_documents(payload, filename, defaults=None):
    rows = payload if isinstance(payload, list) else payload.get("items", [payload]) if isinstance(payload, dict) else []
    docs = []
    for row in rows:
        if isinstance(row, str):
            text, meta = row, {}
        elif isinstance(row, dict):
            text = next((row.get(key) for key in ("text", "content", "body", "description") if row.get(key)), "")
            meta = {key: row[key] for key in ("title", "url", "updated_at", "model_id") if row.get(key) is not None}
        else:
            continue
        if str(text).strip():
            docs.append({"filename": filename, "raw_text": str(text), **(defaults or {}), **meta})
    return docs


class DocumentLoader:
    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".json"}

    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_all_documents(self):
        docs, files = [], []
        for path in sorted(self.data_dir.iterdir()):
            if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue
            try:
                loaded = self._load_path(path)
                if loaded:
                    doc_type = "Model Card" if "card" in path.name.lower() else "Technical Paper"
                    docs.extend({"doc_type": doc_type, **doc} for doc in loaded)
                    files.append(path.name)
            except Exception as exc:
                logger.warning("Skipping %s: %s", path.name, exc)
        return docs, files

    def _load_path(self, path):
        if path.suffix.lower() == ".pdf":
            return self._load_pdf(path)
        if path.suffix.lower() == ".json":
            return _json_documents(json.loads(path.read_text(encoding="utf-8")), path.name)
        return [{"filename": path.name, "raw_text": path.read_text(encoding="utf-8", errors="ignore")}]

    @staticmethod
    def _load_pdf(path):
        with fitz.open(path) as pdf:
            pages = [[block[4] for block in page.get_text("blocks", sort=True) if block[4].strip()] for page in pdf]
        short = [clean_text(block) for page in pages for block in page if len(clean_text(block).split()) <= 16]
        repeated = {text for text, count in Counter(short).items() if count >= max(3, len(pages) // 3)}
        docs, current_section = [], ""
        for page, blocks in enumerate(pages, 1):
            blocks = [block for block in blocks if clean_text(block) not in repeated]
            section_hint = current_section
            for block in blocks:
                candidate = clean_text(block)
                if _is_heading(candidate) and not candidate.isdigit():
                    current_section = candidate
            if blocks:
                docs.append({"filename": path.name, "raw_text": "\n\n".join(blocks), "page": page, "section_hint": section_hint})
        return docs


class _HTMLText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        self.skip += tag in {"script", "style", "noscript"}
        if tag in {"p", "div", "article", "section", "h1", "h2", "h3", "li", "br"}:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"} and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


class URLSource:
    """Loads current JSON, HTML, or text content when the corpus is refreshed."""

    def __init__(self, urls, headers=None, timeout=15, max_bytes=5_000_000):
        self.urls, self.headers, self.timeout, self.max_bytes = list(urls), headers or {}, timeout, max_bytes

    def load(self):
        docs, fetched_at = [], datetime.now(timezone.utc).isoformat()
        for url in self.urls:
            try:
                request = Request(url, headers={"User-Agent": "policy-rag/1.0", **self.headers})
                with urlopen(request, timeout=self.timeout) as response:
                    data, content_type = response.read(self.max_bytes + 1), response.headers.get_content_type()
                    charset = response.headers.get_content_charset() or "utf-8"
                if len(data) > self.max_bytes:
                    raise ValueError(f"source exceeds {self.max_bytes} bytes")
                name = Path(urlsplit(url).path).name or urlsplit(url).netloc
                defaults = {"url": url, "fetched_at": fetched_at, "doc_type": "Live Source"}
                if content_type == "application/json" or name.endswith(".json"):
                    docs.extend(_json_documents(json.loads(data.decode(charset)), name, defaults))
                else:
                    text = data.decode(charset, errors="replace")
                    if content_type == "text/html":
                        parser = _HTMLText()
                        parser.feed(text)
                        text = "".join(parser.parts)
                    docs.append({"filename": name, "raw_text": text, **defaults})
            except Exception as exc:
                logger.warning("Skipping live source %s: %s", url, exc)
        return docs


class Embedder:
    def __init__(self, model_name=EMBEDDER_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def _encode(self, texts, kind):
        values = [str(text) for text in texts]
        name = self.model_name.lower()
        if "e5" in name:
            values = [f"{'query' if kind == 'query' else 'passage'}: {text}" for text in values]
        elif "bge" in name and kind == "query":
            values = [f"Represent this sentence for searching relevant passages: {text}" for text in values]
        return np.asarray(self.model.encode(values, normalize_embeddings=True, show_progress_bar=False), dtype="float32")

    def encode_documents(self, texts):
        return self._encode(texts, "document")

    def encode_queries(self, texts):
        return self._encode(texts, "query")

    def __call__(self, texts):
        return self.encode_documents(texts)


def contextualize(text, meta):
    context = [f"Source: {meta.get('filename', 'Unknown')}"]
    if meta.get("section"):
        context.append(f"Section: {meta['section']}")
    return "\n".join([*context, text])


class HybridRetriever:
    def __init__(self, embeddings, texts, meta, bm25=None, bm25_weight=BM25_WEIGHT, use_reranker=True, reranker=None):
        if len(embeddings) != len(texts) or len(texts) != len(meta):
            raise ValueError("embeddings, texts, and metadata must have equal lengths")
        self.texts, self.meta = list(texts), list(meta)
        self.search_texts = [contextualize(text, item) for text, item in zip(self.texts, self.meta)]
        self.embeddings, self.bm25_weight = l2_normalize(embeddings), bm25_weight
        self.bm25 = BM25Okapi([tokenize(text) for text in self.search_texts])
        self.device = torch_device()
        self.reranker_tokenizer = self.reranker_model = None
        if reranker:
            self.reranker_tokenizer, self.reranker_model = reranker
        elif use_reranker:
            self._load_reranker()

    @property
    def reranker(self):
        return (self.reranker_tokenizer, self.reranker_model) if self.reranker_model is not None else None

    def _load_reranker(self):
        try:
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
            self.reranker_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL).to(self.device).eval()
        except Exception as exc:
            logger.info("Cross-encoder unavailable; using fused retrieval scores. %s", exc)

    def search(self, query, expanded_queries=None, embedder=None, top_k=TOP_K_CANDIDATES, final_k=TOP_K_FINAL, filters=None):
        if not self.texts:
            return []
        if embedder is None:
            raise ValueError("HybridRetriever.search requires an Embedder instance")
        queries = dedupe([query, *(expanded_queries or expand_query(query))])[:MAX_QUERY_VARIANTS]
        mask = self._filter_mask(filters)
        fused = self._fuse(queries, embedder.encode_queries(queries), top_k, mask)
        phrases = query_phrases(query)
        if phrases:
            fused += 0.08 * minmax([sum(phrase in text.lower() for phrase in phrases) for text in self.search_texts])
        pool = self._top(fused, min(top_k, int(mask.sum())), mask)
        scored = self._rerank(query, pool, fused)
        picked = self._pick(scored, final_k, needs_source_diversity(query))
        return [(self.texts[index], float(score), self.meta[index]) for index, score in picked]

    def _fuse(self, queries, query_vectors, top_k, mask):
        fused, depth = np.zeros(len(self.texts), dtype="float32"), min(len(self.texts), max(top_k, 20))
        for number, (query, vector) in enumerate(zip(queries, query_vectors)):
            query_weight = 1.4 if number == 0 else 1.0
            dense, lexical = self.embeddings @ vector, self.bm25.get_scores(tokenize(query))
            for scores, weight in [(dense, 1 - self.bm25_weight), (lexical, self.bm25_weight)]:
                for rank, index in enumerate(self._top(scores, depth, mask), 1):
                    fused[index] += query_weight * weight / (RRF_K + rank)
        return minmax(fused)

    def _filter_mask(self, filters):
        mask = np.ones(len(self.texts), dtype=bool)
        for key, expected in (filters or {}).items():
            accepted = {str(value) for value in expected} if isinstance(expected, (list, tuple, set)) else {str(expected)}
            mask &= np.asarray([str(meta.get(key, "")) in accepted for meta in self.meta])
        return mask

    @staticmethod
    def _top(scores, count, mask):
        indices = np.flatnonzero(mask)
        return indices[np.argsort(np.asarray(scores)[indices])[::-1][:count]] if count and len(indices) else np.asarray([], dtype=int)

    def _rerank(self, query, indices, base_scores):
        ids = list(indices[:RERANK_POOL])
        base = minmax([base_scores[index] for index in ids])
        if not ids or self.reranker_model is None:
            return sorted(zip(ids, base), key=lambda item: item[1], reverse=True)
        with torch.inference_mode():
            inputs = self.reranker_tokenizer([[query, self.search_texts[index]] for index in ids], padding=True, truncation=True, max_length=512, return_tensors="pt")
            logits = self.reranker_model(**{key: value.to(self.device) for key, value in inputs.items()}).logits.cpu().numpy()
        logits = logits[:, -1] if logits.ndim > 1 and logits.shape[-1] > 1 else logits.reshape(-1)
        scores = 0.65 * minmax(logits) + 0.35 * base
        return sorted(zip(ids, scores), key=lambda item: item[1], reverse=True)

    def _pick(self, scored, final_k, diversify):
        remaining, picked, counts = list(scored), [], Counter()
        cap = max(1, (final_k + 1) // 2) if diversify else final_k
        while remaining and len(picked) < final_k:
            eligible = [item for item in remaining if counts[self.meta[item[0]].get("filename", "")] < cap] or remaining

            def utility(item):
                index, relevance = item
                same_source = [
                    chosen
                    for chosen, _ in picked
                    if self.meta[index].get("filename") == self.meta[chosen].get("filename")
                ]
                redundancy = max((float(self.embeddings[index] @ self.embeddings[chosen]) for chosen in same_source), default=0.0)
                adjacent = any(
                    self.meta[index].get("filename") == self.meta[chosen].get("filename")
                    and abs(int(self.meta[index].get("chunk_id", 0)) - int(self.meta[chosen].get("chunk_id", 0))) <= 1
                    for chosen, _ in picked
                )
                return 0.86 * float(relevance) - 0.14 * max(0.0, redundancy) - 0.04 * adjacent

            choice = max(eligible, key=utility)
            remaining.remove(choice)
            picked.append(choice)
            counts[self.meta[choice[0]].get("filename", "")] += 1
        return picked


class PolicyAnalystGenerator:
    def __init__(self, model_name=SUMMARIZER_MODEL):
        self.device = torch_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device).eval()

    def rewrite_queries(self, query, count=3):
        prompt = f"Write {count} short, distinct search queries that retrieve evidence for this question. One query per line.\nQuestion: {query}\nQueries:"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(self.device)
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=64, num_beams=max(3, count), num_return_sequences=count)
        return dedupe(self.tokenizer.decode(output, skip_special_tokens=True).strip(" -0123456789.\n") for output in outputs)

    def generate_briefing(self, query, context_passages, metas, subject, intent):
        if not context_passages:
            return {"answer": "No relevant documents found.", "matches": []}
        sources = []
        for number, (text, meta) in enumerate(zip(context_passages, metas), 1):
            location = f", page {meta['page']}" if meta.get("page") else ""
            sources.append(f"[{number}] {meta.get('filename', 'Unknown')}{location}: {trim_words(text, 170)}")
        prompt = (
            "Answer the question directly in complete sentences, not as a title. Include every requested item. "
            "Use only the evidence, cite claims with [source numbers], preserve exact technical terms, state uncertainty, "
            "and compare sources when asked.\n"
            f"Subject: {subject}\nIntent: {intent}\nQuestion: {query}\n\nEvidence:\n" + "\n\n".join(sources) + "\n\nAnswer:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=300, min_new_tokens=16, num_beams=4, no_repeat_ngram_size=3, repetition_penalty=1.15, early_stopping=True)
        answer = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        answer = re.sub(r"\s+(?:What|How|Why|Which|Who|Where|When|Are|Do|Does|Did|Can|Could|Should|Would)\b[^?]*\?\s*$", "", answer)
        return {"answer": answer, "matches": metas}


def trim_words(text, limit):
    words = text.split()
    return " ".join(words[:limit]) + (" ..." if len(words) > limit else "")


def configured_sources():
    urls = [url.strip() for url in os.getenv("RAG_SOURCE_URLS", "").split(",") if url.strip()]
    return [URLSource(urls)] if urls else []


def build_corpus(data_dir=DATA_DIR, sources=None):
    docs, files = DocumentLoader(data_dir).load_all_documents()
    for source in configured_sources() if sources is None else sources:
        live_docs = source.load()
        docs.extend(live_docs)
        files.extend(doc["filename"] for doc in live_docs)
    chunks, meta = [], []
    for doc in docs:
        for chunk_id, record in enumerate(chunk_records(doc["raw_text"])):
            if not record["section"] and doc.get("section_hint"):
                record["section"] = doc["section_hint"]
                record["text"] = f"{record['section']}\n{record['text']}"
            chunks.append(record["text"])
            meta.append({key: value for key, value in {**doc, **record, "chunk_id": chunk_id}.items() if key not in {"raw_text", "text", "section_hint"} and value not in (None, "")})
    return chunks, meta, dedupe(files)


class EmbeddingCache:
    def __init__(self, data_dir):
        self.path = Path(os.getenv("RAG_CACHE_DIR", str(Path(data_dir) / ".rag_cache"))) / "embeddings.npz"

    @staticmethod
    def key(texts, meta, model_name):
        digest = hashlib.sha256(model_name.encode())
        for text, item in zip(texts, meta):
            digest.update(text.encode("utf-8", errors="ignore"))
            stable_meta = {key: value for key, value in item.items() if key not in {"fetched_at", "score"}}
            digest.update(json.dumps(stable_meta, sort_keys=True, default=str).encode())
        return digest.hexdigest()

    def load(self, key):
        try:
            with np.load(self.path, allow_pickle=False) as data:
                return data["embeddings"] if str(data["key"].item()) == key else None
        except (OSError, KeyError, ValueError):
            return None

    def save(self, key, embeddings):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(f".{self.path.stem}.{os.getpid()}.npz")
            np.savez_compressed(temporary, key=key, embeddings=embeddings)
            os.replace(temporary, self.path)
        except OSError as exc:
            logger.warning("Could not cache embeddings: %s", exc)


class RAGSystem:
    def __init__(self, data_dir=DATA_DIR, load_generator=True, sources=None, use_reranker=True, use_cache=True, embedder_model=EMBEDDER_MODEL):
        self.data_dir, self.sources = data_dir, sources
        self.use_reranker, self.use_cache, self._lock = use_reranker, use_cache, threading.RLock()
        self.embedder, self.retriever = Embedder(embedder_model), None
        self.files = self.refresh()
        self.generator = PolicyAnalystGenerator() if load_generator else None

    def refresh(self):
        with self._lock:
            chunks, meta, files = build_corpus(self.data_dir, self.sources)
            if not chunks:
                raise ValueError(f"No readable documents found in {self.data_dir}")
            cache, embeddings = EmbeddingCache(self.data_dir), None
            key = cache.key(chunks, meta, self.embedder.model_name)
            if self.use_cache:
                embeddings = cache.load(key)
            if embeddings is None:
                embeddings = self.embedder.encode_documents([contextualize(text, item) for text, item in zip(chunks, meta)])
                if self.use_cache:
                    cache.save(key, embeddings)
            reranker = self.retriever.reranker if self.retriever else None
            self.retriever = HybridRetriever(embeddings, chunks, meta, use_reranker=self.use_reranker, reranker=reranker)
            self.files = files
            return files

    def answer(self, query, filters=None, final_k=TOP_K_FINAL):
        with self._lock:
            if self.generator is None:
                raise RuntimeError("Generation is disabled")
            generated = self.generator.rewrite_queries(query) if os.getenv("RAG_GENERATIVE_MULTIQUERY", "0") == "1" else []
            return answer_query(query, self.embedder, self.retriever, self.generator, expand_query(query, generated), filters, final_k)


def build_rag_system(data_dir=DATA_DIR, load_generator=True, sources=None, use_reranker=True, use_cache=True):
    system = RAGSystem(data_dir, load_generator, sources, use_reranker, use_cache)
    return system.embedder, system.retriever, system.generator, system.files


def answer_query(query, embedder, retriever, generator, expanded_queries=None, filters=None, final_k=TOP_K_FINAL):
    variants = expanded_queries or expand_query(query)
    results = retriever.search(query, variants, embedder, final_k=final_k, filters=filters)
    metas = [{**meta, "score": round(score, 4)} for _, score, meta in results]
    subject, intent = infer_policy_subject_and_intent(query)
    response = generator.generate_briefing(query, [text for text, _, _ in results], metas, subject, intent)
    response["query_variants"] = variants
    return response


def main():
    try:
        system = RAGSystem(DATA_DIR)
    except Exception as exc:
        print(f"Error initializing RAG system: {exc}")
        sys.exit(1)
    print(f"AI Policy Insights RAG\nLoaded {len(system.files)} source(s): {', '.join(system.files)}")
    while True:
        query = input("\nQuery (or 'q' to quit): ").strip()
        if query.lower() in {"q", "quit"}:
            break
        if query:
            response = system.answer(query)
            print(f"\nMemo\n----\n{response['answer']}\n\nSources\n-------")
            for meta in response["matches"]:
                location = f" (page {meta['page']})" if meta.get("page") else ""
                print(f"{meta['filename']}{location}")


if __name__ == "__main__":
    main()
