# AI Policy RAG

A concise local RAG pipeline for model cards, technical papers, and current web/JSON sources.

## What improved

- **Retrieval:** PDF page and section metadata, sentence-aware chunks, overlap at sentence boundaries, quality/fast embedding profiles, query/document prefixes for E5 and BGE models, BM25 + dense reciprocal-rank fusion, cross-encoder reranking, metadata filters, and semantic diversity.
- **GenAI:** bounded multi-query retrieval, optional model-generated query rewrites, grounded synthesis with numbered citations, and refreshable JSON/HTML/text sources.
- **Deployment:** cached embeddings, one loaded model set per process, health/query/refresh endpoints, environment configuration, and a container image suitable for horizontal scaling.

## Run locally

```bash
python -m pip install -r requirements.txt
python modelcard_rag.py
```

Run the API:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
curl -X POST http://localhost:8000/query \
  -H 'content-type: application/json' \
  -d '{"question":"Compare GPT-4 and Llama 2 safety testing"}'
```

Add current HTTP sources at startup, then call `POST /refresh` whenever they should be re-fetched:

```bash
RAG_SOURCE_URLS="https://example.org/feed.json,https://example.org/policy.html" uvicorn api:app
```

Supported JSON records use a `text`, `content`, `body`, or `description` field. Optional `title`, `url`, `updated_at`, and `model_id` fields become retrieval metadata.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `RAG_PROFILE` | `quality` | `quality` uses MPNet; `fast` uses MiniLM. |
| `RAG_EMBED_MODEL` | profile model | Any Sentence Transformers model; E5/BGE prefixes are automatic. |
| `RAG_GENERATOR_MODEL` | `google/flan-t5-large` | Grounded answer model. |
| `RAG_CHUNK_WORDS` | `180` | Maximum body words per chunk. |
| `RAG_CHUNK_OVERLAP` | `35` | Approximate whole-sentence overlap. |
| `RAG_TOP_K` | `6` | Evidence passages returned to generation. |
| `RAG_SOURCE_URLS` | empty | Comma-separated live JSON, HTML, or text URLs. |
| `RAG_GENERATIVE_MULTIQUERY` | `0` | Set to `1` for three model-generated query rewrites. |
| `RAG_CACHE_DIR` | `<data>/.rag_cache` | Persistent embedding cache location. |
| `RAG_OFFLINE` | `0` | Set to `1` when all model files are local; the container enables it. |

Keep one API worker per container because models are memory-heavy, mount `.rag_cache` on persistent storage, and scale containers horizontally. Protect `/refresh` behind authentication at the gateway in production.

## Test and containerize

```bash
python -m unittest test_unit.py
docker build -t ai-policy-rag .
docker run --rm -p 8000:8000 ai-policy-rag
```
