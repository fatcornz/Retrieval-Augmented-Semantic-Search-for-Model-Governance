import unittest

import numpy as np

from modelcard_rag import HybridRetriever, chunk_records, expand_query


class TinyEmbedder:
    terms = ["tevv", "test", "evaluation", "verification", "validation", "recipe"]

    def _encode(self, texts):
        values = [[text.lower().count(term) for term in self.terms] for text in texts]
        values = np.asarray(values, dtype="float32")
        return values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-12)

    encode_documents = _encode
    encode_queries = _encode


class RAGUnitTests(unittest.TestCase):
    def test_chunks_preserve_sections_and_boundaries(self):
        text = "Testing and Evaluation\n\n" + "A complete sentence about model testing. " * 35
        chunks = chunk_records(text, max_words=45, overlap=10)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk["section"] == "Testing and Evaluation" for chunk in chunks))
        self.assertTrue(all(len(chunk["text"].split()) <= 50 for chunk in chunks))

    def test_multiquery_adds_domain_terms(self):
        variants = expand_query("How does NIST define TEVV?")
        self.assertGreater(len(variants), 1)
        self.assertIn("verification", " ".join(variants).lower())

    def test_fused_retrieval_finds_semantic_and_lexical_match(self):
        texts = ["TEVV means test evaluation verification and validation.", "A cooking recipe for bread."]
        meta = [{"filename": "nist.pdf", "chunk_id": 0}, {"filename": "food.txt", "chunk_id": 0}]
        embedder = TinyEmbedder()
        retriever = HybridRetriever(embedder.encode_documents(texts), texts, meta, use_reranker=False)
        result = retriever.search("Define TEVV", expand_query("Define TEVV"), embedder, final_k=1)
        self.assertEqual(result[0][2]["filename"], "nist.pdf")

    def test_diversity_does_not_penalize_matching_cross_source_evidence(self):
        embeddings = np.asarray([[1, 0], [0.99, 0.01], [0, 1]], dtype="float32")
        meta = [{"filename": "a.pdf", "chunk_id": 0}, {"filename": "b.pdf", "chunk_id": 0}, {"filename": "b.pdf", "chunk_id": 1}]
        retriever = HybridRetriever(embeddings, ["method a", "method b", "unrelated"], meta, use_reranker=False)
        picked = retriever._pick([(0, 1.0), (1, 0.9), (2, 0.85)], final_k=2, diversify=True)
        self.assertEqual([index for index, _ in picked], [0, 1])


if __name__ == "__main__":
    unittest.main()
