"""
TF-IDF retriever with cosine similarity for RAG.

Provides a simple interface:
- fit(index_dict)
- query(question, top_k)

index_dict schema (from common.build_index):
{
  "chunks": List[str],
  "metadata": List[dict]
}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievalResult:
    chunk: str
    score: float
    metadata: dict


class TfidfRetriever:
    def __init__(self, max_features: int = 50000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words="english",
        )
        self.doc_matrix = None
        self.chunks: List[str] = []
        self.metadata: List[dict] = []

    def fit(self, index: dict) -> None:
        self.chunks = index.get("chunks", [])
        self.metadata = index.get("metadata", [{} for _ in self.chunks])
        if not self.chunks:
            # create a dummy matrix to avoid downstream errors
            self.doc_matrix = None
            return
        self.doc_matrix = self.vectorizer.fit_transform(self.chunks)

    def query(self, question: str, top_k: int = 5) -> List[RetrievalResult]:
        if not question:
            return []
        if self.doc_matrix is None or len(self.chunks) == 0:
            return []
        q_vec = self.vectorizer.transform([question])
        sims = cosine_similarity(q_vec, self.doc_matrix).flatten()
        if sims.size == 0:
            return []
        top_idx = np.argsort(-sims)[:top_k]
        results: List[RetrievalResult] = []
        for i in top_idx:
            results.append(
                RetrievalResult(
                    chunk=self.chunks[int(i)],
                    score=float(sims[int(i)]),
                    metadata=self.metadata[int(i)] if int(i) < len(self.metadata) else {},
                )
            )
        return results


class EmbeddingsRetriever:
    """Retriever using OpenAI text-embedding-3-small embeddings + cosine similarity.

    Requires OPENAI_API_KEY. Falls back silently if not available.
    """

    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self.chunks: List[str] = []
        self.metadata: List[dict] = []
        self.embeddings: np.ndarray | None = None

    def _embed(self, texts: List[str]) -> np.ndarray:
        api_key = None
        try:
            import os
            api_key = os.getenv("OPENAI_API_KEY")
        except Exception:
            api_key = None
        if not api_key:
            return np.zeros((len(texts), 1), dtype=np.float32)
        try:
            from openai import OpenAI  # type: ignore

            client = OpenAI(api_key=api_key)
            # Batch call
            resp = client.embeddings.create(model=self.model, input=texts)
            vecs = [d.embedding for d in resp.data]
            return np.array(vecs, dtype=np.float32)
        except Exception:
            # Legacy fallback
            try:
                import openai  # type: ignore
                openai.api_key = api_key
                resp = openai.Embedding.create(model=self.model, input=texts)
                vecs = [d["embedding"] for d in resp["data"]]
                return np.array(vecs, dtype=np.float32)
            except Exception:
                return np.zeros((len(texts), 1), dtype=np.float32)

    def fit(self, index: dict) -> None:
        self.chunks = index.get("chunks", [])
        self.metadata = index.get("metadata", [{} for _ in self.chunks])
        if not self.chunks:
            self.embeddings = None
            return
        self.embeddings = self._embed(self.chunks)

    def query(self, question: str, top_k: int = 5) -> List[RetrievalResult]:
        if not question or self.embeddings is None or len(self.chunks) == 0:
            return []
        q = self._embed([question])
        if q.shape[1] != self.embeddings.shape[1]:
            return []
        # cosine similarity
        a = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
        b = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-9)
        sims = (a @ b.T).flatten()
        top_idx = np.argsort(-sims)[:top_k]
        results: List[RetrievalResult] = []
        for i in top_idx:
            results.append(
                RetrievalResult(
                    chunk=self.chunks[int(i)],
                    score=float(sims[int(i)]),
                    metadata=self.metadata[int(i)] if int(i) < len(self.metadata) else {},
                )
            )
        return results


