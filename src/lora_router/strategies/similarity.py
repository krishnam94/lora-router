"""Embedding cosine similarity routing strategy.

Default zero-training strategy. Embeds the query and computes cosine similarity
against cached adapter embeddings. Supports pluggable sentence-transformer models
and optional FAISS indexing for scale.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lora_router.registry import AdapterRegistry
from lora_router.strategies.base import BaseStrategy
from lora_router.types import AdapterSelection


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between vector a and matrix b (rows)."""
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norms = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return b_norms @ a_norm


class SimilarityStrategy(BaseStrategy):
    """Route by cosine similarity between query embedding and adapter embeddings.

    Args:
        encoder_name: SentenceTransformer model name.
        use_faiss: Whether to use FAISS for similarity search (better at scale).
        temperature: Softmax temperature for converting similarities to confidences.
    """

    def __init__(
        self,
        encoder_name: str = "all-MiniLM-L6-v2",
        use_faiss: bool = False,
        temperature: float = 0.2,
        encoder: Any = None,
    ) -> None:
        self._encoder_name = encoder_name
        self._use_faiss = use_faiss
        self._temperature = temperature
        self._encoder = encoder
        self._faiss_index: Any = None
        self._faiss_names: list[str] = []

    def _get_encoder(self) -> Any:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self._encoder_name)
        return self._encoder

    def _encode_query(self, query: str) -> np.ndarray:
        encoder = self._get_encoder()
        emb = encoder.encode([query])
        return np.asarray(emb[0], dtype=np.float32)

    def _encode_queries(self, queries: list[str]) -> np.ndarray:
        encoder = self._get_encoder()
        emb = encoder.encode(queries)
        return np.asarray(emb, dtype=np.float32)

    def _build_faiss_index(self, names: list[str], matrix: np.ndarray) -> None:
        import faiss

        dim = matrix.shape[1]
        norm_matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
        self._faiss_index = faiss.IndexFlatIP(dim)
        self._faiss_index.add(norm_matrix.astype(np.float32))
        self._faiss_names = names

    def _softmax_confidence(self, similarities: np.ndarray) -> np.ndarray:
        """Convert raw similarities to calibrated confidence scores via softmax."""
        scaled = similarities / self._temperature
        scaled -= scaled.max()
        exp_scores = np.exp(scaled)
        return exp_scores / (exp_scores.sum() + 1e-8)

    def route(
        self, query: str, registry: AdapterRegistry, top_k: int = 5
    ) -> list[AdapterSelection]:
        names, matrix = registry.get_embedding_matrix()
        if len(names) == 0:
            return []

        query_emb = self._encode_query(query)

        if self._use_faiss:
            if self._faiss_index is None or self._faiss_names != names:
                self._build_faiss_index(names, matrix)
            query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
            distances, indices = self._faiss_index.search(
                query_norm.reshape(1, -1), min(top_k, len(names))
            )
            selected_names = [self._faiss_names[i] for i in indices[0]]
            selected_sims = distances[0]
        else:
            similarities = _cosine_similarity(query_emb, matrix)
            k = min(top_k, len(names))
            top_indices = np.argsort(similarities)[::-1][:k]
            selected_names = [names[i] for i in top_indices]
            selected_sims = similarities[top_indices]

        confidences = self._softmax_confidence(selected_sims)

        return [
            AdapterSelection(
                adapter_name=name,
                confidence=float(conf),
                scores={"similarity": float(sim)},
            )
            for name, conf, sim in zip(selected_names, confidences, selected_sims)
        ]

    def route_batch(
        self, queries: list[str], registry: AdapterRegistry, top_k: int = 5
    ) -> list[list[AdapterSelection]]:
        names, matrix = registry.get_embedding_matrix()
        if len(names) == 0:
            return [[] for _ in queries]

        query_embs = self._encode_queries(queries)
        results = []

        for query_emb in query_embs:
            similarities = _cosine_similarity(query_emb, matrix)
            k = min(top_k, len(names))
            top_indices = np.argsort(similarities)[::-1][:k]
            selected_sims = similarities[top_indices]
            confidences = self._softmax_confidence(selected_sims)

            selections = [
                AdapterSelection(
                    adapter_name=names[idx],
                    confidence=float(conf),
                    scores={"similarity": float(sim)},
                )
                for idx, conf, sim in zip(top_indices, confidences, selected_sims)
            ]
            results.append(selections)

        return results
