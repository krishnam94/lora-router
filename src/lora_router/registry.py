"""Adapter registry with embedding cache and YAML import/export."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml

from lora_router.types import AdapterInfo


class Embedder(Protocol):
    """Protocol for text embedding models."""

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray: ...


class AdapterRegistry:
    """Registry for LoRA adapters with embedding cache.

    Stores adapter metadata, computes/caches embeddings for routing,
    and supports YAML-based import/export.
    """

    def __init__(self, embedder: Embedder | None = None) -> None:
        self._adapters: dict[str, AdapterInfo] = {}
        self._embeddings: dict[str, np.ndarray] = {}
        self._embedder = embedder

    @property
    def adapter_names(self) -> list[str]:
        return list(self._adapters.keys())

    @property
    def size(self) -> int:
        return len(self._adapters)

    def register(
        self,
        adapter: AdapterInfo,
        embedding: np.ndarray | None = None,
        compute_embedding: bool = True,
    ) -> None:
        """Register an adapter. Optionally provide or auto-compute its embedding."""
        self._adapters[adapter.name] = adapter
        if embedding is not None:
            self._embeddings[adapter.name] = np.asarray(embedding, dtype=np.float32)
        elif compute_embedding and self._embedder is not None:
            text = adapter.text_for_embedding
            emb = self._embedder.encode([text])
            self._embeddings[adapter.name] = np.asarray(emb[0], dtype=np.float32)

    def register_many(self, adapters: list[AdapterInfo], **kwargs: Any) -> None:
        """Register multiple adapters."""
        for adapter in adapters:
            self.register(adapter, **kwargs)

    def get(self, name: str) -> AdapterInfo:
        """Get adapter by name. Raises KeyError if not found."""
        return self._adapters[name]

    def get_embedding(self, name: str) -> np.ndarray | None:
        """Get cached embedding for an adapter."""
        return self._embeddings.get(name)

    def get_all_embeddings(self) -> dict[str, np.ndarray]:
        """Get all cached embeddings."""
        return dict(self._embeddings)

    def get_embedding_matrix(self) -> tuple[list[str], np.ndarray]:
        """Get names and stacked embedding matrix for adapters with embeddings.

        Returns:
            Tuple of (adapter_names, embedding_matrix) where matrix is (N, D).
        """
        names = []
        embeddings = []
        for name in self._adapters:
            if name in self._embeddings:
                names.append(name)
                embeddings.append(self._embeddings[name])
        if not embeddings:
            return [], np.array([])
        return names, np.stack(embeddings)

    def remove(self, name: str) -> None:
        """Remove an adapter from the registry."""
        self._adapters.pop(name, None)
        self._embeddings.pop(name, None)

    def list_adapters(self) -> list[AdapterInfo]:
        """List all registered adapters."""
        return list(self._adapters.values())

    def has(self, name: str) -> bool:
        return name in self._adapters

    def recompute_embeddings(self, names: list[str] | None = None) -> None:
        """Recompute embeddings for specified adapters (or all)."""
        if self._embedder is None:
            raise RuntimeError("No embedder set on registry")
        targets = names or list(self._adapters.keys())
        for name in targets:
            if name in self._adapters:
                text = self._adapters[name].text_for_embedding
                emb = self._embedder.encode([text])
                self._embeddings[name] = np.asarray(emb[0], dtype=np.float32)

    def set_embedder(self, embedder: Embedder) -> None:
        """Set or replace the embedding model."""
        self._embedder = embedder

    def to_yaml(self, path: str | Path) -> None:
        """Export registry to YAML file."""
        data = []
        for adapter in self._adapters.values():
            entry = adapter.model_dump()
            data.append(entry)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(
        cls, path: str | Path, embedder: Embedder | None = None
    ) -> AdapterRegistry:
        """Load registry from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        registry = cls(embedder=embedder)
        for entry in data:
            adapter = AdapterInfo(**entry)
            registry.register(adapter, compute_embedding=embedder is not None)
        return registry

    def exclude(self, names: set[str]) -> AdapterRegistry:
        """Create a new registry excluding specified adapter names.

        Shares the same embedder but copies adapter data. Used for OOD evaluation.
        """
        new_registry = AdapterRegistry(embedder=self._embedder)
        for name, adapter in self._adapters.items():
            if name not in names:
                embedding = self._embeddings.get(name)
                new_registry.register(adapter, embedding=embedding, compute_embedding=False)
        return new_registry
