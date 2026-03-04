"""Unit tests for lora_router.registry module."""

from __future__ import annotations

import numpy as np
import pytest

from lora_router.registry import AdapterRegistry
from lora_router.types import AdapterInfo

pytestmark = pytest.mark.unit


class TestAdapterRegistry:
    def test_register_and_get(self, mock_embedder):
        """Register a single adapter and retrieve it."""
        registry = AdapterRegistry(embedder=mock_embedder)
        adapter = AdapterInfo(name="code", description="Code adapter")
        registry.register(adapter)
        retrieved = registry.get("code")
        assert retrieved.name == "code"
        assert retrieved.description == "Code adapter"

    def test_has(self, mock_embedder):
        """has() returns True for registered adapters, False otherwise."""
        registry = AdapterRegistry(embedder=mock_embedder)
        adapter = AdapterInfo(name="code")
        registry.register(adapter)
        assert registry.has("code") is True
        assert registry.has("math") is False

    def test_remove(self, mock_embedder):
        """remove() deletes adapter and its embedding."""
        registry = AdapterRegistry(embedder=mock_embedder)
        adapter = AdapterInfo(name="code", description="Code adapter")
        registry.register(adapter)
        assert registry.has("code")
        registry.remove("code")
        assert registry.has("code") is False
        assert registry.get_embedding("code") is None

    def test_remove_nonexistent_no_error(self, mock_embedder):
        """Removing a nonexistent adapter does not raise."""
        registry = AdapterRegistry(embedder=mock_embedder)
        registry.remove("nonexistent")  # Should not raise

    def test_list_adapters(self, mock_registry):
        """list_adapters() returns all registered AdapterInfo objects."""
        adapters = mock_registry.list_adapters()
        assert len(adapters) == 6
        names = {a.name for a in adapters}
        assert "code" in names
        assert "math" in names

    def test_size(self, mock_registry):
        """size property returns count of registered adapters."""
        assert mock_registry.size == 6

    def test_register_many(self, mock_embedder):
        """register_many() adds multiple adapters at once."""
        registry = AdapterRegistry(embedder=mock_embedder)
        adapters = [
            AdapterInfo(name="a", description="A"),
            AdapterInfo(name="b", description="B"),
            AdapterInfo(name="c", description="C"),
        ]
        registry.register_many(adapters)
        assert registry.size == 3
        assert registry.has("a")
        assert registry.has("b")
        assert registry.has("c")

    def test_get_embedding_matrix_shape(self, mock_registry):
        """get_embedding_matrix() returns (names, matrix) with correct shape."""
        names, matrix = mock_registry.get_embedding_matrix()
        assert len(names) == 6
        assert matrix.shape == (6, 64)  # 6 adapters, 64-dim embeddings

    def test_get_embedding_cached(self, mock_registry):
        """get_embedding() returns the cached embedding for a registered adapter."""
        emb = mock_registry.get_embedding("code")
        assert emb is not None
        assert emb.shape == (64,)
        assert emb.dtype == np.float32

    def test_get_embedding_missing_returns_none(self, mock_registry):
        """get_embedding() returns None for unregistered adapter names."""
        assert mock_registry.get_embedding("nonexistent") is None

    def test_get_raises_key_error(self, mock_registry):
        """get() raises KeyError for missing adapter name."""
        with pytest.raises(KeyError):
            mock_registry.get("nonexistent_adapter")

    def test_exclude_creates_filtered_registry(self, mock_registry):
        """exclude() returns a new registry without the excluded adapters."""
        filtered = mock_registry.exclude({"code", "math"})
        assert filtered.size == 4
        assert filtered.has("code") is False
        assert filtered.has("math") is False
        assert filtered.has("summarize") is True
        assert filtered.has("reasoning") is True

    def test_exclude_preserves_embeddings(self, mock_registry):
        """exclude() copies embeddings for remaining adapters."""
        filtered = mock_registry.exclude({"code"})
        emb = filtered.get_embedding("math")
        original_emb = mock_registry.get_embedding("math")
        assert emb is not None
        assert original_emb is not None
        np.testing.assert_array_equal(emb, original_emb)

    def test_yaml_round_trip(self, mock_embedder, sample_adapters, tmp_path):
        """to_yaml -> from_yaml preserves adapter metadata."""
        registry = AdapterRegistry(embedder=mock_embedder)
        registry.register_many(sample_adapters)

        yaml_path = tmp_path / "adapters.yaml"
        registry.to_yaml(yaml_path)

        loaded = AdapterRegistry.from_yaml(yaml_path, embedder=mock_embedder)
        assert loaded.size == registry.size
        for adapter in sample_adapters:
            assert loaded.has(adapter.name)
            loaded_adapter = loaded.get(adapter.name)
            assert loaded_adapter.description == adapter.description
            assert loaded_adapter.domain == adapter.domain
            assert loaded_adapter.example_queries == adapter.example_queries

    def test_adapter_names_property(self, mock_registry):
        """adapter_names returns list of all registered names."""
        names = mock_registry.adapter_names
        assert len(names) == 6
        assert set(names) == {"code", "math", "summarize", "reasoning", "creative", "multilingual"}

    def test_register_with_explicit_embedding(self, mock_embedder):
        """Providing an explicit embedding bypasses the embedder."""
        registry = AdapterRegistry(embedder=mock_embedder)
        custom_emb = np.ones(64, dtype=np.float32)
        adapter = AdapterInfo(name="custom")
        registry.register(adapter, embedding=custom_emb)

        retrieved = registry.get_embedding("custom")
        assert retrieved is not None
        np.testing.assert_array_almost_equal(retrieved, custom_emb)
