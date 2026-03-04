"""Shared test fixtures for lora-router unit tests."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pytest

from lora_router.registry import AdapterRegistry
from lora_router.types import AdapterInfo


class MockEmbedder:
    """Deterministic embedder that produces consistent 64-dim embeddings from text hash.

    Uses SHA-256 hash of input text to seed numpy RNG, producing reproducible
    embeddings. This lets tests verify routing behavior without downloading models.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        embeddings = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
            rng = np.random.RandomState(seed)
            emb = rng.randn(self.dim).astype(np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            embeddings.append(emb)
        return np.array(embeddings, dtype=np.float32)


@pytest.fixture
def mock_embedder() -> MockEmbedder:
    """A deterministic mock embedder producing 64-dim vectors."""
    return MockEmbedder(dim=64)


@pytest.fixture
def sample_adapters() -> list[AdapterInfo]:
    """Six diverse AdapterInfo objects for testing."""
    return [
        AdapterInfo(
            name="code",
            path="adapters/code-v1",
            description="Generates and debugs Python, JavaScript, and Rust code",
            domain="code",
            example_queries=[
                "Write a Python function to sort a list",
                "Debug this JavaScript async function",
                "Implement a binary search tree in Rust",
            ],
        ),
        AdapterInfo(
            name="math",
            path="adapters/math-v1",
            description="Solves math problems including algebra, calculus, and statistics",
            domain="math",
            example_queries=[
                "Solve the integral of x^2 dx",
                "What is the derivative of sin(x)",
                "Calculate the standard deviation of this dataset",
            ],
        ),
        AdapterInfo(
            name="summarize",
            path="adapters/summarize-v1",
            description="Summarizes long documents, articles, and papers",
            domain="nlp",
            example_queries=[
                "Summarize this research paper in 3 sentences",
                "Give me the key points of this article",
            ],
        ),
        AdapterInfo(
            name="reasoning",
            path="adapters/reasoning-v1",
            description="Step-by-step logical reasoning and chain-of-thought",
            domain="reasoning",
            example_queries=[
                "Walk me through this logic puzzle step by step",
                "Analyze the premises and conclusion of this argument",
                "What logical fallacy is present in this statement",
            ],
        ),
        AdapterInfo(
            name="creative",
            path="adapters/creative-v1",
            description="Creative writing including stories, poems, and scripts",
            domain="creative",
            example_queries=[
                "Write a short story about a robot learning to paint",
                "Compose a haiku about the ocean",
            ],
        ),
        AdapterInfo(
            name="multilingual",
            path="adapters/multilingual-v1",
            description="Translates between English, Spanish, French, and German",
            domain="translation",
            example_queries=[
                "Translate this paragraph to Spanish",
                "How do you say 'good morning' in French",
                "Convert this German text to English",
            ],
        ),
    ]


@pytest.fixture
def mock_registry(
    mock_embedder: MockEmbedder, sample_adapters: list[AdapterInfo]
) -> AdapterRegistry:
    """AdapterRegistry with mock embedder and 6 sample adapters registered."""
    registry = AdapterRegistry(embedder=mock_embedder)
    registry.register_many(sample_adapters)
    return registry


@pytest.fixture
def sample_seqr_signatures(mock_embedder: MockEmbedder) -> dict[str, np.ndarray]:
    """Random projection matrices for SEQR testing.

    Returns adapter_name -> np.ndarray of shape (rank=8, hidden_dim=64).
    Each adapter gets a distinct, reproducible signature.
    """
    rng = np.random.RandomState(42)
    names = ["code", "math", "summarize", "reasoning", "creative", "multilingual"]
    signatures = {}
    for name in names:
        # Generate a random projection matrix of shape (rank=8, hidden_dim=64)
        raw = rng.randn(8, 64).astype(np.float32)
        # Normalize rows so they act as projection directions
        norms = np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8
        signatures[name] = (raw / norms).astype(np.float32)
    return signatures
