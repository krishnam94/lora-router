"""Trained classifier routing strategy.

Uses sklearn classifiers trained on adapter example queries to predict the best
adapter for a given input. Supports calibrated probability estimates as confidence
scores.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from lora_router.registry import AdapterRegistry
from lora_router.strategies.base import BaseStrategy
from lora_router.types import AdapterSelection


class ClassifierStrategy(BaseStrategy):
    """Route using a trained sklearn classifier.

    Trains on adapter example queries (from registry metadata). Each adapter's
    example queries become labeled training data. At inference, classifies the
    query and returns calibrated probability as confidence.

    Args:
        encoder: Embedding model with .encode() method. If None, uses registry's embedder.
        classifier: Pre-trained sklearn classifier. If None, trains LogisticRegression.
        calibrated: Whether to apply Platt scaling for calibrated probabilities.
    """

    def __init__(
        self,
        encoder: Any = None,
        classifier: Any = None,
        calibrated: bool = True,
    ) -> None:
        self._encoder = encoder
        self._classifier = classifier
        self._label_encoder = LabelEncoder()
        self._calibrated = calibrated
        self._is_trained = classifier is not None

    def train(self, registry: AdapterRegistry) -> None:
        """Train the classifier on adapter example queries from the registry.

        Requires adapters to have example_queries populated.
        """
        texts: list[str] = []
        labels: list[str] = []

        for adapter in registry.list_adapters():
            if adapter.has_examples:
                for query in adapter.example_queries:
                    texts.append(query)
                    labels.append(adapter.name)

        if len(texts) < 2 or len(set(labels)) < 2:
            raise ValueError(
                "Need at least 2 adapters with example_queries to train classifier"
            )

        encoder = self._encoder or registry._embedder
        if encoder is None:
            raise RuntimeError("No encoder available for training")

        embeddings = np.asarray(encoder.encode(texts), dtype=np.float32)
        encoded_labels = self._label_encoder.fit_transform(labels)

        base_clf = LogisticRegression(max_iter=1000, solver="lbfgs")

        if self._calibrated and len(texts) >= 10:
            clf = CalibratedClassifierCV(base_clf, cv=min(3, len(set(labels))))
            clf.fit(embeddings, encoded_labels)
        else:
            base_clf.fit(embeddings, encoded_labels)
            clf = base_clf

        self._classifier = clf
        self._is_trained = True

    def route(
        self, query: str, registry: AdapterRegistry, top_k: int = 5
    ) -> list[AdapterSelection]:
        if not self._is_trained or self._classifier is None:
            return []

        encoder = self._encoder or registry._embedder
        if encoder is None:
            return []

        query_emb = np.asarray(encoder.encode([query]), dtype=np.float32)
        probas = self._classifier.predict_proba(query_emb)[0]

        k = min(top_k, len(probas))
        top_indices = np.argsort(probas)[::-1][:k]

        classes = self._label_encoder.classes_
        return [
            AdapterSelection(
                adapter_name=str(classes[idx]),
                confidence=float(probas[idx]),
                scores={"classifier": float(probas[idx])},
            )
            for idx in top_indices
        ]
