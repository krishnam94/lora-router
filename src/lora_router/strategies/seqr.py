"""SEQR-inspired spectral routing strategy.

Training-free, data-free routing using QR decomposition of adapter weight matrices.
Routes by computing activation norms - the adapter whose subspace captures the most
energy from the input hidden state is selected.

Based on: SEQR (arXiv:2509.18093) and SpectR (arXiv:2504.03454).
"""

from __future__ import annotations

import numpy as np

from lora_router.registry import AdapterRegistry
from lora_router.strategies.base import BaseStrategy
from lora_router.types import AdapterSelection


class SEQRStrategy(BaseStrategy):
    """Route by spectral analysis of adapter weight matrices.

    For each adapter, computes the QR decomposition of the B matrix (or B*A product).
    Routes by projecting the query representation into each adapter's subspace
    and selecting the adapter with the highest activation norm.

    This strategy requires pre-computed adapter weight signatures stored in the
    registry's adapter metadata under the key 'weight_signature' (a dict mapping
    layer names to numpy arrays of shape (rank, hidden_dim)).

    For production use, call `compute_signatures()` to extract signatures from
    PEFT adapter weights.

    Args:
        temperature: Softmax temperature for confidence calibration.
        use_qr: If True, use QR decomposition for efficiency. If False, use full SVD.
    """

    def __init__(self, temperature: float = 0.2, use_qr: bool = True) -> None:
        self._temperature = temperature
        self._use_qr = use_qr
        self._signatures: dict[str, np.ndarray] = {}

    def set_signatures(self, signatures: dict[str, np.ndarray]) -> None:
        """Set pre-computed adapter signatures (adapter_name -> projection_matrix).

        Each signature is a matrix of shape (rank, hidden_dim) representing the
        adapter's subspace. Can be computed from adapter weights via
        `compute_signature_from_weights()`.
        """
        self._signatures = signatures

    @staticmethod
    def compute_signature_from_weights(
        lora_A: np.ndarray, lora_B: np.ndarray, use_qr: bool = True
    ) -> np.ndarray:
        """Compute a routing signature from LoRA weight matrices.

        Args:
            lora_A: LoRA A matrix, shape (rank, in_features).
            lora_B: LoRA B matrix, shape (out_features, rank).
            use_qr: Use QR decomposition (faster) vs full SVD.

        Returns:
            Projection matrix of shape (rank, in_features) for routing.
        """
        product = lora_B @ lora_A  # (out_features, in_features)
        if use_qr:
            Q, R = np.linalg.qr(product.T)  # Q: (in_features, rank)
            return R  # (rank, in_features) - the projection basis
        else:
            U, S, Vt = np.linalg.svd(product, full_matrices=False)
            return np.diag(S) @ Vt  # S * Vt: (rank, in_features)

    @staticmethod
    def compute_signatures_from_peft(
        model_name_or_path: str,
        adapter_names: list[str],
        target_module: str = "q_proj",
        layer_index: int = 0,
        use_qr: bool = True,
    ) -> dict[str, np.ndarray]:
        """Extract routing signatures from PEFT adapters on HuggingFace Hub.

        Args:
            model_name_or_path: Base model name/path.
            adapter_names: List of adapter HF IDs or local paths.
            target_module: Which LoRA target module to extract (e.g., 'q_proj').
            layer_index: Which transformer layer to use for the signature.
            use_qr: Use QR vs SVD decomposition.

        Returns:
            Dict mapping adapter name to signature matrix.
        """
        from peft import PeftConfig
        from safetensors.torch import load_file

        signatures = {}
        for adapter_path in adapter_names:
            try:
                PeftConfig.from_pretrained(adapter_path)
                weights = load_file(f"{adapter_path}/adapter_model.safetensors")

                a_key = f"base_model.model.model.layers.{layer_index}.self_attn.{target_module}.lora_A.weight"
                b_key = f"base_model.model.model.layers.{layer_index}.self_attn.{target_module}.lora_B.weight"

                if a_key in weights and b_key in weights:
                    lora_A = weights[a_key].numpy()
                    lora_B = weights[b_key].numpy()
                    sig = SEQRStrategy.compute_signature_from_weights(
                        lora_A, lora_B, use_qr=use_qr
                    )
                    name = adapter_path.split("/")[-1] if "/" in adapter_path else adapter_path
                    signatures[name] = sig
            except Exception:
                continue
        return signatures

    def _compute_activation_norm(self, query_vec: np.ndarray, signature: np.ndarray) -> float:
        """Compute the activation norm of a query projected into an adapter's subspace."""
        projection = signature @ query_vec  # (rank,)
        return float(np.linalg.norm(projection))

    def _softmax_confidence(self, scores: np.ndarray) -> np.ndarray:
        scaled = scores / self._temperature
        scaled -= scaled.max()
        exp_scores = np.exp(scaled)
        return exp_scores / (exp_scores.sum() + 1e-8)

    def route(
        self, query: str, registry: AdapterRegistry, top_k: int = 5
    ) -> list[AdapterSelection]:
        if not self._signatures:
            return []

        # Use query embedding as the hidden state proxy
        names, matrix = registry.get_embedding_matrix()
        if len(names) == 0:
            return []

        # Get query embedding from registry's embedder
        query_emb: np.ndarray | None = None
        if registry._embedder is not None:
            emb = registry._embedder.encode([query])
            query_emb = np.asarray(emb[0], dtype=np.float32)
        else:
            return []

        # Compute activation norms for each adapter with a signature
        adapter_scores: list[tuple[str, float]] = []
        for name in names:
            if name in self._signatures:
                score = self._compute_activation_norm(query_emb, self._signatures[name])
                adapter_scores.append((name, score))

        if not adapter_scores:
            return []

        adapter_scores.sort(key=lambda x: x[1], reverse=True)
        adapter_scores = adapter_scores[:top_k]

        score_values = np.array([s for _, s in adapter_scores])
        confidences = self._softmax_confidence(score_values)

        return [
            AdapterSelection(
                adapter_name=name,
                confidence=float(conf),
                scores={"seqr": float(score)},
            )
            for (name, score), conf in zip(adapter_scores, confidences)
        ]
