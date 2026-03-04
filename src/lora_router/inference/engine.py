"""GPU inference engine for LLaMA-2 with PEFT LoRA adapter swapping.

Loads a base model once to GPU, then hot-swaps LoRA adapters for
per-task inference. Provides the `inference_fn` callback that
`FlanV2Benchmark.evaluate_full()` expects.

Usage:
    engine = InferenceEngine("meta-llama/Llama-2-7b-hf")
    engine.load_adapter("arc_challenge", "adapters/flan_v2/arc_challenge")
    engine.swap_adapter("arc_challenge")
    output = engine.generate("What is the speed of light?")

    # Or as a callback for evaluate_full():
    inference_fn = engine.create_inference_fn()
    result = benchmark.evaluate_full(strategy, registry, inference_fn)
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Manages LLaMA-2 base model with hot-swappable LoRA adapters via PEFT.

    Loads the base model once to GPU. Adapters are loaded on demand and
    swapped without reloading the base weights.
    """

    def __init__(
        self,
        base_model: str = "meta-llama/Llama-2-7b-hf",
        device: str = "auto",
        torch_dtype: str = "float16",
        max_new_tokens: int = 128,
        load_in_8bit: bool = False,
        token: str | None = None,
    ) -> None:
        """Initialize the inference engine.

        Args:
            base_model: HuggingFace model ID or local path.
            device: Device map for accelerate ("auto", "cuda:0", etc.).
            torch_dtype: Model precision ("float16", "bfloat16", "float32").
            max_new_tokens: Default max tokens for generation.
            load_in_8bit: Whether to use 8-bit quantization (saves VRAM).
            token: HuggingFace token for gated models like LLaMA-2.
        """
        self.base_model_name = base_model
        self.device = device
        self.torch_dtype_str = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.load_in_8bit = load_in_8bit
        self.token = token

        self.model: Any = None
        self.tokenizer: Any = None
        self._loaded_adapters: dict[str, str] = {}  # name -> path
        self._active_adapter: str | None = None
        self._base_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Whether the base model is loaded."""
        return self._base_loaded

    @property
    def active_adapter(self) -> str | None:
        """Currently active adapter name."""
        return self._active_adapter

    @property
    def loaded_adapters(self) -> list[str]:
        """List of loaded adapter names."""
        return list(self._loaded_adapters.keys())

    def load_base_model(self) -> None:
        """Load the base LLaMA-2 model and tokenizer to GPU."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.torch_dtype_str, torch.float16)

        logger.info("Loading base model: %s", self.base_model_name)
        start = time.perf_counter()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            token=self.token,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: dict[str, Any] = {
            "device_map": self.device,
            "torch_dtype": torch_dtype,
            "token": self.token,
        }
        if self.load_in_8bit:
            load_kwargs["load_in_8bit"] = True

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            **load_kwargs,
        )

        self._base_loaded = True
        elapsed = time.perf_counter() - start
        logger.info("Base model loaded in %.1fs", elapsed)

    def load_adapter(self, name: str, path: str) -> None:
        """Load a LoRA adapter into the model.

        First adapter loaded uses PeftModel.from_pretrained().
        Subsequent adapters use model.load_adapter().

        Args:
            name: Unique adapter name (e.g., "arc_challenge").
            path: Local path or HuggingFace ID for the adapter.
        """
        if not self._base_loaded:
            raise RuntimeError("Base model not loaded. Call load_base_model() first.")

        if name in self._loaded_adapters:
            logger.debug("Adapter '%s' already loaded, skipping", name)
            return

        from peft import PeftModel

        logger.info("Loading adapter: %s from %s", name, path)
        start = time.perf_counter()

        if not isinstance(self.model, PeftModel):
            # First adapter - wrap base model with PEFT
            self.model = PeftModel.from_pretrained(
                self.model,
                path,
                adapter_name=name,
            )
        else:
            # Subsequent adapters - add to existing PEFT model
            self.model.load_adapter(path, adapter_name=name)

        self._loaded_adapters[name] = path
        self._active_adapter = name
        elapsed = time.perf_counter() - start
        logger.debug("Adapter '%s' loaded in %.2fs", name, elapsed)

    def load_adapters_from_dir(self, adapter_dir: str, task_names: list[str]) -> int:
        """Load multiple adapters from a directory.

        Expects each adapter in adapter_dir/{task_name}/.

        Args:
            adapter_dir: Parent directory containing adapter subdirs.
            task_names: List of task/adapter names to load.

        Returns:
            Number of adapters successfully loaded.
        """
        from pathlib import Path

        adapter_path = Path(adapter_dir)
        loaded = 0

        for name in task_names:
            path = adapter_path / name
            if not path.exists():
                logger.warning("Adapter dir not found: %s", path)
                continue
            try:
                self.load_adapter(name, str(path))
                loaded += 1
            except Exception as e:
                logger.error("Failed to load adapter '%s': %s", name, e)

        logger.info("Loaded %d/%d adapters from %s", loaded, len(task_names), adapter_dir)
        return loaded

    def swap_adapter(self, name: str) -> None:
        """Switch the active LoRA adapter.

        Args:
            name: Adapter name (must be previously loaded).

        Raises:
            KeyError: If adapter is not loaded.
        """
        if name not in self._loaded_adapters:
            raise KeyError(
                f"Adapter '{name}' not loaded. "
                f"Available: {list(self._loaded_adapters.keys())}"
            )

        if self._active_adapter == name:
            return

        self.model.set_adapter(name)
        self._active_adapter = name

    def generate(
        self,
        input_text: str,
        max_new_tokens: int | None = None,
    ) -> str:
        """Generate text using the currently active adapter.

        Args:
            input_text: Input prompt text.
            max_new_tokens: Override default max tokens.

        Returns:
            Generated text (decoded, without the input prompt).
        """
        import torch

        if not self._base_loaded:
            raise RuntimeError("Base model not loaded. Call load_base_model() first.")

        tokens = max_new_tokens or self.max_new_tokens

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.model.device)

        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the new tokens (skip the input)
        generated = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True,
        )
        return generated.strip()

    def create_inference_fn(self) -> Any:
        """Create the inference callback for evaluate_full().

        Returns a callable with signature:
            inference_fn(input_text: str, adapter_name: str) -> str

        The function swaps to the requested adapter and generates output.
        If the adapter isn't loaded, it returns an empty string with a warning.
        """

        def inference_fn(input_text: str, adapter_name: str) -> str:
            if adapter_name not in self._loaded_adapters:
                logger.warning(
                    "Adapter '%s' not loaded, returning empty. "
                    "Available: %s",
                    adapter_name,
                    list(self._loaded_adapters.keys()),
                )
                return ""
            self.swap_adapter(adapter_name)
            return self.generate(input_text)

        return inference_fn

    def unload(self) -> None:
        """Unload model and free GPU memory."""
        import gc

        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        self._loaded_adapters.clear()
        self._active_adapter = None
        self._base_loaded = False

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        logger.info("Model unloaded, GPU memory freed")

    def gpu_memory_info(self) -> dict[str, float]:
        """Get current GPU memory usage in GB.

        Returns:
            Dict with allocated, reserved, and total memory in GB.
        """
        try:
            import torch

            if not torch.cuda.is_available():
                return {"error": -1.0}
            return {
                "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "reserved_gb": torch.cuda.memory_reserved() / 1e9,
                "total_gb": torch.cuda.get_device_properties(0).total_mem / 1e9,
            }
        except ImportError:
            return {"error": -1.0}

    def __repr__(self) -> str:
        status = "loaded" if self._base_loaded else "unloaded"
        n_adapters = len(self._loaded_adapters)
        active = self._active_adapter or "none"
        return (
            f"InferenceEngine(model={self.base_model_name!r}, "
            f"status={status}, adapters={n_adapters}, active={active})"
        )
