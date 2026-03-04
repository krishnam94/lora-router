"""Tests for the InferenceEngine - all mock-based, no GPU required.

Tests cover:
- Engine initialization and state management
- Adapter loading (first + subsequent)
- Adapter swapping
- Generation pipeline
- Inference function bridge for evaluate_full()
- Error handling
- Memory cleanup
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lora_router.inference.engine import InferenceEngine

# -- Fixtures --


@pytest.fixture
def engine() -> InferenceEngine:
    """Create a fresh InferenceEngine (unloaded)."""
    return InferenceEngine(
        base_model="meta-llama/Llama-2-7b-hf",
        device="auto",
        max_new_tokens=64,
    )


@pytest.fixture
def mock_model() -> MagicMock:
    """Create a mock model with generate() support."""
    import torch

    model = MagicMock()
    model.device = torch.device("cpu")
    model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    return model


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    """Create a mock tokenizer."""
    import torch

    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token_id = 0

    # Mock __call__ returns tensors with input_ids
    encoded = MagicMock()
    encoded.__getitem__ = lambda self, key: torch.tensor([[1, 2, 3]])
    encoded.to.return_value = encoded
    tokenizer.return_value = encoded

    tokenizer.decode.return_value = "The answer is 42."
    return tokenizer


@pytest.fixture
def loaded_engine(
    engine: InferenceEngine,
    mock_model: MagicMock,
    mock_tokenizer: MagicMock,
) -> InferenceEngine:
    """Create an engine with mocked base model loaded."""
    engine.model = mock_model
    engine.tokenizer = mock_tokenizer
    engine._base_loaded = True
    return engine


# -- Initialization tests --


class TestEngineInit:
    def test_default_init(self) -> None:
        engine = InferenceEngine()
        assert engine.base_model_name == "meta-llama/Llama-2-7b-hf"
        assert engine.device == "auto"
        assert engine.torch_dtype_str == "float16"
        assert engine.max_new_tokens == 128
        assert not engine.load_in_8bit
        assert engine.token is None

    def test_custom_init(self) -> None:
        engine = InferenceEngine(
            base_model="local/model",
            device="cuda:0",
            torch_dtype="bfloat16",
            max_new_tokens=256,
            load_in_8bit=True,
            token="hf_test",
        )
        assert engine.base_model_name == "local/model"
        assert engine.device == "cuda:0"
        assert engine.torch_dtype_str == "bfloat16"
        assert engine.max_new_tokens == 256
        assert engine.load_in_8bit
        assert engine.token == "hf_test"

    def test_unloaded_state(self, engine: InferenceEngine) -> None:
        assert not engine.is_loaded
        assert engine.active_adapter is None
        assert engine.loaded_adapters == []
        assert engine.model is None
        assert engine.tokenizer is None

    def test_repr_unloaded(self, engine: InferenceEngine) -> None:
        r = repr(engine)
        assert "unloaded" in r
        assert "adapters=0" in r
        assert "active=none" in r


# -- Base model loading tests --


class TestLoadBaseModel:
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_load_base_model(
        self,
        mock_tok_from: MagicMock,
        mock_model_from: MagicMock,
    ) -> None:
        engine = InferenceEngine(base_model="test/model", token="hf_test")

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tok_from.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model_from.return_value = mock_model

        engine.load_base_model()

        assert engine.is_loaded
        # Tokenizer pad_token should be set to eos_token
        assert mock_tokenizer.pad_token == "</s>"
        mock_tok_from.assert_called_once_with("test/model", token="hf_test")
        mock_model_from.assert_called_once()


# -- Adapter loading tests --


class TestAdapterLoading:
    def test_load_adapter_without_base_model(self, engine: InferenceEngine) -> None:
        with pytest.raises(RuntimeError, match="Base model not loaded"):
            engine.load_adapter("test", "/path/to/adapter")

    @patch("peft.PeftModel.from_pretrained")
    def test_load_first_adapter(
        self,
        mock_peft_from: MagicMock,
        loaded_engine: InferenceEngine,
    ) -> None:
        original_model = loaded_engine.model
        wrapped = MagicMock()
        mock_peft_from.return_value = wrapped

        loaded_engine.load_adapter("arc_challenge", "/path/arc_challenge")

        mock_peft_from.assert_called_once_with(
            original_model,
            "/path/arc_challenge",
            adapter_name="arc_challenge",
        )
        assert "arc_challenge" in loaded_engine.loaded_adapters
        assert loaded_engine.active_adapter == "arc_challenge"
        # Model should be wrapped with PeftModel
        assert loaded_engine.model == wrapped

    def test_load_second_adapter(self, loaded_engine: InferenceEngine) -> None:
        # Simulate first adapter already loaded (model is PeftModel)
        from peft import PeftModel

        loaded_engine.model = MagicMock(spec=PeftModel)
        loaded_engine._loaded_adapters["cb"] = "/path/cb"
        loaded_engine._active_adapter = "cb"

        loaded_engine.load_adapter("rte", "/path/rte")

        # Should use load_adapter, not from_pretrained
        loaded_engine.model.load_adapter.assert_called_once_with(
            "/path/rte", adapter_name="rte"
        )
        assert "rte" in loaded_engine.loaded_adapters
        assert loaded_engine.active_adapter == "rte"

    def test_skip_already_loaded(self, loaded_engine: InferenceEngine) -> None:
        loaded_engine._loaded_adapters["cb"] = "/path/cb"
        original_model = loaded_engine.model

        loaded_engine.load_adapter("cb", "/path/cb")

        # Model should not have been modified
        assert loaded_engine.model is original_model

    def test_load_adapters_from_dir(self, loaded_engine: InferenceEngine) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake adapter dirs
            for name in ["arc_challenge", "cb", "rte"]:
                (Path(tmpdir) / name).mkdir()

            with patch.object(loaded_engine, "load_adapter") as mock_load:
                count = loaded_engine.load_adapters_from_dir(
                    tmpdir, ["arc_challenge", "cb", "rte", "missing_task"]
                )

            # missing_task dir doesn't exist, so 3 loaded
            assert mock_load.call_count == 3
            assert count == 3

    def test_load_adapters_from_dir_handles_errors(self, loaded_engine: InferenceEngine) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "arc_challenge").mkdir()
            (Path(tmpdir) / "cb").mkdir()

            def side_effect(name: str, path: str) -> None:
                if name == "cb":
                    raise RuntimeError("PEFT error")

            with patch.object(loaded_engine, "load_adapter", side_effect=side_effect):
                count = loaded_engine.load_adapters_from_dir(tmpdir, ["arc_challenge", "cb"])

            assert count == 1


# -- Adapter swapping tests --


class TestAdapterSwapping:
    def test_swap_to_loaded_adapter(self, loaded_engine: InferenceEngine) -> None:
        loaded_engine._loaded_adapters["cb"] = "/path/cb"
        loaded_engine._loaded_adapters["rte"] = "/path/rte"
        loaded_engine._active_adapter = "cb"

        loaded_engine.swap_adapter("rte")

        loaded_engine.model.set_adapter.assert_called_once_with("rte")
        assert loaded_engine.active_adapter == "rte"

    def test_swap_to_same_adapter_noop(self, loaded_engine: InferenceEngine) -> None:
        loaded_engine._loaded_adapters["cb"] = "/path/cb"
        loaded_engine._active_adapter = "cb"

        loaded_engine.swap_adapter("cb")

        loaded_engine.model.set_adapter.assert_not_called()

    def test_swap_to_unloaded_adapter_raises(self, loaded_engine: InferenceEngine) -> None:
        with pytest.raises(KeyError, match="not loaded"):
            loaded_engine.swap_adapter("nonexistent")


# -- Generation tests --


class TestGeneration:
    def test_generate_basic(
        self,
        loaded_engine: InferenceEngine,
        mock_tokenizer: MagicMock,
    ) -> None:
        result = loaded_engine.generate("What is 2+2?")

        assert result == "The answer is 42."
        mock_tokenizer.assert_called_once()
        loaded_engine.model.generate.assert_called_once()

    def test_generate_custom_max_tokens(self, loaded_engine: InferenceEngine) -> None:
        loaded_engine.generate("Test", max_new_tokens=32)
        loaded_engine.model.generate.assert_called_once()

    def test_generate_without_loaded_model(self, engine: InferenceEngine) -> None:
        with pytest.raises(RuntimeError, match="Base model not loaded"):
            engine.generate("Hello")


# -- Inference function bridge tests --


class TestInferenceFn:
    def test_create_inference_fn(self, loaded_engine: InferenceEngine) -> None:
        loaded_engine._loaded_adapters["cb"] = "/path/cb"
        loaded_engine._active_adapter = "cb"

        fn = loaded_engine.create_inference_fn()
        assert callable(fn)

    def test_inference_fn_swaps_and_generates(self, loaded_engine: InferenceEngine) -> None:
        loaded_engine._loaded_adapters["cb"] = "/path/cb"
        loaded_engine._loaded_adapters["rte"] = "/path/rte"
        loaded_engine._active_adapter = "cb"

        fn = loaded_engine.create_inference_fn()

        with (
            patch.object(loaded_engine, "swap_adapter") as mock_swap,
            patch.object(loaded_engine, "generate", return_value="yes") as mock_gen,
        ):
            result = fn("Is this entailment?", "rte")

        mock_swap.assert_called_once_with("rte")
        mock_gen.assert_called_once_with("Is this entailment?")
        assert result == "yes"

    def test_inference_fn_missing_adapter_returns_empty(self, loaded_engine: InferenceEngine) -> None:
        fn = loaded_engine.create_inference_fn()
        result = fn("Test input", "nonexistent_adapter")
        assert result == ""


# -- Cleanup tests --


class TestUnload:
    def test_unload_clears_state(self, loaded_engine: InferenceEngine) -> None:
        loaded_engine._loaded_adapters["cb"] = "/path/cb"
        loaded_engine._active_adapter = "cb"

        loaded_engine.unload()

        assert loaded_engine.model is None
        assert loaded_engine.tokenizer is None
        assert not loaded_engine.is_loaded
        assert loaded_engine.active_adapter is None
        assert loaded_engine.loaded_adapters == []


# -- Repr tests --


class TestRepr:
    def test_repr_loaded_with_adapters(self, loaded_engine: InferenceEngine) -> None:
        loaded_engine._loaded_adapters["cb"] = "/path/cb"
        loaded_engine._loaded_adapters["rte"] = "/path/rte"
        loaded_engine._active_adapter = "cb"

        r = repr(loaded_engine)
        assert "loaded" in r
        assert "adapters=2" in r
        assert "active=cb" in r


# -- GPU memory info tests --


class TestGpuMemory:
    def test_gpu_memory_no_cuda(self, engine: InferenceEngine) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            info = engine.gpu_memory_info()
        assert "error" in info
