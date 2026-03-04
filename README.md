# lora-router

Intelligent LoRA adapter routing, composition, and serving.

Automatically routes queries to the best LoRA adapter using embedding similarity, spectral analysis, trained classifiers, or multi-signal ensembles. Supports confidence-based adaptive composition with TIES, DARE, and linear merging.

## Install

```bash
pip install lora-router
```

## Quick Start

```python
from lora_router import AdapterInfo, AdapterRegistry, LoRARouter
from lora_router.strategies import SimilarityStrategy

# Register adapters
registry = AdapterRegistry()
registry.register(AdapterInfo(name="code", description="Python coding tasks"))
registry.register(AdapterInfo(name="math", description="Mathematical reasoning"))

# Route a query
strategy = SimilarityStrategy()
router = LoRARouter(registry, strategy)
decision = router.route("Write a sorting algorithm")
print(decision.top_adapter, decision.top_confidence)
```

## License

Apache 2.0
