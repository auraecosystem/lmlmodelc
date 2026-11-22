# LMLModelC

LMLModelC is a native, local-first decoder-only language-model runtime. It provides an inspectable PyTorch reference architecture, explicit configuration, safe checkpoint loading, tokenization interfaces, generation, training utilities, and an optional ONNX Runtime adapter.

## Status

Initial runtime foundation. Checkpoint compatibility is strict and must be demonstrated against a real checkpoint; no pretrained weights or tokenizer are bundled.

## Install

```bash
python -m pip install -e ".[dev]"
```

## Quick start

```python
from lmlmodel import LMLConfig, LMLModel, LMLPipeline

config = LMLConfig(vocab_size=256, hidden_size=128, num_hidden_layers=4,
                   num_attention_heads=4, num_key_value_heads=4,
                   intermediate_size=512, max_position_embeddings=128)
model = LMLModel(config)
pipeline = LMLPipeline(model)
print(pipeline.generate([1, 42, 17], max_new_tokens=8))
```

The example initializes random weights. It is a smoke test, not a pretrained assistant.

## Checkpoints

Use a trusted checkpoint with a `config` mapping and a `state_dict` mapping. Loading uses `torch.load(weights_only=True)`; legacy pickle checkpoints are not loaded unsafely. State-dict loading is strict by default.

## Design principles

- Local-first execution; no network calls in the core runtime.
- Explicit architecture and tokenizer contracts.
- Safe-by-default checkpoint loading.
- Strict tensor-shape and state-dict validation.
- CPU support; CUDA is optional.
- Tests before performance claims.
