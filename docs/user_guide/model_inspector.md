---
description: Use ModelInspector to inspect trained Ludwig models — collect weights, generate summaries, and estimate feature importance without writing custom PyTorch code.
---

# Model Inspector

`ModelInspector` provides introspection utilities for analyzing trained Ludwig models. It
exposes weight collection, architecture summaries, and feature importance proxies without
requiring you to interact with raw PyTorch module trees.

## Getting a ModelInspector

After training, obtain the inspector from a loaded model:

```python
from ludwig.api import LudwigModel
from ludwig.model_inspector import ModelInspector

model = LudwigModel.load("results/experiment_run/model")

inspector = ModelInspector(
    model=model.model,
    config=model.config.to_dict(),
    training_set_metadata=model.training_set_metadata,
)
```

## Model summary

`model_summary()` returns parameter counts, memory footprint, layer inventory, and high-level
architecture information:

```python
summary = inspector.model_summary()
print(summary)
# {
#   'total_parameters': 4_831_042,
#   'trainable_parameters': 4_831_042,
#   'frozen_parameters': 0,
#   'model_size_mb': 18.43,
#   'layer_counts': {'Linear': 24, 'LayerNorm': 8, 'ReLU': 12, ...},
#   'model_type': 'ecd',
#   'combiner_type': 'concat',
#   'num_input_features': 5,
#   'num_output_features': 2,
# }
```

| Field | Description |
|-------|-------------|
| `total_parameters` | Total number of parameters (trainable + frozen) |
| `trainable_parameters` | Parameters that will receive gradient updates |
| `frozen_parameters` | Parameters frozen (e.g., pretrained encoder backbone) |
| `model_size_mb` | Approximate memory footprint in MB |
| `layer_counts` | Count of each PyTorch module type in the model |
| `model_type` | `ecd` or `llm` |
| `combiner_type` | The combiner used (e.g., `concat`, `transformer`, `hypernetwork`) |
| `num_input_features` | Number of input feature encoders |
| `num_output_features` | Number of output feature decoders |

## Collecting weights

`collect_weights()` returns parameter tensors as a list of metadata dicts. Use this for
weight analysis, debugging gradient flow, or building custom interpretability tools:

```python
# All parameters
weights = inspector.collect_weights()
for w in weights[:5]:
    print(w)
# {'name': 'input_features.text.encoder.weight', 'shape': [256, 128],
#  'dtype': 'torch.float32', 'requires_grad': True, 'num_elements': 32768}

# Specific parameters by name
encoder_weights = inspector.collect_weights(
    tensor_names=["input_features.text.encoder.weight"]
)
```

Each entry is a dict with:

| Key | Description |
|-----|-------------|
| `name` | Fully qualified parameter name (PyTorch `named_parameters()` format) |
| `shape` | List of dimension sizes |
| `dtype` | PyTorch dtype as string |
| `requires_grad` | Whether the parameter is trainable |
| `num_elements` | Total number of scalar elements |

!!! note
    `collect_weights()` returns metadata only, not the actual tensor values, to avoid
    copying large tensors unnecessarily. Access the underlying tensor directly with
    `dict(model.model.named_parameters())[name]`.

## Feature importance proxy

`feature_importance_proxy()` estimates feature importance from encoder weight magnitudes.
This is a rough proxy — for rigorous feature importance use SHAP or Ludwig's
`explain` module.

```python
importance = inspector.feature_importance_proxy()
print(importance)
# {'text_description': 0.92, 'age': 0.45, 'income': 1.0, 'category_col': 0.31}
```

Scores are normalized to `[0, 1]`. Higher means the encoder has larger weight magnitudes,
which loosely correlates with the feature being heavily used by the model.

## Practical use cases

### Checking how many parameters are frozen

Useful when fine-tuning with frozen encoders:

```python
summary = inspector.model_summary()
frozen_pct = summary["frozen_parameters"] / summary["total_parameters"] * 100
print(f"{frozen_pct:.1f}% of parameters are frozen")
```

### Finding large layers

```python
weights = inspector.collect_weights()
large = sorted(weights, key=lambda w: w["num_elements"], reverse=True)
for w in large[:10]:
    print(f"{w['name']}: {w['num_elements']:,} params  shape={w['shape']}")
```

### Quick sanity check after fine-tuning

```python
summary = inspector.model_summary()
assert summary["trainable_parameters"] > 0, "No trainable parameters!"
assert summary["model_size_mb"] < 2000, f"Model too large: {summary['model_size_mb']} MB"
print(f"Model OK: {summary['trainable_parameters']:,} trainable params")
```

## See also

- [LudwigModel Python API](api/LudwigModel.md) — `.train()`, `.predict()`, `.evaluate()`
- [Model Export](model_export.md) — export to ONNX, TorchScript, HuggingFace Hub
- [Visualizations](visualizations.md) — training curves, confusion matrices, feature rankings
