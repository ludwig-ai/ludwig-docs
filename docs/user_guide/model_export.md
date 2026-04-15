# Exporting Ludwig Models

Ludwig supports exporting trained models in several formats for deployment.

## Export Formats

### SafeTensors (Default)

SafeTensors is the default format for Ludwig model weights. It provides secure, zero-copy serialization.

```python
model = LudwigModel.load("results/experiment_run/model")
model.export_model("exported/", format="safetensors")
```

CLI:
```bash
ludwig export_model -m results/experiment_run/model -o exported/ -f safetensors
```

### torch.export

`torch.export` is the official replacement for TorchScript (deprecated in PyTorch 2.9). It captures the full computation graph as an `ExportedProgram` at the ATen operator level.

```python
model = LudwigModel.load("results/experiment_run/model")
model.export_model("exported/", format="torch_export")
```

The exported `.pt2` file can be:

- Loaded back via `torch.export.load()`
- Compiled with `torch.compile()` for runtime optimization
- Used as input to the dynamo-based ONNX exporter
- Deployed via ExecuTorch for on-device inference

CLI:
```bash
ludwig export_model -m results/experiment_run/model -o exported/ -f torch_export
```

!!! note
    `export_model` accepts an optional `sample_input` dict mapping input-feature names to
    tensors. When omitted (the default), Ludwig builds a dummy batch from the model's
    training-set metadata — so the snippet above works as-is. Pass a real example only if
    you need to trace against specific shapes or dtypes, for example when exporting models
    that rely on dynamic axes.

### ONNX

ONNX export uses the dynamo-based exporter (`torch.onnx.export(dynamo=True)`) for cross-platform deployment.

```python
model = LudwigModel.load("results/experiment_run/model")
model.export_model("exported/", format="onnx")
```

As with `torch_export`, `sample_input` is optional and is auto-generated from the training-set
metadata when omitted.

The exported ONNX model can be loaded with ONNX Runtime:

```python
import onnxruntime as ort
session = ort.InferenceSession("exported/model.onnx")
```

CLI:
```bash
ludwig export_model -m results/experiment_run/model -o exported/ -f onnx
```

### MLflow

Export to MLflow format for model registry and deployment:

```bash
ludwig export_mlflow -m results/experiment_run/model -o mlflow_model/
```

See [Integrations](integrations.md) for details on the MLflow integration.

## Loading Exported Models

The `load_exported_model` utility auto-detects the format:

```python
from ludwig.utils.model_export import load_exported_model

# torch.export format
program = load_exported_model("exported/model.pt2")

# ONNX format (requires onnxruntime)
session = load_exported_model("exported/model.onnx")
```
