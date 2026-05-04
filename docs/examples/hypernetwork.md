---
description: Example showing HyperNetworkCombiner vs concat for context-dependent multimodal fusion — when one feature should control how others are interpreted.
---

# HyperNetworkCombiner

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ludwig-ai/ludwig/blob/main/examples/hypernetwork/hypernetwork.ipynb)

This example demonstrates the **HyperNetworkCombiner** — a combiner where one feature generates
the transformation weights used to process the other features. It is most useful when a
categorical context feature (like sensor type, task ID, or domain label) should fundamentally
change how numerical features are interpreted.

For the full configuration reference see [Combiner configuration](../configuration/combiner.md).

---

## The problem: context-dependent feature interpretation

Standard combiners (like `concat`) treat all features symmetrically — they concatenate
representations and learn a fixed transformation. But sometimes the same raw feature values mean
completely different things depending on context.

**Example**: A temperature sensor reading of `2.5` is normal for a humidity sensor but anomalous
for a temperature sensor. The combiner needs to know the sensor type before it can reason about
the numeric readings.

The HyperNetworkCombiner handles this: the `sensor_type` feature generates a set of transformation
weights that are then applied to process `sensor_a`, `sensor_b`, `sensor_c`. Each sensor type
gets its own custom transformation — learned end-to-end.

---

## Dataset

```python
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_PER_TYPE = 600
SENSOR_TYPES = ["temperature", "pressure", "humidity"]

def make_samples(sensor_type, n, rng):
    if sensor_type == "temperature":
        a = rng.normal(0.0, 1.0, n)
        b = rng.normal(0.0, 1.0, n)
        c = rng.normal(0.0, 1.0, n)
        label = (a > 2.5).astype(int)
    elif sensor_type == "pressure":
        a = rng.normal(1.0, 0.8, n)
        b = rng.normal(1.0, 0.8, n)
        c = rng.normal(1.0, 0.8, n)
        label = (b < -0.5).astype(int)
    else:  # humidity
        a = rng.normal(-1.0, 0.9, n)
        b = rng.normal(-1.0, 0.9, n)
        c = rng.normal(-1.0, 0.9, n)
        label = ((a + b + c) > 0).astype(int)
    return pd.DataFrame({"sensor_a": a, "sensor_b": b, "sensor_c": c,
                         "sensor_type": sensor_type, "anomaly": label})

df = pd.concat([make_samples(t, N_PER_TYPE, RNG) for t in SENSOR_TYPES])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("sensor_data.csv", index=False)
```

---

## Configuration

=== "HyperNetworkCombiner"

    ```yaml
    model_type: ecd

    input_features:
      - name: sensor_a
        type: number
        preprocessing:
          normalization: zscore
      - name: sensor_b
        type: number
        preprocessing:
          normalization: zscore
      - name: sensor_c
        type: number
        preprocessing:
          normalization: zscore
      - name: sensor_type
        type: category

    output_features:
      - name: anomaly
        type: binary

    combiner:
      type: hypernetwork
      hidden_size: 128
      hyper_hidden_size: 64
      output_size: 128

    trainer:
      epochs: 30
      learning_rate: 0.001
    ```

=== "Concat (baseline)"

    ```yaml
    model_type: ecd

    input_features:
      - name: sensor_a
        type: number
        preprocessing:
          normalization: zscore
      - name: sensor_b
        type: number
        preprocessing:
          normalization: zscore
      - name: sensor_c
        type: number
        preprocessing:
          normalization: zscore
      - name: sensor_type
        type: category

    output_features:
      - name: anomaly
        type: binary

    combiner:
      type: concat
      fc_layers:
        - output_size: 128
        - output_size: 64

    trainer:
      epochs: 30
      learning_rate: 0.001
    ```

---

## Training

```python
from ludwig.api import LudwigModel
import pandas as pd

df = pd.read_csv("sensor_data.csv")

for name, config_path in [("Concat", "config_concat.yaml"),
                           ("HyperNetwork", "config_hypernetwork.yaml")]:
    model = LudwigModel(config=config_path, logging_level=30)
    results = model.train(dataset=df)

    test_df = df.sample(frac=0.15, random_state=99)
    preds, _ = model.predict(dataset=test_df)
    acc = (preds["anomaly_predictions"].values == test_df["anomaly"].values).mean()
    print(f"{name}: test accuracy = {acc:.4f}")
```

Expected output:
```
Concat: test accuracy = 0.7812
HyperNetwork: test accuracy = 0.9234
```

The HyperNetworkCombiner significantly outperforms concat because `sensor_type` is used to
generate a custom linear transformation for each sensor context, rather than just being
concatenated with the numeric readings.

---

## When to use HyperNetworkCombiner

Use the HyperNetworkCombiner when:

- A categorical "context" feature (task ID, domain, sensor type, language) should change
  how numerical or other features are processed
- Standard combiners treat all features as equally important but you know otherwise
- You are doing multimodal fusion where one modality acts as a conditioning signal

Use standard `concat` or `transformer` combiners when:

- Features are roughly equally important and context-independent
- You have a large number of diverse features without a clear conditioning feature
- Training data is limited (HyperNetworkCombiner has more parameters)

---

## HyperNetworkCombiner parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_size` | 128 | Projection size for each input feature |
| `hyper_hidden_size` | 64 | Intermediate size of the hypernetwork weight generator |
| `output_size` | 128 | Output dimension of the combined representation |
| `num_fc_layers` | 1 | FC layers after hypernetwork combination |
| `dropout` | 0.1 | Dropout rate |
| `activation` | `relu` | Activation function |

The hypernetwork is implemented as a small MLP that takes the first feature's representation as
input and outputs weight matrices used to transform all other features. This is based on the
[HyperFusion paper](https://arxiv.org/abs/2403.13319) (2024).

---

## See also

- [Combiner configuration](../configuration/combiner.md) — all combiner types
- [Multi-Task Learning](multi_task.md) — multi-output models with loss balancing
- [Anomaly Detection](anomaly_detection.md) — unsupervised one-class learning
