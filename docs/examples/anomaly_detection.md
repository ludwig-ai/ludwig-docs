# Anomaly Detection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ludwig-ai/ludwig/blob/main/examples/anomaly_detection/anomaly_detection.ipynb)

Anomaly detection is the task of identifying samples that deviate significantly from a learned
definition of "normal". Unlike supervised classification, anomaly detection typically has access
to abundant normal training data but few or no labeled anomalies — making it a natural fit for
**one-class learning** objectives.

Ludwig's `anomaly` output feature implements three hypersphere-based objectives from the
Deep One-Class Classification family:

- **Deep SVDD** — the standard unsupervised baseline (Ruff et al., ICML 2018)
- **Deep SAD** — a semi-supervised extension for when a small set of labeled anomalies is available (Ruff et al., ICLR 2020)
- **DROCC** — adds adversarial perturbations to prevent collapse (Goyal et al., ICML 2020)

In all cases the model outputs an `anomaly_score` equal to the squared distance of the
sample's latent representation from the learned hypersphere centre `c`. Higher scores mean
more anomalous.

## Dataset

This example uses a synthetic sensor dataset with four numeric features:

| Column | Description |
|--------|-------------|
| `sensor_a` | Continuous sensor reading |
| `sensor_b` | Continuous sensor reading |
| `sensor_c` | Continuous sensor reading |
| `timestamp_hour` | Hour of day (0–23) |
| `anomaly` | Label: 0 = normal, 1 = anomalous, −1 = unlabeled (used by Deep SAD) |

Normal samples are drawn from a Gaussian distribution centred at the origin. Anomalous
samples have a large offset from the origin. The train split contains **only normal samples**;
the test split contains a balanced mix of both classes for evaluation.

## Configuration

### Deep SVDD (unsupervised)

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
  - name: timestamp_hour
    type: number
    preprocessing:
      normalization: zscore

output_features:
  - name: anomaly
    type: anomaly
    loss:
      type: deep_svdd
      nu: 0.1   # fraction of training points allowed outside the hypersphere

trainer:
  epochs: 20
  learning_rate: 0.001

combiner:
  type: concat
  fc_layers:
    - output_size: 64
    - output_size: 32
```

The `nu` parameter controls the **soft-boundary** relaxation. Set `nu: 0` for hard-boundary
SVDD where every training point is pulled toward the centre `c`. A small positive value
(e.g. `0.1`) makes the model more robust to noise in the training set.

### Deep SAD (semi-supervised)

```yaml
output_features:
  - name: anomaly
    type: anomaly
    loss:
      type: deep_sad
      eta: 1.0   # weight for the labeled anomaly repulsion term
```

Deep SAD extends Deep SVDD by incorporating **labeled anomaly rows** in the training data.
Normal and unlabeled samples (label `0` or `-1`) are pulled toward `c`; labeled anomalies
(label `1`) are pushed away. You only need a small fraction — even 5–10% labeled anomalies
can measurably improve AUC-ROC on real datasets.

The `eta` parameter controls the strength of the repulsion term for labeled anomalies.

### DROCC (robust unsupervised)

```yaml
output_features:
  - name: anomaly
    type: anomaly
    loss:
      type: drocc
      perturbation_strength: 0.1
      num_perturbation_steps: 5
```

DROCC prevents **hypersphere collapse** — a failure mode where an expressive encoder learns
to map all inputs to the same point, trivially minimising the SVDD loss while learning
nothing. It works by generating adversarial perturbations around each normal training point
and penalising the model if the perturbed points score as normal.

Use DROCC when training with expressive encoders (e.g. transformer-based input features or
deep combiner stacks) that are prone to degenerate solutions.

## Training

### CLI

```bash
# Train Deep SVDD
ludwig train \
  --config config_deep_svdd.yaml \
  --dataset /tmp/sensors_train.csv

# Train Deep SAD (training CSV must include some rows with anomaly=1)
ludwig train \
  --config config_deep_sad.yaml \
  --dataset /tmp/sensors_train_with_labeled_anomalies.csv

# Train DROCC
ludwig train \
  --config config_drocc.yaml \
  --dataset /tmp/sensors_train.csv
```

### Python API

```python
import pandas as pd
import yaml
from ludwig.api import LudwigModel

train_df = pd.read_csv("/tmp/sensors_train.csv")
test_df  = pd.read_csv("/tmp/sensors_test.csv")

# Load config from file (or pass a dict directly)
with open("config_deep_svdd.yaml") as f:
    config = yaml.safe_load(f)

model = LudwigModel(config, logging_level=30)
results = model.train(dataset=train_df)
train_stats = results.train_stats

# Predict — returns a DataFrame; anomaly scores are in the
# 'anomaly_anomaly_score_predictions' column
predictions, _ = model.predict(dataset=test_df)
print(predictions[["anomaly_anomaly_score_predictions"]].describe())
```

## Evaluation

After prediction, each sample has an `anomaly_anomaly_score_predictions` value (the squared
distance from the hypersphere centre). To convert scores into binary predictions you need
a **threshold**:

- **Percentile threshold**: choose the 95th percentile of normal training scores as the
  threshold. Samples above this value are classified as anomalies.
- **Auto threshold**: set `threshold: auto` in the output feature config. Ludwig will
  automatically select the `threshold_percentile`-th percentile of validation scores after
  training.
- **Fixed threshold**: set `threshold: 0.5` (or any float) in the output feature config.

```python
import numpy as np

# Compute threshold from normal validation scores
normal_val_scores = predictions.loc[
    test_df["anomaly"] == 0, "anomaly_anomaly_score_predictions"
].values
threshold = np.percentile(normal_val_scores, 95)

is_anomaly = predictions["anomaly_anomaly_score_predictions"] > threshold
print(f"Flagged as anomalous: {is_anomaly.sum()} / {len(is_anomaly)}")
```

To measure ranking quality (independent of threshold choice), compute **AUC-ROC**:

```python
from sklearn.metrics import roc_auc_score

scores = predictions["anomaly_anomaly_score_predictions"].values
labels = test_df["anomaly"].values  # 0 = normal, 1 = anomalous

auc = roc_auc_score(labels, scores)
print(f"AUC-ROC: {auc:.4f}")
```

## Choosing a loss variant

| Loss | Labels required | Best for | Key parameter |
|------|----------------|----------|---------------|
| `deep_svdd` | None (fully unsupervised) | General-purpose baseline; simple tabular data | `nu` (soft-boundary fraction) |
| `deep_sad` | Small set of labeled anomalies | When a few verified anomalies are available | `eta` (repulsion strength) |
| `drocc` | None (fully unsupervised) | Expressive encoders prone to collapse; structured data | `perturbation_strength`, `num_perturbation_steps` |

**Rules of thumb:**

- Start with `deep_svdd`. It is the simplest and most widely validated variant.
- If you have even a handful of confirmed anomaly examples, try `deep_sad`. A ratio as low
  as 5% labeled anomalies in the training set often improves AUC-ROC by several points.
- Switch to `drocc` if you observe that validation anomaly scores do not increase over
  training epochs (a sign of collapse), or if you are using transformer-based input features.

## Full config reference

For all available parameters for the `anomaly` output feature and its loss functions, see the
[Anomaly Features configuration reference](../../configuration/features/anomaly_features.md).
