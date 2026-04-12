# Anomaly Features

The `anomaly` output feature type enables **unsupervised anomaly detection** using the
[Deep SVDD](https://proceedings.mlr.press/v80/ruff18a.html) (Deep Support Vector Data Description) objective.

Instead of predicting a label, the model learns a compact representation of "normal" data in a latent
hypersphere. At inference time, each sample is scored by its squared distance to the centre of that
hypersphere: samples far from the centre are flagged as anomalies.

```
anomaly_score = ||z - c||²
```

where `z` is the encoder/combiner output and `c` is the hypersphere centre, initialised after the first
training epoch as the mean of all encoder outputs.

This makes Ludwig's anomaly feature a natural fit for multimodal anomaly detection: use any combination
of tabular, image, text, or audio input features with the ECD combiner, and the anomaly decoder will
find a joint hypersphere in the fused latent space.

## Minimal config

```yaml
input_features:
  - name: temperature
    type: number
  - name: vibration
    type: number
  - name: log_message
    type: text

output_features:
  - name: anomaly_score
    type: anomaly
```

Train with only normal (non-anomalous) samples. After training, run `ludwig predict` to obtain a
`anomaly_score_predictions` column with per-sample anomaly scores. Use a threshold on this score (e.g.
the 95th percentile of training scores) to classify samples as normal or anomalous.

## Output columns

| Column | Description |
|--------|-------------|
| `NAME_predictions` | Squared distance to the hypersphere centre (higher = more anomalous) |

## Decoder parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `type` | `anomaly` | Must be `anomaly` |
| `input_size` | auto | Size of the combiner output. Set automatically. |

## Loss

The anomaly feature uses the **Deep SVDD** loss: the mean squared distance of all training samples to
the centre of the hypersphere. The objective pushes normal-sample representations toward the centre.

Optional auxiliary loss functions:

| Loss | Description |
|------|-------------|
| `deep_svdd` | Standard hard-boundary SVDD (default) |
| `deep_sad` | [Deep SAD](https://arxiv.org/abs/1906.02694) — semi-supervised, uses a small set of labelled anomalies during training |
| `drocc` | [DROCC](https://arxiv.org/abs/2002.12718) — adds a perturbation term to improve robustness on structured data |

```yaml
output_features:
  - name: anomaly_score
    type: anomaly
    loss:
      type: drocc
      perturbation_strength: 0.1
      radius: 1.0
```

## Example: anomaly detection on sensor data

```yaml
model_type: ecd

input_features:
  - name: sensor_a
    type: number
  - name: sensor_b
    type: number
  - name: sensor_c
    type: number

output_features:
  - name: anomaly
    type: anomaly

trainer:
  epochs: 20
  learning_rate: 0.001

combiner:
  type: concat
```

Train on a dataset containing only normal sensor readings. Evaluate by scoring held-out normal and
anomalous samples — anomalous samples should have noticeably higher scores.

## Notes

- The hypersphere centre is **not** updated after the first epoch. This is the standard Deep SVDD protocol.
- Use the ECD model type. The `anomaly` output feature is not supported with `llm`.
- For best results, normalise numeric input features (`preprocessing: normalization: zscore`).
