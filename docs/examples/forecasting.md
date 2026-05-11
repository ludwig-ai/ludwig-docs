# Forecasting

Ludwig supports timeseries forecasting end-to-end: training a model on historical data, evaluating it, and
generating multi-step predictions with `model.forecast()`.

## Data formats

Ludwig accepts two data layouts:

**Row-major** — each row is already a pre-embedded window (space-separated floats). Use this when you have
already computed sliding windows externally.

| timeseries_window         | next_value |
| ------------------------- | ---------- |
| 15.07 14.89 14.45 14.30   | 14.12      |
| 14.89 14.45 14.30 14.12   | 13.97      |

**Column-major** — each row is one scalar observation. Ludwig converts to windows automatically via
`window_size` (inputs) and `horizon` (outputs).

| timestamp | temperature |
| --------- | ----------- |
| 2024-01-01 | 15.07 |
| 2024-01-02 | 14.89 |
| 2024-01-03 | 14.45 |
| ...        | ...   |

Column-major is the most common format for real-world timeseries data and is the recommended starting point.

## Column-major example (recommended)

### Dataset

A CSV where each row is one observation:

```
timestamp,temperature
2024-01-01,15.07
2024-01-02,14.89
2024-01-03,14.45
...
```

The `timestamp` column is not required by Ludwig — only the numeric series column is needed.

### Config

```yaml
input_features:
  - name: temperature
    type: timeseries
    preprocessing:
      window_size: 24      # look back 24 steps
      padding_value: 0.0

output_features:
  - name: temperature       # same column name — Ludwig aligns input windows with output targets
    type: timeseries
    preprocessing:
      horizon: 6            # predict 6 steps ahead
    decoder:
      type: projector
    loss:
      type: huber

trainer:
  epochs: 50
  batch_size: 128
  optimizer:
    type: adam
    lr: 0.001
```

### Training

```bash
ludwig train \
  --dataset temperature.csv \
  --config config.yaml \
  --output_directory results
```

Or with the Python API:

```python
import pandas as pd
from ludwig.api import LudwigModel

df = pd.read_csv("temperature.csv")
model = LudwigModel("config.yaml")
results = model.train(dataset=df)
print(f"Model saved to {results.output_directory}")
```

### Forecasting

After training, generate future predictions from a seed dataset:

```python
model = LudwigModel.load("results/experiment_run/model")

# Seed: any DataFrame with the input column. Only the last window_size rows are used.
seed = pd.read_csv("recent_observations.csv")

# Forecast 48 steps ahead
forecast_df = model.forecast(seed, horizon=48)
print(forecast_df)
#    temperature
# 0    14.23
# 1    14.01
# ...
```

The forecast iteratively slides the prediction window: each predicted value is fed back as the next input,
so you can forecast arbitrarily far ahead beyond the trained `horizon`. Preprocessing runs once for the
initial window and then updates in O(1) per step.

## Row-major example

Use this when you have pre-computed windows, or when training and predicting on windows that don't come
from a contiguous series (e.g., independent samples with known context windows).

### Dataset

```
timeseries_window,horizon_values
15.07 14.89 14.45 14.30,14.12 13.97 13.80
14.89 14.45 14.30 14.12,13.97 13.80 13.65
```

### Config

```yaml
input_features:
  - name: timeseries_window
    type: timeseries
    encoder:
      type: transformer
      num_heads: 4
      num_layers: 2

output_features:
  - name: horizon_values
    type: timeseries
    decoder:
      type: projector

trainer:
  epochs: 20
  batch_size: 64
```

## State-of-the-art encoders

Ludwig includes two modern time-series-specific encoders — PatchTST and N-BEATS — that typically outperform
generic sequence encoders on forecasting benchmarks.

### PatchTST

PatchTST segments the input window into short overlapping patches and encodes them with a Transformer. It
is particularly effective on longer input windows (96+ timesteps) and multi-step horizons.

```yaml
input_features:
  - name: temperature
    type: timeseries
    preprocessing:
      window_size: 96
    encoder:
      type: patchtst
      patch_size: 16
      patch_stride: 8
      d_model: 128
      num_heads: 8
      num_layers: 3
      output_size: 256

output_features:
  - name: temperature
    type: timeseries
    preprocessing:
      horizon: 24
    loss:
      type: huber

trainer:
  epochs: 100
  optimizer:
    type: adamw
    lr: 1e-4
```

### N-BEATS

N-BEATS is a pure MLP with doubly-residual stacking. It is fast to train, requires no positional encoding,
and matches Transformer-based models on many univariate benchmarks.

```yaml
input_features:
  - name: temperature
    type: timeseries
    preprocessing:
      window_size: 96
    encoder:
      type: nbeats
      num_stacks: 2
      num_blocks: 3
      layer_size: 256
      output_size: 256

output_features:
  - name: temperature
    type: timeseries
    preprocessing:
      horizon: 24

trainer:
  epochs: 100
  optimizer:
    type: adamw
    lr: 1e-4
```

### Scale-free validation metrics

Use `mean_absolute_scaled_error` (MASE) or `symmetric_mean_absolute_percentage_error` (sMAPE) as validation
metrics when comparing models trained on datasets with different scales, or when interpretable percentage-based
error is preferred.

```yaml
output_features:
  - name: temperature
    type: timeseries
    validation_metric: mean_absolute_scaled_error
```

## Choosing an encoder

| Encoder | When to use |
|---------|-------------|
| `patchtst` | Best accuracy on longer windows (96+), multi-step horizons |
| `nbeats` | Fast training, strong univariate baseline, no positional encoding needed |
| `transformer` | Good accuracy on complex patterns, generic architecture |
| `parallel_cnn` | Fast training, captures local patterns, good baseline |
| `stacked_cnn` | Longer-range dependencies than parallel_cnn |
| `rnn` | Sequential dependencies, moderate sequence length |
| `passthrough` | Pass raw window to the combiner without encoding |

## CLI evaluation

```bash
ludwig evaluate \
  --model_path results/experiment_run/model \
  --dataset test.csv \
  --output_directory evaluation
```

## Notes

- **Direct multi-step forecasting**: the decoder predicts all `horizon` steps simultaneously in one forward
  pass. This is fast and works well for short horizons. For long-horizon forecasting, `model.forecast()` with
  iterative stepping is the recommended approach.
- **Input/output column alignment**: when the output feature `name` matches the input feature `name`, Ludwig
  uses predicted values as inputs for subsequent forecast steps. If they differ, the input window is padded
  with `padding_value` for steps beyond the seed data.
- **Minimum seed length**: `model.forecast()` needs at least `window_size` rows in the seed DataFrame to
  fill the initial context window. Shorter seeds are padded automatically.
