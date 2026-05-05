{% from './macros/includes.md' import render_fields, render_yaml %}

# Preprocessing

Timeseries features are handled as sequence features, with the only difference being that the matrix in the HDF5
preprocessing file uses floats instead of integers.

Since data is continuous, the JSON file, which typically stores vocabulary mappings, isn't needed.

Ludwig supports two data formats for timeseries:

- **Row-major** (default): each row in the dataset is already a space-separated sequence of floats representing one
  complete window. Use `timeseries_length_limit` to cap the window size.
- **Column-major**: each row is a single scalar observation. Ludwig converts to row-major automatically using a sliding
  window controlled by `preprocessing.window_size` (for inputs) or `preprocessing.horizon` (for outputs).

# Input Features

## Preprocessing

{% set preprocessing = get_feature_preprocessing_schema("timeseries") %}
{{ render_yaml(preprocessing) }}

Parameters:

{{ render_fields(schema_class_to_fields(preprocessing)) }}

### Column-major preprocessing with `window_size`

When your dataset has one observation per row (column-major), set `window_size` to the number of past observations
each input window should span:

```yaml
input_features:
  - name: temperature
    type: timeseries
    preprocessing:
      window_size: 24   # use the last 24 observations as context
      padding_value: 0.0
```

Ludwig will slide a window of length `window_size` over the column and produce one row-major embedding per
observation, padding the beginning of the series with `padding_value`.

## Encoders

### Sequence Encoders

Time series encoders are the same as for [Sequence Features](sequence_features.md#input-features), with one exception:

Time series features don't have an embedding layer at the beginning, so the `b x s` placeholders (where `b` is the batch
size and `s` is the sequence length) are directly mapped to a `b x s x 1` tensor and then passed to the different
sequential encoders.

The encoder parameters specified at the feature level are:

- `tied` (default `null`): name of another input feature to tie the weights of the encoder with. It needs to be the name of
a feature of the same type and with the same encoder parameters.

Example timeseries input feature:

```yaml
name: timeseries_column_name
type: timeseries
tied: null
encoder:
    type: parallel_cnn
```

### PatchTST

PatchTST (`type: patchtst`) splits the time series into fixed-length overlapping patches, projects each patch to a
learned embedding, and encodes the sequence of patch embeddings with a Transformer. Processing is
channel-independent: each variate is encoded separately before being combined by the combiner.

Reference: Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers", ICLR 2023.

```yaml
encoder:
  type: patchtst
  patch_size: 16        # length of each patch in timesteps
  patch_stride: 8       # step between consecutive patches (overlap when < patch_size)
  d_model: 128          # patch embedding dimension
  num_heads: 8          # number of Transformer attention heads
  num_layers: 3         # number of Transformer encoder layers
  ffn_dim: 256          # feed-forward hidden dimension
  output_size: 256      # size of the final encoder output vector
  reduce_output: mean   # how to aggregate patch outputs: mean | last | first
```

Parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `patch_size` | `16` | Length of each patch in timesteps |
| `patch_stride` | `8` | Step between consecutive patch start positions. Set equal to `patch_size` for non-overlapping patches |
| `d_model` | `128` | Dimension of the patch embedding and Transformer hidden states |
| `num_heads` | `8` | Number of multi-head attention heads |
| `num_layers` | `3` | Number of Transformer encoder layers |
| `ffn_dim` | `256` | Hidden dimension of the Transformer feed-forward sub-layer |
| `output_size` | `256` | Dimension of the final output vector |
| `reduce_output` | `mean` | Aggregation strategy across patch outputs: `mean`, `last`, or `first` |

### N-BEATS

N-BEATS (`type: nbeats`) is a pure MLP architecture with doubly-residual stacking. Each block produces a
backcast (subtracts its modeled portion from the input) and a forecast contribution. Contributions from all
blocks are summed to produce the final output. The design requires no time-series-specific inductive biases
and achieves strong results on univariate forecasting benchmarks.

Reference: Oreshkin et al., "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting",
ICLR 2020.

```yaml
encoder:
  type: nbeats
  num_stacks: 2         # number of stacks (groups of blocks)
  num_blocks: 3         # number of blocks per stack
  num_layers: 4         # number of FC layers inside each block
  layer_size: 256       # hidden dimension of each FC layer
  output_size: 256      # size of the final encoder output vector
```

Parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_stacks` | `2` | Number of stacks (each stack is a group of residual blocks) |
| `num_blocks` | `3` | Number of blocks per stack |
| `num_layers` | `4` | Number of fully-connected layers inside each block |
| `layer_size` | `256` | Hidden dimension of each fully-connected layer |
| `output_size` | `256` | Dimension of the final output vector |

### Passthrough Encoder

``` mermaid
graph LR
  A["12\n7\n43\n65\n23\n4\n1"] --> B["Cast float32"];
  B --> C["Aggregation\n Reduce\n Operation"];
  C --> ...;
```
{ data-search-exclude }

The passthrough encoder simply transforms each input value into a float value and adds a dimension to the input tensor,
creating a `b x s x 1` tensor where `b` is the batch size and `s` is the length of the sequence.
The tensor is reduced along the `s` dimension to obtain a single vector of size `h` for each element of the batch.
If you want to output the full `b x s x h` tensor, you can specify `reduce_output: null`.
This is useful for `timeseries` features when you want to pass the raw window directly to a downstream combiner
such as the sequence combiner.

{% set encoder = get_encoder_schema("timeseries", "passthrough") %}
{{ render_yaml(encoder, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(encoder, exclude=["type"])) }}

# Output Features

Ludwig supports timeseries as an output feature for forecasting tasks. The decoder projects the combined
representation to a vector of length `horizon` — one predicted value per future timestep. All steps are
predicted simultaneously in a single forward pass (direct multi-step forecasting).

## Preprocessing

{% set preprocessing = get_feature_preprocessing_schema("timeseries_output") %}
{{ render_yaml(preprocessing) }}

Parameters:

{{ render_fields(schema_class_to_fields(preprocessing)) }}

### Column-major preprocessing with `horizon`

When using column-major data, set `horizon` on the output feature to tell Ludwig how many steps ahead each
training target spans:

```yaml
output_features:
  - name: temperature
    type: timeseries
    preprocessing:
      horizon: 12   # predict the next 12 observations
    decoder:
      type: projector
```

The input feature must share the same column name (Ludwig uses it to align input windows with output targets):

```yaml
input_features:
  - name: temperature
    type: timeseries
    preprocessing:
      window_size: 24
output_features:
  - name: temperature
    type: timeseries
    preprocessing:
      horizon: 12
    decoder:
      type: projector
```

## Decoders

### Projector

The projector decoder is a fully-connected layer (or stack of FC layers) that maps the combiner output to a
vector of size `horizon`. This is the recommended decoder for timeseries output.

```yaml
output_features:
  - name: temperature
    type: timeseries
    decoder:
      type: projector
```

## Metrics

Ludwig computes the following metrics for timeseries output features during training and evaluation.

### Mean Absolute Scaled Error (MASE)

MASE normalizes the mean absolute error by the mean absolute error of the in-sample naive one-step forecast
(i.e. the mean absolute difference between consecutive observations). Because the scale cancels out, MASE is
comparable across datasets with different units or magnitudes. A MASE of 1.0 means the model performs
identically to the naive baseline; values below 1.0 indicate the model outperforms it.

```yaml
output_features:
  - name: temperature
    type: timeseries
    validation_metric: mean_absolute_scaled_error
```

### Symmetric Mean Absolute Percentage Error (sMAPE)

sMAPE computes the percentage error symmetrically, using the average of the absolute predicted and actual
values as the denominator. This avoids the asymmetry of standard MAPE (which penalises over-forecasts more
than under-forecasts) and is bounded between 0% and 200%.

```yaml
output_features:
  - name: temperature
    type: timeseries
    validation_metric: symmetric_mean_absolute_percentage_error
```

### Available metrics summary

| Metric key | Description |
|------------|-------------|
| `mean_squared_error` | Mean squared error |
| `mean_absolute_error` | Mean absolute error |
| `mean_absolute_scaled_error` | Scale-free MAE normalised by naive baseline |
| `symmetric_mean_absolute_percentage_error` | Symmetric percentage error (0–200%) |
| `r2` | Coefficient of determination |

## Loss

The default loss for timeseries output features is Huber loss, which is robust to outliers compared to MSE.
You can override it:

```yaml
output_features:
  - name: temperature
    type: timeseries
    loss:
      type: mean_squared_error
```

# Forecasting with `model.forecast()`

After training a model with timeseries input and output features, use `model.forecast()` to generate
multi-step predictions from a seed dataset:

```python
import pandas as pd
from ludwig.api import LudwigModel

model = LudwigModel.load("results/experiment_run/model")

# Seed data — must contain enough rows to fill the input window_size.
# Only the last window_size rows are used as context.
seed_df = pd.read_csv("recent_observations.csv")

# Predict 48 steps ahead, iteratively sliding the window.
forecast_df = model.forecast(seed_df, horizon=48)
print(forecast_df)
# Returns a DataFrame with one column per timeseries output feature,
# and one row per forecasted timestep.
```

`model.forecast()` uses an efficient incremental strategy: it preprocesses the initial window once, then
slides each new prediction into the window in O(1) per step rather than re-running full preprocessing.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | DataFrame / path | required | Seed data containing at least `window_size` rows |
| `horizon` | int | `1` | Number of timesteps to forecast ahead |
| `data_format` | str | `"auto"` | Dataset format (csv, parquet, etc.) |
| `output_directory` | str | `None` | If set, saves forecast results here |
| `output_format` | str | `"parquet"` | Format for saved results |

!!! note
    `model.forecast()` requires the model to have at least one timeseries input feature **and** at least one
    timeseries output feature. If the output feature column name matches the input feature column name,
    Ludwig automatically feeds each predicted value back as input for the next step.
