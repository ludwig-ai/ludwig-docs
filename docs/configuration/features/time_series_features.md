{% from './macros/includes.md' import render_fields, render_yaml %}

# Preprocessing

Timeseries features are handled as sequence features, with the only difference being that the matrix in the HDF5
preprocessing file uses floats instead of integers.

Since data is continuous, the JSON file, which typically stores vocabulary mappings, isn't needed.

# Input Features

## Encoders

### Sequence Encoders

Time series encoders are the same as for [Sequence Features](../sequence_features#sequence-input-features-and-encoders), with one exception:

Time series features don't have an embedding layer at the beginning, so the `b x s` placeholders (where `b` is the batch
size and `s` is the sequence length) are directly mapped to a `b x s x 1` tensor and then passed to the different
sequential encoders.

The encoder parameters specified at the feature level are:

- `tied` (default `null`): name of another input feature to tie the weights of the encoder with. It needs to be the name of
a feature of the same type and with the same encoder parameters.

Example category feature entry in the input features list:

```yaml
name: timeseries_column_name
type: timeseries
tied: null
encoder: 
    type: parallel_cnn
```

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
This encoder is not really useful for `sequence` or `text` features, but may be useful for `timeseries` features, as it
allows for using them without any processing in later stages of the model, like in a sequence combiner for instance.

{% set encoder = get_encoder_schema("timeseries", "passthrough") %}
{{ render_yaml(encoder, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(encoder, exclude=["type"])) }}

# Output Features

There are no time series decoders at the moment.

If this would unlock an interesting use case for your application, please file a GitHub Issue or ping the
[Ludwig Slack](https://join.slack.com/t/ludwig-ai/shared_invite/zt-mrxo87w6-DlX5~73T2B4v_g6jj0pJcQ).
