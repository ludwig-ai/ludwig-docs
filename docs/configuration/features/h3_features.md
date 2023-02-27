{% from './macros/includes.md' import render_fields, render_yaml %}
{% set mv_details = "See [Missing Value Strategy](./input_features.md#missing-value-strategy) for details." %}
{% set norm_details = "See [Normalization](../combiner.md#normalization) for details." %}
{% set details = {"missing_value_strategy": mv_details, "norm": norm_details} %}

H3 is a indexing system for representing geospatial data.
For more details about it refer to <https://eng.uber.com/h3>.

# Preprocessing

Ludwig will parse the H3 64bit encoded format automatically.

{% set preprocessing = get_feature_preprocessing_schema("h3") %}
{{ render_yaml(preprocessing, parent="preprocessing") }}

Parameters:

{{ render_fields(schema_class_to_fields(preprocessing), details=details) }}

Preprocessing parameters can also be defined once and applied to all H3 input features using the [Type-Global Preprocessing](../defaults.md#type-global-preprocessing) section.

# Input Features

Input H3 features are transformed into a int valued tensors of size `N x 19` (where `N` is the size of the dataset and the 19 dimensions
represent 4 H3 resolution parameters (4) - mode, edge, resolution, base cell - and 15 cell coordinate values.

The encoder parameters specified at the feature level are:

- **`tied`** (default `null`): name of another input feature to tie the weights of the encoder with. It needs to be the name of
a feature of the same type and with the same encoder parameters.

Example H3 feature entry in the input features list:

```yaml
name: h3_feature_name
type: h3
tied: null
encoder: 
    type: embed
```

The available encoder parameters are:

- **`type`** (default ``embed``): the possible values are `embed`, `weighted_sum`,  and `rnn`.

Encoder type and encoder parameters can also be defined once and applied to all H3 input features using
the [Type-Global Encoder](../defaults.md#type-global-encoder) section.

## Encoders

### Embed Encoder

This encoder encodes each component of the H3 representation (mode, edge, resolution, base cell and children cells) with embeddings. Children cells with value `0` will be masked out. After the embedding, all embeddings are summed and optionally passed through a stack of fully connected layers.

{% set encoder_embed = get_encoder_schema("h3", "embed") %}
{{ render_yaml(encoder_embed, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(encoder_embed, exclude=["type"]), details=details) }}

### Weighted Sum Embed Encoder

This encoder encodes each component of the H3 representation (mode, edge, resolution, base cell and children cells) with embeddings. Children cells with value `0` will be masked out. After the embedding, all embeddings are summed with a weighted sum (with learned weights) and optionally passed through a stack of fully connected layers.

{% set encoder_weighted_sum = get_encoder_schema("h3", "weighted_sum") %}
{{ render_yaml(encoder_weighted_sum, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(encoder_weighted_sum, exclude=["type"]), details=details) }}

### RNN Encoder

This encoder encodes each component of the H3 representation (mode, edge, resolution, base cell and children cells) with embeddings. Children cells with value `0` will be masked out. After the embedding, all embeddings are passed through an RNN encoder.

The intuition behind this is that, starting from the base cell, the sequence of children cells can be seen as a sequence encoding the path in the tree of all H3 hexes.

{% set encoder_rnn = get_encoder_schema("h3", "rnn") %}
{{ render_yaml(encoder_rnn, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(encoder_rnn, exclude=["type"]), details=details) }}

# Output Features

There is currently no support for H3 as an output feature. Consider using the [`TEXT` type](../../features/text_features).
