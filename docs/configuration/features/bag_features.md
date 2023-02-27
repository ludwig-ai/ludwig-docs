{% from './macros/includes.md' import render_fields, render_yaml %}
{% set mv_details = "See [Missing Value Strategy](./input_features.md#missing-value-strategy) for details." %}
{% set norm_details = "See [Normalization](../combiner.md#normalization) for details." %}
{% set details = {"missing_value_strategy": mv_details, "norm": norm_details} %}

## Preprocessing

Bag features are expected to be provided as a string of elements separated by whitespace, e.g. "elem5 elem0 elem5 elem1".
Bags are similar to [set features](../set_features), the only difference being that elements may appear multiple
times. The bag feature encoder outputs a matrix, similar to a set encoder, except each element of the matrix is a float
value representing the frequency of the respective element in the bag. Embeddings are aggregated by summation, weighted
by the frequency of each element.

{% set preprocessing = get_feature_preprocessing_schema("bag") %}
{{ render_yaml(preprocessing, parent="preprocessing") }}

Parameters:

{{ render_fields(schema_class_to_fields(preprocessing), details=details) }}

## Input Features and Encoders

Bag features have only one encoder type available: `embed`.

The encoder parameters specified at the feature level are:

- **`tied`** (default `null`): name of another input feature to tie the weights of the encoder with. It needs to be the name of
a feature of the same type and with the same encoder parameters.

Example bag feature entry in the input features list:

```yaml
name: bag_column_name
type: bag
tied: null
encoder: 
    type: embed
```

Encoder type and encoder parameters can also be defined once and applied to all bag input features using the
[Type-Global Encoder](../defaults.md#type-global-encoder) section.

### Embed Weighted Encoder

``` mermaid
graph LR
  A["0.0\n1.0\n1.0\n0.0\n0.0\n2.0\n0.0"] --> B["0\n1\n5"];
  B --> C["emb 0\nemb 1\nemb 5"];
  C --> D["Weighted\n Sum\n Operation"];
```
{ data-search-exclude }

The embed weighted encoder first transforms the element frequency vector to sparse integer lists, which are then mapped
to either dense or sparse embeddings (one-hot encodings). Lastly, embeddings are aggregated as a weighted sum where each
embedding is multiplied by its respective element's frequency.
Inputs are of size `b` while outputs are of size `b x h` where `b` is the batch size and `h` is the dimensionality of
the embeddings.

The parameters are the same used for [set input features](../set_features#set-input-features-and-encoders) except for
`reduce_output` which should not be used because the weighted sum already acts as a reducer.

{% set encoder = get_encoder_schema("bag", "embed") %}
{{ render_yaml(encoder, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(encoder, exclude=["type"]), details=details) }}

## Output Features and Decoders

Bag types are not supported as output features at this time.
