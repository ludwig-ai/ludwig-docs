{% from './macros/includes.md' import render_fields, render_yaml %}
{% set mv_details = "See [Missing Value Strategy](./input_features.md#missing-value-strategy) for details." %}
{% set norm_details = "See [Normalization](../combiner.md#normalization) for details." %}
{% set details = {"missing_value_strategy": mv_details, "norm": norm_details} %}

## Preprocessing

Set features are expected to be provided as a string of elements separated by whitespace, e.g. "elem5 elem9 elem6".
The string values are transformed into a binary (int8 actually) valued matrix of size `n x l` (where `n` is the number
of rows in the dataset and `l` is the minimum of the size of the biggest set and a `max_size` parameter) and added to
HDF5 with a key that reflects the name of column in the dataset.
The way sets are mapped into integers consists in first using a tokenizer to map each input string to a sequence of set
elements (by default this is done by splitting on spaces).
Next a dictionary is constructed which maps each unique element to its frequency in the dataset column. Elements are
ranked by frequency and a sequential integer ID is assigned in ascending order from the most frequent to the most rare.
The column name is added to the JSON file, with an associated dictionary containing

1. the mapping from integer to string (`idx2str`)
1. the mapping from string to id (`str2idx`)
1. the mapping from string to frequency (`str2freq`)
1. the maximum size of all sets (`max_set_size`)
1. additional preprocessing information (by default how to fill missing values and what token to use to fill missing values)

{% set preprocessing = get_feature_preprocessing_schema("set") %}
{{ render_yaml(preprocessing, parent="preprocessing") }}

Parameters:

{{ render_fields(schema_class_to_fields(preprocessing), details=details) }}

Preprocessing parameters can also be defined once and applied to all set input features using the [Type-Global Preprocessing](../defaults.md#type-global-preprocessing) section.

## Input Features and Encoders

``` mermaid
graph LR
  A["0\n0\n1\n0\n1\n1\n0"] --> B["2\n4\n5"];
  B --> C["emb 2\nemb 4\nemb 5"];
  C --> D["Aggregation\n Reduce\n Operation"];
```

Set features have one encoder: `embed`, the raw binary values coming from the input placeholders are first transformed to sparse
integer lists, then they are mapped to either dense or sparse embeddings (one-hot encodings), finally they are
reduced on the sequence dimension and returned as an aggregated embedding vector.
Inputs are of size `b` while outputs are of size `b x h` where `b` is the batch size and `h` is the dimensionality of
the embeddings.

The encoder parameters specified at the feature level are:

- **`tied`** (default `null`): name of another input feature to tie the weights of the encoder with. It needs to be the name of
a feature of the same type and with the same encoder parameters.

```yaml
name: set_column_name
type: set
tied: null
encoder: 
    type: embed
```

Encoder type and encoder parameters can also be defined once and applied to all set input features using the [Type-Global Encoder](../defaults.md#type-global-encoder) section.

### Embed Encoder

{% set encoder = get_encoder_schema("set", "embed") %}
{{ render_yaml(encoder, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(encoder, exclude=["type"]), details=details) }}

## Output Features and Decoders

``` mermaid
graph LR
  A["Combiner\n Output"] --> B["Fully\n Connected\n Layers"];
  B --> C["Projection into\n Output Space"];
  C --> D["Sigmoid"];
  subgraph DEC["DECODER.."]
  B
  C
  D
  end
```

Set features can be used when multi-label classification needs to be performed.
There is only one decoder available for set features: a (potentially empty) stack of fully connected layers, followed by
a projection into a vector of size of the number of available classes, followed by a sigmoid.

```yaml
name: set_column_name
type: set
reduce_input: sum
dependencies: []
reduce_dependencies: sum
loss:
    type: sigmoid_cross_entropy
decoder:
    type: classifier
```

Parameters:

- **`reduce_input`** (default `sum`): defines how to reduce an input that is not a vector, but a matrix or a higher order
tensor, on the first dimension (second if you count the batch dimension). Available values are: `sum`, `mean` or `avg`,
`max`, `concat` (concatenates along the first dimension).
- **`dependencies`** (default `[]`): the output features this one is dependent on. For a detailed explanation refer to
[Output Feature Dependencies](../output_features#output-feature-dependencies).
- **`reduce_dependencies`** (default `sum`): defines how to reduce the output of a dependent feature that is not a vector,
but a matrix or a higher order tensor, on the first dimension (second if you count the batch dimension). Available
values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last
vector of the first dimension).
- **`loss`** (default `{type: sigmoid_cross_entropy}`): is a dictionary containing a loss `type`. The only supported loss
`type` for set features is `sigmoid_cross_entropy`. See [Loss](#loss) for details.
- **`decoder`** (default: `{"type": "classifier"}`): Decoder for the desired task. Options: `classifier`. See [Decoder](#decoder) for details.

### Decoder

{% set decoder = get_decoder_schema("set", "classifier") %}
{{ render_yaml(decoder, parent="decoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(decoder, exclude=["type"]), details=details) }}

Decoder type and decoder parameters can also be defined once and applied to all set output features using the [Type-Global Decoder](../defaults.md#type-global-decoder) section.

### Loss

{% set loss_classes = get_loss_schemas("set") %}
{% for loss in loss_classes %}

#### {{ loss.name() }}

{{ render_yaml(loss, parent="loss") }}

Parameters:

{{ render_fields(schema_class_to_fields(loss, exclude=["type"]), details=details) }}
{% endfor %}

Loss type and loss related parameters can also be defined once and applied to all set output features using the [Type-Global Loss](../defaults.md#type-global-loss) section.

###  Metrics

The metrics that are calculated every epoch and are available for set features are `jaccard` (counts the number of
elements in the intersection of prediction and label divided by number of elements in the union) and the `loss` itself.
You can set either of them as `validation_metric` in the `training` section of the configuration if you set the
`validation_field` to be the name of a set feature.
