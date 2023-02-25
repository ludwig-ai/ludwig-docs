{% from './macros/includes.md' import render_fields, render_yaml %}
{% set mv_details = "See [Missing Value Strategy](./input_features.md#missing-value-strategy) for details." %}
{% set norm_details = "See [Normalization](../combiner.md#normalization) for details." %}
{% set details = {"missing_value_strategy": mv_details, "norm": norm_details, "fc_norm": norm_details} %}

Vector features enable providing an ordered set of numerical values within a single feature.

This is useful for providing pre-trained representations or activations obtained from other models or for providing
multivariate inputs and outputs. An interesting use of vector features is the possibility of providing a probability
distribution as output for a multiclass classification problem instead of a single correct class like with a category
feature. Vector output features can also be useful for distillation and noise-aware losses.

## Vector Feature Preprocessing

The data is expected as whitespace separated numerical values. Example: "1.0 0.0 1.04 10.49".  All vectors are expected to be of the same size.

{% set preprocessing = get_feature_preprocessing_schema("vector") %}
{{ render_yaml(preprocessing, parent="preprocessing") }}

Parameters:

{{ render_fields(schema_class_to_fields(preprocessing), details=details) }}

Preprocessing parameters can also be defined once and applied to all vector input features using the [Type-Global Preprocessing](../defaults.md#type-global-preprocessing) section.

## Vector Input Features and Encoders

The vector feature supports two encoders: `dense` and `passthrough`.

The encoder parameters specified at the feature level are:

- **`tied`** (default `null`): name of the input feature to tie the weights of the encoder with. It needs to be the name of
a feature of the same type and with the same encoder parameters.

Example vector feature entry in the input features list:

```yaml
name: vector_column_name
type: vector
tied: null
encoder: 
    type: dense
```

The available encoder parameters are:

- **`type`** (default `dense`): the possible values are `passthrough` and `dense`. `passthrough` outputs the
raw vector values unaltered. `dense` uses a stack of fully connected layers to create an embedding matrix.

Encoder type and encoder parameters can also be defined once and applied to all vector input features using the
[Type-Global Encoder](../defaults.md#type-global-encoder) section.

### Passthrough Encoder

{% set encoder = get_encoder_schema("vector", "passthrough") %}
{{ render_yaml(encoder, parent="encoder") }}

There are no additional parameters for `passthrough` encoder.

### Dense Encoder

For vector features, a dense encoder (stack of fully connected layers) can be used to encode the vector.  

{% set encoder = get_encoder_schema("vector", "dense") %}
{{ render_yaml(encoder, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(encoder, exclude=["type"]), details=details) }}

## Vector Output Features and Decoders

``` mermaid
graph LR
  A["Combiner Output"] --> B["Fully\n Connected\n Layers"];
  B --> C["Projection into\nVector Size"] --> D["Softmax"];
  subgraph DEC["DECODER.."]
  B
  C
  D
  end
```

Vector features can be used when multi-class classification needs to be performed with a noise-aware loss or when the task is multivariate regression.

There is only one decoder available for vector features: a (potentially empty) stack of fully connected layers, followed
by a projection into a tensor of the vector size (optionally followed by a softmax in the case of multi-class classification).

Example vector output feature using default parameters:

```yaml
name: vector_column_name
type: vector
reduce_input: sum
dependencies: []
reduce_dependencies: sum
loss:
    type: sigmoid_cross_entropy
decoder:
    type: projector
```

Parameters:

- **`reduce_input`** (default `sum`): defines how to reduce an input that is not a vector, but a matrix or a higher order tensor, on the first dimension (second if you count the batch dimension). Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension).
- `dependencies` (default `[]`): the output features this one is dependent on. For a detailed explanation refer to [Output Features Dependencies](#output-features-dependencies).
- **`reduce_dependencies`** (default `sum`): defines how to reduce the output of a dependent feature that is not a vector, but a matrix or a higher order tensor, on the first dimension (second if you count the batch dimension). Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension).
- **`softmax`** (default `false`): determines if to apply a softmax at the end of the decoder. It is useful for predicting a vector of values that sum up to 1 and can be interpreted as probabilities.
- **`loss`** (default `{type: mean_squared_error}`): is a dictionary containing a loss `type`. The available loss `type` are `mean_squared_error`, `mean_absolute_error` and `softmax_cross_entropy` (use it only if `softmax` is `true`). See [Loss](#loss) for details.
- **`decoder`** (default: `{"type": "projector"}`): Decoder for the desired task. Options: `projector`. See [Decoder](#decoder) for details.

### Decoder

{% set decoder = get_decoder_schema("vector", "projector") %}
{{ render_yaml(decoder, parent="decoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(decoder, exclude=["type"]), details=details) }}

Decoder type and decoder parameters can also be defined once and applied to all number output features using the [Type-Global Decoder](../defaults.md#type-global-decoder) section.

### Loss

{% set loss_classes = get_loss_schemas("vector") %}
{% for loss in loss_classes %}

#### {{ loss.name() }}

{{ render_yaml(loss, parent="loss") }}

Parameters:

{{ render_fields(schema_class_to_fields(loss, exclude=["type"]), details=details) }}
{% endfor %}

Loss type and loss related parameters can also be defined once and applied to all vector output features using the [Type-Global Loss](../defaults.md#type-global-loss) section.

### Metrics

The metrics that are calculated every epoch and are available for set features are `mean_squared_error`, `mean_absolute_error`, `r2`, and the `loss` itself.

You can set any of them as `validation_metric` in the `training` section of the configuration if you set the
`validation_field` to be the name of a vector feature.
