{% from './macros/includes.md' import render_fields, render_yaml %}
{% set mv_details = "See [Missing Value Strategy](./input_features.md#missing-value-strategy) for details." %}
{% set norm_details = "See [Normalization](../combiner.md#normalization) for details." %}
{% set details = {"missing_value_strategy": mv_details, "norm": norm_details, "norm_params": norm_details} %}

## Binary Features Preprocessing

Binary features are directly transformed into a binary valued vector of length `n` (where `n` is the size of the dataset) and added to the HDF5 with a key that reflects the name of column in the dataset.

{% set preprocessing = get_feature_preprocessing_schema("binary") %}
{{ render_yaml(preprocessing, parent="preprocessing") }}

Parameters:

{{ render_fields(schema_class_to_fields(preprocessing), details=details) }}

Preprocessing parameters can also be defined once and applied to all binary input features using
the [Type-Global Preprocessing](../defaults.md#type-global-preprocessing) section.

## Binary Input Features and Encoders

Binary features have two encoders, `passthrough` and `dense`. The available encoder can be specified using the `type` parameter:

- **`type`** (default `passthrough`): the possible values are `passthrough` and `dense`. `passthrough` outputs the raw integer values unaltered. `dense` randomly initializes a trainable embedding matrix.

The encoder parameters specified at the feature level are:

- **`tied`** (default `null`): name of another input feature to tie the weights of the encoder with. It needs to be the name of a feature of the same type and with the same encoder parameters.

Example binary feature entry in the input features list:

```yaml
name: binary_column_name
type: binary
tied: null
encoder: 
    type: dense
```

Encoder type and encoder parameters can also be defined once and applied to all binary input features using the
[Type-Global Encoder](../defaults.md#type-global-encoder) section.

### Passthrough Encoder

The `passthrough` encoder passes through raw binary values without any transformations. Inputs of size `b` are transformed to outputs of size `b x 1` where `b` is the batch size.

{% set encoder_passthrough = get_encoder_schema("binary", "passthrough") %}
{{ render_yaml(encoder_passthrough, parent="encoder") }}

There are no additional parameters for the `passthrough` encoder.

### Dense Encoder

The `dense` encoder passes the raw binary values through a fully connected layer. Inputs of size `b` are transformed to size `b x h`.

{% set encoder_dense = get_encoder_schema("binary", "dense") %}
{{ render_yaml(encoder_dense, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(encoder_dense, exclude=["type"]), details=details) }}

## Binary Output Features and Decoders

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

Binary output features can be used when a binary classification needs to be performed or when the output is a single probability. There are two decoders available: `regressor` and `passthrough`.

Example binary output feature using default parameters:

```yaml
name: binary_column_name
type: binary
reduce_input: sum
dependencies: []
calibration: false
reduce_dependencies: sum
loss:
    type: cross_entropy
    confidence_penalty: 0
    robust_lambda: 0
    positive_class_weight: 1
decoder:
    fc_layers: null
    num_fc_layers: 0
    activation: relu
    norm: null
    dropout: 0.2
    weights_initializer: glorot_uniform
    bias_initializer: zeros
    threshold: 0.5
```

Parameters:

- **`reduce_input`** (default `sum`): defines how to reduce an input that is not a vector, but a matrix or a higher order tensor, on the first dimension (second if you count the batch dimension). Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension).
- **`dependencies`** (default `[]`): the output features this one is dependent on. For a detailed explanation refer to [Output Features Dependencies](../output_features#output-feature-dependencies).
- **`calibration`** (default `false`): if true, performs calibration by temperature scaling after training is complete.
Calibration uses the validation set to find a scale factor (temperature) which is multiplied with the logits to shift
output probabilities closer to true likelihoods.
- **`reduce_dependencies`** (default `sum`): defines how to reduce the output of a dependent feature that is not a vector, but a matrix or a higher order tensor, on the first dimension (second if you count the batch dimension). Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension).
- **`loss`** (default `{"type": "binary_weighted_cross_entropy"}`): is a dictionary containing a loss `type`. `binary_weighted_cross_entropy` is the only supported loss type for binary output features. See [Loss](#loss) for details.
- **`decoder`** (default: `{"type": "regressor"}`): Decoder for the desired task. Options: `regressor`. See [Decoder](#decoder) for details.

Decoder type and decoder parameters can also be defined once and applied to all binary output features using the [Type-Global Decoder](../defaults.md#type-global-decoder) section.

### Decoder

#### Regressor Decoder

The regressor decoder is a (potentially empty) stack of fully connected layers, followed by a projection into a single number followed by a sigmoid function.

{% set decoder = get_decoder_schema("binary", "regressor") %}
{{ render_yaml(decoder, parent="decoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(decoder, exclude=["type"]), details=details) }}

#### Passthrough Decoder

The `passthrough` decoder passes through raw binary values without any transformations.

{% set decoder = get_decoder_schema("binary", "passthrough") %}
{{ render_yaml(decoder, parent="decoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(decoder, exclude=["type"]), details=details) }}

### Loss

{% set loss = get_loss_schema("binary_weighted_cross_entropy") %}
{{ render_yaml(loss, parent="loss") }}

Parameters:

{{ render_fields(schema_class_to_fields(loss, exclude=["type"]), details=details) }}

Loss and loss related parameters can also be defined once and applied to all binary output features using the [Type-Global Loss](../defaults.md#type-global-loss) section.

### Metrics

The metrics that are calculated every epoch and are available for binary features are the `accuracy`, `loss`,
`precision`, `recall`, `roc_auc` and `specificity`.

You can set either to be the `validation_metric` in the `training` section of the configuration if the `validation_field` is set as the name of a binary feature.
