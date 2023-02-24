{% from './macros/includes.md' import render_fields, render_yaml %}
{% set mv_details = "See [Missing Value Strategy](./input_features.md#missing-value-strategy) for details." %}
{% set nz_details = "See [Normalization](#normalization) for details." %}
{% set norm_details = "See [Normalization](../combiner.md#normalization) for details." %}
{% set details = {"missing_value_strategy": mv_details, "normalization": nz_details, "norm": norm_details, "fc_norm": norm_details} %}

## Number Features Preprocessing

Number features are directly transformed into a float valued vector of length `n` (where `n` is the size of the dataset)
and added to the HDF5 with a key that reflects the name of column in the dataset.
No additional information about them is available in the JSON metadata file.

{% set preprocessing = get_feature_preprocessing_schema("number") %}
{{ render_yaml(preprocessing, parent="preprocessing") }}

Parameters:

{{ render_fields(schema_class_to_fields(preprocessing), details=details) }}

Preprocessing parameters can also be defined once and applied to all number input features using
the [Type-Global Preprocessing](../defaults.md#type-global-preprocessing) section.

### Normalization

Technique to be used when normalizing the number feature types.

Options:

- **`null`**: No normalization is performed.
- **`zscore`**: The mean and standard deviation are computed so that values are shifted to have zero mean and 1 standard deviation.
- **`minmax`**: The minimum is subtracted from values and the result is divided by difference between maximum and minimum.
- **`log1p`**: The value returned is the natural log of 1 plus the original value. Note: `log1p` is defined only for positive values.
- **`iq`**: The median is subtracted from values and the result is divided by the interquartile range (IQR), i.e., the 75th percentile value minus the 25th percentile value. The resulting data has a zero mean and median and a standard deviation of 1. This is useful if your feature has large outliers since the normalization won't be skewed by those values.

The best normalization techniqe to use depends on the distribution of your data, but `zscore` is a good place to start in many cases.

## Number Input Features and Encoders

Number features have two encoders.
One encoder (`passthrough`) simply returns the raw numerical values coming from the input placeholders as outputs.
Inputs are of size `b` while outputs are of size `b x 1` where `b` is the batch size.
The other encoder (`dense`) passes the raw numerical values through fully connected layers.
In this case the inputs of size `b` are transformed to size `b x h`.

The encoder parameters specified at the feature level are:

- **`tied`** (default `null`): name of the input feature to tie the weights of the encoder with. It needs to be the name of
a feature of the same type and with the same encoder parameters.

Example number feature entry in the input features list:

```yaml
name: number_column_name
type: number
tied: null
encoder: 
    type: dense
```

The available encoder parameters:

- **`type`** (default `passthrough`): the possible values are `passthrough`, `dense` and `sparse`. `passthrough` outputs the
raw integer values unaltered. `dense` randomly initializes a trainable embedding matrix, `sparse` uses one-hot encoding.

Encoder type and encoder parameters can also be defined once and applied to all number input features using
the [Type-Global Encoder](../defaults.md#type-global-encoder) section.

### Passthrough Encoder

{% set encoder = get_encoder_schema("number", "passthrough") %}
{{ render_yaml(encoder, parent="encoder") }}

There are no additional parameters for `passthrough` encoder.

### Dense Encoder

{% set encoder = get_encoder_schema("number", "dense") %}
{{ render_yaml(encoder, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(encoder, exclude=["type"]), details=details) }}

## Number Output Features and Decoders

Number features can be used when a regression needs to be performed.
There is only one decoder available for number features: a (potentially empty) stack of fully connected layers, followed
by a projection to a single number.

Example number output feature using default parameters:

```yaml
name: number_column_name
type: number
reduce_input: sum
dependencies: []
reduce_dependencies: sum
loss:
    type: mean_squared_error
decoder:
    type: regressor
```

Parameters:

- **`reduce_input`** (default `sum`): defines how to reduce an input that is not a vector, but a matrix or a higher order
tensor, on the first dimension (second if you count the batch dimension). Available values are: `sum`, `mean` or `avg`,
`max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension).
- **`dependencies`** (default `[]`): the output features this one is dependent on. For a detailed explanation refer to
[Output Feature Dependencies](../output_features#output-feature-dependencies).
- **`reduce_dependencies`** (default `sum`): defines how to reduce the output of a dependent feature that is not a vector,
but a matrix or a higher order tensor, on the first dimension (second if you count the batch dimension). Available
values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last
vector of the first dimension).
- **`loss`** (default `{type: mean_squared_error}`): is a dictionary containing a loss `type`. Options: 
`mean_squared_error`, `mean_absolute_error`, `root_mean_squared_error`, `root_mean_squared_percentage_error`. See [Loss](#loss) for details.
- **`decoder`** (default: `{"type": "regressor"}`): Decoder for the desired task. Options: `regressor`. See [Decoder](#decoder) for details.

### Decoder

{% set decoder = get_decoder_schema("number", "regressor") %}
{{ render_yaml(decoder, parent="decoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(decoder, exclude=["type"]), details=details) }}

Decoder type and decoder parameters can also be defined once and applied to all number output features using the [Type-Global Decoder](../defaults.md#type-global-decoder) section.

### Loss

{% set loss_classes = get_loss_schemas("number") %}
{% for loss in loss_classes %}
#### {{ loss.name() }}

{{ render_yaml(loss, parent="loss") }}

Parameters:

{{ render_fields(schema_class_to_fields(loss, exclude=["type"]), details=details) }}
{% endfor %}

Loss and loss related parameters can also be defined once and applied to all category output features using the [Type-Global Loss](../defaults.md#type-global-loss) section.

### Metrics

The metrics that are calculated every epoch and are available for number features are `mean_squared_error`,
`mean_absolute_error`, `root_mean_squared_error`, `root_mean_squared_percentage_error` and the `loss` itself.
You can set either of them as `validation_metric` in the `training` section of the configuration if you set the
`validation_field` to be the name of a number feature.
