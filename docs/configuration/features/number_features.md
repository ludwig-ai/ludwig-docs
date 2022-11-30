## Number Features Preprocessing

Number features are directly transformed into a float valued vector of length `n` (where `n` is the size of the dataset)
and added to the HDF5 with a key that reflects the name of column in the dataset.
No additional information about them is available in the JSON metadata file.

Parameters available for preprocessing are

- `missing_value_strategy` (default `fill_with_const`): what strategy to follow when there's a missing value in a number
column. The value should be one of `fill_with_const` (replaces the missing value with a specific value specified with
the `fill_value` parameter), `fill_with_mode` (replaces the missing values with the most frequent value in the column),
`fill_with_mean` (replaces the missing values with the mean of the values in the column), `bfill` (replaces the missing values with the next valid value), `ffill` (replaces the missing values with the previous valid value).
- `fill_value` (default `0`): the value to replace the missing values with in case the `missing_value_strategy` is
`fill_with_const`.
- `normalization` (default `null`): technique to be used when normalizing the number feature types. The available
options are `null`, `zscore`, `minmax`, `log1p` and `iq`. 
    - `null`: no normalization is performed. 
    - `zscore`: the mean and standard deviation are computed so that values are shifted to have zero mean and 1 standard deviation. 
    - `minmax`: the minimum is subtracted from values and the result is divided by difference
between maximum and minimum. 
    - `log1p`: the value returned is the natural log of 1 plus the original value. Note: `log1p` is defined only for positive values.
    - `iq`: the median is subtracted from values and the result is divided by the interquartile range (IQR), i.e., the 75th percentile value minus the 25th percentile value. This is useful if your feature has large outliers since the normalization won't be skewed by those values.

Configuration example:

```yaml
name: click_count
type: number
preprocessing:
    missing_value_strategy: fill_with_const
    fill_value: 0
    normalization: null
```

Preprocessing parameters can also be defined once and applied to all number input features using
the [Type-Global Preprocessing](../defaults.md#type-global-preprocessing) section.

## Number Input Features and Encoders

Number features have two encoders.
One encoder (`passthrough`) simply returns the raw numerical values coming from the input placeholders as outputs.
Inputs are of size `b` while outputs are of size `b x 1` where `b` is the batch size.
The other encoder (`dense`) passes the raw numerical values through fully connected layers.
In this case the inputs of size `b` are transformed to size `b x h`.

The encoder parameters specified at the feature level are:

- `tied` (default `null`): name of the input feature to tie the weights of the encoder with. It needs to be the name of
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

- `type` (default `passthrough`): the possible values are `passthrough`, `dense` and `sparse`. `passthrough` outputs the
raw integer values unaltered. `dense` randomly initializes a trainable embedding matrix, `sparse` uses one-hot encoding.

Encoder type and encoder parameters can also be defined once and applied to all number input features using
the [Type-Global Encoder](../defaults.md#type-global-encoder) section.

### Passthrough Encoder

There are no additional parameters for the `passthrough` encoder.

### Dense Encoder

For the `dense` encoder these are the available parameters.

- `fc_layers` (default `null`): a list of dictionaries containing the parameters of all the fully connected
layers. The length of the list determines the number of stacked fully connected layers and the content of each
dictionary determines the parameters for a specific layer. The available parameters for each layer are: `activation`,
`dropout`, `norm`, `norm_params`, `output_size`, `use_bias`, `bias_initializer` and `weights_initializer`. If any of
those values is missing from the dictionary, the default one specified as a parameter of the encoder will be used
instead.
- `num_layers` (default `1`): this is the number of stacked fully connected layers that the input to the feature passes
through. Their output is projected in the feature's output space.
- `output_size` (default `256`): if `output_size` is not already specified in `fc_layers` this is the default
`output_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `glorot_uniform`): initializer for the weight matrix. Options are: `constant`,
`identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`,
`glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
To see the parameters of each initializer, please
refer to [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html).
- `bias_initializer` (default `zeros`):  initializer for the bias vector. Options are: `constant`, `identity`,
`zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`,
`glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other
keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please
refer to [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html).
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be
used for each layer. It indicates the norm of the output and it can be `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters
used with `batch` see the [Torch documentation on batch normalization](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)
or for `layer` see the [Torch documentation on layer normalization](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html).
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default
`activation` that will be used for each layer. It indicates the activation function applied to the output.
- `dropout` (default `0`): dropout rate

Example number feature entry in the input features list:

```yaml
name: number_column_name
type: number
norm: null
tied: null
encoder: 
    type: dense
    num_layers: 1
    output_size: 256
    use_bias: true
    weights_initializer: glorot_uniform
    bias_initializer: zeros
    activation: relu
    dropout: 0
```

## Number Output Features and Decoders

Number features can be used when a regression needs to be performed.
There is only one decoder available for number features: a (potentially empty) stack of fully connected layers, followed
by a projection to a single number.

These are the available parameters of a number output feature

- `reduce_input` (default `sum`): defines how to reduce an input that is not a vector, but a matrix or a higher order
tensor, on the first dimension (second if you count the batch dimension). Available values are: `sum`, `mean` or `avg`,
`max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension).
- `dependencies` (default `[]`): the output features this one is dependent on. For a detailed explanation refer to
[Output Feature Dependencies](../output_features#output-feature-dependencies).
- `reduce_dependencies` (default `sum`): defines how to reduce the output of a dependent feature that is not a vector,
but a matrix or a higher order tensor, on the first dimension (second if you count the batch dimension). Available
values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last
vector of the first dimension).
- `loss` (default `{type: mean_squared_error}`): is a dictionary containing a loss `type`. The available loss types are
`mean_squared_error` and `mean_absolute_error`.

Loss type and loss related parameters can also be defined once and applied to all number output features using the [Type-Global Loss](../defaults.md#type-global-loss) section.

These are the available parameters of a number output feature decoder

- `fc_layers` (default `null`): a list of dictionaries containing the parameters of all the fully connected
layers. The length of the list determines the number of stacked fully connected layers and the content of each
dictionary determines the parameters for a specific layer. The available parameters for each layer are: `activation`,
`dropout`, `norm`, `norm_params`, `output_size`, `use_bias`, `bias_initializer` and `weights_initializer`. If any of
those values is missing from the dictionary, the default one specified as a parameter of the encoder will be used
instead.
- `num_fc_layers` (default 0): this is the number of stacked fully connected layers that the input to the feature passes
through. Their output is projected in the feature's output space.
- `output_size` (default `256`): if `output_size` is not already specified in `fc_layers` this is the default
`output_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default
`activation` that will be used for each layer. It indicates the activation function applied to the output.
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be
used for each layer. It indicates how the output should be normalized and may be one of `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters
used with `batch` see the [Torch documentation on batch normalization](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)
or for `layer` see the [Torch documentation on layer normalization](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html).
- `dropout` (default `0`): dropout rate
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `xavier_uniform`): initializer for the weight matrix. Options are: `constant`,
`identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`,
`glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
To see the parameters of each initializer, please refer to [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html).
- `bias_initializer` (default `zeros`):  initializer for the bias vector. Options are: `constant`, `identity`,
`zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`,
`glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other
keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please
refer to [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html).
- `clip` (default `null`): If not `null` it specifies a minimum and maximum value the predictions will be clipped to.
The value can be either a list or a tuple of length 2, with the first value representing the minimum and the second the
maximum. For instance `(-5,5)` will make it so that all predictions will be clipped to the `[-5,5]` interval.

Decoder type and decoder parameters can also be defined once and applied to all number output features using the [Type-Global Decoder](../defaults.md#type-global-decoder) section.

Example number feature entry (with default parameters) in the output features list:

```yaml
name: number_column_name
type: number
reduce_input: sum
dependencies: []
reduce_dependencies: sum
loss:
    type: mean_squared_error
decoder:
    fc_layers: null
    num_fc_layers: 0
    output_size: 256
    activation: relu
    norm: null
    norm_params: null
    dropout: 0
    use_bias: true
    weights_initializer: glorot_uniform
    bias_initializer: zeros
    clip: null
```

## Number Features Metrics

The metrics that are calculated every epoch and are available for number features are `mean_squared_error`,
`mean_absolute_error`, `root_mean_squared_error`, `root_mean_squared_percentage_error` and the `loss` itself.
You can set either of them as `validation_metric` in the `training` section of the configuration if you set the
`validation_field` to be the name of a number feature.
