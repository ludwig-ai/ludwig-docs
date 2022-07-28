Vector features enable providing an ordered set of numerical values within a single feature.

This is useful for providing pre-trained representations or activations obtained from other models or for providing
multivariate inputs and outputs. An interesting use of vector features is the possibility of providing a probability
distribution as output for a multiclass classification problem instead of a single correct class like with a category
feature. Vector output features can also be useful for distillation and noise-aware losses.

## Vector Feature Preprocessing

The data is expected as whitespace separated numerical values. Example: "1.0 0.0 1.04 10.49".  All vectors are expected to be of the same size.

Preprocessing parameters:

- `vector_size` (default `null`): size of the vector. If not provided, it will be inferred from the data.
- `missing_value_strategy` (default `fill_with_const`): what strategy to follow when there's a missing value. The value should be one of `fill_with_const` (replaces the missing value with a specific value specified with the `fill_value` parameter), `fill_with_mode` (replaces the missing values with the most frequent value in the column), `fill_with_mean` (replaces the missing values with the mean of the values in the column), `backfill` (replaces the missing values with the next valid value).
- `fill_value` (default `""`): the value to replace the missing values with in case the `missing_value_strategy` is `fill_with_const`.

## Vector Feature Encoders

The vector feature supports two encoders: `dense` and `passthrough`.

The encoder parameters specified at the feature level are:

- `tied` (default `null`): name of the input feature to tie the weights of the encoder with. It needs to be the name of
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

- `type` (default `dense`): the possible values are `passthrough` and `dense`. `passthrough` outputs the
raw vector values unaltered. `dense` uses a stack of fully connected layers to create an embedding matrix.

### Passthrough Encoder

There are no additional parameters for the `passthrough` encoder.

### Dense Encoder

For vector features, a dense encoder (stack of fully connected layers) can be used to encode the vector.  It takes the
following parameters:

- `layers` (default `null`): a list of dictionaries containing the parameters of all the fully connected layers. The length of the list determines the number of stacked fully connected layers and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `activation`,
`dropout`, `norm`, `norm_params`, `output_size`, `use_bias`, `bias_initializer` and `weights_initializer`. If any of those values is missing from the dictionary, the default one specified as a parameter of the encoder will be used instead. If both `fc_layers` and `num_fc_layers` are `null`, a default list will be assigned to `fc_layers` with the value `[{output_size: 512}, {output_size: 256}]` (only applies if `reduce_output` is not `null`).
- `num_layers` (default `0`): This is the number of stacked fully connected layers.
- `output_size` (default `256`): if a `output_size` is not already specified in `fc_layers` this is the default `output_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `'glorot_uniform'`): initializer for the weights matrix. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
- `bias_initializer` (default `'zeros'`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be used for each layer. It indicates the norm of the output and it can be `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`. For information on parameters used with `batch` see [Torch's documentation on batch normalization](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) or for `layer` see [Torch's documentation on layer normalization](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html).
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default `activation` that will be used for each layer. It indicates the activation function applied to the output.
- `dropout` (default `0`): dropout rate

Example vector feature entry in the input features list using a dense encoder:

```yaml
name: vector_column_name
type: vector
tied: null
encoder: 
    type: dense
    layers: null
    num_layers: 0
    output_size: 256
    use_bias: true
    weights_initializer: glorot_uniform
    bias_initializer: zeros
    norm: null
    norm_params: null
    activation: relu
    dropout: 0
```

## Vector Feature Decoders

```yaml
name: vector_column_name
type: vector
reduce_input: sum
dependencies: []
reduce_dependencies: sum
loss:
    type: sigmoid_cross_entropy
decoder:
    fc_layers: null
    num_fc_layers: 0
    output_size: 256
    use_bias: true
    weights_initializer: glorot_uniform
    bias_initializer: zeros
    activation: relu
    clip: null
```

Vector features can be used when multi-class classification needs to be performed with a noise-aware loss or when the task is multivariate regression.

There is only one decoder available for vector features: a (potentially empty) stack of fully connected layers, followed
by a projection into a tensor of the vector size (optionally followed by a softmax in the case of multi-class classification).

```
+--------------+   +---------+   +-----------+
|Combiner      |   |Fully    |   |Projection |   +-------+
|Output        +--->Connected+--->into Vector+--->Softmax|
|Representation|   |Layers   |   |Size       |   +-------+
+--------------+   +---------+   +-----------+
```

These are the available parameters of the vector output feature.

- `reduce_input` (default `sum`): defines how to reduce an input that is not a vector, but a matrix or a higher order tensor, on the first dimension (second if you count the batch dimension). Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension).
- `dependencies` (default `[]`): the output features this one is dependent on. For a detailed explanation refer to [Output Features Dependencies](#output-features-dependencies).
- `reduce_dependencies` (default `sum`): defines how to reduce the output of a dependent feature that is not a vector, but a matrix or a higher order tensor, on the first dimension (second if you count the batch dimension). Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension).
- `softmax` (default `false`): determines if to apply a softmax at the end of the decoder. It is useful for predicting a vector of values that sum up to 1 and can be interpreted as probabilities.
- `loss` (default `{type: mean_squared_error}`): is a dictionary containing a loss `type`. The available loss `type` are `mean_squared_error`, `mean_absolute_error` and `softmax_cross_entropy` (use it only if `softmax` is `true`).

These are the available parameters of a vector output feature decoder.

- `fc_layers` (default `null`): a list of dictionaries containing the parameters of all the fully connected layers. The length of the list determines the number of stacked fully connected layers and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `activation`,
`dropout`, `norm`, `norm_params`, `output_size`, `use_bias`, `bias_initializer` and `weights_initializer`. If any of those values is missing from the dictionary, the default one specified as a parameter of the decoder will be used instead.
- `num_fc_layers` (default 0): this is the number of stacked fully connected layers that the input to the feature passes through. Their output is projected in the feature's output space.
- `output_size` (default `256`): if a `output_size` is not already specified in `fc_layers` this is the default `output_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `'glorot_uniform'`): initializer for the weights matrix. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
- `bias_initializer` (default `'zeros'`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default `activation` that will be used for each layer. It indicates the activation function applied to the output.
- `clip` (default `null`): If not `null` it specifies a minimum and maximum value the predictions will be clipped to. The value can be either a list or a tuple of length 2, with the first value representing the minimum and the second the maximum. For instance `(-5,5)` will make it so that all predictions will be clipped in the `[-5,5]` interval.

## Vector Features Measures

The metrics that are calculated every epoch and are available for set features are `mean_squared_error`, `mean_absolute_error`, `r2`, and the `loss` itself.

You can set any of them as `validation_metric` in the `training` section of the configuration if you set the
`validation_field` to be the name of a vector feature.
