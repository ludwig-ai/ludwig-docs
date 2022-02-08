## Numerical Features Preprocessing

Numerical features are directly transformed into a float valued vector of length `n` (where `n` is the size of the dataset) and added to the HDF5 with a key that reflects the name of column in the dataset.
No additional information about them is available in the JSON metadata file.

Parameters available for preprocessing are

- `missing_value_strategy` (default `fill_with_const`): what strategy to follow when there's a missing value in a binary column. The value should be one of `fill_with_const` (replaces the missing value with a specific value specified with the `fill_value` parameter), `fill_with_mode` (replaces the missing values with the most frequent value in the column), `fill_with_mean` (replaces the missing values with the mean of the values in the column), `backfill` (replaces the missing values with the next valid value).
- `fill_value` (default `0`): the value to replace the missing values with in case the `missing_value_strategy` is `fill-value`.
- `normalization` (default `null`): technique to be used when normalizing the numerical feature types. The available options are `null`, `zscore`, `minmax` and `log1p`. If the value is `null` no normalization is performed. If the value is `zscore`, the mean and standard deviation are computed so that values are shifted to have zero mean and 1 standard deviation. If the value is `minmax`, minimun and maximum values are computed and the minimum is subtracted from values and the result is divided by difference between maximum and minimum.  If `normalization` is `log1p` the  value returned is the natural log of 1 plus the original value.  Note: `log1p` is defined only for positive values.

## Numerical Input Features and Encoders

Numerical features have two encoders.
One encoder (`passthrough'`) takes the raw binary values coming from the input placeholders are just returned as outputs.
Inputs are of size `b` while outputs are of size `b x 1` where `b` is the batch size.
The other encoder (`'dense'`) passes the raw binary values through fully connected layers.
In this case the inputs of size `b` are transformed to size `b x h`.

The available encoder parameters are:

- `norm'` (default `null`): norm to apply after the single neuron. It can be `null`, `batch` or `layer`.
- `tied_weights` (default `null`): name of the input feature to tie the weights of the encoder with. It needs to be the name of a feature of the same type and with the same encoder parameters.

There are no additional parameters for the `passthrough` encoder.

## Dense Encoder Parameters

For the `dense` encoder these are the available parameters.

- `num_layers` (default `1`): this is the number of stacked fully connected layers that the input to the feature passes through. Their output is projected in the feature's output space.
- `fc_size` (default `256`): f a `fc_size` is not already specified in `fc_layers` this is the default `fc_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `'glorot_uniform'`): initializer for the weights matrix. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `bias_initializer` (default `'zeros'`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `weights_regularizer` (default `null`): regularizer function applied to the weights matrix.  Valid values are `l1`, `l2` or `l1_l2`.
- `bias_regularizer` (default `null`): regularizer function applied to the bias vector.  Valid values are `l1`, `l2` or `l1_l2`.
- `activity_regularizer` (default `null`): regurlizer function applied to the output of the layer.  Valid values are `l1`, `l2` or `l1_l2`.
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be used for each layer. It indicates the norm of the output and it can be `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters used with `batch` see [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization) or for `layer` see [Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization).
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default `activation` that will be used for each layer. It indicates the activation function applied to the output.
- `dropout` (default `0`): dropout rate

Example numerical feature entry in the input features list:

```yaml
name: numerical_column_name
type: numerical
norm: null
tied_weights: null
encoder: dense
num_layers: 1
fc_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activation: relu
dropout: 0
```

## Numerical Output Features and Decoders

Numerical features can be used when a regression needs to be performed.
There is only one decoder available for numerical features and it is a (potentially empty) stack of fully connected layers, followed by a projection into a single number.

These are the available parameters of a numerical output feature

- `reduce_input` (default `sum`): defines how to reduce an input that is not a vector, but a matrix or a higher order tensor, on the first dimension (second if you count the batch dimension). Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension).
- `dependencies` (default `[]`): the output features this one is dependent on. For a detailed explanation refer to [Output Features Dependencies](#output-features-dependencies).
- `reduce_dependencies` (default `sum`): defines how to reduce the output of a dependent feature that is not a vector, but a matrix or a higher order tensor, on the first dimension (second if you count the batch dimension). Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension).
- `loss` (default `{type: mean_squared_error}`): is a dictionary containing a loss `type`. The available losses `type` are `mean_squared_error` and `mean_absolute_error`.

These are the available parameters of a numerical output feature decoder

- `fc_layers` (default `null`): it is a list of dictionaries containing the parameters of all the fully connected layers. The length of the list determines the number of stacked fully connected layers and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `fc_size`, `norm`, `activation`, `dropout`, `initializer` and `regularize`. If any of those values is missing from the dictionary, the default one specified as a parameter of the decoder will be used instead.
- `num_fc_layers` (default 0): this is the number of stacked fully connected layers that the input to the feature passes through. Their output is projected in the feature's output space.
- `fc_size` (default `256`): if a `fc_size` is not already specified in `fc_layers` this is the default `fc_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default `activation` that will be used for each layer. It indicates the activation function applied to the output.
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be used for each layer. It indicates the norm of the output and it can be `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters used with `batch` see [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization) or for `layer` see [Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization).
- `dropout` (default `0`): dropout rate
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `'glorot_uniform'`): initializer for the weights matrix. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `bias_initializer` (default `'zeros'`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `weights_regularizer` (default `null`): regularizer function applied to the weights matrix.  Valid values are `l1`, `l2` or `l1_l2`.
- `bias_regularizer` (default `null`): regularizer function applied to the bias vector.  Valid values are `l1`, `l2` or `l1_l2`.
- `activity_regularizer` (default `null`): regurlizer function applied to the output of the layer.  Valid values are `l1`, `l2` or `l1_l2`.
- `clip` (default `null`): If not `null` it specifies a minimum and maximum value the predictions will be clipped to. The value can be either a list or a tuple of length 2, with the first value representing the minimum and the second the maximum. For instance `(-5,5)` will make it so that all predictions will be clipped in the `[-5,5]` interval.

Example numerical feature entry (with default parameters) in the output features list:

```yaml
name: numerical_column_name
type: numerical
reduce_input: sum
dependencies: []
reduce_dependencies: sum
loss:
    type: mean_squared_error
fc_layers: null
num_fc_layers: 0
fc_size: 256
activation: relu
norm: null
norm_params: null
dropout: 0
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
clip: null
```

## Numerical Features Measures

The measures that are calculated every epoch and are available for numerical features are `mean_squared_error`, `mean_absolute_error`, `r2` and the `loss` itself.
You can set either of them as `validation_measure` in the `training` section of the configuration if you set the `validation_field` to be the name of a numerical feature.
