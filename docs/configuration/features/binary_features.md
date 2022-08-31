## Binary Features Preprocessing

Binary features are directly transformed into a binary valued vector of length `n` (where `n` is the size of the
dataset) and added to the HDF5 with a key that reflects the name of column in the dataset.

The parameters available for preprocessing are:

- `missing_value_strategy` (default `fill_with_false`): what strategy to follow when there's a missing value in a binary
column. The value should be one of `fill_with_const` (replaces the missing value with a specific value specified with
the `fill_value` parameter), `fill_with_mode` (replaces the missing values with the most frequent value in the column),
`fill_with_mean` (replaces the missing values with the mean of the values in the column), `backfill` (replaces the
missing values with the next valid value), or `fill_with_false` (default, replaces the missing value with False).
- `fill_value` (default `0`): the value to replace the missing values with in case the `missing_value_strategy` is
`fill_with_const`.
- `fallback_true_label`: In case the binary feature doesn't have conventional boolean values, we will interpret the
`fallback_true_label` as 1 (true) and the other values as 0 (False).

## Binary Input Features and Encoders

```yaml
name: binary_column_name
type: binary
tied: null
encoder: passthrough
```

Binary features have two encoders, `passthrough` and `dense`.

The `passthrough` encoder passes through raw binary values without any transformations. Inputs of size `b` are
transformed to outputs of size `b x 1` where `b` is the batch size.

The `dense` encoder passes the raw binary values through a fully connected layer. Inputs of size `b` are transformed to
size `b x h`.

The available encoder parameters are:

- `tied` (default `null`): name of the input feature to tie the weights of the encoder with. It needs to be the name of
a feature of the same type and with the same encoder parameters.

### Passthrough Encoder

There are no additional parameters for the `passthrough` encoder.

### Dense Encoder

For the `dense` encoder these are the available parameters.

- `num_layers` (default `1`): this is the number of stacked fully connected layers that the input to the feature passes
through. Their output is projected in the feature's output space.
- `output_size` (default `256`): if `output_size` is not already specified in `fc_layers` this is the default
`output_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `'glorot_uniform'`): initializer for the weights matrix. Options are: `constant`,
`identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`,
`glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
- `bias_initializer` (default `zeros`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`,
`ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`,
`xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be
used for each layer. It indicates the norm of the output and it can be `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`. For information on parameters
used with `batch` see [Torch's documentation on batch normalization](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)
or for `layer` see [Torch's documentation on layer normalization](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html).
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default
`activation` that will be used for each layer. It indicates the activation function applied to the output.
- `dropout` (default `0`): dropout rate

## Binary Output Features and Decoders

```yaml
name: binary_column_name
type: binary
reduce_input: sum
dependencies: []
reduce_dependencies: sum
loss:
    type: cross_entropy
    confidence_penalty: 0
    robust_lambda: 0
    positive_class_weight: 1
fc_layers: null
num_fc_layers: 0
activation: relu
norm: null
dropout: 0.2
weights_initializer: glorot_uniform
bias_initializer: zeros
threshold: 0.5
```

Binary output features can be used when a binary classification needs to be performed or when the output is a single
probability. There is only one decoder available for binary features and it is a (potentially empty) stack of fully
connected layers, followed by a projection into a single number followed by a sigmoid function.

These are the available parameters of a binary output feature.

- `reduce_input` (default `sum`): defines how to reduce an input that is not a vector, but a matrix or a higher order
tensor, on the first dimension (second if you count the batch dimension). Available values are: `sum`, `mean` or `avg`,
`max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension).
- `calibration` (default `false`): if true, performs calibration by temperature scaling after training is complete.
Calibration uses the validation set to find a scale factor (temperature) which is multiplied with the logits to shift
output probabilities closer to true likelihoods.
- `dependencies` (default `[]`): the output features this one is dependent on. For a detailed explanation refer to
[Output Feature Dependencies](../output_features#output-feature-dependencies).
- `reduce_dependencies` (default `sum`): defines how to reduce the output of a dependent feature that is not a vector,
but a matrix or a higher order tensor, on the first dimension (second if you count the batch dimension). Available
values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last
vector of the first dimension).
- `loss` (default `{type: cross_entropy, confidence_penalty: 0, robust_lambda: 0, positive_class_weight: 1}`): is a
dictionary containing a loss `type` and its hyperparameters. The only available loss `type` is `cross_entropy` (cross
entropy), and the optional parameters are `confidence_penalty` (an additional term that penalizes too confident
predictions by adding a `a * (max_entropy - entropy) / max_entropy` term to the loss, where a is the value of this
parameter), `robust_lambda` (replaces the loss with `(1 - robust_lambda) * loss + robust_lambda / 2` which is useful in
case of noisy labels) and `positive_class_weight` (multiplies the loss for the positive class, increasing its importance).

These are the available parameters of a binary output feature decoder.

- `fc_layers` (default `null`): a list of dictionaries containing the parameters of all the fully connected layers.
The length of the list determines the number of stacked fully connected layers and the content of each dictionary
determines the parameters for a specific layer. The available parameters for each layer are: `activation`, `dropout`,
`norm`, `norm_params`, `output_size`, `use_bias`, `bias_initializer` and `weights_initializer`. If any of those values
is missing from the dictionary, the default one specified as a parameter of the encoder will be used instead.
- `num_fc_layers` (default 0): this is the number of stacked fully connected layers that the input to the feature passes
through. Their output is projected in the feature's output space.
- `output_size` (default `256`): if a `output_size` is not already specified in `fc_layers` this is the default
`output_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default
`activation` that will be used for each layer. It indicates the activation function applied to the output.
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be
used for each layer. It indicates the norm of the output and it can be `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`. For information on parameters
used with `batch` see [Torch's documentation on batch normalization](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)
or for `layer` see [Torch's documentation on layer normalization](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html).
- `dropout` (default `0`): dropout rate
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `'glorot_uniform'`): initializer for the weights matrix. Options are: `constant`,
`identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`,
`glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
- `bias_initializer` (default `'zeros'`): initializer for the bias vector. Options are: `constant`, `identity`, `zeros`,
`ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`,
`xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
- `threshold` (default `0.5`): The threshold above (greater or equal) which the predicted output of the sigmoid will be
mapped to 1.

## Binary Feature Metrics

The only metrics that are calculated every epoch and are available for binary features are the `accuracy` and the `loss`
itself.

You can set either to be the `validation_metric` in the `training` section of the configuration if the
`validation_field` is set as the name of a binary feature.
