H3 is a indexing system for representing geospatial data.
For more details about it refer to: https://eng.uber.com/h3/ .

## H3 Features Preprocessing

Ludwig will parse the H3 64bit encoded format automatically. 
The parameters for preprocessing are:

- `missing_value_strategy` (default `fill_with_const`): what strategy to follow when there's a missing value in a binary column. The value should be one of `fill_with_const` (replaces the missing value with a specific value specified with the `fill_value` parameter), `fill_with_mode` (replaces the missing values with the most frequent value in the column), `fill_with_mean` (replaces the missing values with the mean of the values in the column), `backfill` (replaces the missing values with the next valid value).
- `fill_value` (default `576495936675512319`): the value to replace the missing values with in case the `missing_value_strategy` is `fill_value`. This is a 64bit integer comaptible with the H3 bit layout. The default value encodes mode 1, edge 0, resolution 0, base_cell 0.

Example of a preprocessing specification:

```yaml
name: h3_feature_name
type: h3
preprocessing:
  missing_value_strategy: fill_with_const
  fill_value: 576495936675512319
```


## H3 Input Features and Encoders

Input date features are transformed into a int valued tensors of size `N x 8` (where `N` is the size of the dataset and the 8 dimensions contain year, month, day, weekday, yearday, hour, minute and second) and added to HDF5 with a key that reflects the name of column in the dataset.

Currently there are two encoders supported for dates: Embed Encoder and Wave encoder which can be set by setting `encoder` parameter to `embed` or `wave` in the input feature dictionary in the configuration (`embed` is the default one).

### Embed Encoder

This encoder encodes each components of the H3 representation (mode, edge, resolution, base cell and childern cells) with embeddings.
Chidren cells with value `0` will be masked out.
After the embedding, all embeddings are summed and optionally passed through a stack of fully connected layers.
It takes the following optional parameters:

- `embedding_size` (default `10`): it is the maximum embedding size adopted..
- `embeddings_on_cpu` (default `false`): by default embeddings matrices are stored on GPU memory if a GPU is used, as it allows for faster access, but in some cases the embedding matrix may be really big and this parameter forces the placement of the embedding matrix in regular memory and the CPU is used to resolve them, slightly slowing down the process as a result of data transfer between CPU and GPU memory.
- `fc_layers` (default `null`): it is a list of dictionaries containing the parameters of all the fully connected layers. The length of the list determines the number of stacked fully connected layers and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `fc_size`, `norm`, `activation` and `regularize`. If any of those values is missing from the dictionary, the default one specified as a parameter of the encoder will be used instead.
- `num_fc_layers` (default `0`): This is the number of stacked fully connected layers.
- `fc_size` (default `10`): if a `fc_size` is not already specified in `fc_layers` this is the default `fc_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
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

Example date feature entry in the input features list using an embed encoder:

```yaml
name: h3_column_name
type: h3
encoder: embed
embedding_size: 10
embeddings_on_cpu: false
fc_layers: null
num_fc_layers: 0
fc_size: 10
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
norm: null
norm_params: null
activation: relu
dropout: 0
```

### Weighted Sum Embed Encoder

This encoder encodes each components of the H3 representation (mode, edge, resolution, base cell and childern cells) with embeddings.
Chidren cells with value `0` will be masked out.
After the embedding, all embeddings are summed with a weighted sum (with learned weights) and optionally passed through a stack of fully connected layers.
It takes the following optional parameters:

- `embedding_size` (default `10`): it is the maximum embedding size adopted..
- `embeddings_on_cpu` (default `false`): by default embeddings matrices are stored on GPU memory if a GPU is used, as it allows for faster access, but in some cases the embedding matrix may be really big and this parameter forces the placement of the embedding matrix in regular memory and the CPU is used to resolve them, slightly slowing down the process as a result of data transfer between CPU and GPU memory.
- `should_softmax` (default `false`): determines if the weights of the weighted sum should be passed though a softmax layer before being used.
- `fc_layers` (default `null`): it is a list of dictionaries containing the parameters of all the fully connected layers. The length of the list determines the number of stacked fully connected layers and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `fc_size`, `norm`, `activation` and `regularize`. If any of those values is missing from the dictionary, the default one specified as a parameter of the encoder will be used instead.
- `num_fc_layers` (default `0`): This is the number of stacked fully connected layers.
- `fc_size` (default `10`): if a `fc_size` is not already specified in `fc_layers` this is the default `fc_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
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
- `reduce_output` (default `sum`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).


Example date feature entry in the input features list using an embed encoder:

```yaml
name: h3_column_name
type: h3
encoder: weighted_sum
embedding_size: 10
embeddings_on_cpu: false
should_softmax: false
fc_layers: null
num_fc_layers: 0
fc_size: 10
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
norm: null
norm_params: null
activation: relu
dropout: 0
reduce_output: sum
```

### RNN Encoder

This encoder encodes each components of the H3 representation (mode, edge, resolution, base cell and childern cells) with embeddings.
Chidren cells with value `0` will be masked out.
After the embedding, all embeddings are passed through an RNN encoder.
The intuition behind this is that, starting from the base cell, the sequence of children cells can be seen as a sequence encoding the path in the tree of all H3 hexes, thus the encoding with  recurrent model.
It takes the following optional parameters:

- `embedding_size` (default `10`): it is the maximum embedding size adopted..
- `embeddings_on_cpu` (default `false`): by default embeddings matrices are stored on GPU memory if a GPU is used, as it allows for faster access, but in some cases the embedding matrix may be really big and this parameter forces the placement of the embedding matrix in regular memory and the CPU is used to resolve them, slightly slowing down the process as a result of data transfer between CPU and GPU memory.
- `num_layers` (default `1`): the number of stacked recurrent layers.
- `state_size` (default `256`): the size of the state of the rnn.
- `cell_type` (default `rnn`): the type of recurrent cell to use. Available values are: `rnn`, `lstm`, `lstm_block`, `lstm`, `ln`, `lstm_cudnn`, `gru`, `gru_block`, `gru_cudnn`. For reference about the differences between the cells please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell). We suggest to use the `block` variants on CPU and the `cudnn` variants on GPU because of their increased speed.
- `bidirectional` (default `false`): if `true` two recurrent networks will perform encoding in the forward and backward direction and their outputs will be concatenated.
- `activation` (default `'tanh'`): activation function to use
- `recurrent_activation` (default `'sigmoid'`): activation function to use in the recurrent step
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `unit_forget_bias` (default `true`): If `true`, add 1 to the bias of the forget gate at initialization
- `weights_initializer` (default `'glorot_uniform'`): initializer for the weights matrix. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `recurrent_initializer` (default `'orthogonal'`): initializer for recurrent matrix weights
- `bias_initializer` (default `'zeros'`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `weights_regularizer` (default `null`): regularizer function applied to the weights matrix.  Valid values are `l1`, `l2` or `l1_l2`.
- `recurrent_regularizer` (default `null`): regularizer function applied to recurrent matrix weights
- `bias_regularizer` (default `null`): regularizer function applied to the bias vector.  Valid values are `l1`, `l2` or `l1_l2`.
- `activity_regularizer` (default `null`): regurlizer function applied to the output of the layer.  Valid values are `l1`, `l2` or `l1_l2`.
- `dropout` (default `0.0`): dropout rate
- `recurrent_dropout` (default `0.0`): dropout rate for recurrent state
- `initializer` (default `null`): the initializer to use. If `null`, the default initialized of each variable is used (`glorot_uniform` in most cases). Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `regularize` (default `true`): if `true` the embedding weights are added to the set of weights that get regularized by a regularization loss (if the `regularization_lambda` in `training` is greater than 0).
- `reduce_output` (default `last`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).

Example date feature entry in the input features list using an embed encoder:

```yaml
name: h3_column_name
type: h3
encoder: rnn
embedding_size: 10
embeddings_on_cpu: false
num_layers: 1
cell_type: rnn
state_size: 10
bidirectional: false
activation: tanh
recurrent_activation: sigmoid
use_bias: true
unit_forget_bias: true
weights_initializer: glorot_uniform
recurrent_initializer: orthogonal
bias_initializer: zeros
weights_regularizer: null
recurrent_regularizer: null
bias_regularizer: null
activity_regularizer: null
dropout: 0.0
recurrent_dropout: 0.0
initializer: null
regularize: true
reduce_output: last
```

## H3 Output Features and Decoders

There are no date decoders at the moment (WIP), so H3 cannot be used as output features.

## H3 Features Measures

As no H3 decoders are available at the moment, there are also no date measures.