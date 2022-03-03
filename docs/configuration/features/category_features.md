## Category Features Preprocessing

Category features are transformed into integer valued vectors of size `n` (where `n` is the size of the dataset) and
added to the HDF5 with a key that reflects the name of column in the dataset.
Categories are mapped to integers by first collecting a dictionary of all unique category strings present in the column
of the dataset, ranking them descending by frequency and assigning a sequential integer ID from the most frequent to the
most rare (with 0 assigned to the special unknown placeholder token `<UNK>`).
The column name is added to the JSON file, with an associated dictionary containing

1. the mapping from integer to string (`idx2str`)
1. the mapping from string to id (`str2idx`)
1. the mapping from string to frequency (`str2freq`)
1. the size of the set of all tokens (`vocab_size`)
1. additional preprocessing information (by default how to fill missing values and what token to use to fill missing values)

The parameters available for preprocessing are

- `missing_value_strategy` (default `fill_with_const`): what strategy to follow when there is a missing value in the
category column. The value should be one of `fill_with_const` (replaces the missing value with a specific value
specified with the `fill_value` parameter), `fill_with_mode` (replaces the missing values with the most frequent value
in the column), `fill_with_mean` (replaces the missing values with the mean of the values in the column), `backfill`
(replaces the missing values with the next valid value).
- `fill_value` (default `<UNK>`): the value to replace missing values with when `missing_value_strategy` is
`fill_with_const`.
- `lowercase` (default `false`): if the string has to be lowercased before being handled by the tokenizer.
- `most_common` (default `10000`): the maximum number of most common tokens to be considered. if the data contains more
than this amount, the most infrequent tokens will be treated as unknown.

## Category Input Features and Encoders

Category features have three encoders.
The `passthrough` encoder passes the raw integer values coming from the input placeholders to outputs of size `b x 1`.
The other two encoders map to either `dense` or `sparse` embeddings (one-hot encodings) and returned as outputs of size
`b x h`, where `b` is the batch size and `h` is the dimensionality of the embeddings.

Input feature parameters.

- `encoder` (default `dense`): the possible values are `passthrough`, `dense` and `sparse`. `passthrough` outputs the
raw integer values unaltered. `dense` randomly initializes a trainable embedding matrix, `sparse` uses one-hot encoding.
- `tied_weights` (default `null`): name of the input feature to tie the weights of the encoder with. It needs to be the
name of a feature of the same type and with the same encoder parameters.

Example category feature entry in the input features list:

```yaml
name: category_column_name
type: category
tied_weights: null
encoder: dense
```

The available encoder parameters:

### Dense Encoder

- `embedding_size` (default `256`): the maximum embedding size, the actual size will be
`min(vocabulary_size, embedding_size)` for `dense` representations and exactly `vocabulary_size` for the `sparse`
encoding, where `vocabulary_size` is the number of different strings appearing in the training set in the column the
feature is named after (plus 1 for `<UNK>`).
- `embeddings_on_cpu` (default `false`): by default embedding matrices are stored on GPU memory if a GPU is used, as it
allows for faster access, but in some cases the embedding matrix may be too large. This parameter forces the
placement of the embedding matrix in regular memory and the CPU is used for embedding lookup, slightly slowing down the
process as a result of data transfer between CPU and GPU memory.
- `pretrained_embeddings` (default `null`): by default `dense` embeddings are initialized randomly, but this parameter
allows to specify a path to a file containing embeddings in the [GloVe format](https://nlp.stanford.edu/projects/glove/).
When the file containing the embeddings is loaded, only the embeddings with labels present in the vocabulary are kept,
the others are discarded. If the vocabulary contains strings that have no match in the embeddings file, their embeddings
are initialized with the average of all other embedding plus some random noise to make them different from each other.
This parameter has effect only if `representation` is `dense`.
- `embeddings_trainable` (default `true`): If `true` embeddings are trained during the training process, if `false`
embeddings are fixed. It may be useful when loading pretrained embeddings for avoiding finetuning them. This parameter
has effect only when `representation` is `dense` as `sparse` one-hot encodings are not trainable.
- `dropout` (default `0`): dropout rate.
- `embedding_initializer` (default `null`): the initializer to use. If `null`, the default initialized of each variable
is used (`glorot_uniform` in most cases). Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`,
`uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`,
`he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a
key `type` that identifies the type of initializer and other keys for its parameters, e.g.
`{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to
[TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).

### Sparse Encoder

- `embedding_size` (default `256`): it is the maximum embedding size, the actual size will be
`min(vocabulary_size, embedding_size)` for `dense` representations and exactly `vocabulary_size` for the `sparse` encoding, where `vocabulary_size` is the number of different strings appearing in the training set in the column the feature is named after (plus 1 for `<UNK>`).
- `embeddings_on_cpu` (default `false`): by default embeddings matrices are stored on GPU memory if a GPU is used, as it allows for faster access, but in some cases the embedding matrix may be really big and this parameter forces the placement of the embedding matrix in regular memory and the CPU is used to resolve them, slightly slowing down the process as a result of data transfer between CPU and GPU memory.
- `pretrained_embeddings` (default `null`): by default `dense` embeddings are initialized randomly, but this parameter allows to specify a path to a file containing embeddings in the [GloVe format](https://nlp.stanford.edu/projects/glove/). When the file containing the embeddings is loaded, only the embeddings with labels present in the vocabulary are kept, the others are discarded. If the vocabulary contains strings that have no match in the embeddings file, their embeddings are initialized with the average of all other embedding plus some random noise to make them different from each other. This parameter has effect only if `representation` is `dense`.
- `embeddings_trainable` (default `true`): If `true` embeddings are trained during the training process, if `false` embeddings are fixed. It may be useful when loading pretrained embeddings for avoiding finetuning them. This parameter has effect only when `representation` is `dense` as `sparse` one-hot encodings are not trainable.
- `dropout` (default `false`): determines if there should be a dropout layer after embedding.
- `initializer` (default `null`): the initializer to use. If `null`, the default initialized of each variable is used (`glorot_uniform` in most cases). Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `regularize` (default `true`): if `true` the embedding weights are added to the set of weights that get regularized by a regularization loss (if the `regularization_lambda` in `training` is greater than 0).
- `tied_weights` (default `null`): name of the input feature to tie the weights of the encoder with. It needs to be the name of a feature of the same type and with the same encoder parameters.

Example category feature entry in the input features list:

```yaml
name: category_column_name
type: category
encoder: sparse
tied_weights: null
embedding_size: 256
embeddings_on_cpu: false
pretrained_embeddings: null
embeddings_trainable: true
dropout: 0
initializer: null
regularizer: null
```

## Category Output Features and Decoders

Category features can be used when a multi-class classification needs to be performed.
There is only one decoder available for category features and it is a (potentially empty) stack of fully connected layers, followed by a projection into a vector of size of the number of available classes, followed by a softmax.

```
+--------------+   +---------+   +-----------+
|Combiner      |   |Fully    |   |Projection |   +-------+
|Output        +--->Connected+--->into Output+--->Softmax|
|Representation|   |Layers   |   |Space      |   +-------+
+--------------+   +---------+   +-----------+
```

These are the available parameters of a category output feature

- `reduce_input` (default `sum`): defines how to reduce an input that is not a vector, but a matrix or a higher order tensor, on the first dimension (second if you count the batch dimension). Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension).
- `dependencies` (default `[]`): the output features this one is dependent on. For a detailed explanation refer to [Output Features Dependencies](#output-features-dependencies).
- `reduce_dependencies` (default `sum`): defines how to reduce the output of a dependent feature that is not a vector, but a matrix or a higher order tensor, on the first dimension (second if you count the batch dimension). Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension).
- `loss` (default `{type: softmax_cross_entropy, class_similarities_temperature: 0, class_weights: 1, confidence_penalty: 0, distortion: 1, labels_smoothing: 0, negative_samples: 0, robust_lambda: 0, sampler: null, unique: false}`): is a dictionary containing a loss `type`. The available losses `type` are `softmax_cross_entropy` and `sampled_softmax_cross_entropy`.
- `top_k` (default `3`): determines the parameter `k`, the number of categories to consider when computing the `top_k` measure. It computes accuracy but considering as a match if the true category appears in the first `k` predicted categories ranked by decoder's confidence.

These are the `loss` parameters

- `confidence_penalty` (default `0`): penalizes overconfident predictions (low entropy) by adding an additional term that penalizes too confident predictions by adding a `a * (max_entropy - entropy) / max_entropy` term to the loss, where a is the value of this parameter. Useful in case of noisy labels.
- `robust_lambda` (default `0`): replaces the loss with `(1 - robust_lambda) * loss + robust_lambda / c` where `c` is the number of classes, which is useful in case of noisy labels.
- `class_weights` (default `1`): the value can be a vector of weights, one for each class, that is multiplied to the loss of the datapoints that have that class as ground truth. It is an alternative to oversampling in case of unbalanced class distribution. The ordering of the vector follows the category to integer ID mapping in the JSON metadata file (the `<UNK>` class needs to be included too). Alternatively, the value can be a dictionary with class strings as keys and weights as values, like `{class_a: 0.5, class_b: 0.7, ...}`.
- `class_similarities` (default `null`): if not `null` it is a `c x c` matrix in the form of a list of lists that contains the mutual similarity of classes. It is used if `class_similarities_temperature` is greater than 0. The ordering of the vector follows the category to integer ID mapping in the JSON metadata file (the `<UNK>` class needs to be included too).
- `class_similarities_temperature` (default `0`): is the temperature parameter of the softmax that is performed on each row of `class_similarities`. The output of that softmax is used to determine the supervision vector to provide instead of the one hot vector that would be provided otherwise for each datapoint. The intuition behind it is that errors between similar classes are more tollerable than errors between really different classes.
- `labels_smoothing` (default `0`): If label_smoothing is nonzero, smooth the labels towards `1/num_classes`: `new_onehot_labels = onehot_labels * (1 - label_smoothing) + label_smoothing / num_classes`.
- `negative_samples` (default `0`): if `type` is `sampled_softmax_cross_entropy`, this parameter indicates how many negative samples to use.
- `sampler` (default `null`): options are `fixed_unigram`, `uniform`, `log_uniform`, `learned_unigram`. For a detailed description of the samplers refer to [TensorFlow's documentation](https://www.tensorflow.org/api_guides/python/nn#Candidate_Sampling).
- `distortion` (default `1`): when `loss` is `sampled_softmax_cross_entropy` and the sampler is either `unigram` or `learned_unigram` this is used to skew the unigram probability distribution. Each weight is first raised to the distortion's power before adding to the internal unigram distribution. As a result, distortion = 1.0 gives regular unigram sampling (as defined by the vocab file), and distortion = 0.0 gives a uniform distribution.
- `unique` (default `false`): Determines whether all sampled classes in a batch are unique.

These are the available parameters of a category output feature decoder

- `fc_layers` (default `null`): it is a list of dictionaries containing the parameters of all the fully connected layers. The length of the list determines the number of stacked fully connected layers and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `fc_size`, `norm`, `activation`, `dropout`, `weights_initializer` and `weighs_regularizer`. If any of those values is missing from the dictionary, the default value will be used.
- `num_fc_layers` (default 0): this is the number of stacked fully connected layers that the input to the feature passes through. Their output is projected in the feature's output space.
- `fc_size` (default `256`): if a `fc_size` is not already specified in `fc_layers` this is the default `fc_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default `activation` that will be used for each layer. It indicates the activation function applied to the output.
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be used for each layer. It indicates the norm of the output and it can be `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters used with `batch` see [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization) or for `layer` see [Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization).
- `dropout` (default `false`): determines if there should be a dropout layer after each layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `glorot_uniform`): initializer for the fully connected weight matrix. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `bias_initializer` (default `zeros`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `weights_regularizer` (default `null`): regularizer function applied to the fully connected weights matrix.  Valid values are `l1`, `l2` or `l1_l2`.
- `bias_regularizer` (default `null`): regularizer function applied to the bias vector.  Valid values are `l1`, `l2` or `l1_l2`.
- `activity_regularizer` (default `null`): regurlizer function applied to the output of the layer.  Valid values are `l1`, `l2` or `l1_l2`.

Example category feature entry (with default parameters) in the output features list:

```yaml
name: category_column_name
type: category
reduce_input: sum
dependencies: []
reduce_dependencies: sum
loss:
    type: softmax_cross_entropy
    confidence_penalty: 0
    robust_lambda: 0
    class_weights: 1
    class_similarities: null
    class_similarities_temperature: 0
    labels_smoothing: 0
    negative_samples: 0
    sampler: null
    distortion: 1
    unique: false
fc_layers: null
num_fc_layers: 0
fc_size: 256
activation: relu
norm: null
norm_params: null
dropout: 0
use_biase: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
top_k: 3
```

## Category Features Measures

The measures that are calculated every epoch and are available for category features are `accuracy`, `top_k` (computes accuracy considering as a match if the true category appears in the first `k` predicted categories ranked by decoder's confidence) and the `loss` itself.
You can set either of them as `validation_measure` in the `training` section of the configuration if you set the `validation_field` to be the name of a category feature.
