## Category Features Preprocessing

Category features are transformed into integer valued vectors of size `n` (where `n` is the size of the dataset) and
added to the HDF5 with a key that reflects the name of column in the dataset.
Categories are mapped to integers by first collecting a dictionary of all unique category strings present in the column
of the dataset, ranking them descending by frequency and assigning a sequential integer ID from the most frequent to the
most rare (with 0 assigned to the special unknown placeholder token `<UNK>`).
The column name is added to the JSON file, with an associated dictionary containing

1. the mapping from integer to string (`idx2str`)
2. the mapping from string to id (`str2idx`)
3. the mapping from string to frequency (`str2freq`)
4. the size of the set of all tokens (`vocab_size`)
5. additional preprocessing information (by default how to fill missing values and what token to use to fill missing values)

The parameters available for preprocessing are

- `missing_value_strategy` (default `fill_with_const`): what strategy to follow when there is a missing value in the
category column. The value should be one of `fill_with_const` (replaces the missing value with a specific value
specified with the `fill_value` parameter), `fill_with_mode` (replaces the missing values with the most frequent value
in the column), `fill_with_mean` (replaces the missing values with the mean of the values in the column), `bfill` (replaces the missing values with the next valid value), `ffill` (replaces the missing values with the previous valid value).
- `fill_value` (default `<UNK>`): the value to replace missing values with when `missing_value_strategy` is
`fill_with_const`.
- `lowercase` (default `false`): if the string has to be lowercased before being handled by the tokenizer.
- `most_common` (default `10000`): the maximum number of most common tokens to be considered. If the data contains more
than this amount, the most infrequent tokens will be treated as unknown.

Configuration example:

```yaml
name: color
type: category
preprocessing:
    missing_value_strategy: fill_with_const
    fill_value: <UNK>
    lowercase: false
    most_common: 10000
```

Preprocessing parameters can also be defined once and applied to all category input features using the [Type-Global Preprocessing](../defaults.md#type-global-preprocessing) section.

## Category Input Features and Encoders

Category features have three encoders.
The `passthrough` encoder passes the raw integer values coming from the input placeholders to outputs of size `b x 1`.
The other two encoders map to either `dense` or `sparse` embeddings (one-hot encodings) and returned as outputs of size
`b x h`, where `b` is the batch size and `h` is the dimensionality of the embeddings.

The encoder parameters specified at the feature level are:

- `tied` (default `null`): name of another input feature to tie the weights of the encoder with. It needs to be the name of
a feature of the same type and with the same encoder parameters.

Example category feature entry in the input features list:

```yaml
name: category_column_name
type: category
tied: null
encoder: 
    type: dense
```

The available encoder parameters are:

- `type` (default `dense`): the possible values are `passthrough`, `dense` and `sparse`. `passthrough` outputs the
raw integer values unaltered. `dense` randomly initializes a trainable embedding matrix, `sparse` uses one-hot encoding.

Encoder type and encoder parameters can also be defined once and applied to all category input features using
the [Type-Global Encoder](../defaults.md#type-global-encoder) section.

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
`he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.

### Sparse Encoder

- `embedding_size` (default `256`): it is the maximum embedding size, the actual size will be
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
- `dropout` (default `0`): dropout rate
- `embedding_initializer` (default `null`): the initializer to use. If `null`, the default initialized of each variable
is used (`glorot_uniform` in most cases). Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`,
`uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`,
`he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.

Example category feature entry in the input features list:

```yaml
name: category_column_name
type: category
tied: null
encoder: 
    type: sparse
    embedding_size: 256
    embeddings_on_cpu: false
    pretrained_embeddings: null
    embeddings_trainable: true
    dropout: 0
    embedding_initializer: null
```

## Category Output Features and Decoders

Category features can be used when a multi-class classification needs to be performed.
There is only one decoder available for category features: a (potentially empty) stack of fully connected layers,
followed by a projection into a vector of size of the number of available classes, followed by a softmax.

```
+--------------+   +---------+   +-----------+
|Combiner      |   |Fully    |   |Projection |   +-------+
|Output        +--->Connected+--->into Output+--->Softmax|
|Representation|   |Layers   |   |Space      |   +-------+
+--------------+   +---------+   +-----------+
```

These are the available parameters of a category output feature

- `reduce_input` (default `sum`): defines how to reduce an input that is not a vector, but a matrix or a higher order
tensor, on the first dimension (second if you count the batch dimension). Available values are: `sum`, `mean` or `avg`,
`max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension).
- `calibration` (default `false`): if true, performs calibration by temperature scaling after training is complete.
Calibration uses the validation set to find a scale factor (temperature) which is multiplied with the logits to shift
output probabilities closer to true likelihoods.
- `dependencies` (default `[]`): the output features this one is dependent on. For a detailed explanation refer to
[Output Features Dependencies](../output_features#output-feature-dependencies).
- `reduce_dependencies` (default `sum`): defines how to reduce the output of a dependent feature that is not a vector,
but a matrix or a higher order tensor, on the first dimension (second if you count the batch dimension). Available
values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last
vector of the first dimension).
- `loss` (default `{type: softmax_cross_entropy}`): is a dictionary containing a loss `type`. `softmax_cross_entropy` is
the only supported loss type for category output features.
- `top_k` (default `3`): determines the parameter `k`, the number of categories to consider when computing the `top_k`
measure. It computes accuracy but considering as a match if the true category appears in the first `k` predicted
categories ranked by decoder's confidence.

Decoder type and decoder parameters can also be defined once and applied to all category output features using the [Type-Global Decoder](../defaults.md#type-global-decoder) section.

These are the `loss` parameters

- `confidence_penalty` (default `0`): penalizes overconfident predictions (low entropy) by adding an additional term
that penalizes too confident predictions by adding a `a * (max_entropy - entropy) / max_entropy` term to the loss, where
a is the value of this parameter. Useful in case of noisy labels.
- `robust_lambda` (default `0`): replaces the loss with `(1 - robust_lambda) * loss + robust_lambda / c` where `c` is
the number of classes, which is useful in case of noisy labels.
- `class_weights` (default `1`): the value can be a vector of weights, one for each class, that is multiplied to the
loss of the datapoints that have that class as ground truth. It is an alternative to oversampling in case of unbalanced
class distribution. The ordering of the vector follows the category to integer ID mapping in the JSON metadata file (the
`<UNK>` class needs to be included too). Alternatively, the value can be a dictionary with class strings as keys and
weights as values, like `{class_a: 0.5, class_b: 0.7, ...}`.
- `class_similarities` (default `null`): if not `null` it is a `c x c` matrix in the form of a list of lists that
contains the mutual similarity of classes. It is used if `class_similarities_temperature` is greater than 0. The
ordering of the vector follows the category to integer ID mapping in the JSON metadata file (the `<UNK>` class needs to
be included too).
- `class_similarities_temperature` (default `0`): is the temperature parameter of the softmax that is performed on each
row of `class_similarities`. The output of that softmax is used to determine the supervision vector to provide instead
of the one hot vector that would be provided otherwise for each datapoint. The intuition behind it is that errors
between similar classes are more tolerable than errors between really different classes.

Loss and loss related parameters can also be defined once and applied to all category output features using the [Type-Global Loss](../defaults.md#type-global-loss) section.

These are the available parameters of a category output feature decoder

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
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters
used with `batch` see [Torch's documentation on batch normalization](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)
or for `layer` see [Torch's documentation on layer normalization](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html).
- `dropout` (default `0`): dropout rate
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `glorot_uniform`): initializer for the fully connected weight matrix. Options are:
`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`,
`glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`,
`lecun_uniform`. To see the parameters of each initializer, please refer to [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html).
- `bias_initializer` (default `zeros`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`,
`ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`,
`xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is
possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its
parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to
[torch.nn.init](https://pytorch.org/docs/stable/nn.init.html).

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
    top_k: 3
```

## Category Features Metrics

The measures that are calculated every epoch and are available for category features are `accuracy`, `hits_at_k`
(computes accuracy considering as a match if the true category appears in the first `k` predicted categories ranked by
decoder's confidence) and the `loss` itself.
You can set either of them as `validation_metric` in the `training` section of the configuration if you set the
`validation_field` to be the name of a category feature.
