## Set Features Preprocessing

Set features are expected to be provided as a string of elements separated by whitespace, e.g. "elem5 elem9 elem6".
The string values are transformed into a binary (int8 actually) valued matrix of size `n x l` (where `n` is the number
of rows in the dataset and `l` is the minimum of the size of the biggest set and a `max_size` parameter) and added to
HDF5 with a key that reflects the name of column in the dataset.
The way sets are mapped into integers consists in first using a tokenizer to map each input string to a sequence of set
elements (by default this is done by splitting on spaces).
Next a dictionary is constructed which maps each unique element to its frequency in the dataset column. Elements are
ranked by frequency and a sequential integer ID is assigned in ascending order from the most frequent to the most rare.
The column name is added to the JSON file, with an associated dictionary containing

1. the mapping from integer to string (`idx2str`)
1. the mapping from string to id (`str2idx`)
1. the mapping from string to frequency (`str2freq`)
1. the maximum size of all sets (`max_set_size`)
1. additional preprocessing information (by default how to fill missing values and what token to use to fill missing values)

The parameters available for preprocessing are

- `tokenizer` (default `space`): defines how to transform the raw text content of the dataset column to a set of
elements. The default value `space` splits the string on spaces. Common options include: `underscore` (splits on
underscore), `comma` (splits on comma), `json` (decodes the string into a set or a list through a JSON parser). For all
available options see [Tokenizers](../../preprocessing#tokenizers).
- `missing_value_strategy` (default `fill_with_const`): what strategy to follow when there's a missing value in a set
column. The value should be one of `fill_with_const` (replaces the missing value with a specific value specified with
the `fill_value` parameter), `fill_with_mode` (replaces the missing values with the most frequent value in the column),
`fill_with_mean` (replaces the missing values with the mean of the values in the column), `backfill` (replaces the
missing values with the next valid value).
- `fill_value` (default `0`): the value to replace the missing values with in case the `missing_value_strategy` is
`fill_with_const`.
- `lowercase` (default `false`): if the string has to be lowercased before being handled by the tokenizer.
- `most_common` (default `10000`): the maximum number of most common tokens to be considered. if the data contains more
than this amount, the most infrequent tokens will be treated as unknown.

## Set Input Features and Encoders

Set features have one encoder, the raw binary values coming from the input placeholders are first transformed to sparse
integer lists, then they are mapped to either dense or sparse embeddings (one-hot encodings), finally they are
reduced on the sequence dimension and returned as an aggregated embedding vector.
Inputs are of size `b` while outputs are of size `b x h` where `b` is the batch size and `h` is the dimensionality of
the embeddings.

```
+-+
|0|          +-----+
|0|   +-+    |emb 2|   +-----------+
|1|   |2|    +-----+   |Aggregation|
|0+--->4+---->emb 4+--->Reduce     +->
|1|   |5|    +-----+   |Operation  |
|1|   +-+    |emb 5|   +-----------+
|0|          +-----+
+-+
```

The available encoder parameters are

- `representation` (default `dense`): the possible values are `dense` and `sparse`. `dense` means the embeddings are
initialized randomly, `sparse` means they are initialized to be one-hot encodings.
- `embedding_size` (default `50`): it is the maximum embedding size, the actual size will be
`min(vocabulary_size, embedding_size)` for `dense` representations and exactly `vocabulary_size` for the `sparse`
encoding, where `vocabulary_size` is the number of different strings appearing in the training set in the input column
(plus 2 for `<UNK>` and `<PAD>` tokens if present).
- `embeddings_trainable` (default `true`): If `true` embeddings are trained during the training process, if `false`
embeddings are fixed. It may be useful when loading pretrained embeddings for avoiding finetuning them. This parameter
has effect only when `representation` is `dense` as `sparse` one-hot encodings are not trainable.
- `pretrained_embeddings` (default `null`): by default `dense` embeddings are initialized randomly, but this parameter
allows to specify a path to a file containing embeddings in the [GloVe format](https://nlp.stanford.edu/projects/glove/).
When the file containing the embeddings is loaded, only the embeddings with labels present in the vocabulary are kept,
the others are discarded. If the vocabulary contains strings that have no match in the embeddings file, their embeddings
are initialized with the average of all other embedding plus some random noise to make them different from each other.
This parameter has effect only if `representation` is `dense`.
- `embeddings_on_cpu` (default `false`): by default embedding matrices are stored on GPU memory if a GPU is used, as it
allows for faster access, but in some cases the embedding matrix may be too large. This parameter forces the placement
of the embedding matrix in regular memory and the CPU is used for embedding lookup, slightly slowing down the process as
a result of data transfer between CPU and GPU memory.
- `fc_layers` (default `null`): a list of dictionaries containing the parameters of all the fully connected
layers. The length of the list determines the number of stacked fully connected layers and the content of each
dictionary determines the parameters for a specific layer. The available parameters for each layer are: `activation`,
`dropout`, `norm`, `norm_params`, `output_size`, `use_bias`, `bias_initializer` and `weights_initializer`. If any of
those values is missing from the dictionary, the default one specified as a parameter of the encoder will be used
instead. If both `fc_layers` and `num_fc_layers` are `null`, a default list will be assigned to `fc_layers` with the
value `[{output_size: 512}, {output_size: 256}]` (only applies if `reduce_output` is not `null`).
- `num_fc_layers` (default `1`): this is the number of stacked fully connected layers that the input to the feature
passes through. Their output is projected in the feature's output space.
- `output_size` (default `10`): if `output_size` is not already specified in `fc_layers` this is the default
`output_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `glorot_uniform`): initializer for the weight matrix. Options are: `constant`,
`identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`,
`glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other
keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer,
please refer to [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html).
- `bias_initializer` (default `zeros`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`,
`ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`,
`xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is
possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its
parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to
[torch.nn.init](https://pytorch.org/docs/stable/nn.init.html).
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be
used for each layer. It indicates how the output should be normalized and may be one of `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters
used with `batch` see the [Torch documentation on batch normalization](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)
or for `layer` see the [Torch documentation on layer normalization](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html).
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default
`activation` that will be used for each layer. It indicates the activation function applied to the output.
- `dropout` (default `0`): dropout rate
- `reduce_output` (default `sum`): describes the strategy to use to aggregate the embeddings of the items of the set.
Available values are: `sum`, `mean` or `avg`, `max`, `concat` and  `null` (which does not reduce and returns the full tensor).
- `tied_weights` (default `null`): name of the input feature to tie the weights of the encoder with. It needs to be the
name of a feature of the same type and with the same encoder parameters.

Example set feature entry in the input features list:

```yaml
name: set_column_name
type: set
representation: dense
embedding_size: 50
embeddings_trainable: true
pretrained_embeddings: null
embeddings_on_cpu: false
fc_layers: null
num_fc_layers: 0
output_size: 10
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
norm: null
norm_params: null
activation: relu
dropout: 0.0
reduce_output: sum
tied_weights: null
```

## Set Output Features and Decoders

Set features can be used when multi-label classification needs to be performed.
There is only one decoder available for set features: a (potentially empty) stack of fully connected layers, followed by
a projection into a vector of size of the number of available classes, followed by a sigmoid.

```
+--------------+   +---------+   +-----------+
|Combiner      |   |Fully    |   |Projection |   +-------+
|Output        +--->Connected+--->into Output+--->Sigmoid|
|Representation|   |Layers   |   |Space      |   +-------+
+--------------+   +---------+   +-----------+
```

These are the available parameters of the set output feature

- `reduce_input` (default `sum`): defines how to reduce an input that is not a vector, but a matrix or a higher order
tensor, on the first dimension (second if you count the batch dimension). Available values are: `sum`, `mean` or `avg`,
`max`, `concat` (concatenates along the first dimension).
- `dependencies` (default `[]`): the output features this one is dependent on. For a detailed explanation refer to
[Output Feature Dependencies](../output_features#output-feature-dependencies).
- `reduce_dependencies` (default `sum`): defines how to reduce the output of a dependent feature that is not a vector,
but a matrix or a higher order tensor, on the first dimension (second if you count the batch dimension). Available
values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last
vector of the first dimension).
- `loss` (default `{type: sigmoid_cross_entropy}`): is a dictionary containing a loss `type`. The only supported loss
`type` for set features is `sigmoid_cross_entropy`.

These are the available parameters of a set output feature decoder

- `fc_layers` (default `null`): a list of dictionaries containing the parameters of all the fully connected
layers. The length of the list determines the number of stacked fully connected layers and the content of each
dictionary determines the parameters for a specific layer. The available parameters for each layer are: `activation`,
`dropout`, `norm`, `norm_params`, `output_size`, `use_bias`, `bias_initializer` and `weights_initializer`. If any of
those values is missing from the dictionary, the default one specified as a parameter of the decoder will be used instead.
- `num_fc_layers` (default 0): this is the number of stacked fully connected layers that the input to the feature passes
through. Their output is projected in the feature's output space.
- `output_size` (default `256`): if `output_size` is not already specified in `fc_layers` this is the default
`output_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `glorot_uniform`): initializer for the weight matrix. Options are: `constant`,
`identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`,
`glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other
keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please
refer to [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html).
- `bias_initializer` (default `zeros`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`,
`ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`,
`xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is
possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its
parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to
[torch.nn.init](https://pytorch.org/docs/stable/nn.init.html).
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be
used for each layer. It indicates how the output should be normalized and may be one of `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters
used with `batch` see the [Torch documentation on batch normalization](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)
or for `layer` see the [Torch documentation on layer normalization](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html).
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default
`activation` that will be used for each layer. It indicates the activation function applied to the output.
- `dropout` (default `0`): dropout rate
- `threshold` (default `0.5`): The threshold above (greater or equal) which the predicted output of the sigmoid will be
mapped to 1.

Example set feature entry (with default parameters) in the output features list:

```yaml
name: set_column_name
type: set
reduce_input: sum
dependencies: []
reduce_dependencies: sum
loss:
    type: sigmoid_cross_entropy
fc_layers: null
num_fc_layers: 0
output_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
norm: null
norm_params: null
activation: relu
dropout: 0.0
threshold: 0.5
```

## Set Features Metrics

The metrics that are calculated every epoch and are available for set features are `jaccard` (counts the number of
elements in the intersection of prediction and label divided by number of elements in the union) and the `loss` itself.
You can set either of them as `validation_metric` in the `training` section of the configuration if you set the
`validation_field` to be the name of a sequence feature.
