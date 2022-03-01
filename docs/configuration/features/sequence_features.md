## Sequence Features Preprocessing

Sequence features are transformed into an integer valued matrix of size `n x l` (where `n` is the number of rows and `l`
is the minimum of the length of the longest sequence and a `sequence_length_limit` parameter) and added to HDF5 with a
key that reflects the name of column in the dataset.
The way sequences are mapped into integers consists of first using a tokenizer to map text to sequences of tokens
(default tokenization is done by splitting on spaces).
Next, a dictionary is constructed which maps each unique token to its frequency in the dataset column. Tokens are ranked
by frequency and a sequential integer ID is assigned from the most frequent to the most rare (with 0 being assigned to
`<PAD>` used for padding and 1 assigned to `<UNK>` item).
The column name is added to the JSON file, with an associated dictionary containing

1. the mapping from integer to string (`idx2str`)
1. the mapping from string to id (`str2idx`)
1. the mapping from string to frequency (`str2freq`)
1. the maximum length of all sequences (`sequence_length_limit`)
1. additional preprocessing information (how to fill missing values and what token to use to fill missing values)

The parameters available for preprocessing are

- `sequence_length_limit` (default `256`): the maximum length of the sequence. Sequences that are longer than this value
will be truncated, while sequences that are shorter will be padded.
- `most_common` (default `20000`): the maximum number of most common tokens to be considered. if the data contains more
than this amount, the most infrequent tokens will be treated as unknown.
- `padding_symbol` (default `<PAD>`): the string used as a padding symbol, mapped to the integer ID 0 in the vocabulary.
- `unknown_symbol` (default `<UNK>`): the string used as the unknown placeholder, mapped to the integer ID 1 in the
vocabulary.
- `padding` (default `right`): the direction of the padding. `right` and `left` are available options.
- `tokenizer` (default `space`): defines how to map from the raw string content of the dataset column to a sequence of
elements. For the available options refer to the [Tokenizers](../../preprocessing#tokenizers) section.
- `lowercase` (default `false`): if the string has to be lowercase before being handled by the tokenizer.
- `vocab_file` (default `null`)  filepath string to a UTF-8 encoded file containing the sequence's vocabulary. On each
line the first string until `\t` or `\n` is considered a word.
- `missing_value_strategy` (default `fill_with_const`): what strategy to follow when there's a missing value in the
column. The value should be one of `fill_with_const` (replaces the missing value with the value specified by the
`fill_value` parameter), `fill_with_mode` (replaces the missing values with the most frequent value in the column),
`fill_with_mean` (replaces the missing values with the mean of the values in the column), `backfill` (replaces the
missing values with the next valid value).
- `fill_value` (default `<UNK>`): the value to replace the missing values with in case the `missing_value_strategy` is
`fill_value`.

## Sequence Input Features and Encoders

Sequence features have several encoders and each of them has its own parameters.
Inputs are of size `b` while outputs are of size `b x h` where `b` is the batch size and `h` is the dimensionality of
the output of the encoder.
In case a representation for each element of the sequence is needed (for example for tagging them, or for using an
attention mechanism), one can specify the parameter `reduce_output` to be `null` and the output will be a `b x s x h`
tensor where `s` is the length of the sequence.
Some encoders, because of their inner workings, may require additional parameters to be specified in order to obtain one
representation for each element of the sequence.
For instance the `parallel_cnn` encoder by default pools and flattens the sequence dimension and then passes the
flattened vector through fully connected layers, so in order to obtain the full sequence tensor one has to specify
`reduce_output: null`.

Sequence input feature parameters are

- `encoder` (default `parallel_cnn`): the name of the encoder to use to encode the sequence, one of `embed`,
`parallel_cnn`, `stacked_cnn`, `stacked_parallel_cnn`, `rnn`, `cnnrnn`, `transformer` and `passthrough` (equivalent to
`null` or `None`).
- `tied_weights` (default `null`): name of the input feature to tie the weights of the encoder with. It needs to be the
name of a feature of the same type and with the same encoder parameters.

### Embed Encoder

The embed encoder simply maps each integer in the sequence to an embedding, creating a `b x s x h` tensor where `b` is
the batch size, `s` is the length of the sequence and `h` is the embedding size.
The tensor is reduced along the `s` dimension to obtain a single vector of size `h` for each element of the batch.
If you want to output the full `b x s x h` tensor, you can specify `reduce_output: null`.

```
       +------+
       |Emb 12|
       +------+
+--+   |Emb 7 |
|12|   +------+
|7 |   |Emb 43|   +-----------+
|43|   +------+   |Aggregation|
|65+--->Emb 65+--->Reduce     +->
|23|   +------+   |Operation  |
|4 |   |Emb 23|   +-----------+
|1 |   +------+
+--+   |Emb 4 |
       +------+
       |Emb 1 |
       +------+
```

These are the parameters available for the embed encoder

- `representation` (default `dense`): the possible values are `dense` and `sparse`. `dense` means the embeddings are
initialized randomly, `sparse` means they are initialized to be one-hot encodings.
- `embedding_size` (default `256`): it is the maximum embedding size, the actual size will be
`min(vocabulary_size, embedding_size)` for `dense` representations and exactly `vocabulary_size` for the `sparse`
encoding, where `vocabulary_size` is the number of different strings appearing in the training set (plus 2 for `<UNK>`
and `<PAD>` tokens).
- `embeddings_trainable` (default `true`): If `true` embeddings are trained during the training process, if `false`
embeddings are fixed. It may be useful when loading pretrained embeddings for avoiding finetuning them. This parameter
has effect only when `representation` is `dense`, `sparse` one-hot encodings are not trainable.
- `pretrained_embeddings` (default `null`): by default `dense` embeddings are initialized randomly, but this parameter
allows to specify a path to a file containing embeddings in the [GloVe format](https://nlp.stanford.edu/projects/glove/).
When the file containing the embeddings is loaded, only the embeddings with labels present in the vocabulary are kept,
the others are discarded. If the vocabulary contains strings that have no match in the embeddings file, their embeddings
are initialized with the average of all other embedding plus some random noise to make them different from each other.
This parameter has effect only if `representation` is `dense`.
- `embeddings_on_cpu` (default `false`): by default embedding matrices are stored on GPU memory if a GPU is used, as it
allows for faster access, but in some cases the embedding matrix may be too large. This parameter forces the
placement of the embedding matrix in regular memory and the CPU is used for embedding lookup, slightly slowing down the
process as a result of data transfer between CPU and GPU memory.
- `dropout` (default `0`): dropout rate.
- `weights_initializer` (default `glorot_uniform`): initializer for the weight matrix. Options are: `constant`,
`identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`,
`glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other
keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please
refer to [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html).
- `reduce_output` (default `sum`): defines how to reduce the output tensor along the `s` sequence length dimension if
the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates
along the sequence dimension), `last` (selects the last vector of the sequence dimension) and  `null` (which does not reduce
and returns the full tensor).

Example sequence feature entry in the input features list using an embed encoder:

```yaml
name: sequence_column_name
type: sequence
encoder: embed
tied_weights: null
representation: dense
embedding_size: 256
embeddings_trainable: true
pretrained_embeddings: null
embeddings_on_cpu: false
dropout: 0
weights_initializer: null
reduce_output: sum
```

### Parallel CNN Encoder

The parallel cnn encoder is inspired by
[Yoon Kim's Convolutional Neural Network for Sentence Classification](https://arxiv.org/abs/1408.5882).
It works by first mapping the input integer sequence `b x s` (where `b` is the batch size and `s` is the length of the
sequence) into a sequence of embeddings, then it passes the embedding through a number of parallel 1d convolutional
layers with different filter size (by default 4 layers with filter size 2, 3, 4 and 5), followed by max pooling and
concatenation.
This single vector concatenating the outputs of the parallel convolutional layers is then passed through a stack of
fully connected layers and returned as a `b x h` tensor where `h` is the output size of the last fully connected layer.
If you want to output the full `b x s x h` tensor, you can specify `reduce_output: null`.

```
                   +-------+   +----+
                +-->1D Conv+--->Pool+-+
       +------+ |  |Width 2|   +----+ |
       |Emb 12| |  +-------+          |
       +------+ |                     |
+--+   |Emb 7 | |  +-------+   +----+ |
|12|   +------+ +-->1D Conv+--->Pool+-+
|7 |   |Emb 43| |  |Width 3|   +----+ |           +---------+
|43|   +------+ |  +-------+          | +------+  |Fully    |
|65+--->Emb 65+-+                     +->Concat+-->Connected+->
|23|   +------+ |  +-------+   +----+ | +------+  |Layers   |
|4 |   |Emb 23| +-->1D Conv+--->Pool+-+           +---------+
|1 |   +------+ |  |Width 4|   +----+ |
+--+   |Emb 4 | |  +-------+          |
       +------+ |                     |
       |Emb 1 | |  +-------+   +----+ |
       +------+ +-->1D Conv+--->Pool+-+
                   |Width 5|   +----+
                   +-------+
```

These are the available for an parallel cnn encoder:

- `representation` (default `dense`): the possible values are `dense` and `sparse`. `dense` means the embeddings are
initialized randomly, `sparse` means they are initialized to be one-hot encodings.
- `embedding_size` (default `256`): it is the maximum embedding size, the actual size will be
`min(vocabulary_size, embedding_size)` for `dense` representations and exactly `vocabulary_size` for the `sparse`
encoding, where `vocabulary_size` is the number of unique strings appearing in the training set in the input column
(plus 2 for `<UNK>` and `<PAD>` tokens).
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
allows for faster access, but in some cases the embedding matrix may be too large. This parameter forces the
placement of the embedding matrix in regular memory and the CPU is used for embedding lookup, slightly slowing down the
process as a result of data transfer between CPU and GPU memory.
- `conv_layers` (default `null`): a list of dictionaries containing the parameters of all the convolutional layers. The
length of the list determines the number of parallel convolutional layers and the content of each dictionary determines
the parameters for a specific layer. The available parameters for each layer are: `activation`, `dropout`, `norm`,
`norm_params`, `num_filters`, `filter_size`, `strides`, `padding`, `dilation_rate`, `use_bias`, `pool_function`,
`pool_padding`, `pool_size`, `pool_strides`, `bias_initializer`, `weights_initializer`. If any of those values is
missing from the dictionary, the default one specified as a parameter of the encoder will be used instead. If both
`conv_layers` and `num_conv_layers` are `null`, a default list will be assigned to `conv_layers` with the value
`[{filter_size: 2}, {filter_size: 3}, {filter_size: 4}, {filter_size: 5}]`.
- `num_conv_layers` (default `null`): if `conv_layers` is `null`, this is the number of parallel convolutional layers.
- `filter_size` (default `3`): if a `filter_size` is not already specified in `conv_layers` this is the default
`filter_size` that will be used for each layer. It indicates how wide is the 1d convolutional filter.
- `num_filters` (default `256`): if a `num_filters` is not already specified in `conv_layers` this is the default
`num_filters` that will be used for each layer. It indicates the number of filters, and by consequence the output
channels of the 1d convolution.
- `pool_function` (default `max`):  pooling function: `max` will select the maximum value. Any of `average`, `avg` or
`mean` will compute the mean value.
- `pool_size` (default `null`): if a `pool_size` is not already specified in `conv_layers` this is the default
`pool_size` that will be used for each layer. It indicates the size of the max pooling that will be performed along the
`s` sequence dimension after the convolution operation.
- `fc_layers` (default `null`): a list of dictionaries containing the parameters of all the fully connected
layers. The length of the list determines the number of stacked fully connected layers and the content of each
dictionary determines the parameters for a specific layer. The available parameters for each layer are: `activation`,
`dropout`, `norm`, `norm_params`, `output_size`, `use_bias`, `bias_initializer` and `weights_initializer`. If any of
those values is missing from the dictionary, the default one specified as a parameter of the encoder will be used
instead. If both `fc_layers` and `num_fc_layers` are `null`, a default list will be assigned to `fc_layers` with the
value `[{output_size: 512}, {output_size: 256}]` (only applies if `reduce_output` is not `null`).
- `num_fc_layers` (default `null`): if `fc_layers` is `null`, this is the number of stacked fully connected layers (only
applies if `reduce_output` is not `null`).
- `output_size` (default `256`): if `output_size` is not already specified in `fc_layers` this is the default
`output_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `glorot_uniform`): initializer for the weights matrix. Options are: `constant`,
`identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`,
`glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other
keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please
refer to [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html).
- `bias_initializer` (default `zeros`):  initializer for the bias vector. Options are: `constant`, `identity`,
`zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`,
`glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other
keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please
refer to [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html).
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be
used for each layer. It indicates how the output should be normalized and may be one of `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters
used with `batch` see the [Torch documentation on batch normalization](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)
or for `layer` see the [Torch documentation on layer normalization](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html).
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default
`activation` that will be used for each layer. It indicates the activation function applied to the output.
- `dropout` (default `0`): dropout rate
- `reduce_output` (default `sum`): defines how to reduce the output tensor along the `s` sequence length dimension if
the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates
along the sequence dimension), `last` (selects the last vector of the sequence dimension) and  `null` (which does not
reduce and returns the full tensor).

Example sequence feature entry in the input features list using a parallel cnn encoder:

```yaml
name: sequence_column_name
type: sequence
encoder: parallel_cnn
tied_weights: null
representation: dense
embedding_size: 256
embeddings_on_cpu: false
pretrained_embeddings: null
embeddings_trainable: true
conv_layers: null
num_conv_layers: null
filter_size: 3
num_filters: 256
pool_function: max
pool_size: null
fc_layers: null
num_fc_layers: null
output_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
norm: null
norm_params: null
activation: relu
dropout: 0.0
reduce_output: sum
```

### Stacked CNN Encoder

The stacked cnn encoder is inspired by [Xiang Zhang at all's Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626).
It works by first mapping the input integer sequence `b x s` (where `b` is the batch size and `s` is the length of the
sequence) into a sequence of embeddings, then it passes the embedding through a stack of 1d convolutional layers with
different filter size (by default 6 layers with filter size 7, 7, 3, 3, 3 and 3), followed by an optional final pool and
by a flatten operation.
This single flatten vector is then passed through a stack of fully connected layers and returned as a `b x h` tensor
where `h` is the output size of the last fully connected layer.
If you want to output the full `b x s x h` tensor, you can specify the `pool_size` of all your `conv_layers` to be
`null`  and `reduce_output: null`, while if `pool_size` has a value different from `null` and `reduce_output: null` the
returned tensor will be of shape `b x s' x h`, where `s'` is width of the output of the last convolutional layer.

```
       +------+
       |Emb 12|
       +------+
+--+   |Emb 7 |
|12|   +------+
|7 |   |Emb 43|   +----------------+  +---------+
|43|   +------+   |1D Conv         |  |Fully    |
|65+--->Emb 65+--->Layers          +-->Connected+->
|23|   +------+   |Different Widths|  |Layers   |
|4 |   |Emb 23|   +----------------+  +---------+
|1 |   +------+
+--+   |Emb 4 |
       +------+
       |Emb 1 |
       +------+
```

These are the parameters available for the stack cnn encoder:

- `representation` (default `dense`): the possible values are `dense` and `sparse`. `dense` means the embeddings are
initialized randomly, `sparse` means they are initialized to be one-hot encodings.
- `embedding_size` (default `256`): the maximum embedding size, the actual size will be
`min(vocabulary_size, embedding_size)` for `dense` representations and exactly `vocabulary_size` for the `sparse`
encoding, where `vocabulary_size` is the number of different strings appearing in the training set in the column the
feature is named after (plus 2 for `<UNK>` and `<PAD>` tokens).
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
allows for faster access, but in some cases the embedding matrix may be too large. This parameter forces the
placement of the embedding matrix in regular memory and the CPU is used for embedding lookup, slightly slowing down the
process as a result of data transfer between CPU and GPU memory.
- `conv_layers` (default `null`): a list of dictionaries containing the parameters of all the convolutional layers.
The length of the list determines the number of stacked convolutional layers and the content of each dictionary
determines the parameters for a specific layer. The available parameters for each layer are: `activation`, `dropout`,
`norm`, `norm_params`, `num_filters`, `filter_size`, `strides`, `padding`, `dilation_rate`, `use_bias`, `pool_function`,
`pool_padding`, `pool_size`, `pool_strides`, `bias_initializer`, `weights_initializer`. If any of those values is
missing from the dictionary, the default one specified as a parameter of the encoder will be used instead. If both
`conv_layers` and `num_conv_layers` are `null`, a default list will be assigned to `conv_layers` with the value
`[{filter_size: 7, pool_size: 3}, {filter_size: 7, pool_size: 3}, {filter_size: 3, pool_size: null}, {filter_size: 3, pool_size: null}, {filter_size: 3, pool_size: null}, {filter_size: 3, pool_size: 3}]`.
- `num_conv_layers` (default `null`): if `conv_layers` is `null`, this is the number of stacked convolutional layers.
- `filter_size` (default `3`): if a `filter_size` is not already specified in `conv_layers` this is the default
`filter_size` that will be used for each layer. It indicates how wide is the 1d convolutional filter.
- `num_filters` (default `256`): if a `num_filters` is not already specified in `conv_layers` this is the default
`num_filters` that will be used for each layer. It indicates the number of filters, and by consequence the output channels of the 1d convolution.
- `strides` (default `1`): stride length of the convolution
- `padding` (default `same`):  one of `valid` or `same`.
- `dilation_rate` (default `1`): dilation rate to use for dilated convolution
- `pool_function` (default `max`):  pooling function: `max` will select the maximum value.  Any of `average`, `avg` or
`mean` will compute the mean value.
- `pool_size` (default `null`): if a `pool_size` is not already specified in `conv_layers` this is the default
`pool_size` that will be used for each layer. It indicates the size of the max pooling that will be performed along the
`s` sequence dimension after the convolution operation.
- `pool_strides` (default `null`): factor to scale down
- `pool_padding` (default `same`): one of `valid` or `same`
- `fc_layers` (default `null`): a list of dictionaries containing the parameters of all the fully connected layers.
The length of the list determines the number of stacked fully connected layers and the content of each dictionary
determines the parameters for a specific layer. The available parameters for each layer are: `activation`, `dropout`,
`norm`, `norm_params`, `output_size`, `use_bias`, `bias_initializer` and `weights_initializer`. If any of those values
is missing from the dictionary, the default one specified as a parameter of the encoder will be used instead. If both
`fc_layers` and `num_fc_layers` are `null`, a default list will be assigned to `fc_layers` with the value
`[{output_size: 512}, {output_size: 256}]` (only applies if `reduce_output` is not `null`).
- `num_fc_layers` (default `null`): if `fc_layers` is `null`, this is the number of stacked fully connected layers (only
applies if `reduce_output` is not `null`).
- `output_size` (default `256`): if an `output_size` is not already specified in `fc_layers` this is the default
`output_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `glorot_uniform`): initializer for the weight matrix. Options are: `constant`,
`identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`,
`glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other
keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please
refer to [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html).
- `bias_initializer` (default `zeros`):  initializer for the bias vector. Options are: `constant`, `identity`,
`zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`,
`glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other
keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please
refer to [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html).
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be
used for each layer. It indicates how the output should be normalized and may be one of `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters
used with `batch` see the [Torch documentation on batch normalization](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)
or for `layer` see the [Torch documentation on layer normalization](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html).
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default
`activation` that will be used for each layer. It indicates the activation function applied to the output.
- `dropout` (default `0`): dropout rate
- `reduce_output` (default `max`): defines how to reduce the output tensor of the convolutional layers along the `s`
sequence length dimension if the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`,
`max`, `concat` (concatenates along the sequence dimension), `last` (returns the last vector of the sequence dimension)
and `null` (which does not reduce and returns the full tensor).

Example sequence feature entry in the input features list using a parallel cnn encoder:

```yaml
name: sequence_column_name
type: sequence
encoder: stacked_cnn
tied_weights: null
representation: dense
embedding_size: 256
embeddings_trainable: true
pretrained_embeddings: null
embeddings_on_cpu: false
conv_layers: null
num_conv_layers: null
filter_size: 3
num_filters: 256
strides: 1
padding: same
dilation_rate: 1
pool_function: max
pool_size: null
pool_strides: null
pool_padding: same
fc_layers: null
num_fc_layers: null
output_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
norm: null
norm_params: null
activation: relu
dropout: 0
reduce_output: max
```

### Stacked Parallel CNN Encoder

The stacked parallel cnn encoder is a combination of the Parallel CNN and the Stacked CNN encoders where each layer of
the stack is composed of parallel convolutional layers.
It works by first mapping the input integer sequence `b x s` (where `b` is the batch size and `s` is the length of the
sequence) into a sequence of embeddings, then it passes the embedding through a stack of several parallel 1d
convolutional layers with different filter size, followed by an optional final pool and by a flatten operation.
This single flattened vector is then passed through a stack of fully connected layers and returned as a `b x h` tensor
where `h` is the output size of the last fully connected layer.
If you want to output the full `b x s x h` tensor, you can specify `reduce_output: null`.

```
                   +-------+                      +-------+
                +-->1D Conv+-+                 +-->1D Conv+-+
       +------+ |  |Width 2| |                 |  |Width 2| |
       |Emb 12| |  +-------+ |                 |  +-------+ |
       +------+ |            |                 |            |
+--+   |Emb 7 | |  +-------+ |                 |  +-------+ |
|12|   +------+ +-->1D Conv+-+                 +-->1D Conv+-+
|7 |   |Emb 43| |  |Width 3| |                 |  |Width 3| |                   +---------+
|43|   +------+ |  +-------+ | +------+  +---+ |  +-------+ | +------+  +----+  |Fully    |
|65+--->Emb 65+-+            +->Concat+-->...+-+            +->Concat+-->Pool+-->Connected+->
|23|   +------+ |  +-------+ | +------+  +---+ |  +-------+ | +------+  +----+  |Layers   |
|4 |   |Emb 23| +-->1D Conv+-+                 +-->1D Conv+-+                   +---------+
|1 |   +------+ |  |Width 4| |                 |  |Width 4| |
+--+   |Emb 4 | |  +-------+ |                 |  +-------+ |
       +------+ |            |                 |            |
       |Emb 1 | |  +-------+ |                 |  +-------+ |
       +------+ +-->1D Conv+-+                 +-->1D Conv+-+
                   |Width 5|                      |Width 5|
                   +-------+                      +-------+
```

These are the available parameters for the stack parallel cnn encoder:

- `representation` (default `dense`): the possible values are `dense` and `sparse`. `dense` means the embeddings are
initialized randomly, `sparse` means they are initialized to be one-hot encodings.
- `embedding_size` (default `256`): the maximum embedding size, the actual size will be
`min(vocabulary_size, embedding_size)` for `dense` representations and exactly `vocabulary_size` for the `sparse`
encoding, where `vocabulary_size` is the number of different strings appearing in the training set in the column the
feature is named after (plus 2 for `<UNK>` and `<PAD>` tokens).
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
allows for faster access, but in some cases the embedding matrix may be too large. This parameter forces the
placement of the embedding matrix in regular memory and the CPU is used for embedding lookup, slightly slowing down the
process as a result of data transfer between CPU and GPU memory.
- `stacked_layers` (default `null`): a nested list of lists of dictionaries containing the parameters of the stack of
parallel convolutional layers. The length of the list determines the number of stacked parallel convolutional layers,
length of the sub-lists determines the number of parallel conv layers and the content of each dictionary determines the
parameters for a specific layer. The available parameters for each layer are: `activation`, `dropout`, `norm`,
`norm_params`, `num_filters`, `filter_size`, `strides`, `padding`, `dilation_rate`, `use_bias`, `pool_function`,
`pool_padding`, `pool_size`, `pool_strides`, `bias_initializer`, `weights_initializer`. If any of those values is
missing from the dictionary, the default one specified as a parameter of the encoder will be used instead. If both
`stacked_layers` and `num_stacked_layers` are `null`, a default list will be assigned to `stacked_layers` with the value
`[[{filter_size: 2}, {filter_size: 3}, {filter_size: 4}, {filter_size: 5}], [{filter_size: 2}, {filter_size: 3}, {filter_size: 4}, {filter_size: 5}], [{filter_size: 2}, {filter_size: 3}, {filter_size: 4}, {filter_size: 5}]]`.
- `num_stacked_layers` (default `null`): if `stacked_layers` is `null`, this is the number of elements in the stack of
parallel convolutional layers.
- `filter_size` (default `3`): if a `filter_size` is not already specified in `stacked_layers` this is the default
`filter_size` that will be used for each layer. It indicates how wide is the 1d convolutional filter.
- `num_filters` (default `256`): if a `num_filters` is not already specified in `stacker_layers` this is the default
`num_filters` that will be used for each layer. It indicates the number of filters, and by consequence the output
channels of the 1d convolution.
- `pool_function` (default `max`):  pooling function: `max` will select the maximum value.  Any of `average`, `avg` or
`mean` will compute the mean value.
- `pool_size` (default `null`): if a `pool_size` is not already specified in `stacked_layers` this is the default
`pool_size` that will be used for each layer. It indicates the size of the max pooling that will be performed along the
`s` sequence dimension after the convolution operation.
- `fc_layers` (default `null`): a list of dictionaries containing the parameters of all the fully connected layers.
The length of the list determines the number of stacked fully connected layers and the content of each dictionary
determines the parameters for a specific layer. The available parameters for each layer are: `activation`, `dropout`,
`norm`, `norm_params`, `output_size`, `use_bias`, `bias_initializer` and `weights_initializer`. If any of those values
is missing from the dictionary, the default one specified as a parameter of the encoder will be used instead. If both
`fc_layers` and `num_fc_layers` are `null`, a default list will be assigned to `fc_layers` with the value
`[{output_size: 512}, {output_size: 256}]` (only applies if `reduce_output` is not `null`).
- `num_fc_layers` (default `null`): if `fc_layers` is `null`, this is the number of stacked fully connected layers (only
applies if `reduce_output` is not `null`).
- `output_size` (default `256`): if an `output_size` is not already specified in `fc_layers` this is the default
`output_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `glorot_uniform`): initializer for the weights matrix. Options are: `constant`,
`identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`,
`glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other
keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please
refer to [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html).
- `bias_initializer` (default `zeros`):  initializer for the bias vector. Options are: `constant`, `identity`,
`zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`,
`glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other
keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please
refer to [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html).
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be
used for each layer. It indicates how the output should be normalized and may be one of `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters
used with `batch` see the [Torch documentation on batch normalization](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)
or for `layer` see the [Torch documentation on layer normalization](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html).
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default
`activation` that will be used for each layer. It indicates the activation function applied to the output.
- `dropout` (default `0`): dropout rate
- `reduce_output` (default `sum`): defines how to reduce the output tensor along the `s` sequence length dimension if
the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates
along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce
and returns the full tensor).

Example sequence feature entry in the input features list using a parallel cnn encoder:

```yaml
name: sequence_column_name
type: sequence
encoder: stacked_parallel_cnn
tied_weights: null
representation: dense
embedding_size: 256
embeddings_trainable: true
pretrained_embeddings: null
embeddings_on_cpu: false
stacked_layers: null
num_stacked_layers: null
filter_size: 3
num_filters: 256
pool_function: max
pool_size: null
fc_layers: null
num_fc_layers: null
output_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
norm: null
norm_params: null
activation: relu
dropout: 0
reduce_output: max
```

### RNN Encoder

The rnn encoder works by first mapping the input integer sequence `b x s` (where `b` is the batch size and `s` is the
length of the sequence) into a sequence of embeddings, then it passes the embedding through a stack of recurrent layers
(by default 1 layer), followed by a reduce operation that by default only returns the last output, but can perform other
reduce functions.
If you want to output the full `b x s x h` where `h` is the size of the output of the last rnn layer, you can specify
`reduce_output: null`.

```
       +------+
       |Emb 12|
       +------+
+--+   |Emb 7 |
|12|   +------+
|7 |   |Emb 43|                 +---------+
|43|   +------+   +----------+  |Fully    |
|65+--->Emb 65+--->RNN Layers+-->Connected+->
|23|   +------+   +----------+  |Layers   |
|4 |   |Emb 23|                 +---------+
|1 |   +------+
+--+   |Emb 4 |
       +------+
       |Emb 1 |
       +------+


```

These are the available parameters for the rnn encoder:

- `representation` (default `dense`): the possible values are `dense` and `sparse`. `dense` means the embeddings are
initialized randomly, `sparse` means they are initialized to be one-hot encodings.
- `embedding_size` (default `256`): the maximum embedding size, the actual size will be
`min(vocabulary_size, embedding_size)` for `dense` representations and exactly `vocabulary_size` for the `sparse`
encoding, where `vocabulary_size` is the number of different strings appearing in the training set in the column the
feature is named after (plus 2 for `<UNK>` and `<PAD>` tokens).
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
allows for faster access, but in some cases the embedding matrix may be too large. This parameter forces the
placement of the embedding matrix in regular memory and the CPU is used for embedding lookup, slightly slowing down the
process as a result of data transfer between CPU and GPU memory.
- `num_layers` (default `1`): the number of stacked recurrent layers.
- `state_size` (default `256`): the size of the state of the rnn.
- `cell_type` (default `rnn`): the type of recurrent cell to use. Available values are: `rnn`, `lstm`, `gru`. For
reference about the differences between the cells please refer to
[torch.nn Recurrent Layers](https://pytorch.org/docs/stable/nn.html#recurrent-layers).
- `bidirectional` (default `false`): if `true` two recurrent networks will perform encoding in the forward and backward
direction and their outputs will be concatenated.
- `activation` (default `tanh`): activation function to use.
- `recurrent_activation` (default `sigmoid`): activation function to use in the recurrent step
- `unit_forget_bias` (default `true`): If `true`, add 1 to the bias of the forget gate at initialization
- `recurrent_initializer` (default `orthogonal`): initializer for recurrent matrix weights
- `dropout` (default `0.0`): dropout rate
- `recurrent_dropout` (default `0.0`): dropout rate for recurrent state
- `fc_layers` (default `null`): a list of dictionaries containing the parameters of all the fully connected layers.
The length of the list determines the number of stacked fully connected layers and the content of each dictionary
determines the parameters for a specific layer. The available parameters for each layer are: `activation`, `dropout`,
`norm`, `norm_params`, `output_size`, `use_bias`, `bias_initializer` and `weights_initializer`. If any of those values
is missing from the dictionary, the default one specified as a parameter of the encoder will be used instead. If both
`fc_layers` and `num_fc_layers` are `null`, a default list will be assigned to `fc_layers` with the value
`[{output_size: 512}, {output_size: 256}]` (only applies if `reduce_output` is not `null`).
- `num_fc_layers` (default `null`): if `fc_layers` is `null`, this is the number of stacked fully connected layers (only
applies if `reduce_output` is not `null`).
- `output_size` (default `256`): if an `output_size` is not already specified in `fc_layers` this is the default
`output_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `glorot_uniform`): initializer for the weight matrix. Options are: `constant`,
`identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`,
`glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other
keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please
refer to [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html).
- `bias_initializer` (default `zeros`):  initializer for the bias vector. Options are: `constant`, `identity`,
`zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`,
`glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other
keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please
refer to [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html).
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be
used for each layer. It indicates how the output should be normalized and may be one of `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`. For information on parameters
used with `batch` see the [Torch documentation on batch normalization](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)
or for `layer` see the [Torch documentation on layer normalization](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html).
- `fc_activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default
`activation` that will be used for each layer. It indicates the activation function applied to the output.
- `fc_dropout` (default `0`): dropout rate
- `reduce_output` (default `last`): defines how to reduce the output tensor along the `s` sequence length dimension if
the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates
along the sequence dimension), `last` (returns the last vector of the sequence dimension) and  `null` (which does not
reduce and returns the full tensor).

Example sequence feature entry in the input features list using a parallel cnn encoder:

```yaml
name: sequence_column_name
type: sequence
encoder: rnn
tied_weights: null
representation': dense
embedding_size: 256
embeddings_trainable: true
pretrained_embeddings: null
embeddings_on_cpu: false
num_layers: 1
state_size: 256
cell_type: rnn
bidirectional: false
activation: tanh
recurrent_activation: sigmoid
unit_forget_bias: true
recurrent_initializer: orthogonal
dropout: 0.0
recurrent_dropout: 0.0
fc_layers: null
num_fc_layers: null
output_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
norm: null
norm_params: null
fc_activation: relu
fc_dropout: 0
reduce_output: last
```

### CNN RNN Encoder

The `cnnrnn` encoder works by first mapping the input integer sequence `b x s` (where `b` is the batch size and `s` is
the length of the sequence) into a sequence of embeddings, then it passes the embedding through a stack of convolutional
layers (by default 2), that is followed by a stack of recurrent layers (by default 1), followed by a reduce operation
that by default only returns the last output, but can perform other reduce functions.
If you want to output the full `b x s x h` where `h` is the size of the output of the last rnn layer, you can specify
`reduce_output: null`.

```
       +------+
       |Emb 12|
       +------+
+--+   |Emb 7 |
|12|   +------+
|7 |   |Emb 43|                                +---------+
|43|   +------+   +----------+   +----------+  |Fully    |
|65+--->Emb 65+--->CNN Layers+--->RNN Layers+-->Connected+->
|23|   +------+   +----------+   +----------+  |Layers   |
|4 |   |Emb 23|                                +---------+
|1 |   +------+
+--+   |Emb 4 |
       +------+
       |Emb 1 |
       +------+
```

These are the available parameters of the cnn rnn encoder:

- `representation` (default `dense`): the possible values are `dense` and `sparse`. `dense` means the embeddings are
initialized randomly, `sparse` means they are initialized to be one-hot encodings.
- `embedding_size` (default `256`): the maximum embedding size, the actual size will be
`min(vocabulary_size, embedding_size)` for `dense` representations and exactly `vocabulary_size` for the `sparse`
encoding, where `vocabulary_size` is the number of different strings appearing in the training set in the column the
feature is named after (plus 2 for `<UNK>` and `<PAD>` tokens).
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
allows for faster access, but in some cases the embedding matrix may be too large. This parameter forces the
placement of the embedding matrix in regular memory and the CPU is used for embedding lookup, slightly slowing down the
process as a result of data transfer between CPU and GPU memory.
- `conv_layers` (default `null`): a list of dictionaries containing the parameters of all the convolutional layers.
The length of the list determines the number of stacked convolutional layers and the content of each dictionary
determines the parameters for a specific layer. The available parameters for each layer are: `activation`, `dropout`,
`norm`, `norm_params`, `num_filters`, `filter_size`, `strides`, `padding`, `dilation_rate`, `use_bias`, `pool_function`,
`pool_padding`, `pool_size`, `pool_strides`, `bias_initializer`, `weights_initializer`. If any of those values is
missing from the dictionary, the default one specified as a parameter of the encoder will be used instead. If both
`conv_layers` and `num_conv_layers` are `null`, a default list will be assigned to `conv_layers` with the value
`[{filter_size: 7, pool_size: 3}, {filter_size: 7, pool_size: 3}, {filter_size: 3, pool_size: null}, {filter_size: 3, pool_size: null}, {filter_size: 3, pool_size: null}, {filter_size: 3, pool_size: 3}]`.
- `num_conv_layers` (default `1`): the number of stacked convolutional layers.
- `num_filters` (default `256`): if a `num_filters` is not already specified in `conv_layers` this is the default
`num_filters` that will be used for each layer. It indicates the number of filters, and by consequence the output
channels of the 1d convolution.
- `filter_size` (default `5`): if a `filter_size` is not already specified in `conv_layers` this is the default
`filter_size` that will be used for each layer. It indicates how wide is the 1d convolutional filter.
- `strides` (default `1`): stride length of the convolution
- `padding` (default `same`):  one of `valid` or `same`.
- `dilation_rate` (default `1`): dilation rate to use for dilated convolution
- `conv_activation` (default `relu`): activation for the convolution layer
- `conv_dropout` (default `0.0`): dropout rate for the convolution layer
- `pool_function` (default `max`):  pooling function: `max` will select the maximum value.  Any of `average`, `avg` or
`mean` will compute the mean value.
- `pool_size` (default 2 ): if a `pool_size` is not already specified in `conv_layers` this is the default `pool_size`
that will be used for each layer. It indicates the size of the max pooling that will be performed along the `s` sequence
dimension after the convolution operation.
- `pool_strides` (default `null`): factor to scale down
- `pool_padding` (default `same`): one of `valid` or `same`
- `num_rec_layers` (default `1`): the number of recurrent layers
- `state_size` (default `256`): the size of the state of the rnn.
- `cell_type` (default `rnn`): the type of recurrent cell to use. Available values are: `rnn`, `lstm`, `gru`. For
reference about the differences between the cells please refer to
[torch.nn Recurrent Layers](https://pytorch.org/docs/stable/nn.html#recurrent-layers).
- `bidirectional` (default `false`): if `true` two recurrent networks will perform encoding in the forward and backward
direction and their outputs will be concatenated.
- `activation` (default `tanh`): activation function to use
- `recurrent_activation` (default `sigmoid`): activation function to use in the recurrent step
- `unit_forget_bias` (default `true`): If `true`, add 1 to the bias of the forget gate at initialization
- `recurrent_initializer` (default `orthogonal`): initializer for recurrent matrix weights
- `dropout` (default `0.0`): dropout rate
- `recurrent_dropout` (default `0.0`): dropout rate for recurrent state
- `fc_layers` (default `null`): a list of dictionaries containing the parameters of all the fully connected
layers. The length of the list determines the number of stacked fully connected layers and the content of each
dictionary determines the parameters for a specific layer. The available parameters for each layer are: `activation`,
`dropout`, `norm`, `norm_params`, `output_size`, `use_bias`, `bias_initializer` and `weights_initializer`. If any of
those values is missing from the dictionary, the default one specified as a parameter of the encoder will be used
instead. If both `fc_layers` and `num_fc_layers` are `null`, a default list will be assigned to `fc_layers` with the
value `[{output_size: 512}, {output_size: 256}]` (only applies if `reduce_output` is not `null`).
- `num_fc_layers` (default `null`): if `fc_layers` is `null`, this is the number of stacked fully connected layers (only
applies if `reduce_output` is not `null`).
- `output_size` (default `256`): if an `output_size` is not already specified in `fc_layers` this is the default
`output_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `glorot_uniform`): initializer for the weights matrix. Options are: `constant`,
`identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`,
`glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other
keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please
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
- `fc_activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default
`activation` that will be used for each layer. It indicates the activation function applied to the output.
- `fc_dropout` (default `0`): dropout rate
- `reduce_output` (default `last`): defines how to reduce the output tensor along the `s` sequence length dimension if
the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates
along the sequence dimension), `last` (returns the last vector of the sequence dimension) and `null` (which does not
reduce and returns the full tensor).

Example sequence feature entry in the inputs features list using a cnn rnn encoder:

```yaml
name: sequence_column_name
type: sequence
encoder: cnnrnn
tied_weights: null
representation: dense
embedding_size: 256
embeddings_trainable: true
pretrained_embeddings: null
embeddings_on_cpu: false
conv_layers: null
num_conv_layers: 1
num_filters: 256
filter_size: 5
strides: 1
padding: same
dilation_rate: 1
conv_activation: relu
conv_dropout: 0.0
pool_function: max
pool_size: 2
pool_strides: null
pool_padding: same
num_rec_layers: 1
state_size: 256
cell_type: rnn
bidirectional: false
activation: tanh
recurrent_activation: sigmoid
unit_forget_bias: true
recurrent_initializer: orthogonal
dropout: 0.0
recurrent_dropout: 0.0
fc_layers: null
num_fc_layers: null
output_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
norm: null
norm_params: null
fc_activation: relu
fc_dropout: 0
reduce_output: last
```

### Transformer Encoder

The `transformer` encoder implements a stack of transformer blocks, replicating the architecture introduced in the
[Attention is all you need](https://arxiv.org/abs/1706.03762) paper, and adds am optional stack of fully connected
layers at the end.

```
       +------+
       |Emb 12|
       +------+
+--+   |Emb 7 |
|12|   +------+
|7 |   |Emb 43|   +-------------+   +---------+
|43|   +------+   |             |   |Fully    |
|65+---+Emb 65+---> Transformer +--->Connected+->
|23|   +------+   | Blocks      |   |Layers   |
|4 |   |Emb 23|   +-------------+   +---------+
|1 |   +------+
+--+   |Emb 4 |
       +------+
       |Emb 1 |
       +------+

```

- `representation` (default `dense`): the possible values are `dense` and `sparse`. `dense` means the embeddings are
initialized randomly, `sparse` means they are initialized to be one-hot encodings.
- `embedding_size` (default `256`): the maximum embedding size, the actual size will be
`min(vocabulary_size, embedding_size)` for `dense` representations and exactly `vocabulary_size` for the `sparse`
encoding, where `vocabulary_size` is the number of different strings appearing in the training set in the column the
feature is named after (plus 2 for `<UNK>` and `<PAD>` tokens).
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
allows for faster access, but in some cases the embedding matrix may be too large. This parameter forces the
placement of the embedding matrix in regular memory and the CPU is used for embedding lookup, slightly slowing down the
process as a result of data transfer between CPU and GPU memory.
- `num_layers` (default `1`): number of transformer blocks.
- `hidden_size` (default `256`): the size of the hidden representation within the transformer block. It is usually the
same as the `embedding_size`, but if the two values are different, a projection layer will be added before the first
transformer block.
- `num_heads` (default `8`): number of attention heads in each transformer block.
- `transformer_output_size` (default `256`): Size of the fully connected layer after self attention in the transformer
block. This is usually the same as `hidden_size` and `embedding_size`.
- `dropout` (default `0.1`): dropout rate for the transformer block
- `fc_layers` (default `null`): a list of dictionaries containing the parameters of all the fully connected layers.
The length of the list determines the number of stacked fully connected layers and the content of each dictionary
determines the parameters for a specific layer. The available parameters for each layer are: `activation`, `dropout`,
`norm`, `norm_params`, `output_size`, `use_bias`, `bias_initializer` and `weights_initializer`. If any of those values
is missing from the dictionary, the default one specified as a parameter of the encoder will be used instead. If both
`fc_layers` and `num_fc_layers` are `null`, a default list will be assigned to `fc_layers` with the value
`[{output_size: 512}, {output_size: 256}]` (only applies if `reduce_output` is not `null`).
- `num_fc_layers` (default `0`): This is the number of stacked fully connected layers (only applies if `reduce_output`
is not `null`).
- `output_size` (default `256`): if an `output_size` is not already specified in `fc_layers` this is the default
`output_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `glorot_uniform`): initializer for the weights matrix. Options are: `constant`,
`identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`,
`glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other
keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please
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
- `fc_activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default
`activation` that will be used for each layer. It indicates the activation function applied to the output.
- `fc_dropout` (default `0`): dropout rate
- `reduce_output` (default `last`): defines how to reduce the output tensor along the `s` sequence length dimension if
the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates
along the sequence dimension), `last` (returns the last vector of the sequence dimension) and  `null` (which does not
reduce and returns the full tensor).

Example sequence feature entry in the inputs features list using a Transformer encoder:

```yaml
name: sequence_column_name
type: sequence
encoder: transformer
tied_weights: null
representation: dense
embedding_size: 256
embeddings_trainable: true
pretrained_embeddings: null
embeddings_on_cpu: false
num_layers: 1
hidden_size: 256
num_heads: 8
transformer_output_size: 256
dropout: 0.1
fc_layers: null
num_fc_layers: 0
output_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
norm: null
norm_params: null
fc_activation: relu
fc_dropout: 0
reduce_output: last
```

### Passthrough Encoder

The passthrough decoder simply transforms each input value into a float value and adds a dimension to the input tensor,
creating a `b x s x 1` tensor where `b` is the batch size and `s` is the length of the sequence.
The tensor is reduced along the `s` dimension to obtain a single vector of size `h` for each element of the batch.
If you want to output the full `b x s x h` tensor, you can specify `reduce_output: null`.
This encoder is not really useful for `sequence` or `text` features, but may be useful for `timeseries` features, as it
allows for using them without any processing in later stages of the model, like in a sequence combiner for instance.

```
+--+
|12|
|7 |                    +-----------+
|43|   +------------+   |Aggregation|
|65+--->Cast float32+--->Reduce     +->
|23|   +------------+   |Operation  |
|4 |                    +-----------+
|1 |
+--+
```

These are the parameters available for the passthrough encoder

- `reduce_output` (default `null`): defines how to reduce the output tensor along the `s` sequence length dimension if
the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates
along the sequence dimension), `last` (returns the last vector of the sequence dimension) and  `null` (which does not
reduce and returns the full tensor).

Example sequence feature entry in the input features list using a passthrough encoder:

```yaml
name: sequence_column_name
type: sequence
encoder: passthrough
reduce_output: null
```

## Sequence Output Features and Decoders

Sequential features can be used when sequence tagging (classifying each element of an input sequence) or sequence
generation needs to be performed.  There are two decoders available for those to tasks named `tagger` and `generator`.

These are the available parameters of a sequence output feature

- `reduce_input` (default `sum`): defines how to reduce an input that is not a vector, but a matrix or a higher order
tensor, on the first dimension (second if you count the batch dimension). Available values are: `sum`, `mean` or `avg`,
`max`, `concat` (concatenates along the sequence dimension), `last` (returns the last vector of the sequence dimension).
- `dependencies` (default `[]`): the output features this one is dependent on. For a detailed explanation refer to
[Output Feature Dependencies](../output_features#output-feature-dependencies).
- `reduce_dependencies` (default `sum`): defines how to reduce the output of a dependent feature that is not a vector,
but a matrix or a higher order tensor, on the first dimension (second if you count the batch dimension). Available
values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the sequence dimension), `last` (returns the
last vector of the sequence dimension).
- `loss` (default `{type: softmax_cross_entropy, class_similarities_temperature: 0, class_weights: 1,
confidence_penalty: 0, distortion: 1, labels_smoothing: 0, negative_samples: 0, robust_lambda: 0, sampler: null, unique:
false}`): is a dictionary containing a loss `type`. The available losses `type` are `softmax_cross_entropy` and
`sampled_softmax_cross_entropy`. For details on both losses, please refer to
[Category Output Features and Decoders](../category_features#category-output-features-and-decoders).

### Tagger Decoder

In the case of `tagger` the decoder is a (potentially empty) stack of fully connected layers, followed by a projection
into a tensor of size `b x s x c`, where `b` is the batch size, `s` is the length of the sequence and `c` is the number
of classes, followed by a softmax_cross_entropy.
This decoder requires its input to be shaped as `b x s x h`, where `h` is a hidden dimension, which is the output of a
sequence, text or time series input feature without reduced outputs or the output of a sequence-based combiner.
If a `b x h` input is provided instead, an error will be raised during model building.

```
Combiner
Output

+---+                 +----------+   +-------+
|emb|   +---------+   |Projection|   |Softmax|
+---+   |Fully    |   +----------+   +-------+
|...+--->Connected+--->...       +--->...    |
+---+   |Layers   |   +----------+   +-------+
|emb|   +---------+   |Projection|   |Softmax|
+---+                 +----------+   +-------+
```

These are the available parameters of a tagger decoder:

- `fc_layers` (default `null`): a list of dictionaries containing the parameters of all the fully connected
layers. The length of the list determines the number of stacked fully connected layers and the content of each
dictionary determines the parameters for a specific layer. The available parameters for each layer are: `activation`,
`dropout`, `norm`, `norm_params`, `output_size`, `use_bias`, `bias_initializer` and `weights_initializer`. If any of
those values is missing from the dictionary, the default one specified as a parameter of the decoder will be used instead.
- `num_fc_layers` (default 0): this is the number of stacked fully connected layers that the input to the feature passes
through. Their output is projected in the feature's output space.
- `output_size` (default `256`): if an `output_size` is not already specified in `fc_layers` this is the default
`output_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `glorot_uniform`): initializer for the weights matrix. Options are: `constant`,
`identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`,
`glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other
keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please
refer to [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html).
- `bias_initializer` (default `zeros`):  initializer for the bias vector. Options are: `constant`, `identity`,
`zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`,
`glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other
keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please
refer to [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html).
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be
used for each layer. It indicates how the output should be normalized and may be one of `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters
used with `batch` see the [Torch documentation on batch normalization](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)
or for `layer` see the [Torch documentation on layer normalization](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html).
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default
`activation` that will be used for each layer. It indicates the activation function applied to the output.
- `dropout` (default `0`): dropout rate
- `attention` (default `false`): If `true`, applies a multi-head self attention layer before prediction.
- `attention_embedding_size` (default `256`): the embedding size of the multi-head self attention layer.
- `attention_num_heads` (default `8`): number of attention heads in the multi-head self attention layer.

Example sequence feature entry using a tagger decoder (with default parameters) in the output features list:

```yaml
name: sequence_column_name
type: sequence
decoder: tagger
reduce_input: null
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
output_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
norm: null
norm_params: null
activation: relu
dropout: 0
attention: false
attention_embedding_size: 256
attention_num_heads: 8
```

### Generator Decoder

In the case of `generator` the decoder is a (potentially empty) stack of fully connected layers, followed by an rnn that
generates outputs feeding on its own previous predictions and generates a tensor of size `b x s' x c`, where `b` is the
batch size, `s'` is the length of the generated sequence and `c` is the number of classes, followed by a
softmax_cross_entropy.
During training teacher forcing is adopted, meaning the list of targets is provided as both inputs and outputs (shifted
by 1), while at evaluation time greedy decoding (generating one token at a time and feeding it as input for the next
step) is performed by beam search, using a beam of 1 by default.
In general a generator expects a `b x h` shaped input tensor, where `h` is a hidden dimension.
The `h` vectors are (after an optional stack of fully connected layers) fed into the rnn generator.
One exception is when the generator uses attention, as in that case the expected size of the input tensor is
`b x s x h`, which is the output of a sequence, text or time series input feature without reduced outputs or the output
of a sequence-based combiner.
If a `b x h` input is provided to a generator decoder using an rnn with attention instead, an error will be raised
during model building.

```
                            Output     Output
                               1  +-+    ... +--+    END
                               ^    |     ^     |     ^
+--------+   +---------+       |    |     |     |     |
|Combiner|   |Fully    |   +---+--+ | +---+---+ | +---+--+
|Output  +--->Connected+---+RNN   +--->RNN... +--->RNN   |
|        |   |Layers   |   +---^--+ | +---^---+ | +---^--+
+--------+   +---------+       |    |     |     |     |
                              GO    +-----+     +-----+
```

- `reduce_input` (default `sum`): defines how to reduce an input that is not a vector, but a matrix or a higher order
tensor, on the first dimension (second if you count the batch dimension). Available values are: `sum`, `mean` or `avg`,
`max`, `concat` (concatenates along the sequence dimension), `last` (returns the last vector of the sequence dimension).

These are the available parameters of a Generator decoder:

- `fc_layers` (default `null`): it is a list of dictionaries containing the parameters of all the fully connected
layers. The length of the list determines the number of stacked fully connected layers and the content of each
dictionary determines the parameters for a specific layer. The available parameters for each layer are: `activation`,
`dropout`, `norm`, `norm_params`, `output_size`, `use_bias`, `bias_initializer` and `weights_initializer`. If any of
those values is missing from the dictionary, the default one specified as a parameter of the decoder will be used instead.
- `num_fc_layers` (default 0): this is the number of stacked fully connected layers that the input to the feature passes
through. Their output is projected in the feature's output space.
- `output_size` (default `256`): if an `output_size` is not already specified in `fc_layers` this is the default
`output_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `glorot_uniform`): initializer for the weight matrix. Options are: `constant`,
`identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`,
`glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other
keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please
refer to [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html).
- `bias_initializer` (default `zeros`):  initializer for the bias vector. Options are: `constant`, `identity`,
`zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`,
`glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other
keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please
refer to [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html).
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be
used for each layer. It indicates how the output should be normalized and may be one of `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters
used with `batch` see [Torch documentation on batch normalization](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)
or for `layer` see [Torch documentation on layer normalization](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html).
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default
`activation` that will be used for each layer. It indicates the activation function applied to the output.
- `dropout` (default `0`): dropout rate
- `cell_type` (default `rnn`): the type of recurrent cell to use. Available values are: `rnn`, `lstm`, `gru`. For
reference about the differences between the cells please refer to
[torch.nn Recurrent Layers](https://pytorch.org/docs/stable/nn.html#recurrent-layers).
- `state_size` (default `256`): the size of the state of the rnn.
- `embedding_size` (default `256`): if `tied_target_embeddings` is `false`, the input embeddings and the weights of the
softmax_cross_entropy weights before the softmax_cross_entropy are not tied together and can have different sizes, this
parameter describes the size of the embeddings of the inputs of the generator.
- `beam_width` (default `1`): sampling from the rnn generator is performed using beam search. By default, with a beam of
one, only a greedy sequence using always the most probably next token is generated, but the beam size can be increased.
This usually leads to better performance at the expense of more computation and slower generation.
- `tied_embeddings` (default `null`): if `null` the embeddings of the targets are initialized randomly, while if the
values is the name of an input feature, the embeddings of that input feature will be used as embeddings of the target.
The `vocabulary_size` of that input feature has to be the same as the output feature and it has to have an embedding
matrix (binary and number features will not have one, for instance). In this case the `embedding_size` will be the same
as the `state_size`. This is useful for implementing autoencoders where the encoding and decoding part of the model
share parameters.
- `max_sequence_length` (default `0`):

Example sequence feature entry using a generator decoder (with default parameters) in the output features list:

```yaml
name: sequence_column_name
type: sequence
decoder: generator
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
output_size: 256
use_bias: true
bias_initializer: zeros
weights_initializer: glorot_uniform
norm: null
norm_params: null
activation: relu
dropout: 0
cell_type: rnn
state_size: 256
embedding_size: 256
beam_width: 1
tied_embeddings: null
max_sequence_length: 0
```

## Sequence Features Metrics

The metrics that are calculated every epoch and are available for sequence features are `sequence_accuracy` (counts the
number of datapoints where all the elements of the predicted sequence are correct over the number of all datapoints),
`token_accuracy` (computes the number of elements in all the sequences that are correctly predicted over the number of
all the elements in all the sequences), `last_accuracy` (accuracy considering only the last element of the sequence, it
is useful for being sure special end-of-sequence tokens are generated or tagged), `edit_distance` (the levenshtein
distance between the predicted and ground truth sequence), `perplexity` (the perplexity of the ground truth sequence
according to the model) and the `loss` itself.
You can set either of them as `validation_metric` in the `training` section of the configuration if you set the
`validation_field` to be the name of a sequence feature.
