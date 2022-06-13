## Text Features Preprocessing

Text features are an extension of [sequence features](../sequence_features). Text inputs are processed by a tokenizer
which maps the raw text input into a sequence of tokens. An integer id is assigned to each unique token. Using this
mapping, each text string is converted first to a sequence of tokens, and next to a sequence of integers.

The list of tokens and their integer representations (vocabulary) is stored in the metadata of the model. In the case of
a text output feature, this same mapping is used to post-process predictions to text.

The parameters for text preprocessing are as follows:

- `tokenizer` (default `space_punct`): defines how to map from the raw string content of the dataset column to a
sequence of elements. For all available options see [Tokenizers](../../preprocessing#tokenizers).
- `vocab_file` (default `null`): filepath string to a UTF-8 encoded file containing the sequence's vocabulary. On each
line the first string until `\t` or `\n` is considered a word.
- `max_sequence_length` (default `256`): the maximum length (number of tokens) of the text. Texts that are longer than
this value will be truncated, while texts that are shorter will be padded.
- `most_common` (default `20000`): the maximum number of most common tokens in the vocabulary. If the data contains more
than this amount, the most infrequent symbols will be treated as unknown.
- `padding_symbol` (default `<PAD>`): the string used as a padding symbol. This special token is mapped to the integer
ID 0 in the vocabulary.
- `unknown_symbol` (default `<UNK>`): the string used as an unknown placeholder. This special token is mapped to the
integer ID 1 in the vocabulary.
- `padding` (default `right`): the direction of the padding. `right` and `left` are available options.
- `lowercase` (default `false`): If true, converts the string to lowercase before tokenizing.
- `missing_value_strategy` (default `fill_with_const`): what strategy to follow when there's a missing value in the
dataset. The value should be one of `fill_with_const` (replaces the missing value with a specific value specified with
the `fill_value` parameter), `fill_with_mode` (replaces the missing values with the most frequent value in the column),
`fill_with_mean` (replaces the missing values with the mean of the values in the column), `backfill` (replaces the
missing values with the next valid value).
- `fill_value` (default `""`): the value to replace the missing values with in case the `missing_value_strategy` is
`fill_value`.

Configuration example:

```yaml
name: text_column_name
type: text
preprocessing:
    tokenizer: space_punct
    vocab_file: null
    max_sequence_length: 256
    most_common: 20000
    padding_symbol: <PAD>
    unknown_symbol: <UNK>
    padding: right
    lowercase: false
    missing_value_strategy: fill_with_const
    fill_value: ""
```

!!! note
    If a text feature's encoder specifies a huggingface model, then the tokenizer for that model will be used
    automatically.

## Text Input Features and Encoders

Text input feature parameters are

- `encoder` (default `parallel_cnn`): encoder to use for the input text feature. The available encoders include encoders
used for [Sequence Features](../sequence_features#sequence-input-features-and-encoders) as well as pre-trained text
encoders from the huggingface transformers library: `albert`, `auto_transformer`, `bert`, `camembert`, `ctrl`,
`distilbert`, `electra`, `flaubert`, `gpt`, `gpt2`, `longformer`, `roberta`, `t5`, `mt5`, `transformer_xl`, `xlm`,
`xlmroberta`, `xlnet`.
- `tied` (default `null`): name of the input feature to tie the weights of the encoder with. Tied must name a feature of
the same type with the same encoder parameters.

Example:

```yaml
name: text_column_name
type: text
encoder: bert
trainable: true
```

### Embed Encoder

The embed encoder simply maps each token in the input sequence to an embedding, creating a `b x s x h` tensor where `b`
is the batch size, `s` is the length of the sequence and `h` is the embedding size.
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
|65+--->Emb 65+--->Reduce     +-->
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
encoding, where `vocabulary_size` is the number of unique strings appearing in the training set input column plus the
number of special tokens (`<UNK>`, `<PAD>`, `<SOS>`, `<EOS>`).
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
along the sequence dimension), `last` (selects the last vector of the sequence dimension) and  `null` (which does not
reduce and returns the full tensor).

Example text feature entry in the input features list using an embed encoder:

```yaml
name: text_column_name
type: text
encoder: embed
representation: dense
embedding_size: 256
embeddings_trainable: true
dropout: 0
reduce_output: sum
```

### Parallel CNN Encoder

The parallel cnn encoder is inspired by
[Yoon Kim's Convolutional Neural Network for Sentence Classification](https://arxiv.org/abs/1408.5882).
It works by first mapping the input token sequence `b x s` (where `b` is the batch size and `s` is the length of the
sequence) into a sequence of embeddings, then it passes the embedding through a number of parallel 1d convolutional
layers with different filter size (by default 4 layers with filter size 2, 3, 4 and 5), followed by max pooling and
concatenation.
This single vector concatenating the outputs of the parallel convolutional layers is then passed through a stack of
fully connected layers and returned as a `b x h` tensor where `h` is the output size of the last fully connected layer.
If you want to output the full `b x s x h` tensor, you can specify `reduce_output: null`.

```
                    +-------+   +----+
                 +-->1D Conv+--->Pool+--+
       +------+  |  |Width 2|   +----+  |
       |Emb 12|  |  +-------+           |
       +------+  |                      |
+--+   |Emb 7 |  |  +-------+   +----+  |
|12|   +------+  +-->1D Conv+--->Pool+--+
|7 |   |Emb 43|  |  |Width 3|   +----+  |            +---------+
|43|   +------+  |  +-------+           |  +------+  |Fully    |
|65+-->Emb 65 +--+                      +-->Concat+-->Connected+-->
|23|   +------+  |  +-------+   +----+  |  +------+  |Layers   |
|4 |   |Emb 23|  +-->1D Conv+--->Pool+--+            +---------+
|1 |   +------+  |  |Width 4|   +----+  |
+--+   |Emb 4 |  |  +-------+           |
       +------+  |                      |
       |Emb 1 |  |  +-------+   +----+  |
       +------+  +-->1D Conv+--->Pool+--+
                    |Width 5|   +----+
                    +-------+
```

These are the available parameters for a parallel cnn encoder:

- `representation` (default `dense`): the possible values are `dense` and `sparse`. `dense` means the embeddings are
initialized randomly, `sparse` means they are initialized to be one-hot encodings.
- `embedding_size` (default `256`): it is the maximum embedding size, the actual size will be
`min(vocabulary_size, embedding_size)` for `dense` representations and exactly `vocabulary_size` for the `sparse`
encoding, where `vocabulary_size` is the number of unique strings appearing in the training set input column plus the
number of special tokens (`<UNK>`, `<PAD>`, `<SOS>`, `<EOS>`).
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

Example text feature entry in the input features list using a parallel cnn encoder:

```yaml
name: text_column_name
type: text
encoder: parallel_cnn
representation: dense
embedding_size: 256
embeddings_trainable: true
filter_size: 3
num_filters: 256
pool_function: max
output_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
activation: relu
dropout: 0.0
reduce_output: sum
```

### Stacked CNN Encoder

The stacked cnn encoder is inspired by [Xiang Zhang at all's Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626).
It works by first mapping the input token sequence `b x s` (where `b` is the batch size and `s` is the length of the
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
|7 |   |Emb 43|   +----------------+   +---------+
|43|   +------+   |1D Conv         |   |Fully    |
|65+--->Emb 65+--->Layers          +--->Connected+-->
|23|   +------+   |Different Widths|   |Layers   |
|4 |   |Emb 23|   +----------------+   +---------+
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
encoding, where `vocabulary_size` is the number of unique strings appearing in the training set input column plus the
number of special tokens (`<UNK>`, `<PAD>`, `<SOS>`, `<EOS>`).
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

Example text feature entry in the input features list using a parallel cnn encoder:

```yaml
name: text_column_name
type: text
encoder: stacked_cnn
representation: dense
embedding_size: 256
embeddings_trainable: true
filter_size: 3
num_filters: 256
strides: 1
padding: same
dilation_rate: 1
pool_function: max
pool_padding: same
output_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
activation: relu
dropout: 0
reduce_output: max
```

### Stacked Parallel CNN Encoder

The stacked parallel cnn encoder is a combination of the Parallel CNN and the Stacked CNN encoders where each layer of
the stack is composed of parallel convolutional layers.
It works by first mapping the input token sequence `b x s` (where `b` is the batch size and `s` is the length of the
sequence) into a sequence of embeddings, then it passes the embedding through a stack of several parallel 1d
convolutional layers with different filter size, followed by an optional final pool and by a flatten operation.
This single flattened vector is then passed through a stack of fully connected layers and returned as a `b x h` tensor
where `h` is the output size of the last fully connected layer.
If you want to output the full `b x s x h` tensor, you can specify `reduce_output: null`.

```
                  +-------+                     +-------+
                +->1D Conv+-+                 +->1D Conv+-+
      +------+  | |Width 2| |                 | |Width 2| |
      |Emb 12|  | +-------+ |                 | +-------+ |
      +------+  |           |                 |           |
+--+  |Emb 7 |  | +-------+ |                 | +-------+ |
|12|  +------+  +->1D Conv+-+                 +->1D Conv+-+
|7 |  |Emb 43|  | |Width 3| |                 | |Width 3| |                 +---------+
|43|  +------+  | +-------+ | +------+  +---+ | +-------+ | +------+ +----+ |Fully    |
|65+->Emb 65 +--+           +->Concat+-->...+-+           +->Concat+->Pool+->Connected+-->
|23|  +------+  | +-------+ | +------+  +---+ | +-------+ | +------+ +----+ |Layers   |
|4 |  |Emb 23|  +->1D Conv+-+                 +->1D Conv+-+                 +---------+
|1 |  +------+  | |Width 4| |                 | |Width 4| |
+--+  |Emb 4 |  | +-------+ |                 | +-------+ |
      +------+  |           |                 |           |
      |Emb 1 |  | +-------+ |                 | +-------+ |
      +------+  +->1D Conv+-+                 +->1D Conv+-+
                  |Width 5|                     |Width 5|
                  +-------+                     +-------+
```

These are the available parameters for the stack parallel cnn encoder:

- `representation` (default `dense`): the possible values are `dense` and `sparse`. `dense` means the embeddings are
initialized randomly, `sparse` means they are initialized to be one-hot encodings.
- `embedding_size` (default `256`): the maximum embedding size, the actual size will be
`min(vocabulary_size, embedding_size)` for `dense` representations and exactly `vocabulary_size` for the `sparse`
encoding, where `vocabulary_size` is the number of unique strings appearing in the training set input column plus the
number of special tokens (`<UNK>`, `<PAD>`, `<SOS>`, `<EOS>`).
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

Example text feature entry in the input features list using a parallel cnn encoder:

```yaml
name: text_column_name
type: text
encoder: stacked_parallel_cnn
representation: dense
embedding_size: 256
embeddings_trainable: true
filter_size: 3
num_filters: 256
pool_function: max
output_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
activation: relu
dropout: 0
reduce_output: max
```

### RNN Encoder

The rnn encoder works by first mapping the input token sequence `b x s` (where `b` is the batch size and `s` is the
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
|65+--->Emb 65+--->RNN Layers+-->Connected+-->
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
encoding, where `vocabulary_size` is the number of unique strings appearing in the training set input column plus the
number of special tokens (`<UNK>`, `<PAD>`, `<SOS>`, `<EOS>`).
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

Example text feature entry in the input features list using a parallel cnn encoder:

```yaml
name: text_column_name
type: text
encoder: rnn
representation': dense
embedding_size: 256
embeddings_trainable: true
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
output_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
fc_activation: relu
fc_dropout: 0
reduce_output: last
```

### CNN RNN Encoder

The `cnnrnn` encoder works by first mapping the input token sequence `b x s` (where `b` is the batch size and `s` is
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
|7 |   |Emb 43|                              +---------+
|43|   +------+  +----------+  +----------+  |Fully    |
|65+--->Emb 65+-->CNN Layers+-->RNN Layers+-->Connected+-->
|23|   +------+  +----------+  +----------+  |Layers   |
|4 |   |Emb 23|                              +---------+
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
encoding, where `vocabulary_size` is the number of unique strings appearing in the training set input column plus the
number of special tokens (`<UNK>`, `<PAD>`, `<SOS>`, `<EOS>`).
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
name: text_column_name
type: text
encoder: cnnrnn
representation: dense
embedding_size: 256
embeddings_trainable: true
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
output_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
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
|65+---+Emb 65+---> Transformer +--->Connected+-->
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
encoding, where `vocabulary_size` is the number of unique strings appearing in the training set input column plus the
number of special tokens (`<UNK>`, `<PAD>`, `<SOS>`, `<EOS>`).
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
name: text_column_name
type: text
encoder: transformer
representation: dense
embedding_size: 256
embeddings_trainable: true
num_layers: 1
hidden_size: 256
num_heads: 8
transformer_output_size: 256
dropout: 0.1
num_fc_layers: 0
output_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
fc_activation: relu
fc_dropout: 0
reduce_output: last
```

### Huggingface encoders

All huggingface-based text encoders are configured with the following parameters:

- `pretrained_model_name_or_path` (default is the huggingface default model path for the specified encoder, i.e. `bert-base-uncased` for BERT). This can be either the name of a model or a path where it was downloaded. For details on the variants available refer to the [Hugging Face documentation](https://huggingface.co/docs/transformers/).
- `reduce_output` (default `cls_pooled`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `cls_pooled`, `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).
- `trainable` (default `false`): if `true` the weights of the encoder will be trained, otherwise they will be kept frozen.

!!! note
    Any hyperparameter of any huggingface encoder can be overridden. Check the
    [huggingface documentation](https://huggingface.co/transformers/model_doc/) for which parameters are used for which models.

    ```yaml
    name: text_column_name
    type: text
    encoder: bert
    trainable: true
    num_attention_heads: 16 # Instead of 12
    ```

#### ALBERT Encoder

The `albert` encoder loads a pretrained [ALBERT](https://arxiv.org/abs/1909.11942) (default `albert-base-v2`) model
using the Hugging Face transformers package. Albert is similar to BERT, with significantly lower memory usage and
somewhat faster training time.

#### AutoTransformer

The `auto_transformer` encoder automatically instantiates the model architecture for the specified
`pretrained_model_name_or_path`. Unlike the other HF encoders, `auto_transformer` does not provide a default value for
`pretrained_model_name_or_path`, this is its only mandatory parameter. See the Hugging Face
[AutoModels documentation](https://huggingface.co/transformers/v3.0.2/model_doc/auto.html) for more details.

#### BERT Encoder

The `bert` encoder loads a pretrained [BERT](https://arxiv.org/abs/1810.04805) (default `bert-base-uncased`) model using
the Hugging Face transformers package.

#### CamemBERT Encoder

The `camembert` encoder loads a pretrained [CamemBERT](https://arxiv.org/abs/1911.03894)
(default `jplu/tf-camembert-base`) model using the Hugging Face transformers package. CamemBERT is pre-trained on a
large French language web-crawled text corpus.

#### CTRL Encoder

The `ctrl` encoder loads a pretrained [CTRL](https://arxiv.org/abs/1909.05858) (default `ctrl`) model using the Hugging
Face transformers package. CTRL is a conditional transformer language model trained to condition on control codes that
govern style, content, and task-specific behavior.

#### DistilBERT Encoder

The `distilbert` encoder loads a pretrained [DistilBERT](https://medium.com/huggingface/distilbert-8cf3380435b5)
(default `distilbert-base-uncased`) model using the Hugging Face transformers package. A compressed version of BERT,
60% faster and smaller that BERT.

#### ELECTRA Encoder

The `electra` encoder loads a pretrained [ELECTRA](https://openreview.net/pdf?id=r1xMH1BtvB) model using the Hugging
Face transformers package.

#### FlauBERT Encoder

The `flaubert` encoder loads a pretrained [FlauBERT](https://arxiv.org/abs/1912.05372)
(default `jplu/tf-flaubert-base-uncased`) model using the Hugging Face transformers package. FlauBERT has an architecture
similar to BERT and is pre-trained on a large French language corpus.

#### GPT Encoder

The `gpt` encoder loads a pretrained
[GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
(default `openai-gpt`) model using the Hugging Face transformers package.

#### GPT-2 Encoder

The `gpt2` encoder loads a pretrained
[GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
(default `gpt2`) model using the Hugging Face transformers package.

#### Longformer Encoder

The `longformer` encoder loads a pretrained [Longformer](https://arxiv.org/pdf/2004.05150.pdf)
(default `allenai/longformer-base-4096`) model using the Hugging Face transformers package. Longformer is a good choice
for longer text, as it supports sequences up to 4096 tokens long.

#### RoBERTa Encoder

The `roberta` encoder loads a pretrained [RoBERTa](https://arxiv.org/abs/1907.11692) (default `roberta-base`) model
using the Hugging Face transformers package. Replication of BERT pretraining which may match or exceed the performance
of BERT.

#### Transformer XL Encoder

The `transformer_xl` encoder loads a pretrained [Transformer-XL](https://arxiv.org/abs/1901.02860)
(default `transfo-xl-wt103`) model using the Hugging Face transformers package. Adds novel positional encoding scheme
which improves understanding and generation of long-form text up to thousands of tokens.

#### T5 Encoder

The `t5` encoder loads a pretrained [T5](https://arxiv.org/pdf/1910.10683.pdf) (default `t5-small`) model using the
Hugging Face transformers package. T5 (Text-to-Text Transfer Transformer) is pre-trained on a huge text dataset crawled
from the web and shows good transfer performance on multiple tasks.

#### MT5 Encoder

The `mt5` encoder loads a pretrained [MT5](https://arxiv.org/abs/2010.11934) (default `google/mt5-base`) model using the
Hugging Face transformers package. MT5 is a multilingual variant of T5 trained on a dataset of 101 languages.

#### XLM Encoder

The `xlm` encoder loads a pretrained [XLM](https://arxiv.org/abs/1901.07291) (default `xlm-mlm-en-2048`) model using the
Hugging Face transformers package. Pre-trained by cross-langauge modeling.

#### XLM-RoBERTa Encoder

The `xlmroberta` encoder loads a pretrained [XLM-RoBERTa](https://arxiv.org/abs/1911.02116)
(default `jplu/tf-xlm-reoberta-base`) model using the Hugging Face transformers package. XLM-RoBERTa is a multi-language
model similar to BERT, trained on 100 languages.

#### XLNet Encoder

The `xlnet` encoder loads a pretrained [XLNet](https://arxiv.org/abs/1906.08237) (default `xlnet-base-cased`) model
using the Hugging Face transformers package. XLNet outperforms BERT on a variety of benchmarks.

## Text Output Features and Decoders

Text output features are a special case of [Sequence Features](#sequence-output-features-and-decoders), so all options
of sequence features are available for text features as well.

Text output features can be used for either tagging (classifying each token of an input sequence) or text
generation (generating text by repeatedly sampling from the model). There are two decoders available for these tasks
named `tagger` and `generator` respectively.

The following are the available parameters of a text output feature:

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
confidence_penalty: 0, robust_lambda: 0}`): is a dictionary containing a loss `type`. The only available loss `type` for
text features is `softmax_cross_entropy`. For more details on losses and their options, see also
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
- `num_fc_layers` (default 0): the number of stacked fully connected layers that the input to the feature passes
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

Example text feature entry using a tagger decoder (with default parameters) in the output features list:

```yaml
name: text_column_name
type: text
decoder: tagger
reduce_input: null
dependencies: []
reduce_dependencies: sum
loss:
    type: softmax_cross_entropy
    confidence_penalty: 0
    robust_lambda: 0
    class_weights: 1
    class_similarities_temperature: 0
num_fc_layers: 0
output_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
activation: relu
dropout: 0
attention: false
attention_embedding_size: 256
attention_num_heads: 8
```

### Generator Decoder

In the case of `generator` the decoder is a (potentially empty) stack of fully connected layers, followed by an RNN that
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
If a `b x h` input is provided to a generator decoder using an RNN with attention instead, an error will be raised
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

- `fc_layers` (default `null`): a list of dictionaries containing the parameters of all the fully connected
layers. The length of the list determines the number of stacked fully connected layers and the content of each
dictionary determines the parameters for a specific layer. The available parameters for each layer are: `activation`,
`dropout`, `norm`, `norm_params`, `output_size`, `use_bias`, `bias_initializer` and `weights_initializer`. If any of
those values is missing from the dictionary, the default one specified as a parameter of the decoder will be used instead.
- `num_fc_layers` (default 0): the number of stacked fully connected layers that the input to the feature passes
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
- `embedding_size` (default `256`): The size of the embeddings of the inputs of the generator.
- `beam_width` (default `1`): sampling from the RNN generator is performed using beam search. By default, with a beam of
one, only a greedy sequence using always the most probable next token is generated, but the beam size can be increased.
This usually leads to better performance at the expense of more computation and slower generation.
- `tied` (default `null`): if `null` the embeddings of the targets are initialized randomly. If `tied` names an input
feature, the embeddings of that input feature will be used as embeddings of the target.
The `vocabulary_size` of that input feature has to be the same as the output feature and it has to have an embedding
matrix (binary and number features will not have one, for instance). In this case the `embedding_size` will be the same
as the `state_size`. This is useful for implementing autoencoders where the encoding and decoding part of the model
share parameters.
- `max_sequence_length` (default `256`): The maximum sequence length.

Example text feature entry using a generator decoder in the output features list:

```yaml
name: text_column_name
type: text
decoder: generator
reduce_input: sum
dependencies: []
reduce_dependencies: sum
loss:
    type: softmax_cross_entropy
    confidence_penalty: 0
    robust_lambda: 0
    class_weights: 1
    class_similarities_temperature: 0
num_fc_layers: 0
output_size: 256
use_bias: true
bias_initializer: zeros
weights_initializer: glorot_uniform
activation: relu
dropout: 0
cell_type: rnn
state_size: 256
embedding_size: 256
beam_width: 1
max_sequence_length: 256
```

## Text Features Metrics

The metrics available for text features are the same as for [Sequence Features](../sequence_features#sequence-features-metrics):

- `sequence_accuracy` The rate at which the model predicted the correct sequence.
- `token_accuracy` The number of tokens correctly predicted divided by the total number of tokens in all sequences.
- `last_accuracy` Accuracy considering only the last element of the sequence. Useful to ensure special end-of-sequence
tokens are generated or tagged.
- `edit_distance` Levenshtein distance: the minimum number of single-token edits (insertions, deletions or substitutions)
required to change predicted sequence to ground truth.
- `perplexity` Perplexity is the inverse of the predicted probability of the ground truth sequence, normalized by the
number of tokens. The lower the perplexity, the higher the probability of predicting the true sequence.
- `loss` The value of the loss function.

You can set any of the above as `validation_metric` in the `training` section of the configuration if `validation_field`
names a sequence feature.
