## Text Features Preprocessing

Text features are treated the same as sequence features. Text features are
processed by a tokenizer, which defines how to map from the raw string content
of the dataset column to a sequence of symbols. The metadata includes a list of
all occurring symbols as well as their mapping to symbol indices.

The parameters for text preprocessing are as follows:

- `tokenizer` (default `space_punct`): defines how to map from the raw string content of the dataset column to a sequence of elements. For all
available options see [Tokenizers](../../preprocessing#tokenizers).
- `pretrained_model_name_or_path` (default `null`):
- `vocab_file` (default `null`):
- `max_sequence_length` (default `256`): the maximum length of the text. Texts that are longer than this value will be truncated, while texts that are shorter will be padded.
- `most_common` (default `20000`): the maximum number of most common symbols to be considered. If the data contains more than this amount, the most infrequent symbols will be treated as unknown.
- `padding_symbol` (default `<PAD>`): the string used as a padding symbol. Is is mapped to the integer ID 0 in the vocabulary.
- `unknown_symbol` (default `<UNK>`): the string used as a unknown symbol. Is is mapped to the integer ID 1 in the vocabulary.
- `padding` (default `right`): the direction of the padding. `right` and `left` are available options.
- `lowercase` (default `false`): if the string has to be lowercased before being handled by the tokenizer.
- `missing_value_strategy` (default `fill_with_const`): what strategy to follow when there's a missing value in a binary column. The value should be one of `fill_with_const` (replaces the missing value with a specific value specified with the `fill_value` parameter), `fill_with_mode` (replaces the missing values with the most frequent value in the column), `fill_with_mean` (replaces the missing values with the mean of the values in the column), `backfill` (replaces the missing values with the next valid value).
- `fill_value` (default `""`): the value to replace the missing values with in case the `missing_value_strategy` is `fill-value`.

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
used for [Sequence Features](../sequence_features#sequence-input-features-and-encoders) as well as encoders from the huggingface
transformers library: `bert`, `gpt`, `gpt2`, `xlnet`, `xlm`, `roberta`, `distilbert`, `ctrl`, `camembert`, `albert`,
`t5`, `xlmroberta`, `flaubert`, `electra`, `longformer` and `auto-transformer`.
- `tied` (default `null`): name of the input feature to tie the weights of the encoder with. It needs to be the name of a feature of the same type and with the same encoder parameters.

Example:

```yaml
name: text_column_name
type: text
encoder: bert
trainable: true
```

### Huggingface models

All huggingface-based text encoders are configured with the following parameters:

- `pretrained_model_name_or_path` (default is the huggingface default model path for the specified encoder, i.e. `bert-base-uncased` for BERT). This can be either the name of a model or a path where it was downloaded. For details on the variants available refer to the [Hugging Face documentation](https://huggingface.co/docs/transformers/).
- `reduce_output` (default `cls_pooled`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `cls_pooled`, `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).
- `trainable` (default `false`): if `true` the weights of the encoder will be trained, otherwise they will be kept frozen.

!!! note
    Any hyperparameter of any huggingface encoder can be overridden. Check the [huggingface documentation](https://huggingface.co/transformers/model_doc/) for which parameters are used for which models.

    ```yaml
    name: text_column_name
    type: text
    encoder: bert
    trainable: true
    num_attention_heads: 16 # Instead of 12
    ```

## Text Output Features and Decoders

The decoders are the same used for the [Sequence Features](#sequence-output-features-and-decoders).

Example text input feature using default values:

```yaml
name: sequence_column_name
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
cell_type: rnn
state_size: 256
embedding_size: 256
beam_width: 1
attention: null
tied: null
max_sequence_length: 0
```

## Text Features Metrics

The metrics available for text features are the same as for [Sequence Features](../sequence_features#sequence-features-metrics).
