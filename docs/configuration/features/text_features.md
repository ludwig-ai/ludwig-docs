## Text Features Preprocessing

Text features are treated in the same way of sequence features, with a couple differences.
Two different tokenizations happen, one that splits at every character and one that splits on whitespace and punctuation are used, and two different keys are added to the HDF5 file, one containing the matrix of characters and one containing the matrix of words.
The same thing happens in the JSON file, which contains dictionaries for mapping characters to integers (and the inverse) and words to integers (and their inverse).
In the configuration you are able to specify which level of representation to use, if the character level or the word level.

The parameters available for preprocessing are:

- `char_tokenizer` (default `characters`): defines how to map from the raw string content of the dataset column to a sequence of characters. The default value and only available option is `characters` and the behavior is to split the string at each character.
- `char_vocab_file` (default `null`):
- `char_sequence_length_limit` (default `1024`): the maximum length of the text in characters. Texts that are longer than this value will be truncated, while sequences that are shorter will be padded.
- `char_most_common` (default `70`): the maximum number of most common characters to be considered. if the data contains more than this amount, the most infrequent characters will be treated as unknown.
- `word_tokenizer` (default `space_punct`): defines how to map from the raw string content of the dataset column to a sequence of elements. For the available options refer to the [Tokenizers](#tokenizers)section.
- `pretrained_model_name_or_path` (default `null`):
- `word_vocab_file` (default `null`):
- `word_sequence_length_limit` (default `256`): the maximum length of the text in words. Texts that are longer than this value will be truncated, while texts that are shorter will be padded.
- `word_most_common` (default `20000`): the maximum number of most common words to be considered. If the data contains more than this amount, the most infrequent words will be treated as unknown.
- `padding_symbol` (default `<PAD>`): the string used as a padding symbol. Is is mapped to the integer ID 0 in the vocabulary.
- `unknown_symbol` (default `<UNK>`): the string used as a unknown symbol. Is is mapped to the integer ID 1 in the vocabulary.
- `padding` (default `right`): the direction of the padding. `right` and `left` are available options.
- `lowercase` (default `false`): if the string has to be lowercased before being handled by the tokenizer.
- `missing_value_strategy` (default `fill_with_const`): what strategy to follow when there's a missing value in a binary column. The value should be one of `fill_with_const` (replaces the missing value with a specific value specified with the `fill_value` parameter), `fill_with_mode` (replaces the missing values with the most frequent value in the column), `fill_with_mean` (replaces the missing values with the mean of the values in the column), `backfill` (replaces the missing values with the next valid value).
- `fill_value` (default `""`): the value to replace the missing values with in case the `missing_value_strategy` is `fill-value`.

Example of text preprocessing.

```yaml
name: text_column_name
type: text
level: word
preprocessing:
    char_tokenizer: characters
    char_vocab_file: null
    char_sequence_length_limit: 1024
    char_most_common: 70
    word_tokenizer: space_punct
    pretrained_model_name_or_path: null
    word_vocab_file: null
    word_sequence_length_limit: 256
    word_most_common: 20000
    padding_symbol: <PAD>
    unknown_symbol: <UNK>
    padding: right
    lowercase: false
    missing_value_strategy: fill_with_const
    fill_value: ""
```

### Tokenizers

Several different features perform raw data preprocessing by tokenizing strings (for instance sequence, text and set).
Here are the tokenizers options you can specify for those features:

- `characters`: splits every character of the input string in a separate token.
- `space`: splits on space characters using the regex `\s+`.
- `space_punct`: splits on space characters and punctuation using the regex `\w+|[^\w\s]`.
- `underscore`: splits on the underscore character `_`.
- `comma`: splits on the underscore character `,`.
- `untokenized`: treats the whole string as a single token.
- `stripped`: treats the whole string as a single token after removing spaces at the beginnign and at the end of the string.
- `hf_tokenizer`: uses the Hugging Face AutoTokenizer which uses a `pretrained_model_name_or_path` parameter to decide which tokenizer to load.
- Language specific tokenizers: [spaCy](https://spacy.io) based language tokenizers.

The [spaCy](https://spacy.io) based tokenizers are functions that use the powerful tokenization and NLP preprocessing models provided the library.
Several languages are available: English (code `en`), Italian (code `it`), Spanish (code `es`), German (code `de`), French (code `fr`), Portuguese (code `pt`), Dutch (code `nl`), Greek (code `el`), Chinese (code `zh`), Danish (code `da`), Dutch (code `el`), Japanese (code `ja`), Lithuanian (code `lt`), Norwegian (code `nb`), Polish (code `pl`), Romanian (code `ro`) and Multi (code `xx`, useful in case you have a dataset containing different languages).
For each language different functions are available:

- `tokenize`: uses spaCy tokenizer,
- `tokenize_filter`: uses spaCy tokenizer and filters out punctuation, numbers, stopwords and words shorter than 3 characters,
- `tokenize_remove_stopwords`: uses spaCy tokenizer and filters out stopwords,
- `lemmatize`: uses spaCy lemmatizer,
- `lemmatize_filter`: uses spaCy lemmatizer and filters out punctuation, numbers, stopwords and words shorter than 3 characters,
- `lemmatize_remove_stopwords`: uses spaCy lemmatize and filters out stopwords.

In order to use these options, you have to download the the spaCy model:

```
python -m spacy download <language_code>
```

and provide `<language>_<function>` as `tokenizer` like: `english_tokenizer`, `italian_lemmatize_filter`, `multi_tokenize_filter` and so on.
More details on the models can be found in the [spaCy documentation](https://spacy.io/models).

## Text Input Features and Encoders

Text input feature parameters are

- `encoder` (default `parallel_cnn`): encoder to use for the input text feature. The available encoders come from [Sequence Features](#sequence-input-features-and-encoders) and these text specific encoders: `bert`, `gpt`, `gpt2`, `xlnet`, `xlm`, `roberta`, `distilbert`, `ctrl`, `camembert`, `albert`, `t5`, `xlmroberta`, `flaubert`, `electra`, `longformer` and `auto-transformer`.
- `level` (default `word`): `word` specifies using text words, `char` use individual characters.
- `tied_weights` (default `null`): name of the input feature to tie the weights of the encoder with. It needs to be the name of a feature of the same type and with the same encoder parameters.

### BERT Encoder

The `bert` encoder loads a pretrained [BERT](https://arxiv.org/abs/1810.04805) model using the Hugging Face transformers package.

- `pretrained_model_name_or_path` (default `bert-base-uncased`): it can be either the name of a model or a path where it was downloaded. For details on the variants available refer to the [Hugging Face documentation](https://huggingface.co/transformers/model_doc/bert.html).
- `reduced_output` (default `cls_pooled`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `cls_pool`, `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).
- `trainable` (default `false`): if `true` the weights of the encoder will be trained, otherwise they will be kept frozen.

### GPT Encoder

The `gpt` encoder loads a pretrained [GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) model using the Hugging Face transformers package.

- `pretrained_model_name_or_path` (default `openai-gpt`): it can be either the name of a model or a path where it was downloaded. For details on the variants available refer to the [Hugging Face documentation](https://huggingface.co/transformers/model_doc/gpt.html).
- `reduced_output` (default `sum`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).
- `trainable` (default `false`): if `true` the weights of the encoder will be trained, otherwise they will be kept frozen.

### GPT-2 Encoder

The `gpt2` encoder loads a pretrained [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) model using the Hugging Face transformers package.

- `pretrained_model_name_or_path` (default `gpt2`): it can be either the name of a model or a path where it was downloaded. For details on the variants available refer to the [Hugging Face documentation](https://huggingface.co/transformers/model_doc/gpt2.html).
- `reduced_output` (default `sum`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).
- `trainable` (default `false`): if `true` the weights of the encoder will be trained, otherwise they will be kept frozen.

### XLNet Encoder

The `xlnet` encoder loads a pretrained [XLNet](https://arxiv.org/abs/1906.08237) model using the Hugging Face transformers package.

- `pretrained_model_name_or_path` (default `xlnet-base-cased`): it can be either the name of a model or a path where it was downloaded. For details on the variants available refer to the [Hugging Face documentation](https://huggingface.co/transformers/model_doc/xlnet.html).
- `reduced_output` (default `sum`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).
- `trainable` (default `false`): if `true` the weights of the encoder will be trained, otherwise they will be kept frozen.

### XLM Encoder

The `xlm` encoder loads a pretrained [XLM](https://arxiv.org/abs/1901.07291) model using the Hugging Face transformers package.

- `pretrained_model_name_or_path` (default `xlm-mlm-en-2048`): it can be either the name of a model or a path where it was downloaded. For details on the variants available refer to the [Hugging Face documentation](https://huggingface.co/transformers/model_doc/xlm.html).
- `reduced_output` (default `sum`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).
- `trainable` (default `false`): if `true` the weights of the encoder will be trained, otherwise they will be kept frozen.

### RoBERTa Encoder

The `roberta` encoder loads a pretrained [RoBERTa](https://arxiv.org/abs/1907.11692) model using the Hugging Face transformers package.

- `pretrained_model_name_or_path` (default `roberta-base`): it can be either the name of a model or a path where it was downloaded. For details on the variants available refer to the [Hugging Face documentation](https://huggingface.co/transformers/model_doc/roberta.html).
- `reduced_output` (default `cls_pooled`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `cls_pool`, `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).
- `trainable` (default `false`): if `true` the weights of the encoder will be trained, otherwise they will be kept frozen.

### DistilBERT Encoder

The `distilbert` encoder loads a pretrained [DistilBERT](https://medium.com/huggingface/distilbert-8cf3380435b5) model using the Hugging Face transformers package.

- `pretrained_model_name_or_path` (default `istilbert-base-uncased`): it can be either the name of a model or a path where it was downloaded. For details on the variants available refer to the [Hugging Face documentation](https://huggingface.co/transformers/model_doc/distilbert.html).
- `reduced_output` (default `sum`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).
- `trainable` (default `false`): if `true` the weights of the encoder will be trained, otherwise they will be kept frozen.

### CTRL Encoder

The `ctrl` encoder loads a pretrained [CTRL](https://arxiv.org/abs/1909.05858) model using the Hugging Face transformers package.

- `pretrained_model_name_or_path` (default `ctrl`): it can be either the name of a model or a path where it was downloaded. For details on the variants available refer to the [Hugging Face documentation](https://huggingface.co/transformers/model_doc/ctrl.html).
- `reduced_output` (default `sum`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).
- `trainable` (default `false`): if `true` the weights of the encoder will be trained, otherwise they will be kept frozen.

### CamemBERT Encoder

The `camembert` encoder loads a pretrained [CamemBERT](https://arxiv.org/abs/1911.03894) model using the Hugging Face transformers package.

- `pretrained_model_name_or_path` (default `jplu/tf-camembert-base`): it can be either the name of a model or a path where it was downloaded. For details on the variants available refer to the [Hugging Face documentation](https://huggingface.co/transformers/model_doc/camembert.html).
- `reduced_output` (default `cls_pooled`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `cls_pool`, `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).
- `trainable` (default `false`): if `true` the weights of the encoder will be trained, otherwise they will be kept frozen.

### ALBERT Encoder

The `albert` encoder loads a pretrained [ALBERT](https://arxiv.org/abs/1909.11942) model using the Hugging Face transformers package.

- `pretrained_model_name_or_path` (default `albert-base-v2`): it can be either the name of a model or a path where it was downloaded. For details on the variants available refer to the [Hugging Face documentation](https://huggingface.co/transformers/model_doc/albert.html).
- `reduced_output` (default `cls_pooled`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `cls_pool`, `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).
- `trainable` (default `false`): if `true` the weights of the encoder will be trained, otherwise they will be kept frozen.

### T5 Encoder

The `t5` encoder loads a pretrained [T5](https://arxiv.org/pdf/1910.10683.pdf) model using the Hugging Face transformers package.

- `pretrained_model_name_or_path` (default `t5-small`): it can be either the name of a model or a path where it was downloaded. For details on the variants available refer to the [Hugging Face documentation](https://huggingface.co/transformers/model_doc/t5.html).
- `reduced_output` (default `sum`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).
- `trainable` (default `false`): if `true` the weights of the encoder will be trained, otherwise they will be kept frozen.

### XLM-RoBERTa Encoder

The `xlmroberta` encoder loads a pretrained [XLM-RoBERTa](https://arxiv.org/abs/1911.02116) model using the Hugging Face transformers package.

- `pretrained_model_name_or_path` (default `jplu/tf-xlm-reoberta-base`): it can be either the name of a model or a path where it was downloaded. For details on the variants available refer to the [Hugging Face documentation](https://huggingface.co/transformers/model_doc/xlmroberta.html).
- `reduced_output` (default `cls_pooled`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `cls_pool`, `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).
- `trainable` (default `false`): if `true` the weights of the encoder will be trained, otherwise they will be kept frozen.

### FlauBERT Encoder

The `flaubert` encoder loads a pretrained [FlauBERT](https://arxiv.org/abs/1912.05372) model using the Hugging Face transformers package.

- `pretrained_model_name_or_path` (default `jplu/tf-flaubert-base-uncased`): it can be either the name of a model or a path where it was downloaded. For details on the variants available refer to the [Hugging Face documentation](https://huggingface.co/transformers/model_doc/flaubert.html).
- `reduced_output` (default `sum`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).
- `trainable` (default `false`): if `true` the weights of the encoder will be trained, otherwise they will be kept frozen.

### ELECTRA Encoder

The `electra` encoder loads a pretrained [ELECTRA](https://openreview.net/pdf?id=r1xMH1BtvB) model using the Hugging Face transformers package.

- `pretrained_model_name_or_path` (default `google/electra-small-discriminator`): it can be either the name of a model or a path where it was downloaded. For details on the variants available refer to the [Hugging Face documentation](https://huggingface.co/transformers/model_doc/electra.html).
- `reduced_output` (default `sum`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).
- `trainable` (default `false`): if `true` the weights of the encoder will be trained, otherwise they will be kept frozen.

### Longformer Encoder

The `longformer` encoder loads a pretrained [Longformer](https://arxiv.org/pdf/2004.05150.pdf) model using the Hugging Face transformers package.

- `pretrained_model_name_or_path` (default `allenai/longformer-base-4096`): it can be either the name of a model or a path where it was downloaded. For details on the variants available refer to the [Hugging Face documentation](https://huggingface.co/transformers/model_doc/longformer.html).
- `reduced_output` (default `cls_pooled`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `cls_pool`, `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).
- `trainable` (default `false`): if `true` the weights of the encoder will be trained, otherwise they will be kept frozen.

### Auto-Transformer Encoder

The `auto_transformer` encoder loads a pretrained model using the Hugging Face transformers package.
It's the best option for customly trained models that don't fit into the other pretrained transformers encoders.

- `pretrained_model_name_or_path`: it can be either the name of a model or a path where it was downloaded. For details on the available models to the [Hugging Face documentation](https://huggingface.co/transformers/model_doc/auto.html#tfautomodel).
- `reduced_output` (default `sum`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).
- `trainable` (default `false`): if `true` the weights of the encoder will be trained, otherwise they will be kept frozen.

### Example usage

Example text input feature encoder usage:

```yaml
name: text_column_name
type: text
level: word
encoder: bert
tied_weights: null
pretrained_model_name_or_path: bert-base-uncased
reduced_output: cls_pooled
trainable: false
```

## Text Output Features and Decoders

The decoders are the same used for the [Sequence Features](#sequence-output-features-and-decoders).
The only difference is that you can specify an additional `level` parameter with possible values `word` or `char` to force to use the text words or characters as inputs (by default the encoder will use `word`).

Example text input feature using default values:

```yaml
name: sequence_column_name
type: text
level: word
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
fc_size: 256
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
cell_type: rnn
state_size: 256
embedding_size: 256
beam_width: 1
attention: null
tied_embeddings: null
max_sequence_length: 0
```

## Text Features Measures

The measures are the same used for the [Sequence Features](#sequence-features-measures).
