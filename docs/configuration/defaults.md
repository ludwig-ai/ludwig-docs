The top-level `defaults` section specifies type-global:

1. Preprocessing
2. Encoder
3. Decoder
4. Loss

Any configurations set in the `defaults` section apply to all features of that particular feature type.
Any default `preprocessing` and `encoder` configurations will be applied to all input features
of that feature type, while `decoder` and `loss` configurations will be applied to all output features
of that feature type.

These parameters can be set for individual features through the
[input feature configuration](../features/input_features#preprocessing) or
[output feature configuration](../features/output_features#decoders).

!!! Note
    **Feature-specific configurations override global defaults**: When a parameter is defined for a
    specific feature and also modified via defaults, the feature specific configuration overrides
    the value set in the defaults section for that particular parameter.

```yaml
input_features:
  - 
    name: title
    type: text
    preprocessing:
        most_common: 10
  - 
    name: summary
    type: text
....
defaults:
    text:
        preprocessing:
            most_common: 100
```

In the example config above, the `most_common` preprocessing value for `title` will be
set to 10 instead of taking on the default value of 100, while `summary` will now have its
`most_common` preprocessing value set to 100.

## Defining Defaults

The defaults section of Ludwig has the following structure:

```yaml
defaults:
    <feature_type_1>:
        preprocessing:
            parameter_1: value
            ...
        encoder:
            parameter_1: value
            ...
        decoder:
            parameter_1: value
            ...
        loss:
            parameter_1: value
            ...
    <feature_type_2>:
        preprocessing:
            parameter_1: value
            ...
        encoder:
            parameter_1: value
            ...
        decoder:
            parameter_1: value
            ...
        loss:
            parameter_1: value
            ...
    ...
```

Each of the sections `preprocessing`, `encoder`, `decoder` and `loss` are optional so you can
define one or more as you need.

### Type-Global Preprocessing

Specify preprocessing policies that apply globally across all input features of a
certain data type. For example:

```yaml
defaults:
    category:
        preprocessing:
            missing_value_strategy: fill_with_const
            fill_value: <UNK>
```

The preprocessing parameters that each data type accepts can be found in [datatype-specific documentation](../features/supported_data_types).

Note that different features with the same datatype may require different preprocessing. Type-global preprocessing works
in tandem with feature-specific preprocessing configuration parameters, however, feature-specific configurations override the global settings.

#### Tokenizers

Sequence, text, and set features tokenize features as part of preprocessing. There are several tokenization options that can be specified:

- `characters`: splits every character of the input string in a separate token.
- `space`: splits on space characters using the regex `\s+`.
- `space_punct`: splits on space characters and punctuation using the regex `\w+|[^\w\s]`.
- `underscore`: splits on the underscore character `_`.
- `comma`: splits on the underscore character `,`.
- `untokenized`: treats the whole string as a single token.
- `stripped`: treats the whole string as a single token after removing spaces at the beginning and at the end of the string.
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

In order to use these options, you must first download the the spaCy model:

```
python -m spacy download <language_code>
```

and provide `<language>_<function>` as `tokenizer` like: `english_tokenizer`, `italian_lemmatize_filter`, `multi_tokenize_filter` and so on.
More details on the models can be found in the [spaCy documentation](https://spacy.io/models).

### Type-Global Encoder

Specify the encoder type and encoder related parameters across all input features of a
certain data type. This encoder will be shared across all features of this particular
feature type. For example:

```yaml
defaults:
    text:
        encoder:
            type: stacked_cnn
            embedding_size: 128
            num_filters: 512
```

!!! Note
    The encoder `type` is a required parameter when defining a default encoder for a feature type or
    changing the default value for a parameter for the encoder, since the parameters are tied to
    specific encoders. Only one default encoder can be defined for all features of that particular type.

The encoder types and parameters that each data type accepts can be found in
[datatype-specific documentation](../features/supported_data_types).

### Type-Global Decoder

Specify the decoder type and decoder related parameters across all output features of a
certain data type. For example:

```yaml
defaults:
    text:
        decoder:
            type: generator
            output_size: 128
            bias_initializer: he_normal
```

!!! Note
    The decoder `type` is a required parameter when defining a default decoder for a feature type or
    changing the default value for a parameter for the decoder, since the parameters are tied to
    specific decoders. Only one default decoder can be defined for all features of that particular type.

The decoder types and parameters that each data type accepts can be found in [datatype-specific documentation](../features/supported_data_types).

### Type-Global Loss

Specify the loss type and loss related parameters across all output features of a
certain data type. For example:

```yaml
defaults:
    text:
        loss:
            type: softmax_cross_entropy
            confidence_penalty: 0.1
```

The loss types and parameters that each data type accepts can be found in [datatype-specific documentation](../features/supported_data_types).

## Defaults Example

Following is a full example of a Ludwig configuration with type-global defaults.

```yaml title="config.yaml"
input_features:
  - 
    name: title
    type: text
  - 
    name: body
    type: text
  - 
    name: num_characters
    type: number
    preprocessing:
        normalization: zscore
combiner:
  type: concat
  num_fc_layers: 1
output_features:
  - 
    name: spam
    type: category
defaults:
    text:
        preprocessing:
            most_common: 10000
        encoder:
            type: rnn
            cell_type: lstm
            num_layers: 2
training:
  learning_rate: 0.001
  optimizer:
    type: adam
```

Example CLI command:

```
ludwig train --dataset spam.csv --config_str "{input_features: [{name: title, type: text}, {name: body, type: text}, {name: num_characters, type: number, preprocessing: {normalization: zscore}}], output_features: [{name: spam, type: category}], combiner: {type: concat, num_fc_layers: 1}, defaults: {text: {preprocessing: {word_vocab_size: 10000}, encoder: {type: rnn, cell_type: lstm, num_layers: 2}}}, training: {learning_rate: 0.001, optimizer: {type: adam}}"
```
