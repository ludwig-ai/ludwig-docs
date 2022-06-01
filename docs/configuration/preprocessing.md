The top-level `preprocessing` section specifies:

1. dataset splitting (train, validation, test)
2. data balancing
3. type-global preprocessing

## Dataset Splitting

Split data into train, validation, and test.

By default, Ludwig looks for a column named `split` (case-sensitive) which is
expected to consist of 3 possible values that correspond to different datasets:

- `0`: train
- `1`: validation
- `2`: test

If the data does not contain the  `split` column, then data is randomly split
based on splitting percentages, defined by `split_probabilities`.

If `force_split` is `true`, the the split column in the dataset is ignored and
the dataset is randomly split based on splitting percentages, defined by
`split_probabilities`.

Summary of parameters:

- `force_split` (default `false`): if `true` the `split` column in the dataset
    file is ignored and the dataset is randomly split. If `false` the `split`
    column is used if available.
- `split_probabilities` (default `[0.7, 0.1, 0.2]`): the proportion of the
    dataset data to end up in training, validation and test, respectively. The
    three values must sum to 1.0.
- `stratify` (default `null`): if `null` the split is random, otherwise you can
    specify the name of a `category` feature and the split will be stratified on
    that feature.

```yaml
preprocessing:
    force_split: false
    split_probabilities: [0.7, 0.1, 0.2]
    stratify: null
    category: {...}
    sequence: {...}
    text: {...}
    ...
```

## Data Balancing

Users working with imbalanced datasets can specify an oversampling or
undersampling parameter which will balance the data during preprocessing.

In this example, Ludwig will oversample the minority class to achieve a 50%
representation in the overall dataset.

```yaml
preprocessing:
    oversample_minority: 0.5
```

In this example, Ludwig will undersample the majority class to achieve a 70%
representation in the overall dataset.

```yaml
preprocessing:
    undersample_majority: 0.7
```

Data balancing is only supported for binary output features. Additionally,
specifying both oversampling and undersampling parameters simultaneously is not
supported.

## Type-global Preprocessing

Specify preprocessing policies that apply globally across all features of a
certain data type. For example:

```yaml
preprocessing:
    category:
        missing_value_strategy: fill_with_const
        fill_value: <UNK>
```

The preprocessing parameters that each data type accepts can be found in [datatype-specific documentation](../features/supported_data_types).

Note that different features with the same datatype may require different preprocessing. Type-global preprocessing works
in tandem with feature-specific preprocessing configuration parameters, which override the global settings.

For example, a document classification model may have two text input features, one for the title of the document and one for the body.

As the length of the title is much shorter than the length of the body, the parameter `word_length_limit` should be set to `10` for the title and `2000` for the body, but we want both features to share the same vocabulary, with `most_common_words: 10000`.

The way to do this is adding a `preprocessing` key inside the title `input_feature` dictionary and one in the `body` input feature dictionary containing the desired parameter and value.

```yaml
preprocessing:
    text:
        most_common_word: 10000
input_features:
    -
        name: title
        type: text
        preprocessing:
            word_length_limit: 20
    -
        name: body
        type: text
        preprocessing:
            word_length_limit: 2000
```

## Feature-specific preprocessing

To configure feature-specific preprocessing, please check [datatype-specific documentation](../features/supported_data_types).

## Tokenizers

Sequence, text, and set features tokenize features as part of preprocessing. There are several tokenization options that
can be specified:

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
