The `preprocessing` section of the configuration makes it possible to specify datatype specific parameters to perform data preprocessing.
The preprocessing dictionary contains one key of each datatype, but you have to specify only the ones that apply to your case, the other ones will be kept as defaults.
Moreover, the preprocessing dictionary contains parameters related to how to split the data that are not feature specific.

- `force_split` (default `false`): if `true` the `split` column in the dataset file is ignored and the dataset is randomly split. If `false` the `split` column is used if available.
- `split_probabilities` (default `[0.7, 0.1, 0.2]`): the proportion of the dataset data to end up in training, validation and test, respectively. The three values have to sum up to one.
- `stratify` (default `null`): if `null` the split is random, otherwise you can specify the name of a `category` feature and the split will be stratified on that feature.

Example preprocessing dictionary (showing default values):

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

The details about the preprocessing parameters that each datatype accepts will be provided in the datatype-specific sections.

It is important to point out that different features with the same datatype may require different preprocessing.
For instance a document classification model may have two text input features, one for the title of the document and one for the body.

As the length of the title is much shorter than the length of the body, the parameter `word_length_limit` should be set to 10 for the title and 2000 for the body, but both of them share the same parameter `most_common_words` with value 10000.

The way to do this is adding a `preprocessing` key inside the title `input_feature` dictionary and one in the `body` input feature dictionary containing the desired parameter and value.
The configuration will look like:

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
