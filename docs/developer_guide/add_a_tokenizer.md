Tokenizers transform a text string into a sequence of tokens. Ludwig will call the tokenizer for each row in the dataset
during preprocessing. It will then collect a list of unique tokens and assign an integer index to each unique token.
This ordered list of unique tokens is called the __vocabulary__, and it will be used by encoders to convert tokens to
embeddings and by decoders to convert output predictions to tokens.

A tokenizer is primarily responsible for splitting a string into a list of tokens, and may optionally perform other
processing usually for the purpose of reducing the size of the vocabulary. Some examples of processing tokenizers may
perform include:

- Splitting on a delimiter ex. underscore "_" or comma ",".
- Removing punctuation characters
- Removing stop words, for example "a", "an", "the" in English.
- Lemmatization: Grouping inflected forms of the same word i.e. "car", "cars", "car's", "cars'" -> "car"

A tokenizer, once registered, can be used to preprocess any text input column by specifying its name as the value of
`tokenizer` in the `preprocessing` config dictionary:

```yaml
input_features:
    -   name: title
        type: text
        preprocessing:
            tokenizer: <NEW_TOKENIZER>
```

Tokenizers are defined in `ludwig/utils/tokenizers.py`. To add a tokenizer, define a new subclass of `BaseTokenizer` and
add it to the registry.

# 1. Add a new tokenizer class

Tokenizers may define an optional constructor which can receive arguments from the config. Most tokenizers have no
config parameters, and do not need a constructor. For an example of a tokenizer which uses a parameter in its
constructor, see `HFTokenizer`.

The `__call__` method does the actual processing, is called with a single string argument, and is expected to return a
list of strings. The tokenizer will be called once for each example in the dataset during preprocessing. Preprocessed
token sequences will be cached on disk in .hdf5 files and re-used in training and validation, thus the tokenizer will
not be called during training.

```python
class NewTokenizer(BaseTokenizer):
    def __init__(self, **kwargs):
        super().__init__()
        # Initialize any variables or state

    def __call__(self, text: str) -> List[str]:
        # tokenized_text = result of tokenizing
        return tokenized_text
```

# 2. Add the tokenizer to the registry

Tokenizer names are mapped to their implementations in the `tokenizer_registry` dictionary at the bottom of
`ludwig/utils/tokenizers.py`.
```python
tokenizer_registry = {
    "characters": CharactersToListTokenizer,
    "space": SpaceStringToListTokenizer,
    ...
    "new_tokenizer": NewTokenizer,  # Add your tokenizer as a new entry in the registry.
```
