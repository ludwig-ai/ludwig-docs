# Overview

Ludwig data preprocessing performs a few different operations on the incoming dataset:

1. **Computing metadata** like vocabulary, vocabulary size, and sequence lengths. This allows Ludwig to create
    dictionaries like `idx2str` or `str2idx` to map between raw data values to tensor values.
2. **Handling missing values** any rows/examples that have missing feature values are filled in with constants or other
   example-derived values (see [Preprocessing Configuration](../../configuration/preprocessing)).
3. (optional) **Splitting dataset** into train, validation, and test based on splitting percentages, or using explicitly
   specified splits.
4. (optional) **Balancing data** which can be useful for datasets with heavily underrepresented or overrepresented
   classes.

Data preprocessing maps raw data to two files: 1) an HDF5 file containing tensors and 2) a JSON file of metadata.
The HDF5 and JSON files are saved in the same directory as the input dataset, unless `--skip_save_processed_input` is
used. The two files will serve as a cache to help avoid performing the same preprocessing again, which can be time
consuming.

The preprocessing process is highly customizable via the [Preprocessing section](../../configuration/preprocessing) of
the Ludwig config. The basic assumption is always that all data is UTF-8 encoded and contains one row for each
example and one column for each feature.

It's helpful to assign types to each feature. Some types assume a specific format, and different types will have
different ways of mapping raw data into tensors. From v0.5, users also have the option to rely on Ludwig AutoML to
assign types automatically.

# Preprocessing for different data types

Each datatype is preprocessed in a different way, using different parameters and different tokenizers.
Details on how to set those parameters for each feature type and for each specific feature is described in the
[Configuration - Preprocessing](../../configuration/preprocessing) section.

## Binary features

`Binary` features are directly transformed into a binary valued vector of length `n` (where `n` is the size of the
dataset) and added to the HDF5 with a key that reflects the name of column in the dataset. No additional information
about them is available in the JSON metadata file.

## Number features

`Number` features are directly transformed into a float valued vector of length `n` (where `n` is the size of the
dataset) and added to the HDF5 with a key that reflects the name of column in the dataset. No additional information
about them is available in the JSON metadata file.

## Category features

`Category` features are transformed into an integer valued vector of size `n` (where `n` is the size of the dataset)
and added to the HDF5 with a key that reflects the name of column in the dataset.

The way categories are mapped into integers consists of first collecting a dictionary of all the different category
strings present in the column of the dataset, then rank them by frequency and then assign them an increasing integer ID
from the most frequent to the most rare (with 0 being assigned to a `<UNK>` token).  The column name is added to the
JSON file, with an associated dictionary containing:

1. the mapping from integer to string (`idx2str`)
2. the mapping from string to id (`str2idx`)
3. the mapping from string to frequency (`str2freq`)
4. the size of the set of all tokens (`vocab_size`)
5. additional preprocessing information (by default how to fill missing values
   and what token to use to fill missing values)

## Set features

`Set` features are transformed into a binary (int8 actually) valued matrix of size `n x l` (where `n` is the size of the
dataset and `l` is the minimum of the size of the biggest set and a `max_size` parameter) and added to HDF5 with a key
that reflects the name of column in the dataset.

The way sets are mapped into integers consists in first using a tokenizer to map from strings to sequences of set items
(by default this is done by splitting on spaces).  Then a dictionary of all the different set item strings present in
the column of the dataset is collected, then they are ranked by frequency and an increasing integer ID is assigned to
them from the most frequent to the most rare (with 0 being assigned to `<PAD>` used for padding and 1 assigned to
`<UNK>` item).  The column name is added to the JSON file, with an associated dictionary containing:

1. the mapping from integer to string (`idx2str`)
1. the mapping from string to id (`str2idx`)
1. the mapping from string to frequency (`str2freq`)
1. the maximum size of all sets (`max_set_size`)
1. additional preprocessing information (by default how to fill missing values
   and what token to use to fill missing values)

## Bag features

`Bag` features are treated in the same way of set features, with the only difference being that the matrix had float
values (frequencies).

## Sequence Features

Sequence features by default are managed by `space` tokenizers. This splits the content of the feature value into a list
of strings using space.

| before tokenizer       | after tokenizer            |
| ---------------------- | -------------------------- |
| "token3 token4 token2" | \[token3, token4, token2\] |
| "token3 token1"        | \[token3, token1\]         |

Computing metadata: A list `idx2str` and two dictionaries `str2idx` and `str2freq` are created containing all the tokens
in all the lists obtained by splitting all the rows of the column and an integer id is assigned to each of them (in
order of frequency).

```json
{
    "column_name": {
        "idx2str": [
            "<PAD>",
            "<UNK>",
            "token3",
            "token2",
            "token4",
            "token1"
        ],
        "str2idx": {
            "<PAD>": 0,
            "<UNK>": 1,
            "token3": 2,
            "token2": 3,
            "token4": 4,
            "token1": 5
        },
        "str2freq": {
            "<PAD>":  0,
            "<UNK>":  0,
            "token3": 2,
            "token2": 1,
            "token4": 1,
            "token1": 1
        }
    }
}
```

Finally, a numpy matrix is created with sizes `n x l` where `n` is the number of rows in the column and `l` is the
minimum of the longest tokenized list and a `max_length` parameter that can be set. All sequences shorter than `l` are
right-padded to the `max_length` (though this behavior may also be modified through a parameter).

| after tokenizer            | numpy matrix |
| -------------------------- | ------------ |
| \[token3, token4, token2\] | 2 4 3        |
| \[token3, token1\]         | 2 5 0        |

The final result matrix is saved in the HDF5 with the name of the original column in the dataset as key, while the
mapping from token to integer ID (and its inverse mapping) is saved in the JSON file.

A frequency-ordered vocabulary dictionary is created which maps tokens to integer IDs. Special symbols like `<PAD>`,
`<START>`, `<STOP>`, and `<UNK>` have specific indices. By default, we use `[0, 1, 2, 3]`, but these can be overridden
manually.

If a `huggingface` encoder is specified, then that encoder's special symbol indices will be used instead.

The computed metadata includes:

1. the mapping from integer to string (`idx2str`)
2. the mapping from string to id (`str2idx`)
3. the mapping from string to frequency (`str2freq`)
4. the maximum length of all sequences (`sequence_length_limit`)
5. additional preprocessing information (by default how to fill missing values
   and what token to use to fill missing values)

## Text features

`Text` features are treated in the same way of sequence features, with a couple differences. Two different tokenizations
happen, one that splits at every character and one that uses a custom tokenizer. Two different keys are added to the
HDF5 file, one for the matrix of characters and one for the matrix of symbols.

The same thing happens in the JSON file, where there are two sets of dictionaries, one for mapping characters to
integers (and the inverse) and symbols to integers (and their inverse).

If a `huggingface` encoder is specified, then that encoder's tokenizer will be used for the symbol-based tokenizer.

In the configuration users can specify which level of representation to use: the character level or the symbol level.

## Timeseries features

`Timeseries` features are treated in the same way of sequence features, with the only difference being that the matrix
in the HDF5 file does not have integer values, but float values. The JSON file has no additional mapping information.

## Image features

`Image` features are transformed into a int8 valued tensor of size `n x h x w x c` (where `n` is the size of the dataset
and `h x w` is a specific resizing of the image that can be set, and `c` is the number of color channels) and added to
HDF5 with a key that reflects the name of column in the dataset.

The column name is added to the JSON file, with an associated dictionary containing preprocessing information about the
sizes of the resizing.
