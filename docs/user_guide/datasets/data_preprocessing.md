# Dataset preprocessing

Ludwig data preprocessing maps raw data in a supported dataset into an HDF5 file containing tensors and a JSON file containing mappings from strings to tensors when needed.
This mapping is performed when a UTF-8 encoded data is provided as input and both HDF5 and JSON files are saved in the same directory as the input dataset, unless the argument `--skip_save_processed_input` is used (both in `train` and `experiment` commands).
The reason to save those files is both to provide a cache and avoid performing the preprocessing again (as, depending on the type of features involved, it could be time consuming) and to provide the needed mappings to be able to map unseen data into tensors.

The preprocessing process is personalizable to fit the specifics of your data format, but the basic assumption is always that your UTF-8 encoded dataset contains one row for each datapoint and one column for each feature (either input or output), and that you are able to determine the type of that column among the ones supported by Ludwig.
The reason for that is that each data type is mapped into tensors in a different way and expects the content to be formatted in a specific way.
Different datatypes may have different tokenizers that format the values of a cell.

For instance, the value of a cell of a sequence feature column by default is managed by a `space` tokenizer, that splits the content of the value into a list of strings using space.

| before tokenizer       | after tokenizer            |
| ---------------------- | -------------------------- |
| "token3 token4 token2" | \[token3, token4, token2\] |
| "token3 token1"        | \[token3, token1\]         |

Then a list `idx2str` and two dictionaries `str2idx` and `str2freq` are created containing all the tokens in all the lists obtained by splitting all the rows of the column and an integer id is assigned to each of them (in order of frequency).

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

Finally, a numpy matrix is created with sizes `n x l` where `n` is the number of rows in the column and `l` is the minimum of the longest tokenized list and a `max_length` parameter that can be set.
All sequences shorter than `l` are padded on the right (but this behavior may also be modified through a parameter).

| after tokenizer            | numpy matrix |
| -------------------------- | ------------ |
| \[token3, token4, token2\] | 2 4 3        |
| \[token3, token1\]         | 2 5 0        |

The final result matrix is saved in the HDF5 with the name of the original column in the dataset as key, while the mapping from token to integer ID (and its inverse mapping) is saved in the JSON file.

Each datatype is preprocessed in a different way, using different parameters and different tokenizers.
Details on how to set those parameters for each feature type and for each specific feature will be described in the [Configuration - Preprocessing](#preprocessing) section.

`Binary` features are directly transformed into a binary valued vector of length `n` (where `n` is the size of the dataset) and added to the HDF5 with a key that reflects the name of column in the dataset.
No additional information about them is available in the JSON metadata file.

`Numerical` features are directly transformed into a float valued vector of length `n` (where `n` is the size of the dataset) and added to the HDF5 with a key that reflects the name of column in the dataset.
No additional information about them is available in the JSON metadata file.

`Category` features are transformed into an integer valued vector of size `n` (where `n` is the size of the dataset) and added to the HDF5 with a key that reflects the name of column in the dataset.
The way categories are mapped into integers consists in first collecting a dictionary of all the different category strings present in the column of the dataset, then rank them by frequency and then assign them an increasing integer ID from the most frequent to the most rare (with 0 being assigned to a `<UNK>` token).  The column name is added to the JSON file, with an associated dictionary containing:

1. the mapping from integer to string (`idx2str`)
1. the mapping from string to id (`str2idx`)
1. the mapping from string to frequency (`str2freq`)
1. the size of the set of all tokens (`vocab_size`)
1. additional preprocessing information (by default how to fill missing values
   and what token to use to fill missing values)

`Set` features are transformed into a binary (int8 actually) valued matrix of size `n x l` (where `n` is the size of the dataset and `l` is the minimum of the size of the biggest set and a `max_size` parameter) and added to HDF5 with a key that reflects the name of column in the dataset.
The way sets are mapped into integers consists in first using a tokenizer to map from strings to sequences of set items (by default this is done by splitting on spaces).  Then a dictionary of all the different set item strings present in the column of the dataset is collected, then they are ranked by frequency and an increasing integer ID is assigned to them from the most frequent to the most rare (with 0 being assigned to `<PAD>` used for padding and 1 assigned to `<UNK>` item).  The column name is added to the JSON file, with an associated dictionary containing:

1. the mapping from integer to string (`idx2str`)
1. the mapping from string to id (`str2idx`)
1. the mapping from string to frequency (`str2freq`)
1. the maximum size of all sets (`max_set_size`)
1. additional preprocessing information (by default how to fill missing values
   and what token to use to fill missing values)

`Bag` features are treated in the same way of set features, with the only difference being that the matrix had float values (frequencies).

`Sequence` features are transformed into an integer valued matrix of size `n x l` (where `n` is the size of the dataset and `l` is the minimum of the length of the longest sequence and a `sequence_length_limit` parameter) and added to HDF5 with a key that reflects the name of column in the dataset.
The way sets are mapped into integers consists in first using a tokenizer to map from strings to sequences of tokens (by default this is done by splitting on spaces).
Then a dictionary of all the different token strings present in the column of the dataset is collected, then they are ranked by frequency and an increasing integer ID is assigned to them from the most frequent to the most rare (with 0 being assigned to `<PAD>` used for padding and 1 assigned to `<UNK>` item).
The column name is added to the JSON file, with an associated dictionary containing:

1. the mapping from integer to string (`idx2str`)
1. the mapping from string to id (`str2idx`)
1. the mapping from string to frequency (`str2freq`)
1. the maximum length of all sequences (`sequence_length_limit`)
1. additional preprocessing information (by default how to fill missing values
   and what token to use to fill missing values)

`Text` features are treated in the same way of sequence features, with a couple differences.
Two different tokenizations happen, one that splits at every character and one that uses a spaCy based tokenizer (and removes stopwords), and two different keys are added to the HDF5 file, one for the matrix of characters and one for the matrix of words.
The same thing happens in the JSON file, where there are dictionaries for mapping characters to integers (and the inverse) and words to integers (and their inverse).
In the configuration you are able to specify which level of representation to use: the character level or the word level.

`Timeseries` features are treated in the same way of sequence features, with the only difference being that the matrix in the HDF5 file does not have integer values, but float values.
Moreover, there is no need for any mapping in the JSON file.

`Image` features are transformed into a int8 valued tensor of size `n x h x w x c` (where `n` is the size of the dataset and `h x w` is a specific resizing of the image that can be set, and `c` is the number of color channels) and added to HDF5 with a key that reflects the name of column in the dataset.
The column name is added to the JSON file, with an associated dictionary containing preprocessing information about the sizes of the resizing.
