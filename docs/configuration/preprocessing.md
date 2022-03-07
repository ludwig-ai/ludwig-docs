The top-level `preprocessing` section specifies:

1. dataset splitting (train, validation, test)
2. data balancing
3. type-global preprocessing

## Dataset Splitting

Split data into train, validation, and test based on splitting percentages.

- `force_split` (default `false`): if `true` the `split` column in the dataset file is ignored and the dataset is randomly split. If `false` the `split` column is used if available.
- `split_probabilities` (default `[0.7, 0.1, 0.2]`): the proportion of the dataset data to end up in training, validation and test, respectively. The three values have to sum up to one.
- `stratify` (default `null`): if `null` the split is random, otherwise you can specify the name of a `category` feature and the split will be stratified on that feature.

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

If you are working with imbalanced classes, you can specify an oversampling or undersampling parameter which will balance the data in the specified manner. For example:

This example will trigger the preprocessing pipeline to oversample the minority class until it has a 50% representation relative to the majority class.

```yaml
preprocessing:
    oversample_minority: 0.5
```

This example will trigger the preprocessing pipeline to undersample the majority class until the minority class has a 70% representation relative to the majority class.

```yaml
preprocessing:
    undersample_majority: 0.7
```

Data balancing is only supported for binary output classes currently. Additionally, specifying both parameters at the same time is also not supported currently.

## Type-global Preprocessing

Specify preprocessing policies that apply globally across all features of a certain data type. For example:

```yaml
preprocessing:
    category:
        missing_value_strategy: fill_with_const
        fill_value: <UNK>
```

The preprocessing parameters that each data type accepts can be found in [datatype-specific documentation](../../features/supported_data_types).

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

To configure feature-specific preprocessing, please check [datatype-specific documentation](../../features/supported_data_types).
