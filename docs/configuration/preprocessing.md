The top-level `preprocessing` section specifies:

1. dataset splitting (train, validation, test)
2. data balancing

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

## Feature-specific preprocessing

To configure feature-specific preprocessing, please check [datatype-specific documentation](../features/supported_data_types).
