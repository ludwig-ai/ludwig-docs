The top-level `preprocessing` section specifies:

1. Dataset Splitting (train, validation, test)
2. Dataset Balancing

## Dataset Splitting

When training Ludwig models, we generally split the data into train, validation, and test sets. However, there are
different ways we may want to do this depending on the problem at hand.

There a few different methods that Ludwig uses to split to provided data. Each of these methods is specified in a nested
dictionary under `split`. The following splitting methods are currently supported by Ludwig:

### Random Split

By default, Ludwig will randomly split the data into train, validation, and test sets according to split probabilities,
which by default are: `[0.7, 0.1, 0.2]`. However, you can specify different splitting probabilities if you'd like. For
instance, if I wanted my dataset to be split according to a *60% train*, *15% validation*, and *25% test* regime, I
could specify that with the following split config:

```yaml
preprocessing:
    split: 
        type: random
        probabilities: [0.6, 0.15, 0.25]
```

### Fixed Split

Sometimes, you may want to only have a *train* and a *test* split. Other times you may want to define your own split
according to a dataset specific characteristic. In the event that you want to provide your own split, you can utilize
fixed splitting where you provide a specific column that Ludwig will use to perform the dataset splitting. This column
should contain the following values for each split you want to include in your training/eval:

- `0`: train
- `1`: validation
- `2`: test

*Note: Your dataset must contain a train split while the validation and test splits are encouraged, but optional*

The following config is an example that would perform fixed splitting using a column named `split`:

```yaml

preprocessing:
    split:
        type: fixed
        column: split
```

### Stratified Split

Sometimes you may want to split your data according to a particular column's distribution to maintain the same
representation of this distribution across all your dataset splits. This is where stratified splitting comes in. In
order to perform this type of splitting, you specify the name of the column you want to perform stratified splitting on
and the split probabilities. For example:

```yaml

preprocessing:
    split:
        type: stratify
        column: color
        probabilities: [0.7, 0.1, 0.2]
```

*Note: This split method is only supported with a local Pandas backend. We are working to include a dask version of
splitting in future releases*

### Datetime Split

Another common use case is splitting a column according to a datetime column where you may want to have the data split
in a temporal order. This is useful for situations like [backtesting](https://en.wikipedia.org/wiki/Backtesting) where
a user wants to make sure that a model trained on historical data would have performed well on unseen future data. With
this strategy, we order the data by date (ascending) and then split according to the `split_probabilties`. For example,
if split_probabilities: [0.7, 0.1, 0.2], then the earliest *70%* of the data will be used for training, the middle *10%*
used for validation, and the last *20%* used for testing. The following config shows how to specify this type of
splitting using a datetime column named `created_ts`:

```yaml
preprocessing:
    split:
        type: datetime
        column: created_ts
        probabilities: [0.7, 0.1, 0.2]
```

## Data Balancing

Users working with imbalanced datasets can specify an oversampling or
undersampling parameter which will balance the data during preprocessing.

### Oversampling

In this example, Ludwig will oversample the minority class to achieve a 50%
representation in the overall dataset.

```yaml
preprocessing:
    oversample_minority: 0.5
```

### Undersampling

In this example, Ludwig will undersample the majority class to achieve a 70%
representation in the overall dataset.

```yaml
preprocessing:
    undersample_majority: 0.7
```

*Note: Dataset balancing is only supported for binary output features currently, we are working to add category support 
in future releases.*\
*Note: Specifying both oversampling and undersampling parameters simultaneously is not supported.*

### Sample Ratio

Sometimes users may want to train on a sample of their input training data (maybe there's too much, and we only need 20%).
In order to achieve this, a user can specify a `sample_ratio` to specify the ratio of the dataset to use for training.
By default, the sample ratio is 1.0, so if not specified, all the data will be used for training. For example, if I only
want to use 30% of my input data, I could specify a config like this:

```yaml
preprocessing:
    sample_ratio: 0.3
```
