The top-level `preprocessing` section specifies dataset splitting (train,
validation, test), and dataset balancing.

```yaml
preprocessing:
    split: 
        type: random
        probabilities: [0.7, 0.2, 0.1]
```

## Dataset Splitting

Data splitting is an important aspect of machine learning to train and evaluate
machine learning models.

Ludwig supports splitting data into train, validation, and test sets, and this
is configured using the top-level `preprocessing` section of the Ludwig config.

There is no set guideline or metric for how the data should be split; it may
depend on the size of your data or the type of problem.

There are a few different methods that Ludwig uses to split data. Each of these
methods is specified under the `split` subsection.

The following splitting methods are currently supported by Ludwig:

### Random Split

By default, Ludwig will randomly split the data into train, validation, and test
sets according to split probabilities, which by default are: `[0.7, 0.1, 0.2]`.

However, you can specify different splitting probabilities if you'd like. For
instance, if you want your dataset to be split according to a *60% train*,
*15% validation*, and *25% test* regime, you would use this config:

```yaml
preprocessing:
    split: 
        type: random
        probabilities: [0.6, 0.15, 0.25]
```

### Fixed Split

For users with pre-defined split that you want to use across experiments, Ludwig
supports fixed dataset splitting.

Provide an additional column in your data called `split` with the following
values for each split you want to include in your training/validation/test
subset.

- `0`: train
- `1`: validation
- `2`: test

!!! note
    
    Your dataset must contain a train split while the validation and test splits
    are encouraged, but technically optional.

The following config is an example that would perform fixed splitting using a
column named `split`:

```yaml
preprocessing:
    split:
        type: fixed
        column: split
```

### Stratified Split

Sometimes you may want to split your data according to a particular column's
distribution to maintain the same representation of this distribution across all
your data subsets.

In order to perform stratified splitting, you specify the name of the column you
want to perform stratified splitting on and the split probabilities. For
example:

```yaml
preprocessing:
    split:
        type: stratify
        column: color
        probabilities: [0.7, 0.1, 0.2]
```

This helps ensure that the distribution of the values of the `color` feature are
roughly the same across data subsets.

!!! note

    This split method is only supported with a local Pandas backend. We are
    actively working on including support for other data sources like Dask.

### Datetime Split

Another common use case is splitting a column according to a datetime column
where you may want to have the data split in a temporal order.

This is useful for situations like
[backtesting](https://en.wikipedia.org/wiki/Backtesting) where a user wants to
make sure that a model trained on historical data would have performed well on
unseen future data.

If we were to use a uniformly random split strategy in these cases, then the
model may not generalize well if the data distribution is subject to change over
time. Splitting the training from the test data along the time dimension is one
way to avoid this false sense of confidence, by showing how well the model
should do on unseen data from the future.

For datetime-based splitting, we order the data by date (ascending) and then
split according to the `split_probabilties`. For example, if
`split_probabilities: [0.7, 0.1, 0.2]`, then the earliest *70%* of the data will
be used for training, the middle *10%* used for validation, and the last *20%*
used for testing.

The following config shows how to specify this type of splitting using a
datetime column named `created_ts`:

```yaml
preprocessing:
    split:
        type: datetime
        column: created_ts
        probabilities: [0.7, 0.1, 0.2]
```

### Hash Split

Hash splitting deterministically assigns each row to a split based on a hash
of a provided "key" column. This is a useful alternative to random splitting when
such a key is available for a couple of reasons:

- *To prevent data leakage.* For example, imagine you are predicting which products a user is likely to buy. If a user
appears in both the train and test splits, then it may appear that your model is generalizing better than it actually is. In these cases,
hashing on the user ID column will ensure that every sample for a user is assigned to the same split.
- *To ensure consistent assignment of rows to splits as the underlying dataset evolves over time.* 
Though random splitting is determinstic between runs due to the use a random seed, if the underlying
dataset changes (e.g., new rows are added over time), then rows may move into different splits. Hashing on a primary
key will ensure that all existing rows retain their original splits as new rows are added over time.

```yaml
preprocessing:
    split: 
        type: hash
        column: user_id
        probabilities: [0.6, 0.15, 0.25]
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

!!! warning
    
    Dataset balancing is **only supported for binary output features** currently.
    We are working to add category support in a future release.

!!! note
    
    Specifying both oversampling and undersampling parameters simultaneously is
    not supported.

### Sample Ratio

Sometimes users may want to train on a sample of their input training data
(maybe there's too much, and we only need 20%). In order to achieve this, a user
can specify a `sample_ratio` to specify the ratio of the dataset to use for
training.

By default, the sample ratio is 1.0, so if not specified, all the data will be
used for training. For example, if you only want to use 30% of my input data,
you could specify a config like this:

```yaml
preprocessing:
    sample_ratio: 0.3
```

## Feature-specific preprocessing

To configure feature-specific preprocessing, please check
[datatype-specific documentation](../features/supported_data_types).
