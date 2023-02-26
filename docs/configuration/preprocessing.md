{% from './macros/includes.md' import render_fields, render_yaml %}

The top-level `preprocessing` section specifies dataset splitting (train, validation, test), sample ratio (undersampling the minority class or oversampling the majority class) and dataset balancing.

```yaml
preprocessing:
    sample_ratio: 1.0
    oversample_minority: 0.5
    split: 
        type: random
        probabilities: [0.7, 0.2, 0.1]
```

# Dataset Splitting

Data splitting is an important aspect of machine learning to train and evaluate
machine learning models.

Ludwig supports splitting data into train, validation, and test sets, and this
is configured using the top-level `preprocessing` section of the Ludwig config.

There is no set guideline or metric for how the data should be split; it may
depend on the size of your data or the type of problem.

There are a few different methods that Ludwig uses to split data. Each of these
methods is specified under the `split` subsection.

The following splitting methods are currently supported by Ludwig:

## Random Split

By default, Ludwig will randomly split the data into train, validation, and test
sets according to split probabilities, which by default are: `[0.7, 0.1, 0.2]`.

{% set random_split = get_split_schema("random") %}
{{ render_yaml(random_split, parent="split") }}

However, you can specify different splitting probabilities if you'd like by setting
the probabilities for each of the 3 datasets (so that they sum up to 1)

## Fixed Split

If you have a column denoting pre-defined splits (train, validation and test) that you want to use across experiments, Ludwig supports using fixed dataset splits.

The following config is an example that would perform fixed splitting using a column named `split` in the dataset:

{% set fixed_split = get_split_schema("fixed") %}
{{ render_yaml(fixed_split, parent="split") }}

Within the data itself, we would ensure that there is a column called `split` with the following
values for each row in the column based on the split we want to map that row to:

- `0`: train
- `1`: validation
- `2`: test

!!! note

    Your dataset must contain a train split. However, the validation and test splits
    are encouraged, but optional.

## Stratified Split

Sometimes you may want to split your data according to a particular column's distribution to maintain the same representation of this distribution across all your data subsets. This may
be particularly useful when you have more than one class and your dataset is imbalanced.

In order to perform stratified splitting, you specify the name of the column you want to perform stratified splitting on and the split probabilities.

The following config is an example that would perform stratified splitting for a column `color`:

{% set stratify_split = get_split_schema("stratify", col_name="color") %}
{{ render_yaml(stratify_split, parent="split") }}

This helps ensure that the distribution of the values in `color` are
roughly the same across data subsets.

!!! note

    This split method is only supported with a local Pandas backend. We are
    actively working on including support for other data sources like Dask.

## Datetime Split

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

{% set datetime_split = get_split_schema("datetime", col_name="created_ts") %}
{{ render_yaml(datetime_split, parent="split") }}

## Hash Split

Hash splitting deterministically assigns each sample to a split based on a hash
of a provided "key" column. This is a useful alternative to random splitting when
such a key is available for a couple of reasons:

- **To prevent data leakage**: 
For example, imagine you are predicting which users are likely to churn in a given month. If a user
appears in both the train and test splits, then it may seem that your model is generalizing better than it actually is. In these cases,
hashing on the user ID column will ensure that every sample for a user is assigned to the same split.
- **To ensure consistent assignment of samples to splits as the underlying dataset evolves over time**: 
Though random splitting is determinstic between runs due to the use of a random seed, if the underlying
dataset changes (e.g., new samples are added over time), then samples may move into different splits. Hashing on a primary
key will ensure that all existing samples retain their original splits as new samples are added over time.

{% set hash_split = get_split_schema("hash", col_name="user_id") %}
{{ render_yaml(hash_split, parent="split") }}

# Data Balancing

Users working with imbalanced datasets can specify an oversampling or
undersampling parameter which will balance the data during preprocessing.

!!! warning

    Dataset balancing is **only supported for binary output features** currently.
    We are working to add category support in a future release.

!!! note

    Specifying both oversampling and undersampling parameters simultaneously is
    not supported.

## Oversampling

In this example, Ludwig will oversample the minority class to achieve a 50%
representation in the overall dataset.

```yaml
preprocessing:
    oversample_minority: 0.5
```

## Undersampling

In this example, Ludwig will undersample the majority class to achieve a 70%
representation in the overall dataset.

```yaml
preprocessing:
    undersample_majority: 0.7
```

# Sample Ratio

Sometimes users may want to train on a sample of their input training data (maybe 
there's too much, and we only need 20%, or we want to try out ideas on a smaller
subset of our data). In order to achieve this, a user can specify a `sample_ratio` 
to indicate the ratio of the dataset to use for training.

By default, the sample ratio is 1.0, so if not specified, all the data will be
used for training. For example, if you only want to use 30% of my input data,
you could specify a config like this:

```yaml
preprocessing:
    sample_ratio: 0.3
```

# Feature-specific preprocessing

To configure feature-specific preprocessing, please check
[datatype-specific documentation](../features/supported_data_types).
