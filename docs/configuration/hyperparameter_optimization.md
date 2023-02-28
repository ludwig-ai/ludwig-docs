{% from './macros/includes.md' import render_fields, render_yaml %}

The `hyperopt` section of the Ludwig configuration defines what metrics to optimize for, which parameters to optimize,
search strategy and execution strategy.

{% set hyperopt = get_hyperopt_schema() %}
{{ render_yaml(hyperopt, parent="hyperopt") }}

# Hyperopt configuration parameters

{{ render_fields(schema_class_to_fields(hyperopt)) }}

# Defining hyperparameter search spaces

In the `parameters` section, hyperparameters are dot( `.`) separate names. The parts of the hyperparameter names separated by `.` are references to nested sections in the Ludwig configuration.
For instance, to reference the `learning_rate`, in the `trainer` section one would use the name `trainer.learning_rate`.
If the parameter to reference is inside an input or output feature, the name of that feature will be used as starting point.
For instance, for referencing the `cell_type` of the `encoder` for the `title` feature, use the name `title.encoder.cell_type`.

## Numeric Hyperparameters

- `space`: Use [Ray Tune's Search Space](https://docs.ray.io/en/latest/tune/api_docs/search_space.html) types, e.g., `uniform`, `quniform`, `loguniform`, `choice`, etc.  Refer the cited page for details.

For numeric `spaces`, these define the range where the value is generated

- `lower`: the minimum value the parameter can have
- `upper`: the maximum value the parameter can have
- `q`: quantization number, used in `spaces` such as `quniform`, `qloguniform`, `qrandn`, `qrandint`, `qlograndint`
- `base`: defines the base of the log for `loguniform`, `qloguniform`, `lograndint` and `qlograndint`

!!! note

    Depending on the specific numeric `space`, the `upper` parameter may be inclusive or excluse.  Refer to the [Ray Tune documentation](https://docs.ray.io/en/latest/tune/api_docs/search_space.html#tune-uniform) for the specific distribution for details.

**Float example**: Uniform floating point random values (in log space) between 0.001 and 0.1

```yaml
trainer.learning_rate:
  space: loguniform
  lower: 0.001
  upper: 0.1
```

**Integer example**: Uniform random integer values 1, 2, 3

```yaml
combiner.num_fc_layers:
  space: randint
  lower: 1
  upper: 4
```

**Quantized Example**: Uniform random floating point values such a 0, 0.1, 0.2, ..., 0.9

```yaml
my_output_feature.decoder.dropout:
  space: quniform
  lower: 0
  upper: 1
  q: 0.1
```

## Categorical Hyperparameters

- `space`: Use `choice`.
- `categories`: a list of possible values. The type of each value of the list is general, i.e., they could be strings,
integers, floats and anything else, even entire dictionaries.  The values will be a uniform random selection.

Example:

```yaml
title.encoder.cell_type:
  space: choice
  categories: [rnn, gru, lstm]
```

## Hyperparameters in a Grid

For `space`: `grid_search`

- `values`: is a list of values to use in creating the grid search space.  The type of each value of the list is general, i.e., they could be strings,
integers, floats and anything else, even entire dictionaries.

Example:

```yaml
title.encoder.cell_type:
  space: grid_search
  values: [rnn, gru, lstm]
```

## More comprehensive example

```yaml
hyperopt:
  parameters:
    trainer.learning_rate:
      space: loguniform
      lower: 0.001
      upper: 0.1
    combiner.num_fc_layers:
      space: randint
      lower: 2
      upper: 6
    title.encoder.cell_type:
      space: grid_search
      values: ["rnn", "gru"]
    title.encoder.bidirectional:
      space: choice
      categories: [True, False]
    title.encoder.fc_layers:
      space: choice
      categories:
        - [{"output_size": 512}, {"output_size": 256}]
        - [{"output_size": 512}]
        - [{"output_size": 256}]
```

### Default Hyperopt Parameters

In addition to defining hyperopt parameters for individual input or output features (like the `title` feature
in the example above), default parameters can be specified for entire feature types (for example, the encoder
to use for all text features in your dataset). Read more about default hyperopt parameters [here](../user_guide/hyperopt.md#default-hyperopt-parameters).

### Nested Ludwig Config Parameters

Ludwig also allows partial or full Ludwig configs to be sampled from the hyperopt search space.
Read more about nested Ludwig config parameters [here](../user_guide/hyperopt.md#nested-ludwig-config-parameters).

# Search Algorithm

Ray Tune supports its own collection of [search algorithms](https://docs.ray.io/en/master/tune/api_docs/suggestion.html), specified by the `search_alg` section of the hyperopt config:

```yaml
search_alg:
  type: variant_generator
```

You can find the full list of supported search algorithm names in Ray Tune's [create_searcher](https://github.com/ray-project/ray/blob/master/python/ray/tune/suggest/__init__.py) function. Please note these algorithms require installation of additional packages.  As of this version of Ludwig, Ludwig installs the packages for the search algorithm `hyperopt`.  For all other search algorithms, the user is expected to install the required packages.

# Executor

## Ray Tune Executor

{% set executor = get_hyperopt_executor_schema() %}

{{ render_yaml(executor, parent="executor") }}

The `ray` executor is used to enable [Ray Tune](https://docs.ray.io/en/master/tune/index.html) for both local and distributed hyperopt across a cluster of machines.

**Parameters:**

{{ render_fields(schema_class_to_fields(executor)) }}

## Scheduler

Ray Tune also allows you to specify a [scheduler](https://docs.ray.io/en/master/tune/api_docs/schedulers.html) to support features like early stopping and other population-based strategies that may pause and resume trials during trainer. Ludwig exposes the complete scheduler API in the `scheduler` section of the `executor` config.

You can find the full list of supported schedulers in Ray Tune's [create_scheduler](https://github.com/ray-project/ray/blob/master/python/ray/tune/schedulers/__init__.py) function.

Example:

```yaml
executor:
  type: ray
  cpu_resources_per_trial: 2
  gpu_resources_per_trial: 1
  kubernetes_namespace: ray
  time_budget_s: 7200
  scheduler:
    type: async_hyperband
    time_attr: training_iteration
    reduction_factor: 4
```

**Running Ray Executor:**

See the section on [Running Ludwig with Ray](../../user_guide/distributed_training#ray) for guidance on setting up your
Ray cluster.

# Full hyperparameter optimization example

Following is a full example of a Ludwig configuration with hyperparameter optimization.

Example YAML:

```yaml
input_features:
  -
    name: title
    type: text
    encoder:
        type: rnn
        cell_type: lstm
        num_layers: 2
combiner:
  type: concat
  num_fc_layers: 1
output_features:
  -
    name: class
    type: category
defaults:
  text:
    preprocessing:
      word_vocab_size: 10000
training:
  learning_rate: 0.001
  optimizer:
    type: adam
hyperopt:
  goal: maximize
  output_feature: class
  metric: accuracy
  split: validation
  parameters:
    trainer.learning_rate:
      space: loguniform
      lower: 0.0001
      upper: 0.1
    trainer.optimizer.type:
      space: choice
      categories: [sgd, adam, adagrad]
    preprocessing.text.word_vocab_size:
      space: qrandint
      lower: 700
      upper: 1200
      q: 5
    combiner.num_fc_layers:
      space: randint
      lower: 1
      upper: 5
    title.encoder.cell_type:
      space: choice
      values: [rnn, gru, lstm]
  search_alg:
    type: random
  executor:
    type: ray
    num_samples: 12
```

Example CLI command:

```
ludwig hyperopt --dataset reuters-allcats.csv --config_str "{input_features: [{name: title, type: text, encoder: {type: rnn, cell_type: lstm, num_layers: 2}}], output_features: [{name: class, type: category}], training: {learning_rate: 0.001}, hyperopt: {goal: maximize, output_feature: class, metric: accuracy, split: validation, parameters: {trainer.learning_rate: {space: loguniform, lower: 0.0001, upper: 0.1}, title.encoder.cell_type: {space: choice, categories: [rnn, gru, lstm]}}, search_alg: {type: variant_generator},executor: {type: ray, num_samples: 10}}}"
```
