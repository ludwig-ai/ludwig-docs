The `hyperopt` section of the Ludwig configuration defines what metrics to optimize for, which parameters to optimize,
search strategy and execution strategy.

```yaml
hyperopt:
  goal: minimize
  output_feature: combined
  metric: loss
  split: validation
  parameters:
    title.encoder.cell_type: ... # title is a text feature type
    title.encoder.num_layers: ... 
    combiner.num_fc_layers: ...
    section.encoder.embedding_size: ...
    preprocessing.text.vocab_size: ...
    trainer.learning_rate: ...
    trainer.optimizer.type: ...
    ...
  search_alg:
    type: variant_generator  # random, hyperopt, bohb, ...
    # search_alg parameters...
  executor:
    type: ray
    num_samples: ...
    scheduler:
      type: fifo  # hb_bohb, asynchyperband, ...
      # scheduler parameters...
```

# Hyperopt configuration parameters

- `goal` which indicates if to minimize or maximize a metric or a loss of any of the output features on any of the dataset splits. Available values are: `minimize` (default) or `maximize`.
- `output_feature` is a `str` containing the name of the output feature that we want to optimize the metric or loss of. Available values are `combined` (default) or the name of any output feature provided in the configuration. `combined` is a special output feature that allows to optimize for the aggregated loss and metrics of all output features.
- `metric` is the metric that we want to optimize for. The default one is `loss`, but depending on the type of the feature defined in `output_feature`, different metrics and losses are available. Check the metrics section of the specific output feature type to figure out what metrics are available to use.
- `split` is the split of data that we want to compute our metric on. By default it is the `validation` split, but you have the flexibility to specify also `train` or `test` splits.
- `parameters` section consists of a set of hyperparameters to optimize. They are provided as keys (the names of the parameters) and values associated with them (that define the search space). The values vary depending on the type of the hyperparameter. Syntax for this section is based on [Ray Tune's Search Space parameters](https://docs.ray.io/en/latest/tune/api_docs/search_space.html).
- `search_alg` section specifies the algorithm to sample the defined `parameters` space. Candidate algorithms are those found in [Ray Tune's Search Algorithms](https://docs.ray.io/en/latest/tune/api_docs/suggestion.html).
- `executor` section specifies how to execute the hyperparameter optimization. The execution could happen locally in a serial manner or in parallel across multiple workers and with GPUs as well if available.  The `executor` section includes specification for work scheduling and the number of samples to generate.

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

The `ray` executor is used to enable [Ray Tune](https://docs.ray.io/en/master/tune/index.html) for both local and distributed hyperopt across a cluster of machines.

**Parameters:**

- `num_samples`: This parameter, along with the `space` specifications in the `parameters` section, controls how many trials are generated (default: 1).

!!! note

    * If all the hyperparameters in the `parameters` section have non-`grid_search` specifications (e.g., `uniform`, `randn`, `choice`, etc.), then the number of trials will be `num_samples`.
    * If all the hyperparameters have `grid_search`, then the number of trials will be the product of the number of values specified for each hyperparameter.  In this case, `num_samples` should be set to 1.  For example, if there are three `grid_search` hyperparameters, with 2, 4 and 4 values, respectively.  The number of trials will be 2 X 4 X 4 = 32, where each trial is a unique combination of the three `grid_search` hyperparameter values.
    * If there is a mixture of `grid_search` and non-`grid_search` spaces, the number of trials will be product of the number of values specified for each `grid_search` hyperpameter multiplied by the value of `num_samples`.  To illustrate this point, we take the three `grid_search` hyperparameters described in the preceding bullet item and add 2 hyperparameters with `uniform` and `randint` spaces.  With `num_samples = 10`, for each unique combination of values from the `grid_search` hyperparameters, 10 trials will be generated with random values selected for the `uniform` and `randint` hyperparameters.  This will lead to a total of 32 X 10 = 320 trials.

- `cpu_resources_per_trial`: The number of CPU cores allocated to each trial (default: 1).
- `gpu_resources_per_trial`: The number of GPU devices allocated to each trial (default: 0).
- `kubernetes_namespace`: When running on Kubernetes, provide the namespace of the Ray cluster to sync results between pods. See the [Ray docs](https://docs.ray.io/en/master/_modules/ray/tune/integration/kubernetes.html) for more info.
- `time_budget_s`: The number of seconds for the entire hyperopt run.
- `max_concurrent_trials`: The maximum number of trials to train concurrently. Defaults to `auto` if not specified.
    - If you're using a Ray backend, `auto` will infer the max_concurrent_trials that can be set given your cluster configuration to prevent trials from stalling.
    - If you're using a Local or Horovod backend, `auto` will set max_concurrent_trials to None.

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
