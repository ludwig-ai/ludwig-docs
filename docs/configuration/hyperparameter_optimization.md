The `hyperopt` section of the Ludwig configuration contains what metrics to optimize for, which parameters to optimize,
sampler, and execution strategy.

```yaml
hyperopt:
  goal: minimize
  output_feature: combined
  metric: loss
  split: validation
  parameters:
    text.cell_type: ...
    text.num_layers: ...
    combiner.num_fc_layers: ...
    section.embedding_size: ...
    preprocessing.text.vocab_size: ...
    trainer.learning_rate: ...
    trainer.optimizer.type: ...
    ...
  sampler:
    type: grid  # random, ...
    # sampler parameters...
  executor:
    type: serial  # parallel, ...
    # executor parameters...
```

# Hyperopt configuration parameters

- `goal` which indicates if to minimize or maximize a metric or a loss of any of the output features on any of the dataset splits. Available values are: `minimize` (default) or `maximize`.
- `output_feature` is a `str` containing the name of the output feature that we want to optimize the metric or loss of. Available values are `combined` (default) or the name of any output feature provided in the configuration. `combined` is a special output feature that allows to optimize for the aggregated loss and metrics of all output features.
- `metric` is the metric that we want to optimize for. The default one is `loss`, but depending on the type of the feature defined in `output_feature`, different metrics and losses are available. Check the metrics section of the specific output feature type to figure out what metrics are available to use.
- `split` is the split of data that we want to compute our metric on. By default it is the `validation` split, but you have the flexibility to specify also `train` or `test` splits.
- `parameters` section consists of a set of hyperparameters to optimize. They are provided as keys (the names of the parameters) and values associated with them (that define the search space). The values vary depending on the type of the hyperparameter. Types can be `float`, `int` and `category`.
- `sampler` section contains the sampler type to be used for sampling hyper-paramters values and its configuration. Currently available sampler types are `grid` and `random`. The sampler configuration parameters modify the sampler behavior, for instance for `random` you can set how many random samples to draw.
- `executor` section specifies how to execute the hyperparameter optimization. The execution could happen locally in a serial manner or in parallel across multiple workers and with GPUs as well if available.

# Defining hyperparameter search spaces

In the `parameters` section, `.` is used to reference an parameter nested inside a section of the configuration.
For instance, to reference the `learning_rate`, one would have to use the name `trainer.learning_rate`.
If the parameter to reference is inside an input or output feature, the name of that feature will be be used as starting point.
For instance, for referencing the `cell_type` of the `text` feature, use the name `text.cell_type`.

## Numerical ranges

- `space`: Use `linear` or `log`.
- `range.low`: the minimum value the parameter can have
- `range.high`: the maximum value the parameter can have
- `steps`: (optional) number of steps to break down a range.

For instance `range: (0.0, 1.0), steps: 3` would yield `[0.0, 0.5, 1.0]` as potential values to sample from, while if
`steps` is not specified, the full range between `0.0` and `1.0` will be used.

Float example:

```yaml
trainer.learning_rate:
  space: linear
  range:
    low: 0.001
    high: 0.1
  steps: 4
```

Integer example:

```yaml
combiner.num_fc_layers:
  space: linear
  range:
    low: 1
    high: 4
```

## Categorical selection

- `space`: Use `choice`.
- `categories`: a list of possible values. The type of each value of the list is not important (they could be strings,
integers, floats and anything else, even entire dictionaries).

Example:

```yaml
text.cell_type:
  categories: [rnn, gru, lstm]
  space: choice
```

# Sampler

## Grid sampler

The `grid` sampler creates a search space by exhaustively selecting all elements from the outer product of all possible
combinations of hyperparameter values provided in the `parameters` section.

To use `grid` sampling with `float` parameters, it is required to specify the number of `steps`.

Example:

```yaml
sampler:
  type: grid
```

## Random sampler

The `random` sampler samples hyperparameter values randomly from the parameters search space.
`num_samples` (default: `10`) can be specified in the `sampler` section.

Example:

```yaml
sampler:
  type: random
  num_samples: 10
```

## PySOT sampler

The `pysot` sampler uses the [pySOT](https://arxiv.org/pdf/1908.00420.pdf) package for asynchronous surrogate
optimization. This package implements many popular methods from Bayesian optimization and surrogate optimization.[^1]

[^1]:
    By default, pySOT uses the Stochastic RBF (SRBF) method by [Regis and Shoemaker](https://pubsonline.informs.org/doi/10.1287/ijoc.1060.0182).
    SRBF starts by evaluating a symmetric Latin hypercube design of size `2 * d + 1`, where d is the number of
    hyperparameters that are optimized. When these points have been evaluated, SRBF fits a radial basis function surrogate and uses this surrogate together with an acquisition function to select the next sample(s). More details are available on the GitHub page: <https://github.com/dme65/pySOT>.

!!! tip

    We recommend using at least `10 * d` total samples to allow the algorithm to converge.

Example:

```yaml
sampler:
  type: pysot
  num_samples: 10
```

## Ray Tune sampler

The `ray` sampler is used in conjunction with the `ray` executor to enable [Ray Tune](https://docs.ray.io/en/master/tune/index.html) for distributed hyperopt across a cluster of machines.

Ray Tune supports its own collection of [search algorithms](https://docs.ray.io/en/master/tune/api_docs/suggestion.html), specified by the `search_alg` section of the sampler config:

```yaml
sampler:
  type: ray
  search_alg:
    type: ax
```

You can find the full list of supported search algorithm names in Ray Tune's [create_searcher](https://github.com/ray-project/ray/blob/master/python/ray/tune/suggest/__init__.py) function.

Ray Tune also allows you to specify a [scheduler](https://docs.ray.io/en/master/tune/api_docs/schedulers.html) to support features like early stopping and other population-based strategies that may pause and resume trials during trainer. Ludwig exposes the complete scheduler API in the `scheduler` section of the config:

```yaml
sampler:
  type: ray
  search_alg:
    random_state_seed: 42
    type: hyperopt
  scheduler:
    type: async_hyperband
    time_attr: training_iteration
    reduction_factor: 4
```

You can find the full list of supported schedulers in Ray Tune's [create_scheduler](https://github.com/ray-project/ray/blob/master/python/ray/tune/schedulers/__init__.py) function.

Other config options, including `parameters`, `num_samples`, and `goal` work the same for Ray Tune as they do for other sampling strategies in Ludwig. The `parameters` will be converted from the Ludwig format into a Ray Tune [search space](https://docs.ray.io/en/master/tune/api_docs/search_space.html). However, note that the `space` field of the Ludwig config should conform to the Ray Tune [distribution names](https://docs.ray.io/en/master/tune/api_docs/search_space.html#random-distributions-api). For example:

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
    text.cell_type:
      space: grid_search
      values: ["rnn", "gru"]
    text.bidirectional:
      space: choice
      categories: [True, False]
    text.fc_layers:
      space: choice
      categories:
        - [{"output_size": 512}, {"output_size": 256}]
        - [{"output_size": 512}]
        - [{"output_size": 256}]
  goal: minimize
```

# Executor

## Serial Executor

The `serial` executor performs hyperparameter optimization locally in a serial manner, executing the elements in the set
of sampled parameters obtained by the selected sampler one at a time.

Example:

```yaml
executor:
  type: serial
```

## Ray Tune Executor

The `ray` executor is used in conjunction with the `ray` sampler to enable [Ray Tune](https://docs.ray.io/en/master/tune/index.html) for distributed hyperopt across a cluster of machines.

**Parameters:**

- `cpu_resources_per_trial`: The number of CPU cores allocated to each trial (default: 1).
- `gpu_resources_per_trial`: The number of GPU devices allocated to each trial (default: 0).
- `kubernetes_namespace`: When running on Kubernetes, provide the namespace of the Ray cluster to sync results between pods. See the [Ray docs](https://docs.ray.io/en/master/_modules/ray/tune/integration/kubernetes.html) for more info.
- `time_budget_s`: The number of seconds for the entire hyperopt run.

Example:

```yaml
executor:
  type: ray
  cpu_resources_per_trial: 2
  gpu_resources_per_trial: 1
  kubernetes_namespace: ray
  time_budget_s: 7200
```

**Running Ray Executor:**

See the section on [Running Ludwig with Ray](../../user_guide/distributed_training#ray) for guidance on setting up your
Ray cluster.

# Full hyperparameter optimization example

Example YAML:

```yaml
input_features:
  -
    name: text
    type: text
    encoder: rnn
    cell_type: lstm
    num_layers: 2
combiner:
  type: concat
  num_fc_layers: 1
output_features:
  -
    name: class
    type: category
preprocessing:
  text:
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
      type: float
      low: 0.0001
      high: 0.1
      steps: 4
      scale: log
    trainer.optimizer.type:
      type: category
      values: [sgd, adam, adagrad]
    preprocessing.text.word_vocab_size:
      type: int
      low: 700
      high: 1200
      steps: 5
    combiner.num_fc_layers:
      type: int
      low: 1
      high: 5
    text.cell_type:
      type: category
      values: [rnn, gru, lstm]
  sampler:
    type: random
    num_samples: 12
  executor:
    type: parallel
    num_workers: 4
```

Example CLI command:

```
ludwig hyperopt --dataset reuters-allcats.csv --config "{input_features: [{name: text, type: text, encoder: rnn, cell_type: lstm, num_layers: 2}], output_features: [{name: class, type: category}], training: {learning_rate: 0.001}, hyperopt: {goal: maximize, output_feature: class, metric: accuracy, split: validation, parameters: {trainer.learning_rate: {type: float, low: 0.0001, high: 0.1, steps: 4, scale: log}, text.cell_type: {type: category, values: [rnn, gru, lstm]}}, sampler: {type: grid}, executor: {type: serial}}}"
```
