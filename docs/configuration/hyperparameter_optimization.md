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
- `search_alg` section specifies the algorittm to sample the defined `parameters` space. Candidate algorithms are those found in [Ray Tune's Search Algorithms](https://docs.ray.io/en/latest/tune/api_docs/suggestion.html).  
- `executor` section specifies how to execute the hyperparameter optimization. The execution could happen locally in a serial manner or in parallel across multiple workers and with GPUs as well if available.  The `executor` section includes spefication for work scheduling and the number of samples to generate.

# Defining hyperparameter search spaces

In the `parameters` section, `.` is used to reference an parameter nested inside a section of the configuration.
For instance, to reference the `learning_rate`, one would have to use the name `trainer.learning_rate`.
If the parameter to reference is inside an input or output feature, the name of that feature will be be used as starting point.
For instance, for referencing the `cell_type` of the `text` feature, use the name `text.cell_type`.

## Number ranges

- `space`: Use [Ray Tune's Search Space](https://docs.ray.io/en/latest/tune/api_docs/search_space.html) types, e.g., `uniform`, `quniform`, `loguniform`, `choice`, etc.  Refer the cited page for details.

For numeric `spaces`, these define the space
- `lower`: the minimum value the parameter can have
- `upper`: the maximum value the parameter can have
- `q`: quantization number, used in `spaces` such as `quniform`, `qloguniform`, `qrandn`, `qrandint`, `qlograndint`
- `base`: defines the base of the log for `loguniform`, `qloguniform`, `lograndint` and `qlograndint`

Float example:

```yaml
trainer.learning_rate:
  space: loguniform
  lower: 0.001
  uppper: 0.1
```

Integer example:

```yaml
combiner.num_fc_layers:
  space: randint
  lower: 1
  upper: 4
```

Quantized Example:

```yaml
my_output_feature.dropout:
  space: quniform
  lower: 0
  upper: 1
  q: 0.1
```


## Categorical selection

- `space`: Use `choice`.
- `categories`: a list of possible values. The type of each value of the list is not important (they could be strings,
integers, floats and anything else, even entire dictionaries).  The values will be a uniform random selection.

Example:

```yaml
text.cell_type:
  categories: [rnn, gru, lstm]
  space: choice
```


## Grid Space

For `grid_search` space
- `values`: is a list of values to use in creating a grid search space.

Example:

```yaml
text.cell_type:
  space: grid_search
  values: [rnn, gru, lstm]
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
ludwig hyperopt --dataset reuters-allcats.csv --config_str "{input_features: [{name: text, type: text, encoder: rnn, cell_type: lstm, num_layers: 2}], output_features: [{name: class, type: category}], training: {learning_rate: 0.001}, hyperopt: {goal: maximize, output_feature: class, metric: accuracy, split: validation, parameters: {trainer.learning_rate: {type: float, low: 0.0001, high: 0.1, steps: 4, scale: log}, text.cell_type: {type: category, values: [rnn, gru, lstm]}}, sampler: {type: grid}, executor: {type: serial}}}"
```
