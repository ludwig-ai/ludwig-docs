# Hyper-parameter optimization configuration

In order to perform hyper-parameter optimization, its configuration has to be provided inside the Ludwig configuration as a root key `hyperopt`.
Its configuration contains what metric to optimize, which parameters to optimize, which sampler to use, and how to execute the optimization.

The different parameters that could be defined in the `hyperopt` configuration are:

- `goal` which indicates if to minimize or maximize a metric or a loss of any of the output features on any of the dataset splits. Available values are: `minimize` (default) or `maximize`.
- `output_feature` is a `str` containing the name of the output feature that we want to optimize the metric or loss of. Available values are `combined` (default) or the name of any output feature provided in the configuration. `combined` is a special output feature that allows to optimize for the aggregated loss and metrics of all output features.
- `metric` is the metric that we want to optimize for. The default one is `loss`, but depending on the type of the feature defined in `output_feature`, different metrics and losses are available. Check the metrics section of the specific output feature type to figure out what metrics are available to use.
- `split` is the split of data that we want to compute our metric on. By default it is the `validation` split, but you have the flexibility to specify also `train` or `test` splits.
- `parameters` section consists of a set of hyper-parameters to optimize. They are provided as keys (the names of the parameters) and values associated with them (that define the search space). The values vary depending on the type of the hyper-parameter. Types can be `float`, `int` and `category`.
- `sampler` section contains the sampler type to be used for sampling hyper-paramters values and its configuration. Currently available sampler types are `grid` and `random`. The sampler configuration parameters modify the sampler behavior, for instance for `random` you can set how many random samples to draw.
- `executor` section specifies how to execute the hyper-parameter optimization. The execution could happen locally in a serial manner or in parallel across multiple workers and with GPUs as well if available.

Example:

```yaml
hyperopt:
  goal: minimize
  output_feature: combined
  metric: loss
  split: validation
  parameters:
    utterance.cell_type: ...
    utterance.num_layers: ...
    combiner.num_fc_layers: ...
    section.embedding_size: ...
    preprocessing.text.vocab_size: ...
    training.learning_rate: ...
    training.optimizer.type: ...
    ...
  sampler:
    type: grid  # random, ...
    # sampler parameters...
  executor:
    type: serial  # parallel, ...
    # executor parameters...
```

In the `parameters` section, `.` is used to reference an parameter nested inside a section of the configuration.
For instance, to reference the `learning_rate`, one would have to use the name `training.learning_rate`.
If the parameter to reference is inside an input or output feature, the name of that feature will be be used as starting point.
For instance, for referencing the `cell_type` of the `utterance` feature, use the name `utterance.cell_type`.

# Hyper-parameters

## Float parameters

For a `float` value, the parameters to specify are:

- `low`: the minimum value the parameter can have
- `high`: the maximum value the parameter can have
- `scale`: `linear` (default) or `log`
- `steps`: OPTIONAL number of steps.

For instance `range: (0.0, 1.0), steps: 3` would yield `[0.0, 0.5, 1.0]` as potential values to sample from, while if `steps` is not specified, the full range between `0.0` and `1.0` will be used.

Example:

```yaml
training.learning_rate:
  type: real
  low: 0.001
  high: 0.1
  steps: 4
  scale: linear
```

## Int parameters

For an `int` value, the parameters to specify are:

- `low`: the minimum value the parameter can have
- `high`: the maximum value the parameter can have
- `steps`: OPTIONAL number of steps.

For instance `range: (0, 10), steps: 3` would yield `[0, 5, 10]` for the search, while if `steps` is not specified, `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]` will be used.

Example:

```yaml
combiner.num_fc_layers:
  type: int
  low: 1
  high: 4
```

## Category parameters

For a `category` value, the parameters to specify are:

- `values`: a list of possible values. The type of each value of the list is not important (they could be strings, integers, floats and anything else, even entire dictionaries).

Example:

```yaml
utterance.cell_type:
  type: category
  values: [rnn, gru, lstm]
```

# Sampler

## Grid sampler

The `grid` sampler creates a search space by exhaustively selecting all elements from the outer product of all possible values of the hyper-parameters provided in the `parameters` section.
For `float` parameters, it is required to specify the number of `steps`.

Example:

```yaml
sampler:
  type: grid
```

## Random sampler

The `random` sampler samples hyper-parameter values randomly from the parameters search space.
`num_samples` (default: `10`) can be specified in the `sampler` section.

Example:

```yaml
sampler:
  type: random
  num_samples: 10
```

## PySOT sampler

The `pysot` sampler uses the [pySOT](https://arxiv.org/pdf/1908.00420.pdf) package for asynchronous surrogate optimization.
This package implements many popular methods from Bayesian optimization and surrogate optimization.
By default, pySOT uses the Stochastic RBF (SRBF) method by [Regis and Shoemaker](https://pubsonline.informs.org/doi/10.1287/ijoc.1060.0182).
SRBF starts by evaluating a symmetric Latin hypercube design of size `2 * d + 1`, where d is the number of hyperparameters that are optimized.
When these points have been evaluated, SRBF fits a radial basis function surrogate and uses this surrogate together with an acquisition function to select the next sample(s).
We recommend using at least `10 * d` total samples to allow the algorithm to converge.

More details are available on the GitHub page: https://github.com/dme65/pySOT.

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

Ray Tune also allows you to specify a [scheduler](https://docs.ray.io/en/master/tune/api_docs/schedulers.html) to support features like early stopping and other population-based strategies that may pause and resume trials during training. Ludwig exposes the complete scheduler API in the `scheduler` section of the config:

```yaml
sampler:
  type: ray
  search_alg:
    type: bohb
  scheduler:
    type: hb_bohb
    time_attr: training_iteration
    reduction_factor: 4
```

You can find the full list of supported schedulers in Ray Tune's [create_scheduler](https://github.com/ray-project/ray/blob/master/python/ray/tune/schedulers/__init__.py) function.

Other config options, including `parameters`, `num_samples`, and `goal` work the same for Ray Tune as they do for other sampling strategies in Ludwig. The `parameters` will be converted from the Ludwig format into a Ray Tune [search space](https://docs.ray.io/en/master/tune/api_docs/search_space.html). However, note that the `space` field of the Ludwig config should conform to the Ray Tune [distribution names](https://docs.ray.io/en/master/tune/api_docs/search_space.html#random-distributions-api). For example:

```yaml
hyperopt:
  parameters:
    training.learning_rate:
      space: loguniform
      lower: 0.001
      upper: 0.1
    combiner.num_fc_layers:
      space: randint
      lower: 2
      upper: 6
    utterance.cell_type:
      space: grid_search
      values: ["rnn", "gru"]
    utterance.bidirectional:
      space: choice
      categories: [True, False]
    utterance.fc_layers:
      space: choice
      categories:
        - [{"fc_size": 512}, {"fc_size": 256}]
        - [{"fc_size": 512}]
        - [{"fc_size": 256}]
  goal: minimize
```

# Executor

## Serial Executor

The `serial`executor performs hyper-parameter optimization locally in a serial manner, executing the elements in the set of sampled parameters obtained by the selected sampler one at a time.

Example:

```yaml
executor:
  type: serial
```

## Parallel Executor

The `parallel` executor performs hyper-parameter optimization in parallel, executing the elements in the set of sampled parameters obtained by the selected sampler at the same time.
The maximum numer of parallel workers that train and evaluate models is defined by the parameter `num_workers` (default: `2`).

In case of training with GPUs, the `gpus` argument provided to the command line interface contains the list of GPUs to use, while if no `gpus` parameter is provided, all available GPUs will be used.
The `gpu_fraction` argument can be provided as well, but it gets modified according to the `num_workers` to execute tasks parallely.
For example, if `num_workers: 4` and 2 GPUs are available, if the provided `gpu_fraction` is above `0.5`, if will be replaced by `0.5`.
An `epsilon` (default: `0.01`) parameter is also provided to allow for additional free GPU memory: the GPU franction to use is defined as `(#gpus / #workers) - epsilon`.

Example:

```yaml
executor:
  type: parallel
  num_workers: 2
  epsilon: 0.01
```

## Ray Tune Executor

The `ray` executor is used in conjunction with the `ray` sampler to enable [Ray Tune](https://docs.ray.io/en/master/tune/index.html) for distributed hyperopt across a cluster of machines.

**Parameters:**

- `cpu_resources_per_trial`: The number of CPU cores allocated to each trial (default: 1).
- `gpu_resources_per_trial`: The number of GPU devices allocated to each trial (default: 0).
- `kubernetes_namespace`: When running on Kubernetes, provide the namespace of the Ray cluster to sync results between pods. See the [Ray docs](https://docs.ray.io/en/master/_modules/ray/tune/integration/kubernetes.html) for more info.

Example:

```yaml
executor:
  type: ray
  cpu_resources_per_trial: 2
  gpu_resources_per_trial: 1
  kubernetes_namespace: ray
```

**Running Ray Executor:**

See the section on [Running Ludwig with Ray](https://ludwig-ai.github.io/ludwig-docs/user_guide/#running-ludwig-with-ray) for guidance on setting up your Ray cluster.

## Fiber Executor

[Fiber](https://github.com/uber/fiber) is a Python distributed computing library for modern computer clusters.
The `fiber` executor performs hyper-parameter optimization in parallel on a computer cluster so that massive parallelism can be achieved.
Check [this](https://uber.github.io/fiber/platforms/) for supported cluster types.

Fiber Executor requires `fiber` to be installed:

```bash
pip install fiber
```

**Parameters:**

- `num_workers`: The number of parallel workers that is used to train and evaluate models. The default value is `2`.
- `num_cpus_per_worker`: How many CPU cores are allocated per worker.
- `num_gpus_per_worker`: How many GPUs are allocated per worker.
- `fiber_backend`: Fiber backend to use. This needs to be set if you want to run hyper-parameter optimization on a cluster. The default value is `local`. Available values are `local`, `kubernetes`, `docker`. Check [Fiber's documentation](https://uber.github.io/fiber/platforms/) for details on the supported platforms.

Example:

```yaml
executor:
  type: fiber
  num_workers: 10
  fiber_backend: kubernetes
  num_cpus_per_worker: 2
  num_gpus_per_worker: 1
```

**Running Fiber Executor:**

Fiber runs on a computer cluster and uses Docker to encapsulate all the code and dependencies.
To run a hyper-parameter search powered by Fiber, you have to create a Docker file to encapsulate your code and dependencies.

Example Dockerfile:

```dockerfile
FROM tensorflow/tensorflow:1.15.2-gpu-py3

RUN apt-get -y update && apt-get -y install git libsndfile1

RUN git clone --depth=1 https://github.com/ludwig-ai/ludwig.git
RUN cd ludwig/ \
    && pip install -r requirements.txt -r requirements_text.txt \
          -r requirements_image.txt -r requirements_audio.txt \
          -r requirements_serve.txt -r requirements_viz.txt \
    && python setup.py install

RUN pip install fiber

RUN mkdir /data
ADD train.csv /data/data.csv
ADD hyperopt.yaml /data/hyperopt.yaml

WORKDIR /data
```

In this Dockerfile, the data `data.csv` is embedded in the docker together with `hyperopt.yaml` that specifies the model and hyper-parameter optimization parameters.
If your data is too big to be added directly in the docker image, refer to the [Fiber's documentation](https://uber.github.io/fiber/advanced/#working-with-persistent-storage) for instructions on how to work with shared persistent storage for Fiber workers.
An example `hyperopt.yaml` looks like:

```yaml
input_features:
  -
    name: x
    type: numerical
output_features:
  -
    name: y
    type: category
training:
  epochs: 1
hyperopt:
  sampler:
    type: random
    num_samples: 50
  executor:
    type: fiber
    num_workers: 10
    fiber_backend: kubernetes
    num_cpus_per_worker: 2
    num_gpus_per_worker: 1
  parameters:
    training.learning_rate:
      type: float
      low: 0.0001
      high: 0.1
    y.num_fc_layers:
      type: int
      low: 0
      high: 2
```

Running hyper-parameter optimization with Fiber is a little bit different from other executors because there is docker building and pushing involved, so the `fiber run` command, which takes care of those aspects, is used to run hyper-parameter optimization on a cluster:

`fiber run ludwig hyperopt --dataset train.csv -cf hyperopt.yaml`

Check out [Fiber's documentation](https://uber.github.io/fiber/getting-started/#running-on-a-computer-cluster) for more details on running on clusters.

# Full hyper-parameter optimization example

Example YAML:

```yaml
input_features:
  -
    name: utterance
    type: text
    encoder: rnn
    cell_type: lstm
    num_layers: 2
  -
    name: section
    type: category
    representation: dense
    embedding_size: 100
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
    training.learning_rate:
      type: float
      low: 0.0001
      high: 0.1
      steps: 4
      scale: log
    training.optimizaer.type:
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
    utterance.cell_type:
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
ludwig hyperopt --dataset reuters-allcats.csv --config "{input_features: [{name: utterance, type: text, encoder: rnn, cell_type: lstm, num_layers: 2}], output_features: [{name: class, type: category}], training: {learning_rate: 0.001}, hyperopt: {goal: maximize, output_feature: class, metric: accuracy, split: validation, parameters: {training.learning_rate: {type: float, low: 0.0001, high: 0.1, steps: 4, scale: log}, utterance.cell_type: {type: category, values: [rnn, gru, lstm]}}, sampler: {type: grid}, executor: {type: serial}}}"
```
