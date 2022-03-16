The hyperparameter optimization design in Ludwig is based on two abstract
interfaces: `HyperoptSampler` and `HyperoptExecutor`.

See [Hyperopt configuration](../../configuration/hyperparameter_optimization)
for examples of how the sampler and executor are configured.

# HyperoptSampler

`HyperoptSampler` dictates how to sample hyperparameters values.

The sampler is configured by `sampler` section of the `hyperopt` section of the
Ludwig configuration.

Each hyperparameter that should be sampled is declared in `hyperopt.parameters`,
which also specifies additional constraints that the `Sampler` should honor. For
example:

```yaml
hyperopt:
    goal: minimize
    output_feature: combined
    metric: loss
    split: validation
    parameters:
        trainer.learning_rate:
            space: linear
            range:
                low: 0.001
                high: 0.1
            steps: 4
        text.fc_layers:
            space: choice
            categories:
                - [{"output_size": 512}, {"output_size": 256}]
                - [{"output_size": 512}]
                - [{"output_size": 256}]
```

Here, `trainer.learning_rate` is sampled in continuously while `text.fc_layers`
is sampled discretely.

!!! note

    Different `HyperoptSampler`s are described [here](../../configuration/hyperparameter_optimization#sampler).

# HyperoptExecutor

`HyperoptExecutor` dictates how to execute the hyperparameter optimization,
which operates independently of how hyperparameters are actually sampled.

A `HyperoptExecutor` uses a `HyperoptSampler` to sample hyperparameters values,
usually initializes an execution context, like a multithread pool for instance,
and executes the hyperparameter optimization according to the sampler.

First, a new batch of parameters values is sampled from the `HyperoptSampler`.
Then, sampled parameters values are merged with the seed Ludwig configuration,
with the sampled parameters values overriding the seed's.

Training is executed and validation losses and metrics are collected.
A `(sampled_parameters, statistics)` pair is provided to the
`HyperoptSampler.update` function to inform the next sample of hyperparameters.

The loop is repeated until all the samples are sampled.

Finally, `HyperoptExecutor.execute` returns a list of dictionaries that each
contain: the sampled parameters, metric scores, and other training, validation,
and test statistics.

The returned list is printed and saved to disk, so that it can also be used as
input to
[hyperparameter optimization visualizations](../../user_guide/hyperopt).

!!! note

    Different `HyperoptExecutor`s are described [here](../../configuration/hyperparameter_optimization#executor)

# Adding a HyperoptSampler

## 1. Add a new sampler class

The source code for the base `HyperoptSampler` class is in
`ludwig/hyperopt/sampling.py`.

Classes extending the base class should be defined in this file.

### `__init__`

```python
def __init__(self, goal: str, parameters: Dict[str, Any]):
```

The parameters of the base `HyperoptStrategy` class constructor are:

- `goal` which indicates if to minimize or maximize a metric or a loss of any of the output features on any of the splits which is defined in the `hyperopt` section
- `parameters` which contains all hyperparameters to optimize with their types and ranges / values.

Example:

```python
goal = "minimize"
parameters = {
    "training.learning_rate": {
        "type": "float",
        "low": 0.001,
        "high": 0.1,
        "steps": 4,
        "scale": "linear"
    },
    "combiner.num_fc_layers": {
        "type": "int",
        "low": 2,
        "high": 6,
        "steps": 3
    }
}

sampler = GridSampler(goal, parameters)
```

### `sample`

```python
def sample(self) -> Dict[str, Any]:
```

`sample` is a method that yields a new sample according to the sampler.
It returns a set of parameters names and their values.
If `finished()` returns `True`, calling `sample` would return a `IndexError`.

Example returned value:

```python
{'training.learning_rate': 0.005, 'combiner.num_fc_layers': 2, 'utterance.cell_type': 'gru'}
```

### `sample_batch`

```python
def sample_batch(self, batch_size: int = 1) -> List[Dict[str, Any]]:
```

`sample_batch` method returns a list of sampled parameters of length equal to or less than `batch_size`.
If `finished()` returns `True`, calling `sample_batch` would return a `IndexError`.

Example returned value:

```python
[{'training.learning_rate': 0.005, 'combiner.num_fc_layers': 2, 'utterance.cell_type': 'gru'}, {'training.learning_rate': 0.015, 'combiner.num_fc_layers': 3, 'utterance.cell_type': 'lstm'}]
```

### `update`

```python
def update(
    self,
    sampled_parameters: Dict[str, Any],
    metric_score: float
):
```

`update` updates the sampler with the results of previous computation.

- `sampled_parameters` is a dictionary of sampled parameters.
- `metric_score` is the value of the optimization metric obtained for the specified sample.

It is not needed for stateless strategies like grid and random, but is needed for stateful strategies like bayesian and evolutionary ones.

Example:

```python
sampled_parameters = {
    'training.learning_rate': 0.005,
    'combiner.num_fc_layers': 2,
    'utterance.cell_type': 'gru'
}
metric_score = 2.53463

sampler.update(sampled_parameters, metric_score)
```

### `update_batch`

```python
def update_batch(
    self,
    parameters_metric_tuples: Iterable[Tuple[Dict[str, Any], float]]
):
```

`update_batch` updates the sampler with the results of previous computation in batch.

- `parameters_metric_tuples` a list of pairs of sampled parameters and their respective metric value.

It is not needed for stateless strategies like grid and random, but is needed for stateful strategies like bayesian and evolutionary ones.

Example:

```python
sampled_parameters = [
    {
        'training.learning_rate': 0.005,
        'combiner.num_fc_layers': 2,
        'utterance.cell_type': 'gru'
    },
    {
        'training.learning_rate': 0.015,
        'combiner.num_fc_layers': 5,
        'utterance.cell_type': 'lstm'
    }
]
metric_scores = [2.53463, 1.63869]

sampler.update_batch(zip(sampled_parameters, metric_scores))
```

### `finished`

```python
def finished(self) -> bool:
```

The `finished` method return `True` when all samples have been sampled, return `False` otherwise.

## 2. Add the new sampler class to the corresponding sampler registry

The `sampler_registry` contains a mapping between `sampler` names in the
`hyperopt` section of model definition and `HyperoptSampler` sub-classes.

Add the new sampler to the registry:

```python
sampler_registry = {
    "random": RandomSampler,
    "grid": GridSampler,
    ...,
    "new_sampler_name": NewSamplerClass
}
```

# Adding a HyperoptExecutor

## 1. Add a new executor class

The source code for the base `HyperoptExecutor` class is in the
`ludwig/utils/hyperopt_utils.py` module. Classes extending the base class should
be defined in the module.

### `__init__`

```python
def __init__(
    self,
    hyperopt_sampler: HyperoptSampler,
    output_feature: str,
    metric: str,
    split: str
)
```

The parameters of the base `HyperoptExecutor` class constructor are

- `hyperopt_sampler` is a `HyperoptSampler` object that will be used to sample hyperparameters values
- `output_feature` is a `str` containing the name of the output feature that we want to optimize the metric or loss of. Available values are `combined` (default) or the name of any output feature provided in the model definition. `combined` is a special output feature that allows to optimize for the aggregated loss and metrics of all output features.
- `metric` is the metric that we want to optimize for. The default one is `loss`, but depending on the tye of the feature defined in `output_feature`, different metrics and losses are available. Check the metrics section of the specific output feature type to figure out what metrics are available to use.
- `split` is the split of data that we want to compute our metric on. By default it is the `validation` split, but you have the flexibility to specify `train` or `test` splits.

Example:

```python
goal = "minimize"
parameters = {
    "training.learning_rate": {
        "type": "float",
        "low": 0.001,
        "high": 0.1,
        "steps": 4,
        "scale": "linear"
    },
    "combiner.num_fc_layers": {
        "type": "int",
        "low": 2,
        "high": 6,
        "steps": 3
    }
}
output_feature = "combined"
metric = "loss"
split = "validation"

grid_sampler = GridSampler(goal, parameters)
executor = SerialExecutor(grid_sampler, output_feature, metric, split)
```

### `execute`

```python
def execute(
    self,
    config,
    dataset=None,
    training_set=None,
    validation_set=None,
    test_set=None,
    training_set_metadata=None,
    data_format=None,
    experiment_name="hyperopt",
    model_name="run",
    model_load_path=None,
    model_resume_path=None,
    skip_save_training_description=False,
    skip_save_training_statistics=False,
    skip_save_model=False,
    skip_save_progress=False,
    skip_save_log=False,
    skip_save_processed_input=False,
    skip_save_unprocessed_output=False,
    skip_save_predictions=False,
    skip_save_eval_stats=False,
    output_directory="results",
    gpus=None,
    gpu_memory_limit=None,
    allow_parallel_threads=True,
    use_horovod=None,
    random_seed=default_random_seed,
    debug=False,
    **kwargs
):
```

The `execute` method executes the hyperparameter optimization.
It can leverage the `run_experiment` function to obtain training and eval statistics and the `self.get_metric_score` function to extract the metric score from the eval results according to `self.output_feature`, `self.metric` and `self.split`.

## 2. Add the new executor class to the corresponding executor registry

The `executor_registry` contains a mapping between `executor` names in the `hyperopt` section of model definition and `HyperoptExecutor` sub-classes.
To make a new executor available, add it to the registry:

```python
executor_registry = {
    "serial": SerialExecutor,
    "parallel": ParallelExecutor,
    "fiber": FiberExecutor,
    "new_executor_name": NewExecutorClass
}
```
