Codebase Structure
==================

The codebase is organized in a modular, datatype / feature centric way so that adding a feature for a new datatype is pretty straightforward and requires isolated code changes.
All the datatype specific logic lives in the corresponding feature module, all of which are under `ludwig/features/`.

Feature classes contain raw data preprocessing logic specific to each data type in datatype mixin clases, like `BinaryFeatureMixin`, `NumericalFeatureMixin`, `CategoryFeatureMixin`, and so on.
Those classes contain data preprocessing functions to obtain feature metadata (`get_feature_meta`, one-time dataset-wide operation to collect things like min, max, average, vocabularies, etc.) and to transform raw data into tensors using the previously calculated metadata (`add_feature_data`, which usually work on a a dataset row basis).
Output features also contain datatype-specific logic to compute data postprocessing, to transform model predictions back into data space, and output metrics such as loss, accuracy, etc..

Encoders and decoders are modularized as well (they are under `ludwig/encoders` and `ludwig/decoders` respectively) so that they can be used by multiple features.
For example sequence encoders are shared among text, sequence, and timeseries features.

Various model architecture components which can be reused are also split into dedicated modules, for example convolutional modules, fully connected modules, etc.. which are available in `ludwig/modules`.

The training logic resides in `ludwig/models/trainer.py` which initializes a training session, feeds the data, and executes training.
The prediction logic resides in `ludwig/models/predictior.py` instead.

The command line interface is managet by the `ludwig/cli.py` script, which imports the various scripts in `ludwig/` that perform the different command line commands.

The programmatic interface (which is also used by the CLI commands) is available in the `ludwig/api.py` script.

Hyper-parameter optimization logic is implemented in the scripts in the `ludwig/hyperopt` package. 

The `ludwig/utils` package contains various utilities used by all other packages.

Finally the `ludwig/contrib` packages contains user contributed code that integrates with external libraries.


Adding an Encoder
=================

 1. Add a new encoder class
---------------------------

Source code for encoders lives under `ludwig/encoders`.
New encoder objects should be defined in the corresponding files, for example all new sequence encoders should be added to `ludwig/encoders/sequence_encoders.py`.

All the encoder parameters should be provided as arguments in the constructor with their default values set.
For example the `StackedRNN` encoder takes the following list of arguments in its constructor:

```python
def __init__(
    self,
    should_embed=True,
    vocab=None,
    representation='dense',
    embedding_size=256,
    embeddings_trainable=True,
    pretrained_embeddings=None,
    embeddings_on_cpu=False,
    num_layers=1,
    state_size=256,
    cell_type='rnn',
    bidirectional=False,
    activation='tanh',
    recurrent_activation='sigmoid',
    unit_forget_bias=True,
    recurrent_initializer='orthogonal',
    recurrent_regularizer=None,
    dropout=0.0,
    recurrent_dropout=0.0,
    fc_layers=None,
    num_fc_layers=0,
    fc_size=256,
    use_bias=True,
    weights_initializer='glorot_uniform',
    bias_initializer='zeros',
    weights_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    norm=None,
    norm_params=None,
    fc_activation='relu',
    fc_dropout=0,
    reduce_output='last',
    **kwargs
):
```

Typically all the modules the encoder relies upon are initialized in the encoder's constructor (in the case of the `StackedRNN` encoder these are `EmbedSequence` and `RecurrentStack` modules) so that at the end of the constructor call all the layers are fully described.

Actual computation of activations takes place inside the `call` method of the encoder.
All encoders should have the following signature:

```python
def call(self, inputs, training=None, mask=None):
```

__Inputs__

- __inputs__ (tf.Tensor): input tensor.
- __training__ (bool, default: `None`): boolean indicating whether we are currently training the model or performing inference for prediction.
- __mask__ (tf.Tensor, default: `None`): binary tensor indicating which of the values in the inputs tensor should be masked out.

__Return__

- __hidden__ (tf.Tensor): feature encodings.

The shape of the input tensor and the expected tape of the output tensor varies across feature types.

Encoders are initialized as class member variables in input features object constructors and called inside their `call` methods.


 2. Add the new encoder class to the corresponding encoder registry
-------------------------------------------------------------------

Mapping between encoder names in the model definition and encoder classes in the codebase is done by encoder registries: for example sequence encoder registry is defined in `ludwig/features/sequence_feature.py` inside the `SequenceInputFeature` as:

```python
sequence_encoder_registry = {
    'stacked_cnn': StackedCNN,
    'parallel_cnn': ParallelCNN,
    'stacked_parallel_cnn': StackedParallelCNN,
    'rnn': StackedRNN,
    ...
}
```

All you have to do to make you new encoder available as an option in the model definition is to add it to the appropriate registry.


Adding a Decoder
================

 1. Add a new decoder class
---------------------------

Source code for decoders lives under `ludwig/decoders/`.
New decoder objects should be defined in the corresponding files, for example all new sequence decoders should be added to `ludwig/decoders/sequence_decoders.py`.

All the decoder parameters should be provided as arguments in the constructor with their default values set.
For example the `SequenceGeneratorDecoder` decoder takes the following list of arguments in its constructor:

```python
def __init__(
    self,
    num_classes,
    cell_type='rnn',
    state_size=256,
    embedding_size=64,
    beam_width=1,
    num_layers=1,
    attention=None,
    tied_embeddings=None,
    is_timeseries=False,
    max_sequence_length=0,
    use_bias=True,
    weights_initializer='glorot_uniform',
    bias_initializer='zeros',
    weights_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    reduce_input='sum',
    **kwargs
):
```

Decoders are initialized as class member variables in output feature object constructors and called inside `call` methods.

 2. Add the new decoder class to the corresponding decoder registry
-------------------------------------------------------------------

Mapping between decoder names in the model definition and decoder classes in the codebase is done by encoder registries: for example sequence encoder registry is defined in `ludwig/features/sequence_feature.py` inside the `SequenceOutputFeature` as:

```python
sequence_decoder_registry = {
    'generator': Generator,
    'tagger': Tagger
}
```

All you have to do to make you new decoder available as an option in the model definition is to add it to the appropriate registry.


Adding a new Feature Type
=========================

 1. Add a new feature class
---------------------------

Souce code for feature classes lives under `ludwig/features`.
Input and output feature classes are defined in the same file, for example `CategoryInputFeature` and `CategoryOutputFeature` are defined in `ludwig/features/category_feature.py`.

An input features inherit from the `InputFeature` and corresponding mixin feature classes, for example `CategoryInputFeature` inherits from `CategoryFeatureMixin` and `InputFeature`.

Similarly, output features inherit from the `OutputFeature` and corresponding base feature classes, for example `CategoryOutputFeature` inherits from `CategoryFeatureMixin` and `OutputFeature`.

Feature parameters are provided in a dictionary of key-value pairs as an argument to the input or output feature constructor which contains default parameter values as well.

### Input features

All input features should implement `__init__` and `call` methods with the following signatures:


#### `__init__`

```python
def __init__(self, feature, encoder_obj=None):
```

__Inputs__


- __feature__: (dict) contains all feature parameters.
- __encoder_obj__: (*Encoder, default: `None`) is an encoder object of the type supported (a cateory encoder, binary encoder, etc.). It is used only when two input features share the encoder.


#### `call`

```python
def call(self, inputs, training=None, mask=None):
```

__Inputs__

- __inputs__ (tf.Tensor): input tensor.
- __training__ (bool, default: `None`): boolean indicating whether we are currently training the model or performing inference for prediction.
- __mask__ (tf.Tensor, default: `None`): binary tensor indicating which of the values in the inputs tensor should be masked out.

__Return__

- __hidden__ (tf.Tensor): feature encodings.


### Output features

All input features should implement `__init__`, `logits` and `predictions` methods with the following signatures:


#### `__init__`

```python
def __init__(self, feature, encoder_obj=None):
```

__Inputs__


- __feature__ (dict): contains all feature parameters.
- __decoder_obj__ (*Decoder, default: `None`): is a decoder object of the type supported (a cateory decoder, binary decoder, etc.). It is used only when two output features share the decoder.

#### `logits`

```python
def call(self, inputs, **kwargs):
```

__Inputs__

- __inputs__ (dict): input dictionary that is the output of the combiner.

__Return__

- __hidden__ (tf.Tensor): feature logits.

#### `predictions`

```python
def call(self, inputs, **kwargs):
```

__Inputs__

- __inputs__ (dict): input dictionary that contains the output of the combiner and the logits function.

__Return__

- __hidden__ (dict): contains predictions, probabilities and logits.


 2. Add the new feature class to the corresponding feature registry
-------------------------------------------------------------------

Input and output feature registries are defined in `ludwig/features/feature_registries.py`.


Hyper-parameter optimization
============================

The hyper-parameter optimization design in Ludwig is based on two abstract interfaces: `HyperoptSampler` and `HyperoptExecutor`. 

`HyperoptSampler` represents the sampler adopted for sampling hyper-parameters values.
 Which sampler to use is defined in the `sampler` section of the model definition.
A `Sampler` uses the `parameters` defined in the `hyperopt` section of the YAML model definition and a `goal` , either to minimize or maximize.
Each sub-class of `HyperoptSampler` that implements its abstract methods samples parameters according to their definition and type differently (see [User Guide](user_guide.md#hyper-parameter-optimization) for details), like using a random search (implemented in `RandomSampler`), or a grid serach (implemented in `GridSampler`, or bayesian optimization or evolutionary techniques.
 
`HyperoptExecutor` represents the method used to execute the hyper-parameter optimization, independently of how the values for the hyperparameters are sampled.
Available implementations are a serial executor that executes the training with the different sampled hyper-parameters values one at a time (implemented in `SerialExecutor`), a parallel executor that runs the training using sampled hyper-parameters values in parallel on the same machine (implemented in the `ParallelExecutor`), and a [Fiber](https://uber.github.io/fiber/)-based executor that enables to run the training using sampled hyper-parameters values in parallel on multiple machines within a cluster. 
A `HyperoptExecutor` uses a `HyperoptSampler` to sample hyper-parameters values, usually initializes an execution context, like a multithread pool for instance, and executes the hyper-parameter optimization according to the sampler.
First, a new batch of parameters values is sampled from the `HyperoptSampler`.
Then, sampled parameters values are merged with the basic model definition parameters specified, with the sampled parameters values overriding the ones in the basic model definition they refer to.
Training is executed using the merged model definition and training and validation losses and metrics are collected.
A `(sampled_parameters, statistics)` pair is provided to the `HyperoptSampler.update` function and the loop is repeated until all the samples are sampled.
At the end, `HyperoptExecutor.execute` returns a list of dictionaries that include a parameter sample, its metric score, and its training and test statistics.
The returned list is printed and saved to disk, so that it can also be used as input to [hyper-parameter optimization visualizations](user_guide.md#hyper-parameter-optimization-visualization).


Adding a HyperoptSampler
-------------------------

### 1. Add a new sampler class

The source code for the base `HyperoptSampler` class is in the `ludwig/hyperopt/sampling.py` module.
Classes extending the base class should be defined in the same module.

#### `__init__`
```python
def __init__(self, goal: str, parameters: Dict[str, Any]):
```

The parameters of the base `HyperoptStrategy` class constructor are:
- `goal` which indicates if to minimize or maximize a metric or a loss of any of the output features on any of the splits which is defined in the `hyperopt` section
- `parameters` which contains all hyper-parameters to optimize with their types and ranges / values.

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

#### `sample`
```python
def sample(self) -> Dict[str, Any]:
```

`sample` is a method that yields a new sample according to the sampler.
It returns a set of parameters names and their values.
If `finished()` returns `True`, calling `sample` would return a `IndexError`.

Example returned value:
```
{'training.learning_rate': 0.005, 'combiner.num_fc_layers': 2, 'utterance.cell_type': 'gru'}
```

#### `sample_batch`
```python
def sample_batch(self, batch_size: int = 1) -> List[Dict[str, Any]]:
```

`sample_batch` method returns a list of sampled parameters of length equal to or less than `batch_size`.
If `finished()` returns `True`, calling `sample_batch` would return a `IndexError`. 

Example returned value:
```
[{'training.learning_rate': 0.005, 'combiner.num_fc_layers': 2, 'utterance.cell_type': 'gru'}, {'training.learning_rate': 0.015, 'combiner.num_fc_layers': 3, 'utterance.cell_type': 'lstm'}]
```

#### `update`
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

#### `update_batch`
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

#### `finished`
```python
def finished(self) -> bool:
```

The `finished` method return `True` when all samples have been sampled, return `False` otherwise.


### 2. Add the new sampler class to the corresponding sampler registry

The `sampler_registry` contains a mapping between `sampler` names in the `hyperopt` section of model definition and `HyperoptSampler` sub-classes.
To make a new sampler available, add it to the registry:
```
sampler_registry = {
    "random": RandomSampler,
    "grid": GridSampler,
    ...,
    "new_sampler_name": NewSamplerClass
}
```


Adding a HyperoptExecutor
-------------------------

### 1. Add a new executor class

The source code for the base `HyperoptExecutor` class is in the `ludwig/utils/hyperopt_utils.py` module.
Classes extending the base class should be defined in the module.

#### `__init__`
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
- `hyperopt_sampler` is a `HyperoptSampler` object that will be used to sample hyper-parameters values
- `output_feature` is a `str` containing the name of the output feature that we want to optimize the metric or loss of. Available values are `combined` (default) or the name of any output feature provided in the model definition. `combined` is a special output feature that allows to optimize for the aggregated loss and metrics of all output features.
- `metric` is the metric that we want to optimize for. The default one is `loss`, but depending on the tye of the feature defined in `output_feature`, different metrics and losses are available. Check the metrics section of the specific output feature type to figure out what metrics are available to use.
- `split` is the split of data that we want to compute our metric on. By default it is the `validation` split, but you have the flexibility to specify also `train` or `test` splits.

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

#### `execute`
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

The `execute` method executes the hyper-parameter optimization.
It can leverage the `run_experiment` function to obtain training and eval statistics and the `self.get_metric_score` function to extract the metric score from the eval results according to `self.output_feature`, `self.metric` and `self.split`.


### 2. Add the new executor class to the corresponding executor registry

The `executor_registry` contains a mapping between `executor` names in the `hyperopt` section of model definition and `HyperoptExecutor` sub-classes.
To make a new executor available, add it to the registry:
```
executor_registry = {
    "serial": SerialExecutor,
    "parallel": ParallelExecutor,
    "fiber": FiberExecutor,
    "new_executor_name": NewExecutorClass
}
```


Adding a new Integration
========================

Ludwig provides an open-ended method of third-party system
integration. This makes it easy to integrate other systems or services
with Ludwig without having users do anything other than adding a flag
to the command line interface.

To contribute an integration, follow these steps:

1. Create a Python file in `ludwig/contribs/` with an obvious name. In this example, it is called `mycontrib.py`.
2. Inside that file, create a class with the following structure, renaming `MyContribution` to a name that is associated with the third-party system:

```python
class MyContribution():
    @staticmethod
    def import_call(argv, *args, **kwargs):
        # This is called when your flag is used before any other imports.

    def experiment(self, *args, **kwargs):
        # See: ludwig/experiment.py and ludwig/cli.py

    def experiment_save(self, *args, **kwargs):
        # See: ludwig/experiment.py

    def train_init(self, experiment_directory, experiment_name, model_name,
                   resume, output_directory):
        # See: ludwig/train.py

    def train(self, *args, **kwargs):
        # See: ludwig/train.py and ludwig/cli.py

    def train_model(self, *args, **kwargs):
        # See: ludwig/train.py

    def train_save(self, *args, **kwargs):
        # See: ludwig/train.py

    def train_epoch_end(self, progress_tracker):
        # See: ludwig/models/model.py

    def predict(self, *args, **kwargs):
        # See: ludwig/predict.py and ludwig/cli.py

    def predict_end(self, test_stats):
        # See: ludwig/predict.py

    def test(self, *args, **kwargs):
        # See: ludwig/test.py and ludwig/cli.py

    def visualize(self, *args, **kwargs):
        # See: ludwig/visualize.py and ludwig/cli.py

    def visualize_figure(self, fig):
        # See ludwig/utils/visualization_utils.py

    def serve(self, *args, **kwargs):
        # See ludwig/utils/serve.py and ludwig/cli.py

    def collect_weights(self, *args, **kwargs):
        # See ludwig/collect.py and ludwig/cli.py

    def collect_activations(self, *args, **kwargs):
        # See ludwig/collect.py and ludwig/cli.py
```

If your integration does not handle a particular action, you can simply remove the method, or do nothing (e.g., `pass`).

If you would like to add additional actions not already handled by the
above, add them to the appropriate calling location, add the
associated method to your class, and add them to this
documentation. See existing calls as a pattern to follow.

3. In the file `ludwig/contribs/__init__.py` add an import in this pattern, using your names:

```python
from .mycontrib import MyContribution
```

4. In the file `ludwig/contribs/__init__.py` in the `contrib_registry["classes"]` dictionary, add a key/value pair where the key is your flag, and the value is your class name, like:

```python
contrib_registry = {
    ...,
    "classes": {
        ...,
        "myflag": MyContribution,
    }
}
```

5. Submit your contribution as a pull request to the Ludwig github repository.


Style Guidelines
================

We expect contributions to mimic existing patterns in the codebase and demonstrate good practices: the code should be concise, readable, PEP8-compliant, and conforming to 80 character line length limit.

Tests
=====

We are using ```pytest``` to run tests.
To install all the required dependencies for testing, please do `pip install ludwig[test]`.
Current test coverage is limited to several integration tests which ensure end-to-end functionality but we are planning to expand it.

Checklist
---------

Before running tests, make sure 
1. Your environment is properly setup.
2. You have write access on the machine. Some of the tests require saving data to disk.

Running tests
-------------

To run all tests, just run
```python -m pytest``` from the ludwig root directory.
Note that you don't need to have ludwig module installed and in this case
code change will take effect immediately.

To run a single test, run
``` 
python -m pytest path_to_filename -k "test_method_name"
```

Example
-------

```
python -m pytest tests/integration_tests/test_experiment.py -k "test_visual_question_answering"
```
