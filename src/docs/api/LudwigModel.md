<span style="float:right;">[[source]](https://github.com/uber/ludwig/blob/master/ludwig/api.py#L70)</span>
# LudwigModel class

```python
ludwig.api.LudwigModel(
  config,
  logging_level=40,
  use_horovod=None,
  gpus=None,
  gpu_memory_limit=None,
  allow_parallel_threads=True
)
```

Class that allows access to high level Ludwig functionalities.

__Inputs__


- __config__ (Union[str, dict]): in-memory representation of
    model definition or string path to a YAML model definition file.
- __logging_level__ (int): Log level that will be sent to stderr.
- __use_horovod__ (bool): use Horovod for distributed training.
Will be set automatically if `horovodrun` is used to launch
the training script.
- __gpus__ (Union[str, int, List[int]], default: `None`): GPUs
to use (it uses the same syntax of CUDA_VISIBLE_DEVICES)
- __gpu_memory_limit__ (int: default: `None`): maximum memory in MB to
allocate per GPU device.
- __allow_parallel_threads__ (bool, default: `True`): allow TensorFlow
to use multithreading parallelism to improve performance at the
cost of determinism.

__Example usage:__


```python
from ludwig.api import LudwigModel
```

Train a model:

```python
config = {...}
ludwig_model = LudwigModel(config)
train_stats, _, _  = ludwig_model.train(dataset=file_path)
```

or

```python
train_stats, _, _ = ludwig_model.train(dataset=dataframe)
```

If you have already trained a model you can load it and use it to predict

```python
ludwig_model = LudwigModel.load(model_dir)
```

Predict:

```python
predictions, _ = ludwig_model.predict(dataset=file_path)
```

or

```python
predictions, _ = ludwig_model.predict(dataset=dataframe)
```

Evaluation:

```python
eval_stats, _, _ = ludwig_model.evaluate(dataset=file_path)
```

or

```python
eval_stats, _, _ = ludwig_model.evaluate(dataset=dataframe)
```


---
# LudwigModel methods

## collect_activations


```python
collect_activations(
  layer_names,
  dataset,
  data_format=None,
  split='full',
  batch_size=128,
  debug=False
)
```


Loads a pre-trained model model and input data to collect the values
of the activations contained in the tensors.

__Inputs__

- __layer_names__ (list): list of strings for layer names in the model
to collect activations.
- __dataset__ (Union[str, Dict[str, list], pandas.DataFrame]): source
containing the data to make predictions.
- __data_format__ (str, default: `None`): format to interpret data
sources. Will be inferred automatically if not specified.  Valid
formats are `'auto'`, `'csv'`, `'df'`, `'dict'`, `'excel'`, `'feather'`,
`'fwf'`, `'hdf5'` (cache file produced during previous training),
`'html'` (file containing a single HTML `<table>`), `'json'`, `'jsonl'`,
`'parquet'`, `'pickle'` (pickled Pandas DataFrame), `'sas'`, `'spss'`,
`'stata'`, `'tsv'`.
:param: split: (str, default= `'full'`): if the input dataset contains
a split column, this parameter indicates which split of the data
to use. Possible values are `'full'`, `'training'`, `
'validation'`, `'test'`.
- __batch_size__ (int, default: 128): size of batch to use when making
predictions.
- __debug__ (bool, default: `False`): if `True` turns on `tfdbg`
with `inf_or_nan` checks.

__Return__

- __return__ (list): list of collected tensors.


---
## collect_weights


```python
collect_weights(
  tensor_names=None
)
```


Load a pre-trained model and collect the tensors with a specific name

__Inputs__

- __tensor_names__ (list, default: `None`): List of tensor names to collect
weights

__Return__

- __return__ (list): List of tensors


---
## create_model


```python
create_model(
  config,
  random_seed=42
)
```


Instantiates Encoder-Combiner-Decoder (ECD) object

__Inputs__

- __config__ (dict): Ludwig model definition
- __random_seed__ (int, default: ludwig default random seed): Random
seed used for weights initialization,
splits and any other random function.

__Return__

- __return__ (ludwig.models.ECD): Instance of the Ludwig model object.


---
## evaluate


```python
ludwig.evaluate(
  dataset=None,
  data_format=None,
  split='full',
  batch_size=128,
  skip_save_unprocessed_output=True,
  skip_save_predictions=True,
  skip_save_eval_stats=True,
  collect_predictions=False,
  collect_overall_stats=False,
  output_directory='results',
  return_type=<class 'pandas.core.frame.DataFrame'>,
  debug=False
)
```


This function is used to predict the output variables given the
input variables using the trained model and compute test statistics
like performance measures, confusion matrices and the like.

__Inputs__

- __dataset__ (Union[str, dict, pandas.DataFrame]): source containing
the entire dataset to be evaluated.
- __data_format__ (str, default: `None`): format to interpret data
sources. Will be inferred automatically if not specified.  Valid
formats are `'auto'`, `'csv'`, `'df'`, `'dict'`, `'excel'`, `'feather'`,
`'fwf'`, `'hdf5'` (cache file produced during previous training),
`'html'` (file containing a single HTML `<table>`), `'json'`, `'jsonl'`,
`'parquet'`, `'pickle'` (pickled Pandas DataFrame), `'sas'`, `'spss'`,
`'stata'`, `'tsv'`.
:param: split: (str, default= `'full'`): if the input dataset contains
a split column, this parameter indicates which split of the data
to use. Possible values are `'full'`, `'training'`, `
'validation'`, `'test'`.
- __batch_size__ (int, default: 128): size of batch to use when making
predictions.
- __skip_save_unprocessed_output__ (bool, default: `True`): if this
parameter is `False`, predictions and their probabilities are saved
in both raw unprocessed numpy files containing tensors and as
postprocessed CSV files (one for each output feature).
If this parameter is `True`, only the CSV ones are saved and the
numpy ones are skipped.
- __skip_save_predictions__ (bool, default: `True`): skips saving
test predictions CSV files.
- __skip_save_eval_stats__ (bool, default: `True`): skips saving
test statistics JSON file.
- __collect_predictions__ (bool, default: `False`): if `True`
collects post-processed predictions during eval.
- __collect_overall_stats__ (bool, default: False): if `True`
collects overall stats during eval.
- __output_directory__ (str, default: `'results'`): the directory that
will contain the training statistics, TensorBoard logs, the saved
model and the training progress files.
- __return_type__ (Union[str, dict, pandas.DataFrame], default: pandas.DataFrame): indicates
the format to of the returned predictions.
- __debug__ (bool, default: `False`): If `True` turns on `tfdbg`
    with `inf_or_nan` checks.


__Return__

- __return__ (`evaluation_statistics`, `predictions`, `output_directory`):
`evaluation_statistics` dictionary containing evaluation performance
    statistics,
`postprocess_predictions` contains predicted values,
`output_directory` is location where results are stored.


---
## experiment


```python
experiment(
  dataset=None,
  training_set=None,
  validation_set=None,
  test_set=None,
  training_set_metadata=None,
  data_format=None,
  experiment_name='experiment',
  model_name='run',
  model_load_path=None,
  model_resume_path=None,
  eval_split='test',
  skip_save_training_description=False,
  skip_save_training_statistics=False,
  skip_save_model=False,
  skip_save_progress=False,
  skip_save_log=False,
  skip_save_processed_input=False,
  skip_save_unprocessed_output=False,
  skip_save_predictions=False,
  skip_save_eval_stats=False,
  skip_collect_predictions=False,
  skip_collect_overall_stats=False,
  output_directory='results',
  random_seed=42,
  debug=False
)
```



Trains a model on a dataset's training and validation splits and
uses it to predict on the test split.
It saves the trained model and the statistics of training and testing.

__Inputs__

- __dataset__ (Union[str, dict, pandas.DataFrame], default: `None`):
source containing the entire dataset to be used in the experiment.
If it has a split column, it will be used for splitting (0 for train,
1 for validation, 2 for test), otherwise the dataset will be
randomly split.
- __training_set__ (Union[str, dict, pandas.DataFrame], default: `None`):
source containing training data.
- __validation_set__ (Union[str, dict, pandas.DataFrame], default: `None`):
source containing validation data.
- __test_set__ (Union[str, dict, pandas.DataFrame], default: `None`):
source containing test data.
- __training_set_metadata__ (Union[str, dict], default: `None`):
metadata JSON file or loaded metadata.  Intermediate preprocess
structure containing the mappings of the input
dataset created the first time an input file is used in the same
directory with the same name and a '.meta.json' extension.
- __data_format__ (str, default: `None`): format to interpret data
sources. Will be inferred automatically if not specified.  Valid
formats are `'auto'`, `'csv'`, `'df'`, `'dict'`, `'excel'`, `'feather'`,
`'fwf'`, `'hdf5'` (cache file produced during previous training),
`'html'` (file containing a single HTML `<table>`), `'json'`, `'jsonl'`,
`'parquet'`, `'pickle'` (pickled Pandas DataFrame), `'sas'`, `'spss'`,
`'stata'`, `'tsv'`.
- __experiment_name__ (str, default: `'experiment'`): name for
the experiment.
- __model_name__ (str, default: `'run'`): name of the model that is
being used.
- __model_load_path__ (str, default: `None`): if this is specified the
loaded model will be used as initialization
(useful for transfer learning).
- __model_resume_path__ (str, default: `None`): resumes training of
the model from the path specified. The model definition is restored.
In addition to model definition, training statistics and loss for
epoch and the state of the optimizer are restored such that
training can be effectively continued from a previously interrupted
training process.
- __eval_split__ (str, default: `test`): split on which
to perform evaluation. Valid values are `training`, `validation`
and `test`.
- __skip_save_training_description__ (bool, default: `False`): disables
saving the description JSON file.
- __skip_save_training_statistics__ (bool, default: `False`): disables
saving training statistics JSON file.
- __skip_save_model__ (bool, default: `False`): disables
saving model weights and hyperparameters each time the model
improves. By default Ludwig saves model weights after each epoch
the validation metric improves, but if the model is really big
that can be time consuming if you do not want to keep
the weights and just find out what performance can a model get
with a set of hyperparameters, use this parameter to skip it,
but the model will not be loadable later on and the returned model
will have the weights obtained at the end of training, instead of
the weights of the epoch with the best validation performance.
- __skip_save_progress__ (bool, default: `False`): disables saving
progress each epoch. By default Ludwig saves weights and stats
after each epoch for enabling resuming of training, but if
the model is really big that can be time consuming and will uses
twice as much space, use this parameter to skip it, but training
cannot be resumed later on.
- __skip_save_log__ (bool, default: `False`): disables saving
TensorBoard logs. By default Ludwig saves logs for the TensorBoard,
but if it is not needed turning it off can slightly increase the
overall speed.
- __skip_save_processed_input__ (bool, default: `False`): if input
dataset is provided it is preprocessed and cached by saving an HDF5
and JSON files to avoid running the preprocessing again. If this
parameter is `False`, the HDF5 and JSON file are not saved.
- __skip_save_unprocessed_output__ (bool, default: `False`): by default
predictions and their probabilities are saved in both raw
unprocessed numpy files containing tensors and as postprocessed
CSV files (one for each output feature). If this parameter is True,
only the CSV ones are saved and the numpy ones are skipped.
- __skip_save_predictions__ (bool, default: `False`): skips saving test
predictions CSV files
- __skip_save_eval_stats__ (bool, default: `False`): skips saving test
statistics JSON file
- __skip_collect_predictions__ (bool, default: `False`): skips
collecting post-processed predictions during eval.
- __skip_collect_overall_stats__ (bool, default: `False`): skips
collecting overall stats during eval.
- __output_directory__ (str, default: `'results'`): the directory that
will contain the training statistics, TensorBoard logs, the saved
model and the training progress files.
- __gpus__ (list, default: `None`): list of GPUs that are available
for training.
- __gpu_memory_limit__ (int, default: `None`): maximum memory in MB to
allocate per GPU device.
- __allow_parallel_threads__ (bool, default: `True`): allow TensorFlow
to use multithreading parallelism to improve performance at
the cost of determinism.
- __use_horovod__ (bool, default: `None`): flag for using horovod.
- __random_seed__ (int: default: 42): random seed used for weights
initialization, splits and any other random function.
- __debug__ (bool, default: `False): if `True` turns on `tfdbg` with
`inf_or_nan` checks.

__Return__

- __return__ (Tuple[dict, dict, tuple, str)) `(evaluation_statistics, training_statistics, preprocessed_data, output_directory):`
`evaluation_statistics` dictionary with evaluation performance
    statistics on the test_set,
`training_statistics` is a dictionary of training statistics for
    each output
feature containing loss and metrics values for each epoch,
`preprocessed_data` tuple containing preprocessed
`(training_set, validation_set, test_set)`, `output_directory`
filepath string to where results are stored.


---
## load


```python
load(
  model_dir,
  logging_level=40,
  use_horovod=None,
  gpus=None,
  gpu_memory_limit=None,
  allow_parallel_threads=True
)
```


This function allows for loading pretrained models

__Inputs__


- __model_dir__ (str): path to the directory containing the model.
   If the model was trained by the `train` or `experiment` command,
   the model is in `results_dir/experiment_dir/model`.
- __logging_level__ (int, default: 40): log level that will be sent to
stderr.
- __use_horovod__ (bool, default: `None`): use Horovod for distributed
training. Will be set
automatically if `horovodrun` is used to launch the training script.
- __gpus__ (Union[str, int, List[int]], default: `None`): GPUs
to use (it uses the same syntax of CUDA_VISIBLE_DEVICES)
- __gpu_memory_limit__ (int: default: `None`): maximum memory in MB to
allocate per GPU device.
- __allow_parallel_threads__ (bool, default: `True`): allow TensorFlow
to use
multithreading parallelism to improve performance at the cost of
determinism.

__Return__


- __return__ (LudwigModel): a LudwigModel object


__Example usage__


```python
ludwig_model = LudwigModel.load(model_dir)
```


---
## load_weights


```python
load_weights(
  model_dir
)
```



Loads weights from a pre-trained model

__Inputs__

- __model_dir__ (str): filepath string to location of a pre-trained
model

__Return__

- __return__ ( `Non): `None`

__Example usage__


```python
ludwig_model.load_weights(model_dir)
```


---
## predict


```python
ludwig.predict(
  dataset=None,
  data_format=None,
  split='full',
  batch_size=128,
  skip_save_unprocessed_output=True,
  skip_save_predictions=True,
  output_directory='results',
  return_type=<class 'pandas.core.frame.DataFrame'>,
  debug=False
)
```



Using a trained model, make predictions from the provided dataset.

__Inputs__

- __dataset__ (Union[str, dict, pandas.DataFrame]): source containing
the entire dataset to be evaluated.
- __data_format__ (str, default: `None`): format to interpret data
sources. Will be inferred automatically if not specified.  Valid
formats are `'auto'`, `'csv'`, `'df'`, `'dict'`, `'excel'`, `'feather'`,
`'fwf'`, `'hdf5'` (cache file produced during previous training),
`'html'` (file containing a single HTML `<table>`), `'json'`, `'jsonl'`,
`'parquet'`, `'pickle'` (pickled Pandas DataFrame), `'sas'`, `'spss'`,
`'stata'`, `'tsv'`.
:param: split: (str, default= `'full'`): if the input dataset contains
a split column, this parameter indicates which split of the data
to use. Possible values are `'full'`, `'training'`, `
'validation'`, `'test'`.
- __batch_size__ (int, default: 128): size of batch to use when making
predictions.
- __skip_save_unprocessed_output__ (bool, default: `True`): if this
parameter is `False`, predictions and their probabilities are saved
in both raw unprocessed numpy files containing tensors and as
postprocessed CSV files (one for each output feature).
If this parameter is `True`, only the CSV ones are saved and the
numpy ones are skipped.
- __skip_save_predictions__ (bool, default: `True`): skips saving
test predictions CSV files.
- __output_directory__ (str, default: `'results'`): the directory that
will contain the training statistics, TensorBoard logs, the saved
model and the training progress files.
- __return_type__ (Union[str, dict, pandas.DataFrame], default: pd.DataFrame):
indicates the format of the returned predictions.
- __debug__ (bool, default: `False`): If `True` turns on `tfdbg`
    with `inf_or_nan checks`.

__Return__


- __return__ (Tuple[Union[dict, pd.DataFrame], str]) `(predictions, output_directory):`
`predictions` predictions from the provided dataset,
`output_directory` filepath string to where data was stored.


---
## save


```python
save(
  save_path
)
```


This function allows to save models on disk

__Inputs__


- __ save_path__ (str): path to the directory where the model is
    going to be saved. Both a JSON file containing the model
    architecture hyperparameters and checkpoints files containing
    model weights will be saved.

__Return__


- __return__ (None): `None`

__Example usage__


```python
ludwig_model.save(save_path)
```


---
## save_config


```python
save_config(
  save_path
)
```



Save model definition to specoficed location.

__Inputs__


- __save_path__ (str): filepath string to save model definition as a
JSON file.

__Return__

- __return__ ( `Non): `None`


---
## save_savedmodel


```python
save_savedmodel(
  save_path
)
```


This function allows to save models on disk

__Inputs__


- __ save_path__ (str): path to the directory where the SavedModel
    is going to be saved.

__Return__


- __return__ ( `Non): `None`

__Example usage__


```python
ludwig_model.save_for_serving(save_path)
```


---
## set_logging_level


```python
set_logging_level(
  logging_level
)
```



Sets level for log messages.

__Inputs__


- __logging_level__ (int): Set/Update the logging level. Use logging
constants like `logging.DEBUG` , `logging.INFO` and `logging.ERROR`.

__Return__


- __return__ ( `None): `None`
 
---
## train


```python
train(
  dataset=None,
  training_set=None,
  validation_set=None,
  test_set=None,
  training_set_metadata=None,
  data_format=None,
  experiment_name='api_experiment',
  model_name='run',
  model_resume_path=None,
  skip_save_training_description=False,
  skip_save_training_statistics=False,
  skip_save_model=False,
  skip_save_progress=False,
  skip_save_log=False,
  skip_save_processed_input=False,
  output_directory='results',
  random_seed=42,
  debug=False
)
```


This function is used to perform a full training of the model on the
specified dataset.

During training if the skip parameters are False
the model and statistics will be saved in a directory
`[output_dir]/[experiment_name]_[model_name]_n` where all variables are
resolved to user specified ones and `n` is an increasing number
starting from 0 used to differentiate among repeated runs.

__Inputs__


- __dataset__ (Union[str, dict, pandas.DataFrame], default: `None`):
source containing the entire dataset to be used in the experiment.
If it has a split column, it will be used for splitting
(0 for train, 1 for validation, 2 for test),
otherwise the dataset will be randomly split.
- __training_set__ (Union[str, dict, pandas.DataFrame], default: `None`):
source containing training data.
- __validation_set__ (Union[str, dict, pandas.DataFrame], default: `None`):
source containing validation data.
- __test_set__ (Union[str, dict, pandas.DataFrame], default: `None`):
source containing test data.
- __training_set_metadata__ (Union[str, dict], default: `None`):
metadata JSON file or loaded metadata. Intermediate preprocess
structure containing the mappings of the input
dataset created the first time an input file is used in the same
directory with the same name and a '.meta.json' extension.
- __data_format__ (str, default: `None`): format to interpret data
sources. Will be inferred automatically if not specified.  Valid
formats are `'auto'`, `'csv'`, `'df'`, `'dict'`, `'excel'`,
`'feather'`, `'fwf'`,
`'hdf5'` (cache file produced during previous training),
`'html'` (file containing a single HTML `<table>`),
`'json'`, `'jsonl'`, `'parquet'`,
`'pickle'` (pickled Pandas DataFrame),
`'sas'`, `'spss'`, `'stata'`, `'tsv'`.
- __experiment_name__ (str, default: `'experiment'`): name for
the experiment.
- __model_name__ (str, default: `'run'`): name of the model that is
being used.
- __model_resume_path__ (str, default: `None`): resumes training of
the model from the path specified. The model definition is restored.
In addition to model definition, training statistics, loss for each
epoch and the state of the optimizer are restored such that
training can be effectively continued from a previously interrupted
training process.
- __skip_save_training_description__ (bool, default: `False`):
disables saving the description JSON file.
- __skip_save_training_statistics__ (bool, default: `False`):
disables saving training statistics JSON file.
- __skip_save_model__ (bool, default: `False`): disables
saving model weights and hyperparameters each time the model
improves. By default Ludwig saves model weights after each epoch
the validation metric improves, but if the model is really big
that can be time consuming if you do not want to keep
the weights and just find out what performance can a model get
with a set of hyperparameters, use this parameter to skip it,
but the model will not be loadable later on and the returned model
will have the weights obtained at the end of training, instead of
the weights of the epoch with the best validation performance.
- __skip_save_progress__ (bool, default: `False`): disables saving
progress each epoch. By default Ludwig saves weights and stats
after each epoch for enabling resuming of training, but if
the model is really big that can be time consuming and will uses
twice as much space, use this parameter to skip it, but training
cannot be resumed later on.
- __skip_save_log__ (bool, default: `False`): disables saving
TensorBoard logs. By default Ludwig saves logs for the TensorBoard,
but if it is not needed turning it off can slightly increase the
overall speed.
- __skip_save_processed_input__ (bool, default: `False`): if input
dataset is provided it is preprocessed and cached by saving an HDF5
and JSON files to avoid running the preprocessing again. If this
parameter is `False`, the HDF5 and JSON file are not saved.
- __output_directory__ (str, default: `'results'`): the directory that
will contain the training statistics, TensorBoard logs, the saved
model and the training progress files.
- __random_seed__ (int, default: `42`): a random seed that will be
   used anywhere there is a call to a random number generator: data
   splitting, parameter initialization and training set shuffling
- __debug__ (bool, default: `False`):  if `True` turns on `tfdbg` with
`inf_or_nan` checks.


__Return__


- __return__ (Tuple[dict, Union[dict, pd.DataFrame], str]): tuple containing
`(training_statistics, preprocessed_data, output_directory)`.
`training_statistics` is a dictionary of training statistics
for each output feature containing loss and metrics values
for each epoch.
`preprocessed_data` is the tuple containing these three data sets
`(training_set, validation_set, test_set)`.
`output_directory` filepath to where training results are stored.
 
---
## train_online


```python
train_online(
  dataset,
  training_set_metadata=None,
  data_format='auto',
  random_seed=42,
  debug=False
)
```


Performs one epoch of training of the model on `dataset`.

__Inputs__


- __dataset__ (Union[str, dict, pandas.DataFrame], default: `None`):
source containing the entire dataset to be used in the experiment.
If it has a split column, it will be used for splitting (0 for train,
1 for validation, 2 for test), otherwise the dataset will be
randomly split.
- __training_set_metadata__ (Union[str, dict], default: `None`):
metadata JSON file or loaded metadata.  Intermediate preprocess
structure containing the mappings of the input
dataset created the first time an input file is used in the same
directory with the same name and a '.meta.json' extension.
- __data_format__ (str, default: `None`): format to interpret data
sources. Will be inferred automatically if not specified.  Valid
formats are `'auto'`, `'csv'`, `'df'`, `'dict'`, `'excel'`, `'feather'`,
`'fwf'`, `'hdf5'` (cache file produced during previous training),
`'html'` (file containing a single HTML `<table>`), `'json'`, `'jsonl'`,
`'parquet'`, `'pickle'` (pickled Pandas DataFrame), `'sas'`, `'spss'`,
`'stata'`, `'tsv'`.
- __random_seed__ (int, default: `42`): a random seed that is going to be
   used anywhere there is a call to a random number generator: data
   splitting, parameter initialization and training set shuffling
- __debug__ (bool, default: `False`): If `True` turns on `tfdbg`
    with `inf_or_nan` checks.

__Return__


- __return__ (None): `None`
 
----

# Module functions

----

## kfold_cross_validate


```python
ludwig.api.kfold_cross_validate(
  num_folds,
  config,
  dataset=None,
  data_format=None,
  skip_save_training_description=False,
  skip_save_training_statistics=False,
  skip_save_model=False,
  skip_save_progress=False,
  skip_save_log=False,
  skip_save_processed_input=False,
  skip_save_predictions=False,
  skip_save_eval_stats=False,
  skip_collect_predictions=False,
  skip_collect_overall_stats=False,
  output_directory='results',
  random_seed=42,
  gpus=None,
  gpu_memory_limit=None,
  allow_parallel_threads=True,
  use_horovod=None,
  logging_level=20,
  debug=False
)
```


Performs k-fold cross validation and returns result data structures.

__Inputs__


- __num_folds__ (int): number of folds to create for the cross-validation
- __config__ (Union[dict, str]): model specification
   required to build a model. Parameter may be a dictionary or string
   specifying the file path to a yaml configuration file.  Refer to the
   [User Guide](http://ludwig.ai/user_guide/#model-definition)
   for details.
- __dataset__ (Union[str, dict, pandas.DataFrame], default: `None`):
source containing the entire dataset to be used for k_fold processing.
- __data_format__ (str, default: `None`): format to interpret data
    sources. Will be inferred automatically if not specified.  Valid
    formats are `'auto'`, `'csv'`, `'df'`, `'dict'`, `'excel'`, `'feather'`,
    `'fwf'`,
    `'html'` (file containing a single HTML `<table>`), `'json'`, `'jsonl'`,
    `'parquet'`, `'pickle'` (pickled Pandas DataFrame), `'sas'`, `'spss'`,
    `'stata'`, `'tsv'`.  Currenlty `hdf5` format is not supported for
    k_fold cross validation.
- __skip_save_training_description__ (bool, default: `False`): disables
    saving the description JSON file.
- __skip_save_training_statistics__ (bool, default: `False`): disables
    saving training statistics JSON file.
- __skip_save_model__ (bool, default: `False`): disables
saving model weights and hyperparameters each time the model
improves. By default Ludwig saves model weights after each epoch
the validation metric improves, but if the model is really big
that can be time consuming if you do not want to keep
the weights and just find out what performance can a model get
with a set of hyperparameters, use this parameter to skip it,
but the model will not be loadable later on and the returned model
will have the weights obtained at the end of training, instead of
the weights of the epoch with the best validation performance.
- __skip_save_progress__ (bool, default: `False`): disables saving
   progress each epoch. By default Ludwig saves weights and stats
   after each epoch for enabling resuming of training, but if
   the model is really big that can be time consuming and will uses
   twice as much space, use this parameter to skip it, but training
   cannot be resumed later on.
- __skip_save_log__ (bool, default: `False`): disables saving TensorBoard
   logs. By default Ludwig saves logs for the TensorBoard, but if it
   is not needed turning it off can slightly increase the
   overall speed.
- __skip_save_processed_input__ (bool, default: `False`): if input
dataset is provided it is preprocessed and cached by saving an HDF5
and JSON files to avoid running the preprocessing again. If this
parameter is `False`, the HDF5 and JSON file are not saved.
- __skip_save_predictions__ (bool, default: `False`): skips saving test
    predictions CSV files.
- __skip_save_eval_stats__ (bool, default: `False`): skips saving test
    statistics JSON file.
- __skip_collect_predictions__ (bool, default: `False`): skips collecting
    post-processed predictions during eval.
- __skip_collect_overall_stats__ (bool, default: `False`): skips collecting
    overall stats during eval.
- __output_directory__ (str, default: `'results'`): the directory that
will contain the training statistics, TensorBoard logs, the saved
model and the training progress files.
- __random_seed__ (int, default: `42`): Random seed
    used for weights initialization,
   splits and any other random function.
- __gpus__ (list, default: `None`): list of GPUs that are available
    for training.
- __gpu_memory_limit__ (int, default: `None`): maximum memory in MB to
    allocate per GPU device.
- __allow_parallel_threads__ (bool, default: `True`): allow TensorFlow to
    use multithreading parallelism
   to improve performance at the cost of determinism.
- __use_horovod__ (bool, default: `None`): flag for using horovod
- __debug__ (bool, default: `False`): If `True` turns on tfdbg
    with `inf_or_nan` checks.
- __logging_level__ (int, default: INFO): log level to send to stderr.


__Return__


- __return__ (tuple(kfold_cv_statistics, kfold_split_indices), dict): a tuple of
    dictionaries `kfold_cv_statistics`: contains metrics from cv run.
     `kfold_split_indices`: indices to split training data into
     training fold and test fold.
 
----

## hyperopt


```python
ludwig.hyperopt.run.hyperopt(
  config,
  dataset=None,
  training_set=None,
  validation_set=None,
  test_set=None,
  training_set_metadata=None,
  data_format=None,
  experiment_name='hyperopt',
  model_name='run',
  skip_save_training_description=False,
  skip_save_training_statistics=False,
  skip_save_model=False,
  skip_save_progress=False,
  skip_save_log=False,
  skip_save_processed_input=False,
  skip_save_unprocessed_output=False,
  skip_save_predictions=False,
  skip_save_eval_stats=False,
  skip_save_hyperopt_statistics=False,
  output_directory='results',
  gpus=None,
  gpu_memory_limit=None,
  allow_parallel_threads=True,
  use_horovod=None,
  random_seed=42,
  debug=False
)
```


This method performs an hyperparameter optimization.

__Inputs__


- __config__ (Union[str, dict]): model definition which defines
the different parameters of the model, features, preprocessing and
training.  If `str`, filepath to yaml configuration file.
- __dataset__ (Union[str, dict, pandas.DataFrame], default: `None`):
source containing the entire dataset to be used in the experiment.
If it has a split column, it will be used for splitting (0 for train,
1 for validation, 2 for test), otherwise the dataset will be
randomly split.
- __training_set__ (Union[str, dict, pandas.DataFrame], default: `None`):
source containing training data.
- __validation_set__ (Union[str, dict, pandas.DataFrame], default: `None`):
source containing validation data.
- __test_set__ (Union[str, dict, pandas.DataFrame], default: `None`):
source containing test data.
- __training_set_metadata__ (Union[str, dict], default: `None`):
metadata JSON file or loaded metadata.  Intermediate preprocess
structure containing the mappings of the input
dataset created the first time an input file is used in the same
directory with the same name and a '.meta.json' extension.
- __data_format__ (str, default: `None`): format to interpret data
sources. Will be inferred automatically if not specified.  Valid
formats are `'auto'`, `'csv'`, `'df'`, `'dict'`, `'excel'`, `'feather'`,
`'fwf'`, `'hdf5'` (cache file produced during previous training),
`'html'` (file containing a single HTML `<table>`), `'json'`, `'jsonl'`,
`'parquet'`, `'pickle'` (pickled Pandas DataFrame), `'sas'`, `'spss'`,
`'stata'`, `'tsv'`.
- __experiment_name__ (str, default: `'experiment'`): name for
the experiment.
- __model_name__ (str, default: `'run'`): name of the model that is
being used.
- __skip_save_training_description__ (bool, default: `False`): disables
saving the description JSON file.
- __skip_save_training_statistics__ (bool, default: `False`): disables
saving training statistics JSON file.
- __skip_save_model__ (bool, default: `False`): disables
saving model weights and hyperparameters each time the model
improves. By default Ludwig saves model weights after each epoch
the validation metric improves, but if the model is really big
that can be time consuming if you do not want to keep
the weights and just find out what performance can a model get
with a set of hyperparameters, use this parameter to skip it,
but the model will not be loadable later on and the returned model
will have the weights obtained at the end of training, instead of
the weights of the epoch with the best validation performance.
- __skip_save_progress__ (bool, default: `False`): disables saving
progress each epoch. By default Ludwig saves weights and stats
after each epoch for enabling resuming of training, but if
the model is really big that can be time consuming and will uses
twice as much space, use this parameter to skip it, but training
cannot be resumed later on.
- __skip_save_log__ (bool, default: `False`): disables saving
TensorBoard logs. By default Ludwig saves logs for the TensorBoard,
but if it is not needed turning it off can slightly increase the
overall speed.
- __skip_save_processed_input__ (bool, default: `False`): if input
dataset is provided it is preprocessed and cached by saving an HDF5
and JSON files to avoid running the preprocessing again. If this
parameter is `False`, the HDF5 and JSON file are not saved.
- __skip_save_unprocessed_output__ (bool, default: `False`): by default
predictions and their probabilities are saved in both raw
unprocessed numpy files containing tensors and as postprocessed
CSV files (one for each output feature). If this parameter is True,
only the CSV ones are saved and the numpy ones are skipped.
- __skip_save_predictions__ (bool, default: `False`): skips saving test
predictions CSV files.
- __skip_save_eval_stats__ (bool, default: `False`): skips saving test
statistics JSON file.
- __skip_save_hyperopt_statistics__ (bool, default: `False`): skips saving
hyperopt stats file.
- __output_directory__ (str, default: `'results'`): the directory that
will contain the training statistics, TensorBoard logs, the saved
model and the training progress files.
- __gpus__ (list, default: `None`): list of GPUs that are available
for training.
- __gpu_memory_limit__ (int, default: `None`): maximum memory in MB to
allocate per GPU device.
- __allow_parallel_threads__ (bool, default: `True`): allow TensorFlow
to use multithreading parallelism to improve performance at
the cost of determinism.
- __use_horovod__ (bool, default: `None`): flag for using horovod.
- __random_seed__ (int: default: 42): random seed used for weights
initialization, splits and any other random function.
- __debug__ (bool, default: `False): if `True` turns on `tfdbg` with
`inf_or_nan` checks.

__Return__


- __return__ (List[dict]): The results for the hyperparameter optimization
 