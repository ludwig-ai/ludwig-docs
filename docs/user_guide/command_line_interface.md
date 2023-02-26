# Commands

Ludwig provides several functions through its command line interface.

| Mode                                          | Description                                                                       |
| --------------------------------------------- | --------------------------------------------------------------------------------- |
| [`train`](#train)                             | Trains a model                                                                    |
| [`predict`](#predict)                         | Predicts using a pretrained model                                                 |
| [`evaluate`](#evaluate)                       | Evaluate a pretrained model's performance                                         |
| [`experiment`](#experiment)                   | Runs a full experiment training a model and evaluating it                         |
| [`hyperopt`](#hyperopt)                       | Perform hyperparameter optimization                                               |
| [`serve`](#serve)                             | Serves a pretrained model                                                         |
| [`visualize`](#visualize)                     | Visualizes experiment results                                                     |
| [`init_config`](#init_config)                 | Initialize a user config from a dataset and targets                               |
| [`render_config`](#render_config)             | Renders the fully populated config with all defaults set                          |
| [`collect_summary`](#collect_summary)         | Prints names of weights and layers activations to use with other collect commands |
| [`collect_weights`](#collect_weights)         | Collects tensors containing a pretrained model weights                            |
| [`collect_activations`](#collect_activations) | Collects tensors for each datapoint using a pretrained model                      |
| [`export_torchscript`](#export_torchscript)   | Exports Ludwig models to Torchscript                                              |
| [`export_neuropod`](#export_neuropod)         | Exports Ludwig models to Neuropod                                                 |
| [`export_mlflow`](#export_mlflow)             | Exports Ludwig models to MLflow                                                   |
| [`preprocess`](#preprocess)                   | Preprocess data and saves it into HDF5 and JSON format                            |
| [`synthesize_dataset`](#synthesize_dataset)   | Creates synthetic data for testing purposes                                       |

These are described in detail below.

# train

Train a model from your data.

```bash
ludwig train [options]
```

or with:

```bash
python -m ludwig.train [options]
```

from within Ludwig's main directory.

These are the available arguments:

```
usage: ludwig train [options]

This script trains a model

optional arguments:
  -h, --help            show this help message and exit
  --output_directory OUTPUT_DIRECTORY
                        directory that contains the results
  --experiment_name EXPERIMENT_NAME
                        experiment name
  --model_name MODEL_NAME
                        name for the model
  --dataset DATASET     input data file path. If it has a split column, it
                        will be used for splitting (0: train, 1: validation,
                        2: test), otherwise the dataset will be randomly split
  --training_set TRAINING_SET
                        input train data file path
  --validation_set VALIDATION_SET
                        input validation data file path
  --test_set TEST_SET   input test data file path
  --training_set_metadata TRAINING_SET_METADATA
                        input metadata JSON file path. An intermediate
                        preprocessed  containing the mappings of the input
                        file created the first time a file is used, in the
                        same directory with the same name and a .json
                        extension
  --data_format {auto,csv,excel,feather,fwf,hdf5,htmltables,json,jsonl,parquet,pickle,sas,spss,stata,tsv}
                        format of the input data
  -sspi, --skip_save_processed_input
                        skips saving intermediate HDF5 and JSON files
  -c CONFIG, --config CONFIG
                        Path to the YAML file containing the model configuration
  -cs CONFIG_STR, --config_str CONFIG_STRING
                        JSON or YAML serialized string of the model configuration. Ignores --config
  -mlp MODEL_LOAD_PATH, --model_load_path MODEL_LOAD_PATH
                        path of a pretrained model to load as initialization
  -mrp MODEL_RESUME_PATH, --model_resume_path MODEL_RESUME_PATH
                        path of the model directory to resume training of
  -sstd, --skip_save_training_description
                        disables saving the description JSON file
  -ssts, --skip_save_training_statistics
                        disables saving training statistics JSON file
  -ssm, --skip_save_model
                        disables saving weights each time the model improves.
                        By default Ludwig saves weights after each epoch the
                        validation metric improves, but if the model is really
                        big that can be time consuming. If you do not want to
                        keep the weights and just find out what performance
                        can a model get with a set of hyperparameters, use
                        this parameter to skip it
  -ssp, --skip_save_progress
                        disables saving weights after each epoch. By default
                        ludwig saves weights after each epoch for enabling
                        resuming of training, but if the model is really big
                        that can be time consuming and will save twice as much
                        space, use this parameter to skip it
  -ssl, --skip_save_log
                        disables saving TensorBoard logs. By default Ludwig
                        saves logs for the TensorBoard, but if it is not
                        needed turning it off can slightly increase the
                        overall speed
  -rs RANDOM_SEED, --random_seed RANDOM_SEED
                        a random seed that is going to be used anywhere there
                        is a call to a random number generator: data
                        splitting, parameter initialization and training set
                        shuffling
  -g GPUS [GPUS ...], --gpus GPUS [GPUS ...]
                        list of gpus to use
  -gml GPU_MEMORY_LIMIT, --gpu_memory_limit GPU_MEMORY_LIMIT
                        maximum memory in MB to allocate per GPU device
  -dpt, --disable_parallel_threads
                        disable Torch from using multithreading for
                        reproducibility
  -b BACKEND, --backend BACKEND
                        specifies backend to use for parallel / distributed execution,
                        defaults to local execution or Horovod if called using horovodrun
```

When Ludwig trains a model it creates two intermediate files, one HDF5 and one JSON.
The HDF5 file contains the data mapped to numpy ndarrays, while the JSON file
contains the mappings from the values in the tensors to their original labels.

For instance, for a categorical feature with 3 possible values, the HDF5 file
will contain integers from 0 to 3 (with 0 being a `<UNK>` category), while the
JSON file will contain a `idx2str` list containing all tokens
(`[<UNK>, label_1, label_2, label_3]`), a `str2idx` dictionary
(`{"<UNK>": 0, "label_1": 1, "label_2": 2, "label_3": 3}`) and a `str2freq`
dictionary (`{"<UNK>": 0, "label_1": 93, "label_2": 55, "label_3": 24}`).

The reason to have those  intermediate files is two-fold: on one hand, if you are going to train your model again Ludwig will try to load them instead of recomputing all tensors, which saves a considerable amount of time, and on the other hand when you want to use your model to predict, data has to be mapped to tensors in exactly the same way it was mapped during training, so you'll be required to load the JSON metadata file in the `predict` command.

The first time you provide a UTF-8 encoded dataset (`--dataset`), the HDF5 and JSON files are created, from the second time on Ludwig will load them instead of the dataset even if you specify the dataset (it looks in the same directory for files names in the same way but with a different extension), finally you can directly specify the HDF5 and JSON files.

As the mapping from raw data to tensors depends on the type of feature that you specify in your configuration, if you change type (for instance from `sequence` to `text`) you also have to redo the preprocessing, which is achieved by deleting the HDF5 and JSON files.
Alternatively you can skip saving the HDF5 and JSON files specifying `--skip_save_processed_input`.

Splitting between train, validation and test set can be done in several ways.
This allows for a few possible input data scenarios:

- one single UTF-8 encoded dataset file is provided (`-dataset`). In this case if the dataset contains a `split` column with values `0` for training, `1` for validation and `2` for test, this split will be used. If you want to ignore the split column and perform a random split, use a `force_split` argument in the configuration. In the case when there is no split column, a random `70-20-10` split will be performed. You can set the percentages and specify if you want stratified sampling in the configuration preprocessing section.
- you can provide separate UTF-8 encoded training, validation and test sets  (`--training_set`, `--validation_set`, `--test_set`).
- the HDF5 and JSON file indications specified in the case of a single dataset file apply also in the multiple files case, with the only difference that you need to specify only one JSON file (`--train_set_metadata_json`).

The validation set is optional, but if absent the training will continue until the end of the training epochs, while when there's a validation set the default behavior is to perform early stopping after the validation measure does not improve for a certain amount of epochs. The test set is optional too.

Other optional arguments are `--output_directory`, `--experiment_name` and `--model name`.
By default the output directory is `./results`.
That directory will contain a directory named `[experiment_name]_[model_name]_0`
if model name and experiment name are specified.
If the same combination of experiment and model name is used again, the integer
at the end of the name will be increased.
If neither of them is specified the directory will be named `run_0`.
The directory will contain

- `description.json` - a file containing a description of the training process with all the information to reproduce it.
- `training_statistics.json` - a file containing records of all measures and losses for each epoch.
- `model` - a directory containing model hyperparameters, weights, checkpoints and logs (for TensorBoard).

The configuration can be provided either as a string (`--config_str`)
or as YAML file (`--config`).

Details on how to write your configuration are provided in the [Configuration](#configuration) section.

During training Ludwig saves two sets of weights for the model, one that is the
weights at the end of the epoch where the best performance on the validation
measure was achieved and one that is the weights at the end of the latest epoch.
The reason for keeping the second set is to be able to resume training in case
the training process gets interrupted somehow.

To resume training using the latest weights and the whole history of progress so far you have to specify the `--model_resume_path` argument.
You can avoid saving the latest weights and the overall progress so far by using the argument `--skip_save_progress`, but you will not be able to resume it afterwards.

Another available option is to load a previously trained model as an initialization for a new training process.
In this case Ludwig will start a new training process, without knowing any progress of the previous model, no training statistics, nor the number of epochs the model has been trained on so far.

It's not resuming training, just initializing training with a previously trained model with the same configuration, and it is accomplished through the `--model_load_path` argument.

You can specify a random seed to be used by the python environment, python random package, numpy and Torch with the `--random_seed` argument.
This is useful for reproducibility.

Be aware that due to asynchronicity in the Torch's GPU execution, when training on GPU results may not be reproducible.

You can manage which GPUs on your machine are used with the `--gpus` argument, which accepts a string identical to the format of `CUDA_VISIBLE_DEVICES` environment variable, namely a list of integers separated by comma.
You can also specify the maximum amount of GPU memory which will be allocated per device with `--gpu_memory_limit`.
By default all of memory is allocated.
If less than all of memory is allocated, Torch will need more GPU memory it will try to increase this amount.

If parameter `--backend` is set, will use the given backend for distributed processing (Horovod or Ray).

Finally the `--logging_level` argument lets you set the amount of logging that you want to see during training.

Example:

```bash
ludwig train --dataset reuters-allcats.csv --config "{input_features: [{name: text, type: text, encoder: {type: parallel_cnn}}], output_features: [{name: class, type: category}]}"
```

# predict

This command lets you use a previously trained model to predict on new data.
You can call it with:

```bash
ludwig predict [options]
```

or with:

```bash
python -m ludwig.predict [options]
```

from within Ludwig's main directory.

These are the available arguments:

```
usage: ludwig predict [options]

This script loads a pretrained model and uses it to predict

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     input data file path
  --data_format {auto,csv,excel,feather,fwf,hdf5,htmltables,json,jsonl,parquet,pickle,sas,spss,stata,tsv}
                        format of the input data
  -s {training,validation,test,full}, --split {training,validation,test,full}
                        the split to test the model on
  -m MODEL_PATH, --model_path MODEL_PATH
                        model to load
  -od OUTPUT_DIRECTORY, --output_directory OUTPUT_DIRECTORY
                        directory that contains the results
  -ssuo, --skip_save_unprocessed_output
                        skips saving intermediate NPY output files
  -sstp, --skip_save_predictions
                        skips saving predictions CSV files
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        size of batches
  -g GPUS, --gpus GPUS  list of gpu to use
  -gml GPU_MEMORY_LIMIT, --gpu_memory_limit GPU_MEMORY_LIMIT
                        maximum memory in MB to allocate per GPU device
  -dpt, --disable_parallel_threads
                        disable Torch from using multithreading for
                        reproducibility
  -b BACKEND, --backend BACKEND
                        specifies backend to use for parallel / distributed execution,
                        defaults to local execution or Horovod if called using horovodrun
  -dbg, --debug         enables debugging mode
  -l {critical,error,warning,info,debug,notset}, --logging_level {critical,error,warning,info,debug,notset}
                        the level of logging to use
```

The same distinction between UTF-8 encoded dataset files and HDF5 / JSON files explained in the [train](#train) section also applies here.
In either case, the JSON metadata file obtained during training is needed in order to map the new data into tensors.
If the new data contains a split column, you can specify which split to use to calculate the predictions with the `--split` argument. By default it's `full` which means all the splits will be used.

A model to load is needed, and you can specify its path with the `--model_path` argument.
If you trained a model previously and got the results in, for instance,
`./results/experiment_run_0`, you have to specify
`./results/experiment_run_0/model` for using it to predict.

You can specify an output directory with the argument `--output-directory`, by
default it will be `./result_0`, with increasing numbers if a directory with the same name is present.

The directory will contain a prediction CSV file and a probability CSV file for
each output feature, together with raw NPY files containing raw tensors.
You can specify not to save the raw NPY output files with the argument `skip_save_unprocessed_output`.

A specific batch size for speeding up the prediction can be specified using the argument `--batch_size`.

Finally the `--logging_level`, `--debug`, `--gpus`, `--gpu_memory_limit` and `--disable_parallel_threads`  related arguments behave exactly like described in the train command section.

Example:

```bash
ludwig predict --dataset reuters-allcats.csv --model_path results/experiment_run_0/model/
```

# evaluate

This command lets you use a previously trained model to predict on new data and
evaluate the performance of the prediction compared to ground truth.
You can call it with:

```bash
ludwig evaluate [options]
```

or with:

```bash
python -m ludwig.evaluate [options]
```

from within Ludwig's main directory.

These are the available arguments:

```
usage: ludwig evaluate [options]

This script loads a pretrained model and evaluates its performance by
comparing its predictions with ground truth.

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     input data file path
  --data_format {auto,csv,excel,feather,fwf,hdf5,htmltables,json,jsonl,parquet,pickle,sas,spss,stata,tsv}
                        format of the input data
  -s {training,validation,test,full}, --split {training,validation,test,full}
                        the split to test the model on
  -m MODEL_PATH, --model_path MODEL_PATH
                        model to load
  -od OUTPUT_DIRECTORY, --output_directory OUTPUT_DIRECTORY
                        directory that contains the results
  -ssuo, --skip_save_unprocessed_output
                        skips saving intermediate NPY output files
  -sses, --skip_save_eval_stats
                        skips saving intermediate JSON eval statistics
  -scp, --skip_collect_predictions
                        skips collecting predictions
  -scos, --skip_collect_overall_stats
                        skips collecting overall stats
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        size of batches
  -g GPUS, --gpus GPUS  list of gpu to use
  -gml GPU_MEMORY_LIMIT, --gpu_memory_limit GPU_MEMORY_LIMIT
                        maximum memory in MB to allocate per GPU device
  -dpt, --disable_parallel_threads
                        disable Torch from using multithreading for
                        reproducibility
  -b BACKEND, --backend BACKEND
                        specifies backend to use for parallel / distributed execution,
                        defaults to local execution or Horovod if called using horovodrun
  -dbg, --debug         enables debugging mode
  -l {critical,error,warning,info,debug,notset}, --logging_level {critical,error,warning,info,debug,notset}
                        the level of logging to use
```

All parameters are the same of [predict](#predict) and the behavior is the same.
The only difference isthat `evaluate` requires the dataset to contain also columns with the same name of output features.
This is needed because `evaluate` compares the predictions produced by the model with the ground truth and will save all those statistics in a `test_statistics.json` file in the result directory.

Note that the data must contain columns for each output feature with ground
truth output values in order to compute the performance statistics.
If you receive an error regarding a missing output feature column in your data,
it means that the data does not contain the columns for each output feature to use as ground truth.

Example:

```bash
ludwig evaluate --dataset reuters-allcats.csv --model_path results/experiment_run_0/model/
```

# experiment

This command combines training and evaluation into a single handy command. You
can request a k-fold cross validation run by specifying the `--k_fold`
parameter.

You can call it with:

```bash
ludwig experiment [options]
```

or with:

```bash
python -m ludwig.experiment [options]
```

from within Ludwig's main directory.

These are the available arguments:

```
usage: ludwig experiment [options]

This script trains and evaluates a model

optional arguments:
  -h, --help            show this help message and exit
  --output_directory OUTPUT_DIRECTORY
                        directory that contains the results
  --experiment_name EXPERIMENT_NAME
                        experiment name
  --model_name MODEL_NAME
                        name for the model
  --dataset DATASET     input data file path. If it has a split column, it
                        will be used for splitting (0: train, 1: validation,
                        2: test), otherwise the dataset will be randomly split
  --training_set TRAINING_SET
                        input train data file path
  --validation_set VALIDATION_SET
                        input validation data file path
  --test_set TEST_SET   input test data file path
  --training_set_metadata TRAINING_SET_METADATA
                        input metadata JSON file path. An intermediate
                        preprocessed  containing the mappings of the input
                        file created the first time a file is used, in the
                        same directory with the same name and a .json
                        extension
  --data_format {auto,csv,excel,feather,fwf,hdf5,htmltables,json,jsonl,parquet,pickle,sas,spss,stata,tsv}
                        format of the input data
  -es {training,validation,test,full}, --eval_split {training,validation,test,full}
                        the split to evaluate the model on
  -sspi, --skip_save_processed_input
                        skips saving intermediate HDF5 and JSON files
  -ssuo, --skip_save_unprocessed_output
                        skips saving intermediate NPY output files
  -kf K_FOLD, --k_fold K_FOLD
                        number of folds for a k-fold cross validation run
  -skfsi, --skip_save_k_fold_split_indices
                        disables saving indices generated to split training
                        data set for the k-fold cross validation run, but if
                        it is not needed turning it off can slightly increase
                        the overall speed
  -c CONFIG, --config CONFIG
                        Path to the YAML file containing the model configuration
  -cs CONFIG_STR, --config_str CONFIG_STRING
                        JSON or YAML serialized string of the model configuration. Ignores --config
  -mlp MODEL_LOAD_PATH, --model_load_path MODEL_LOAD_PATH
                        path of a pretrained model to load as initialization
  -mrp MODEL_RESUME_PATH, --model_resume_path MODEL_RESUME_PATH
                        path of the model directory to resume training of
  -sstd, --skip_save_training_description
                        disables saving the description JSON file
  -ssts, --skip_save_training_statistics
                        disables saving training statistics JSON file
  -sstp, --skip_save_predictions
                        skips saving test predictions CSV files
  -sstes, --skip_save_eval_stats
                        skips saving eval statistics JSON file
  -ssm, --skip_save_model
                        disables saving model weights and hyperparameters each
                        time the model improves. By default Ludwig saves model
                        weights after each epoch the validation metric
                        improves, but if the model is really big that can be
                        time consuming if you do not want to keep the weights
                        and just find out what performance a model can get
                        with a set of hyperparameters, use this parameter to
                        skip it,but the model will not be loadable later on
  -ssp, --skip_save_progress
                        disables saving progress each epoch. By default Ludwig
                        saves weights and stats after each epoch for enabling
                        resuming of training, but if the model is really big
                        that can be time consuming and will uses twice as much
                        space, use this parameter to skip it, but training
                        cannot be resumed later on
  -ssl, --skip_save_log
                        disables saving TensorBoard logs. By default Ludwig
                        saves logs for the TensorBoard, but if it is not
                        needed turning it off can slightly increase the
                        overall speed
  -rs RANDOM_SEED, --random_seed RANDOM_SEED
                        a random seed that is going to be used anywhere there
                        is a call to a random number generator: data
                        splitting, parameter initialization and training set
                        shuffling
  -g GPUS [GPUS ...], --gpus GPUS [GPUS ...]
                        list of GPUs to use
  -gml GPU_MEMORY_LIMIT, --gpu_memory_limit GPU_MEMORY_LIMIT
                        maximum memory in MB to allocate per GPU device
  -dpt, --disable_parallel_threads
                        disable Torch from using multithreading for
                        reproducibility
  -b BACKEND, --backend BACKEND
                        specifies backend to use for parallel / distributed execution,
                        defaults to local execution or Horovod if called using horovodrun
  -dbg, --debug         enables debugging mode
  -l {critical,error,warning,info,debug,notset}, --logging_level {critical,error,warning,info,debug,notset}
                        the level of logging to use
```

The parameters combine parameters from both [train](#train) and [test](#test) so
refer to those sections for an in depth explanation.
The output directory will contain the outputs both commands produce.

Example:

```bash
ludwig experiment --dataset reuters-allcats.csv --config "{input_features: [{name: text, type: text, encoder: {type: parallel_cnn}}], output_features: [{name: class, type: category}]}"
```

# hyperopt

This command lets you perform an hyperparameter search with a given sampler and parameters.
You can call it with:

```bash
ludwig hyperopt [options]
```

or with:

```bash
python -m ludwig.hyperopt [options]
```

from within Ludwig's main directory.

These are the available arguments:

```bash
usage: ludwig hyperopt [options]

This script searches for optimal Hyperparameters

optional arguments:
  -h, --help            show this help message and exit
  -sshs, --skip_save_hyperopt_statistics
                        skips saving hyperopt statistics file
  --output_directory OUTPUT_DIRECTORY
                        directory that contains the results
  --experiment_name EXPERIMENT_NAME
                        experiment name
  --model_name MODEL_NAME
                        name for the model
  --dataset DATASET     input data file path. If it has a split column, it
                        will be used for splitting (0: train, 1: validation,
                        2: test), otherwise the dataset will be randomly split
  --training_set TRAINING_SET
                        input train data file path
  --validation_set VALIDATION_SET
                        input validation data file path
  --test_set TEST_SET   input test data file path
  --training_set_metadata TRAINING_SET_METADATA
                        input metadata JSON file path. An intermediate
                        preprocessed file containing the mappings of the input
                        file created the first time a file is used, in the
                        same directory with the same name and a .json
                        extension
  --data_format {auto,csv,excel,feather,fwf,hdf5,htmltables,json,jsonl,parquet,pickle,sas,spss,stata,tsv}
                        format of the input data
  -sspi, --skip_save_processed_input
                        skips saving intermediate HDF5 and JSON files
  -c CONFIG, --config CONFIG
                        Path to the YAML file containing the model configuration
  -cs CONFIG_STR, --config_str CONFIG_STRING
                        JSON or YAML serialized string of the model configuration. Ignores --config
  -mlp MODEL_LOAD_PATH, --model_load_path MODEL_LOAD_PATH
                        path of a pretrained model to load as initialization
  -mrp MODEL_RESUME_PATH, --model_resume_path MODEL_RESUME_PATH
                        path of the model directory to resume training of
  -sstd, --skip_save_training_description
                        disables saving the description JSON file
  -ssts, --skip_save_training_statistics
                        disables saving training statistics JSON file
  -ssm, --skip_save_model
                        disables saving weights each time the model improves.
                        By default Ludwig saves weights after each epoch the
                        validation metric improves, but if the model is really
                        big that can be time consuming. If you do not want to
                        keep the weights and just find out what performance
                        can a model get with a set of hyperparameters, use
                        this parameter to skip it
  -ssp, --skip_save_progress
                        disables saving weights after each epoch. By default
                        ludwig saves weights after each epoch for enabling
                        resuming of training, but if the model is really big
                        that can be time consuming and will save twice as much
                        space, use this parameter to skip it
  -ssl, --skip_save_log
                        disables saving TensorBoard logs. By default Ludwig
                        saves logs for the TensorBoard, but if it is not
                        needed turning it off can slightly increase the
                        overall speed
  -rs RANDOM_SEED, --random_seed RANDOM_SEED
                        a random seed that is going to be used anywhere there
                        is a call to a random number generator: data
                        splitting, parameter initialization and training set
                        shuffling
  -g GPUS [GPUS ...], --gpus GPUS [GPUS ...]
                        list of gpus to use
  -gml GPU_MEMORY_LIMIT, --gpu_memory_limit GPU_MEMORY_LIMIT
                        maximum memory in MB to allocate per GPU device
  -b BACKEND, --backend BACKEND
                        specifies backend to use for parallel / distributed execution,
                        defaults to local execution or Horovod if called using horovodrun
  -dbg, --debug         enables debugging mode
  -l {critical,error,warning,info,debug,notset}, --logging_level {critical,error,warning,info,debug,notset}
                        the level of logging to use
```

The parameters combine parameters from both [train](#train) and [test](#test) so refer to those sections for an in depth explanation. The output directory will contain a `hyperopt_statistics.json` file that summarizes the results obtained.

In order to perform an hyperparameter optimization, the `hyperopt` section needs to be provided within the configuration.
In the `hyperopt` section you will be able to define what metric to optimize, what parameters, what sampler to use to optimize them and how to execute the optimization.
For details on the `hyperopt` section see the detailed description in the [Hyperparameter Optimization](#hyperparameter-optimization) section.

# serve

This command lets you load a pre-trained model and serve it on an http server.

You can call it with:

```bash
ludwig serve [options]
```

or with

```bash
python -m ludwig.serve [options]
```

from within Ludwig's main directory.

These are the available arguments:

```
usage: ludwig serve [options]

This script serves a pretrained model

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PATH, --model_path MODEL_PATH
                        model to load
  -l {critical,error,warning,info,debug,notset}, --logging_level {critical,error,warning,info,debug,notset}
                        the level of logging to use
  -p PORT, --port PORT  port for server (default: 8000)
  -H HOST, --host HOST  host for server (default: 0.0.0.0)
```

The most important argument is `--model_path` where you have to specify the path of the model to load.

Once running, you can make a POST request on the `/predict` endpoint to run inference on the form data submitted.

!!! note

    `ludwig serve` will automatically use GPUs for serving, if avaiable to the
    machine-local torch environment.

## Example curl

__File__

`curl http://0.0.0.0:8000/predict -X POST -F 'image_path=@path_to_image/example.png'`

__Text__

`curl http://0.0.0.0:8000/predict -X POST -F 'english_text=words to be translated'`

__Both Text and File__

`curl http://0.0.0.0:8000/predict -X POST -F 'text=mixed together with' -F 'image=@path_to_image/example.png'`

__Batch prediction__

You can also make a POST request on the `/batch_predict` endpoint to run inference on multiple samples at once.

Requests must be submitted as form data, with one of fields being `dataset`: a JSON encoded string representation of the data to be predicted.

The `dataset` JSON string is expected to be in the Pandas "split" format to reduce payload size. This format divides the dataset into three parts:

1. columns: `List[str]`
1. index (optional): `List[Union[str, int]]`
1. data: `List[List[object]]`

Additional form fields can be used to provide file resources like images that are referenced within the dataset.

Batch prediction example:

`curl http://0.0.0.0:8000/batch_predict -X POST -F 'dataset={"columns": ["a", "b"], "data": [[1, 2], [3, 4]]}'`

# visualize

This command lets you visualize training and prediction statistics, alongside with comparing different models performances and predictions.
You can call it with:

```bash
ludwig visualize [options]
```

or with:

```bash
python -m ludwig.visualize [options]
```

from within Ludwig's main directory.

These are the available arguments:

```
usage: ludwig visualize [options]

This script analyzes results and shows some nice plots.

optional arguments:
  -h, --help            show this help message and exit
  -g GROUND_TRUTH, --ground_truth GROUND_TRUTH
                        ground truth file
  -gm GROUND_TRUTH_METADATA, --ground_truth_metadata GROUND_TRUTH_METADATA
                        input metadata JSON file
  -od OUTPUT_DIRECTORY, --output_directory OUTPUT_DIRECTORY
                        directory where to save plots.If not specified, plots
                        will be displayed in a window
  -ff {pdf,png}, --file_format {pdf,png}
                        file format of output plots
  -v {binary_threshold_vs_metric,calibration_1_vs_all,calibration_multiclass,compare_classifiers_multiclass_multimetric,compare_classifiers_performance_changing_k,compare_classifiers_performance_from_pred,compare_classifiers_performance_from_prob,compare_classifiers_performance_subset,compare_classifiers_predictions,compare_classifiers_predictions_distribution,compare_performance,confidence_thresholding,confidence_thresholding_2thresholds_2d,confidence_thresholding_2thresholds_3d,confidence_thresholding_data_vs_acc,confidence_thresholding_data_vs_acc_subset,confidence_thresholding_data_vs_acc_subset_per_class,confusion_matrix,frequency_vs_f1,hyperopt_hiplot,hyperopt_report,learning_curves,roc_curves,roc_curves_from_test_statistics}, --visualization {binary_threshold_vs_metric,calibration_1_vs_all,calibration_multiclass,compare_classifiers_multiclass_multimetric,compare_classifiers_performance_changing_k,compare_classifiers_performance_from_pred,compare_classifiers_performance_from_prob,compare_classifiers_performance_subset,compare_classifiers_predictions,compare_classifiers_predictions_distribution,compare_performance,confidence_thresholding,confidence_thresholding_2thresholds_2d,confidence_thresholding_2thresholds_3d,confidence_thresholding_data_vs_acc,confidence_thresholding_data_vs_acc_subset,confidence_thresholding_data_vs_acc_subset_per_class,confusion_matrix,frequency_vs_f1,hyperopt_hiplot,hyperopt_report,learning_curves,roc_curves,roc_curves_from_test_statistics}
                        type of visualization
  -f OUTPUT_FEATURE_NAME, --output_feature_name OUTPUT_FEATURE_NAME
                        name of the output feature to visualize
  -gts GROUND_TRUTH_SPLIT, --ground_truth_split GROUND_TRUTH_SPLIT
                        ground truth split - 0:train, 1:validation, 2:test
                        split
  -tf THRESHOLD_OUTPUT_FEATURE_NAMES [THRESHOLD_OUTPUT_FEATURE_NAMES ...], --threshold_output_feature_names THRESHOLD_OUTPUT_FEATURE_NAMES [THRESHOLD_OUTPUT_FEATURE_NAMES ...]
                        names of output features for 2d threshold
  -pred PREDICTIONS [PREDICTIONS ...], --predictions PREDICTIONS [PREDICTIONS ...]
                        predictions files
  -prob PROBABILITIES [PROBABILITIES ...], --probabilities PROBABILITIES [PROBABILITIES ...]
                        probabilities files
  -trs TRAINING_STATISTICS [TRAINING_STATISTICS ...], --training_statistics TRAINING_STATISTICS [TRAINING_STATISTICS ...]
                        training stats files
  -tes TEST_STATISTICS [TEST_STATISTICS ...], --test_statistics TEST_STATISTICS [TEST_STATISTICS ...]
                        test stats files
  -hs HYPEROPT_STATS_PATH, --hyperopt_stats_path HYPEROPT_STATS_PATH
                        hyperopt stats file
  -mn MODEL_NAMES [MODEL_NAMES ...], --model_names MODEL_NAMES [MODEL_NAMES ...]
                        names of the models to use as labels
  -tn TOP_N_CLASSES [TOP_N_CLASSES ...], --top_n_classes TOP_N_CLASSES [TOP_N_CLASSES ...]
                        number of classes to plot
  -k TOP_K, --top_k TOP_K
                        number of elements in the ranklist to consider
  -ll LABELS_LIMIT, --labels_limit LABELS_LIMIT
                        maximum numbers of labels. If labels in dataset are
                        higher than this number, "rare" label
  -ss {ground_truth,predictions}, --subset {ground_truth,predictions}
                        type of subset filtering
  -n, --normalize       normalize rows in confusion matrix
  -m METRICS [METRICS ...], --metrics METRICS [METRICS ...]
                        metrics to display in threshold_vs_metric
  -pl POSITIVE_LABEL, --positive_label POSITIVE_LABEL
                        label of the positive class for the roc curve
  -l {critical,error,warning,info,debug,notset}, --logging_level {critical,error,warning,info,debug,notset}
                        the level of logging to use
```

As the `--visualization` parameters suggests, there is a vast number of visualizations readily available.
Each of them requires a different subset of this command's arguments, so they will be described one by one in the [Visualizations](#visualizations) section.

# init_config

Initialize a user config from a dataset and targets.

```
usage: ludwig init_config [options]

This script initializes a valid config from a dataset.

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        input data file path
  -t TARGET, --target TARGET
                        target(s) to predict as output features of the model
  --time_limit_s TIME_LIMIT_S
                        time limit to train the model in seconds when using hyperopt
  --suggested SUGGESTED
                        use suggested config from automl, otherwise only use inferred types and return a minimal config
  --hyperopt HYPEROPT   include automl hyperopt config
  --random_seed RANDOM_SEED
                        seed for random number generators used in hyperopt to improve repeatability
  --use_reference_config USE_REFERENCE_CONFIG
                        refine hyperopt search space by setting first search point from stored reference model config
  -o OUTPUT, --output OUTPUT
                        output initialized YAML config path
```

# render_config

Renders the fully populated config with all defaults set.

```
usage: ludwig render_config [options]

This script renders the full config from a user config.

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        output rendered YAML config path
```

# collect_summary

This command loads a pretrained model and prints names of weights and layers activations to use with `collect_weights` or `collect_activations`.

```bash
ludwig collect_summary [options]
```

or with:

```bash
python -m ludwig.collect names [options]
```

from within Ludwig's main directory.

These are the available arguments:

```
usage: ludwig collect_summary [options]

This script loads a pretrained model and print names of weights and layer activations.

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PATH, --model_path MODEL_PATH
                        model to load
  -l {critical,error,warning,info,debug,notset}, --logging_level {critical,error,warning,info,debug,notset}
                        the level of logging to use
```

# collect_weights

This command lets you load a pre-trained model and collect the tensors with a specific name in order to save them in a NPY format.
This may be useful in order to visualize the learned weights (for instance collecting embedding matrices) and for some post-hoc analyses.
You can call it with:

```bash
ludwig collect_weights [options]
```

or with:

```bash
python -m ludwig.collect weights [options]
```

from within Ludwig's main directory.

These are the available arguments:

```
usage: ludwig collect_weights [options]

This script loads a pretrained model and uses it collect weights.

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PATH, --model_path MODEL_PATH
                        model to load
  -t TENSORS [TENSORS ...], --tensors TENSORS [TENSORS ...]
                        tensors to collect
  -od OUTPUT_DIRECTORY, --output_directory OUTPUT_DIRECTORY
                        directory that contains the results
  -dbg, --debug         enables debugging mode
  -l {critical,error,warning,info,debug,notset}, --logging_level {critical,error,warning,info,debug,notset}
                        the level of logging to use
```

The three most important arguments are `--model_path` where you have to specify the path of the model to load, `--tensors` that lets you specify a list of tensor names in the Torch graph that contain the weights you want to collect, and finally `--output_directory` that lets you specify where the NPY files (one for each tensor name specified) will be saved.

In order to figure out the names of the tensors containing the weights you want
to collect, use the `collect_summary` command.

# collect_activations

This command lets you load a pre-trained model and input data and collects the
values of activations contained in tensors with a specific name in order to save
them in a NPY format.

This may be useful in order to visualize the activations (for instance
collecting the last layer's activations as embeddings representations of the
input datapoint) and for some post-hoc analyses.

You can call it with:

```bash
ludwig collect_activations [options]
```

or with:

```bash
python -m ludwig.collect activations [options]
```

from within Ludwig's main directory.

These are the available arguments:

```
usage: ludwig collect_activations [options]

This script loads a pretrained model and uses it collect tensors for each
datapoint in the dataset.

optional arguments:
  -h, --help            show this help message and exit
  --dataset  DATASET    filepath for input dataset
  --data_format DATA_FORMAT  format of the dataset.  Valid values are auto,
                        csv, excel, feature, fwf, hdf5, html, tables, json,
                        json, jsonl, parquet, pickle, sas, spss, stata, tsv
  -s {training,validation,test,full}, --split {training,validation,test,full}
                        the split to test the model on
  -m MODEL_PATH, --model_path MODEL_PATH
                        model to load
  -lyr LAYER [LAYER ..], --layers LAYER [LAYER ..]
                        layers to collect
  -od OUTPUT_DIRECTORY, --output_directory OUTPUT_DIRECTORY
                        directory that contains the results
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        size of batches
  -g GPUS, --gpus GPUS  list of gpu to use
  -gml GPU_MEMORY, --gpu_memory_limit GPU_MEMORY
                        maximum memory in MB of gpu memory to allocate per
                        GPU device
  -dpt, --disable_parallel_threads
                        disable Torch from using multithreading
                        for reproducibility
  -b BACKEND, --backend BACKEND
                        specifies backend to use for parallel / distributed execution,
                        defaults to local execution or Horovod if called using horovodrun
  -dbg, --debug         enables debugging mode
  -l {critical,error,warning,info,debug,notset}, --logging_level {critical,error,warning,info,debug,notset}
                        the level of logging to use
```

The data related and runtime related arguments (GPUs, batch size, etc.) are the
same as the ones used in [predict](#predict), you can refer to that section for
an explanation.

The collect-specific arguments, `--model_path`, `--tensors` and
`--output_directory`, are the same used in [collect_weights](#collect_weights),
you can refer to that section for an explanation.

# export_torchscript

Exports a pre-trained model to Torch's `torchscript` format.

```bash
ludwig export_torchscript [options]
```

or with:

```bash
python -m ludwig.export torchscript [options]
```

These are the available arguments:

```
usage: ludwig export_torchscript [options]

This script loads a pretrained model and saves it as torchscript.

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PATH, --model_path MODEL_PATH
                        model to load
  -mo, --model_only     Script and export the model only.
  -d DEVICE, --device DEVICE
                        Device to use for torchscript tracing (e.g. "cuda" or "cpu"). Ideally, this is the same as the device used
                        when the model is loaded.
  -op OUTPUT_PATH, --output_path OUTPUT_PATH
                        path where to save the export model. If not specified, defaults to model_path.
  -l {critical,error,warning,info,debug,notset}, --logging_level {critical,error,warning,info,debug,notset}
                        the level of logging to use
```

For more information, see [TorchScript Export](/user_guide/model_export/#torchscript-export)

# export_neuropod

A Ludwig model can be exported as a [Neuropod](https://github.com/uber/neuropod), a mechanism that allows it to be executed in a framework agnostic way.

In order to export a Ludwig model as a Neuropod, first make sure the `neuropod` package is installed in your environment together with the appropriate backend (only use Python 3.7+), then run the following command:

```bash
ludwig export_neuropod [options]
```

or with:

```bash
python -m ludwig.export neuropod [options]
```

These are the available arguments:

```
usage: ludwig export_neuropod [options]

This script loads a pretrained model and uses it collect weights.

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PATH, --model_path MODEL_PATH
                        model to load
  -mn MODEL_NAME, --model_name MODEL_NAME
                        model name
  -od OUTPUT_PATH, --output_path OUTPUT_PATH
                        path where to save the export model
  -l {critical,error,warning,info,debug,notset}, --logging_level {critical,error,warning,info,debug,notset}
                        the level of logging to use
```

This functionality has been tested with `neuropod==0.2.0`.

# export_mlflow

A Ludwig model can be exported as an [mlflow.pyfunc](https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html) model, which allows it to be executed in a framework agnostic way.

There are two ways to export a Ludwig model to MLflow:

1. Convert a saved model directory on disk to the MLflow format on disk.
1. Register a saved model directory on disk or in an existing MLflow experiment to an MLflow model registry.

For the first approach, you only need to provide the location of the saved Ludwig model locally and the location where the model should be written to on local disk:

```bash
ludwig export_mlflow --model_path /saved/ludwig/model --output_path /exported/mlflow/model
```

For the second, you will need to provide a registered model name used by the model registry:

```bash
ludwig export_mlflow --model_path /saved/ludwig/model --output_path relative/model/path --registered_model_name my_ludwig_model
```

# preprocess

Preprocess data and saves it into HDF5 and JSON format.
The preprocessed files can be then used for performing training, prediction and evaluation.
The advantage is that, being the data already preprocessed, if multiple models have to be trained on the same data, the preprocessed files act as a cache to avoid performing preprocessing multiple times.

```
ludwig preprocess [options]
```

or with:

```
python -m ludwig.preprocess [options]
```

These are the available arguments:

```
usage: ludwig preprocess [options]

This script preprocess a dataset

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     input data file path. If it has a split column, it
                        will be used for splitting (0: train, 1: validation,
                        2: test), otherwise the dataset will be randomly split
  --training_set TRAINING_SET
                        input train data file path
  --validation_set VALIDATION_SET
                        input validation data file path
  --test_set TEST_SET   input test data file path
  --training_set_metadata TRAINING_SET_METADATA
                        input metadata JSON file path. An intermediate
                        preprocessed  containing the mappings of the input
                        file created the first time a file is used, in the
                        same directory with the same name and a .json
                        extension
  --data_format {auto,csv,excel,feather,fwf,hdf5,htmltables,json,jsonl,parquet,pickle,sas,spss,stata,tsv}
                        format of the input data
  -pc PREPROCESSING_CONFIG, --preprocessing_config PREPROCESSING_CONFIG
                        preprocessing config. Uses the same format of config,
                        but ignores encoder specific parameters, decoder
                        specific parameters, combiner and training parameters
  -pcf PREPROCESSING_CONFIG_FILE, --preprocessing_config_file PREPROCESSING_CONFIG_FILE
                        YAML file describing the preprocessing. Ignores
                        --preprocessing_config.Uses the same format of config,
                        but ignores encoder specific parameters, decoder
                        specific parameters, combiner and training parameters
  -rs RANDOM_SEED, --random_seed RANDOM_SEED
                        a random seed that is going to be used anywhere there
                        is a call to a random number generator: data
                        splitting, parameter initialization and training set
                        shuffling
  -dbg, --debug         enables debugging mode
  -l {critical,error,warning,info,debug,notset}, --logging_level {critical,error,warning,info,debug,notset}
                        the level of logging to use
```

# synthesize_dataset

Creates synthetic data for testing purposes depending on the feature list parameters provided in YAML format.

```
ludwig synthesize_dataset [options]
```

or with:

```
python -m ludwig.data.dataset_synthesizer [options]
```

These are the available arguments:

```
usage: ludwig synthesize_dataset [options]

This script generates a synthetic dataset.

optional arguments:
  -h, --help            show this help message and exit
  -od OUTPUT_PATH, --output_path OUTPUT_PATH
                        output CSV file path
  -d DATASET_SIZE, --dataset_size DATASET_SIZE
                        size of the dataset
  -f FEATURES, --features FEATURES
                        list of features to generate in YAML format. Provide a
                        list containing one dictionary for each feature, each
                        dictionary must include a name, a type and can include
                        some generation parameters depending on the type

Process finished with exit code 0

```

Example:

```sh
ludwig synthesize_dataset --features="[ \
  {name: text, type: text}, \
  {name: category, type: category}, \
  {name: number, type: number}, \
  {name: binary, type: binary}, \
  {name: set, type: set}, \
  {name: bag, type: bag}, \
  {name: sequence, type: sequence}, \
  {name: timeseries, type: timeseries}, \
  {name: date, type: date}, \
  {name: h3, type: h3}, \
  {name: vector, type: vector}, \
  {name: image, type: image} \
]" --dataset_size=10 --output_path=synthetic_dataset.csv
```

The available parameters depend on the feature type.

__binary__

- `prob` (float, default: `0.5`): probability of generating `true`.
- `cycle` (boolean, default: `false`): cycle through values instead of sampling.

__number__

- `min` (float, default: `0`): minimum value of the range of values to generate.
- `max` (float, default: `1`): maximum value of the range of values to generate.

__category__

- `vocab_size` (int, default: `10`): size of the vocabulary to sample from.
- `cycle` (boolean, default: `false`): cycle through values instead of sampling.

__sequence__

- `vocab_size` (int, default: `10`): size of the vocabulary to sample from.
- `max_len` (int, default: `10`): maximum length of the generated sequence.
- `min_len` (int, default: `null`): if `null` all sequences will be of size `max_len`. If a value is provided, the length will be randomly determined between `min_len` and `max_len`.

__set__

- `vocab_size` (int, default: `10`): size of the vocabulary to sample from.
- `max_len` (int, default: `10`): maximum length of the generated set.

__bag__

- `vocab_size` (int, default: `10`): size of the vocabulary to sample from.
- `max_len` (int, default: `10`): maximum length of the generated set.

__text__

- `vocab_size` (int, default: `10`): size of the vocabulary to sample from.
- `max_len` (int, default: `10`): maximum length of the generated sequence, lengths will be randomly sampled between `max_len - 20%` and `max_len`.

__timeseries__

- `max_len` (int, default: `10`): maximum length of the generated sequence.
- `min` (float, default: `0`): minimum value of the range of values to generate.
- `max` (float, default: `1`): maximum value of the range of values to generate.

__audio__

- `destination_folder` (str): folder where the generated audio files will be saved.
- `preprocessing: {audio_file_length_limit_in_s}` (int, default: `1`): length of the generated audio in seconds.

__image__

- `destination_folder` (str): folder where the generated image files will be saved.
- `preprocessing: {height}` (int, default: `28`): height of the generated image in pixels.
- `preprocessing: {width}` (int, default: `28`): width of the generated image in pixels.
- `preprocessing: {num_channels}` (int, default: `1`): number of channels of the generated images. Valid values are `1`, `3`, `4`.
- `preprocessing: {infer_image_dimensions}` (boolean, default: `true`): whether to transform differently-sized images to the same width/height dimensions. Target dimensions are inferred by taking the average dimensions of the first `infer_image_sample_size` images, then applying `infer_image_max_height` and `infer_image_max_width`. This parameter has no effect if explicit `width` and `height` are specified.
- `preprocessing: {infer_image_sample_size}` (int, default `100`): sample size of `infer_image_dimensions`.
- `preprocessing: {infer_image_max_height}` (int, default `256`): maximum height of an image transformed using `infer_image_dimensions`.
- `preprocessing: {infer_image_max_width}` (int, default `256`): maximum width of an image transformed using `infer_image_dimensions`.

__date__

No parameters.

__h3__

No parameters.

__vector__

- `vector_size` (int, default: `10`): size of the vectors to generate.
