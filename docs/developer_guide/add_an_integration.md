Ludwig provides an open-ended method of third-party system integration. This makes it easy to integrate other systems or
services with Ludwig which can be enabled simply by passing a flag to the command line interface.

To contribute an integration, follow these steps:

# 1. Create a Python file in `ludwig/contribs/`

The file should have an obvious name associated with the third-party system it integrates with i.e. `comet.py`,
`wandb.py`. In this example, it is called `my_callback.py`.

# 2. Create Callback class

Create a new class implementing the `ludwig.callbacks.Callback` interface. The new class should have a name associated
with the third-party system it integrates with, matching its file name.

```python
from ludwig.callbacks import Callback

class MyCallback(Callback):
```
   
# 3. Implement callback methods

Ludwig provides the following callbacks which you can implement to add functionality to Ludwig. All the following
methods are optional:

```python
 def on_cmdline(self, cmd: str, *args: List[str]):
     """Called when Ludwig is run on the command line with the callback enabled.

     :param cmd: The Ludwig subcommand being run, ex.
                 "train", "evaluate", "predict", ...
     :param args: The full list of command-line arguments (sys.argv).
     """
     pass

 def on_preprocess_start(self, config: Dict[str, Any]):
     """Called before preprocessing starts.

     :param config: The config dictionary.
     """
     pass

 def on_preprocess_end(
         self,
         training_set,
         validation_set,
         test_set,
         training_set_metadata: Dict[str, Any]
    ):
     """Called after preprocessing ends.

     :param training_set: The training set.
     :type training_set: ludwig.dataset.base.Dataset
     :param validation_set: The validation set.
     :type validation_set: ludwig.dataset.base.Dataset
     :param test_set: The test set.
     :type test_set: ludwig.dataset.base.Dataset
     :param training_set_metadata: Values inferred from the training set,
            including preprocessing settings, vocabularies, feature statistics,
            etc. Same as training_set_metadata.json.
     """

     pass

 def on_hyperopt_init(self, experiment_name: str):
     """Called to initialize state before hyperparameter optimization begins.

     :param experiment_name: The name of the current experiment.
     """
     pass

 def on_hyperopt_preprocessing_start(self, experiment_name: str):
     """Called before data preprocessing for hyperparameter optimization begins.

     :param experiment_name: The name of the current experiment.
     """
     pass

 def on_hyperopt_preprocessing_end(self, experiment_name: str):
     """Called after data preprocessing for hyperparameter optimization is
     completed.

     :param experiment_name: The name of the current experiment.
     """
     pass

 def on_hyperopt_start(self, experiment_name: str):
     """Called before any hyperparameter optimization trials are started.

     :param experiment_name: The name of the current experiment.
     """
     pass

 def on_hyperopt_end(self, experiment_name: str):
     """Called after all hyperparameter optimization trials are completed.

     :param experiment_name: The name of the current experiment.
     """
     pass

 def on_hyperopt_trial_start(self, parameters: Dict[str, Any]):
     """Called before the start of each hyperparameter optimization trial.

     :param parameters: The complete dictionary of parameters for this
            hyperparameter optimization experiment.
     """
     pass

 def on_hyperopt_trial_end(self, parameters: Dict[str, Any]):
     """Called after the end of each hyperparameter optimization trial.

     :param parameters: The complete dictionary of parameters for this
            hyperparameter optimization experiment.
     """
     pass

 def on_train_init(
     self,
     base_config: Dict[str, Any],
     experiment_directory: str,
     experiment_name: str,
     model_name: str,
     output_directory: str,
     resume: Union[str, None],
 ):
     """Called after preprocessing, but before the creation of the model and
     trainer objects.

     :param base_config: The user-specified config, before the insertion of
            defaults or inferred values.
     :param experiment_directory: The experiment directory, same as
            output_directory if no experiment specified.
     :param experiment_name: The experiment name.
     :param model_name: The model name.
     :param output_directory: file path to where training results are stored.
     :param resume: model directory to resume training from, or None.
     """
     pass

 def on_train_start(
     self,
     model,
     config: Dict[str, Any],
     config_fp: Union[str, None],
 ):
     """Called after creation of trainer, before the start of training.

     :param model: The ludwig model.
     :type model: ludwig.utils.torch_utils.LudwigModule
     :param config: The config dictionary.
     :param config_fp: The file path to the config, or none if config was passed
            to stdin.
     """
     pass

 def on_train_end(self, output_directory: str):
     """Called at the end of training, before the model is saved.

     :param output_directory: file path to where training results are stored.
     """
     pass

 def on_trainer_train_setup(self, trainer, save_path: str, is_coordinator: bool):
     """Called in every trainer (distributed or local) before training starts.

     :param trainer: The trainer instance.
     :type trainer: trainer: ludwig.models.Trainer
     :param save_path: The path to the directory model is saved in.
     :param is_coordinator: Is this trainer the coordinator.
     """
     pass

 def on_trainer_train_teardown(
         self, trainer, progress_tracker, is_coordinator: bool
     ):
     """Called in every trainer (distributed or local) after training completes.

     :param trainer: The trainer instance.
     :type trainer: ludwig.models.trainer.Trainer
     :param progress_tracker: An object which tracks training progress.
     :type progress_tracker: ludwig.models.trainer.ProgressTracker
     :param is_coordinator: Is this trainer the coordinator.
     """
     pass

 def on_batch_start(self, trainer, progress_tracker, save_path: str):
     """Called on coordinator only before each batch.

     :param trainer: The trainer instance.
     :type trainer: ludwig.models.trainer.Trainer
     :param progress_tracker: An object which tracks training progress.
     :type progress_tracker: ludwig.models.trainer.ProgressTracker
     :param save_path: The path to the directory model is saved in.
     """
     pass

 def on_batch_end(self, trainer, progress_tracker, save_path: str):
     """Called on coordinator only after each batch.

     :param trainer: The trainer instance.
     :type trainer: ludwig.models.trainer.Trainer
     :param progress_tracker: An object which tracks training progress.
     :type progress_tracker: ludwig.models.trainer.ProgressTracker
     :param save_path: The path to the directory model is saved in.
     """
     pass

 def on_epoch_start(self, trainer, progress_tracker, save_path: str):
     """Called on coordinator only before the start of each epoch.

     :param trainer: The trainer instance.
     :type trainer: ludwig.models.trainer.Trainer
     :param progress_tracker: An object which tracks training progress.
     :type progress_tracker: ludwig.models.trainer.ProgressTracker
     :param save_path: The path to the directory model is saved in.
     """
     pass

 def on_epoch_end(self, trainer, progress_tracker, save_path: str):
     """Called on coordinator only after the end of each epoch.

     :param trainer: The trainer instance.
     :type trainer: ludwig.models.trainer.Trainer
     :param progress_tracker: An object which tracks training progress.
     :type progress_tracker: ludwig.models.trainer.ProgressTracker
     :param save_path: The path to the directory model is saved in.
     """
     pass

 def on_validation_start(self, trainer, progress_tracker, save_path: str):
     """Called on coordinator before validation starts.

     :param trainer: The trainer instance.
     :type trainer: ludwig.models.trainer.Trainer
     :param progress_tracker: An object which tracks training progress.
     :type progress_tracker: ludwig.models.trainer.ProgressTracker
     :param save_path: The path to the directory model is saved in.
     """
     pass

 def on_validation_end(self, trainer, progress_tracker, save_path: str):
     """Called on coordinator after validation is complete.

     :param trainer: The trainer instance.
     :type trainer: ludwig.models.trainer.Trainer
     :param progress_tracker: An object which tracks training progress.
     :type progress_tracker: ludwig.models.trainer.ProgressTracker
     :param save_path: The path to the directory model is saved in.
     """
     pass

 def on_test_start(self, trainer, progress_tracker, save_path: str):
     """Called on coordinator before testing starts.

     :param trainer: The trainer instance.
     :type trainer: ludwig.models.trainer.Trainer
     :param progress_tracker: An object which tracks training progress.
     :type progress_tracker: ludwig.models.trainer.ProgressTracker
     :param save_path: The path to the directory model is saved in.
     """
     pass

 def on_test_end(self, trainer, progress_tracker, save_path: str):
     """Called on coordinator after testing ends.

     :param trainer: The trainer instance.
     :type trainer: ludwig.models.trainer.Trainer
     :param progress_tracker: An object which tracks training progress.
     :type progress_tracker: ludwig.models.trainer.ProgressTracker
     :param save_path: The path to the directory model is saved in.
     """
     pass

 def on_build_metadata_start(self, df, mode: str):
     """Called before building metadata for dataset.

     :param df: The dataset.
     :type df: pd.DataFrame
     :param mode: "prediction", "training", or None.
     """
     pass

 def on_build_metadata_end(self, df, mode):
     """Called after building dataset metadata.

     :param df: The dataset.
     :type df: pd.DataFrame
     :param mode: "prediction", "training", or None.
     """
     pass

 def on_build_data_start(self, df, mode):
     """Called before build_data, which does preprocessing, handling missing
     values, adding metadata to training_set_metadata.

     :param df: The dataset.
     :type df: pd.DataFrame
     :param mode: "prediction", "training", or None.
     """
     pass

 def on_build_data_end(self, df, mode):
     """Called after build_data completes.

     :param df: The dataset.
     :type df: pd.DataFrame
     :param mode: "prediction", "training", or None.
     """
     pass

 def on_evaluation_start(self):
     """Called before preprocessing for evaluation."""
     pass

 def on_evaluation_end(self):
     """Called after evaluation is complete."""
     pass

 def on_visualize_figure(self, fig):
     """Called after a visualization is generated.

     :param fig: The figure.
     :type fig: matplotlib.figure.Figure
     """
     pass

 def prepare_ray_tune(
         self,
         train_fn: Callable,
         tune_config: Dict[str, Any],
         tune_callbacks: List[Callable]
     ):
     """Configures Ray Tune callback and config.

     :param train_fn: The function which runs the experiment trial.
     :param tune_config: The ray tune configuration dictionary.
     :param tune_callbacks: List of callbacks (not used yet).

     :returns: Tuple[Callable, Dict] The train_fn and tune_config, which will be
               passed to ray tune.
     """
     return train_fn, tune_config

 @staticmethod
 def preload():
     """Will always be called when Ludwig CLI is invoked, preload gives the
     callback an opportunity to import or create any shared resources.

     Importing required 3rd-party libraries should be done here i.e. import wandb.
     preload is guaranteed to be called before any other callback method, and will
     only be called once per process.
     """
     pass
```

If you would like to add additional actions not already handled by the above:

1. Add them to the appropriate calling location.
2. Add the associated method to your callback class.
3. Write a docstring, and add it to this documentation page.

See existing calls in `ludwig/callbacks.py` as a pattern to follow.

# 4. Import the new callback

In `ludwig/contribs/__init__.py` add an import in this pattern, using your module and class names:

```python
from my_callback import MyCallback
```

# 5. Register a flag for the callback

In `ludwig/contribs/__init__.py` in the `contrib_registry["classes"]` dictionary, add a key/value pair where the key is
the flag which enables the callback and the value is the class:

```python
contrib_registry = {
    ...,
    "myflag": MyCallback,
}
```
