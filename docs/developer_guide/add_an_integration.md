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
