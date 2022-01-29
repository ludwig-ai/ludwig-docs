Python API
==========

Ludwig functionalities can also be accessed through a Python programmatic API.
The API consists of one `LudwigModel` class that can be initialized with a configuration dictionary and then can be trained with a dataset (either in memory or loaded from file).
Pretrained models can be loaded and can be used to obtain predictions on s new dataset (either in memory or loaded from file).

A detailed documentation of all the functions available in `LudwigModel` is provided in the [API documentation](../api.md).

Training a Model
================

To train a model one has first to initialize it using the initializer `LudwigModel()` and a configuration dictionary, and then calling the `train()` function using either a dataframe or a dataset file.

```python
from ludwig.api import LudwigModel

config = {...}
model = LudwigModel(config)
training_statistics, preprocessed_data, output_directory = model.train(dataset=dataset_file_path)
# or
training_statistics, preprocessed_data, output_directory = model.train(dataset=dataframe)
```

`config` is a dictionary that has the same key-value structure of a configuration YAML file, as it's technically equivalent as parsing the YAML file into a Python dictionary.
Note that all null values should be provided as Python `None` instead of the YAML `null`, and the same applies for `True/False` instead of `true/false`. 
`train_statistics` is a dictionary of training statistics
for each output feature containing loss and metrics values
for each epoch.
The contents are exactly the same of the `training_statistics.json` file produced by the `experiment` and `train` commands.
`preprocessed_data` is the tuple containing these three data sets
`(training_set, validation_set, test_set)`.
`output_directory` is the filepath where training results are stored.


Loading a Pre-trained Model
===========================

In order to load a pre-trained Ludwig model you have to call the static function `load()` of the `LudwigModel` class providing the path containing the model.

```python
from ludwig.api import LudwigModel

model = LudwigModel.load(model_path)
```

Predicting
==========

Either a newly trained model or a pre-trained loaded model can be used for predicting on new data using the `predict()` function of the model object.
The dataset has to contain columns with the same names of all the input features of the model.

```python
predictions, output_directory = model.predict(dataset=dataset_file_path)
#or
predictions, output_directory = model.predict(dataset=dataframe)
```

`predictions` will be a dataframe containing the prediction and confidence score / probability of all output features.  `output_directory` filepath to prediction interim files.

If you want to compute also measures on the quality of the predictions you can run:

```python
evaluation_statistics, predictions, output_directory = model.evaluate(dataset=dataset_file_path)
#or
evaluation_statistics, predictions, output_directory = model.evaluate(dataset=dataframe)
```

In this case the dataset should also contain columns with the same names of all the output features, as their content is going to be used as ground truth to compare the predictions against and compute the measures and `evaluation_statistics` will be a dictionary containing several measures of quality depending on the type of each output feature (e.g. `category` features will have an accuracy measure and a confusion matrix, among other measures, associated to them, while `numerical` features will have measures like mean squared loss and R2 among others).
