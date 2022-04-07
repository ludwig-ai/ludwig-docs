This is a complete example of training a model for binary classification.

These interactive notebooks follow the steps of this example:

**TODO: point notebook URL to ludwig-ai/ludwig-docs repo before PR merge**

- Ludwig CLI: [![Adult Census Income Classification with Ludwig CLI](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jimthompson5802/ludwig-docs/blob/expanded-tabular-data-example/docs/examples/adult_census_income_colab_notebooks/Adult_Census_Income_Classification_with_Ludwig_CLI.ipynb)
- Ludwig Python API: [![Adult Census Income Classification with Ludwig API](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jimthompson5802/ludwig-docs/blob/expanded-tabular-data-example/docs/examples/adult_census_income_colab_notebooks/Adult_Income_Census_Classification_with_Ludwig_API.ipynb)

## Download the Adult Census Income dataset

[Adult Census Income](https://archive.ics.uci.edu/ml/datasets/adult) is an extract of 1994 Census data for predicting whether a person's income exceeds $50K per year.  The data set consists of over 49K records with 14 attributes with missing data.
```shell
ludwig datasets download ....
```

This command will create a dataset `mnist_dataset.csv` in the current directory.

The columns in the dataset are

|column| description |
|------|-------------|




## Train

The Ludwig configuration file describes the machine learning task.  There is a vast array of options to control the learning process.  This example only covers a small fraction of the options.  Only the options used in this example are described.  Please refer to the [Configuration Section](../../configuration) for all the details.

First it defines the `input_features`.  

Next the `output_features` are defined.  

The last section in this configuration file describes options for how the the [`trainer`](../../configuration/trainer/) will operate.  In this example the `trainer` will process the training data for 5 epochs.

With `config.yaml`:

```yaml
input_features:


output_features:

trainer:
  epochs: 5
```

```shell
ludwig train \
  --dataset mnist_dataset.csv \
  --config config.yaml
```

## Evaluate

```shell
ludwig evaluate --model_path results/experiment_run/model \
                 --dataset mnist_dataset.csv \
                 --split test \
                 --output_directory test_results
```

## Visualize Metrics

Confusion Matrix and class entropy

```shell
ludwig visualize --visualization confusion_matrix \
                  --ground_truth_metadata results/experiment_run/model/training_set_metadata.json \
                  --test_statistics test_results/test_statistics.json \
                  --output_directory visualizations \
                  --file_format png
```

![confusion matrix and entropy]()
Learning Curves

```shell
                 --ground_truth_metadata results/experiment_run/model/training_set_metadata.json \
                  --training_statistics results/experiment_run/training_statistics.json \
                  --file_format png \
                  --output_directory visualizations
```

![confusion learning curves]()

## Predictions

Make predictions from test images

```shell
ludwig predict --model_path results/experiment_run/model \
                --dataset mnist_dataset.csv \
                --split test \
                --output_directory predictions
```

Sample test images displaying true("label") and predicted("pred") labels.
![mnist sample predictions]()