This example shows how to build a text classifier with Ludwig.

These interactive notebooks follow the steps of this example:

- Ludwig CLI: [![Text Classification with Ludwig CLI](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ludwig-ai/ludwig-docs/blob/daniel/text_classification/docs/examples/notebooks/Text_Classification_with_Ludwig_CLI.ipynb)
- Ludwig Python API: [![Text Classification with Ludwig Python API](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ludwig-ai/ludwig-docs/blob/daniel/text_classification/docs/examples/notebooks/Text_Classification_with_Ludwig_Python_API.ipynb)

We'll be using AG's news topic classification dataset, a common benchmark dataset for text classification. This dataset
is a subset of the full AG news dataset, constructed by choosing 4 largest classes from the original corpus. Each class
contains 30,000 training samples and 1,900 testing samples. The total number of training samples is 120,000 with 7,600
total testing samples.

This dataset contains three columns:

| column      | description                                                |
|-------------|------------------------------------------------------------|
| class_index | 1-4: "world", "sports", "business", "sci/tech" respectively |
| title       | Title of the news article                                  |
| description | Description of the news article                            |

Ludwig also provides several other text classification benchmark datasets which can be used, including:

- [Amazon Reviews](https://s3.amazonaws.com/amazon-reviews-pds/readme.html)
- [BBC News](https://www.kaggle.com/competitions/learn-ai-bbc/overview)
- [IMDB](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- [Yelp Reviews](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset)

## Download Dataset

Downloads the AG news dataset to the current working directory.

```bash
ludwig datasets download agnews
```

This command will download the dataset and write to `agnews.csv` in the current directory.

The CSV file contains the above three columns plus an additional `split` column which is one of 0: train, 1: test,
2: validation.

Sample (description text omitted for space):

```
class_index,title,description,split
3,Carlyle Looks Toward Commercial Aerospace (Reuters),...,0
3,Oil and Economy Cloud Stocks' Outlook (Reuters),...,0
3,Iraq Halts Oil Exports from Main Southern Pipeline (Reuters),...,0
```

## Train

### Define ludwig config

The Ludwig config declares the machine learning task. It tells Ludwig what to predict, what columns to use as input, and optionally specifies the model type and hyperparameters.

Here, for simplicity, we'll try to predict **class_index** from **title**.

With `config.yaml`:

```yaml
input_features:
    -
        name: title
        type: text
        level: word
        encoder: parallel_cnn
output_features:
    -
        name: class_index
        type: category
preprocessing:
    force_split: true
    split_probabilities: [0.7, 0.1, 0.2]
trainer:
    epochs: 3
```

### Create and train a model

```bash
ludwig experiment \
    --dataset agnews.csv \
    --config config.yaml
```

## Evaluate

Generates predictions and performance statistics for the test set.

```bash
ludwig evaluate \
    --model_path results/experiment_run/model \
    --dataset agnews.csv \
    --split test \
    --output_directory test_results
```

## Visualize Metrics

Visualizes confusion matrix, which gives an overview of classifier performance for each class.

```bash
ludwig visualize \
    --visualization confusion_matrix \
    --ground_truth_metadata results/experiment_run/model/training_set_metadata.json \
    --test_statistics test_results/test_statistics.json \
    --output_directory visualizations \
    --file_format png
```

Visualizes learning curves, which show how performance metrics changed over time during training.

```bash
ludwig visualize \
    --visualization learning_curves \
    --ground_truth_metadata results/experiment_run/model/training_set_metadata.json \
    --training_statistics results/experiment_run/training_statistics.json \
    --file_format png \
    --output_directory visualizations
```

## Make Predictions on New Data

Lastly we'll show how to generate predictions for new data.

The following are some recent news headlines. Feel free to edit or add your own strings to text_to_predict to see how
the newly trained model classifies them.

With `text_to_predict.csv`:

```
title
Google may spur cloud cybersecurity M&A with $5.4B Mandiant buy
Europe struggles to meet mounting needs of Ukraine's fleeing millions
How the pandemic housing market spurred buyer's remorse across America
```

```bash
ludwig predict \
    --model_path results/experiment_run/model \
    --dataset text_to_predict.csv \
    --output_directory predictions
```
