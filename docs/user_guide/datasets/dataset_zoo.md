The Ludwig Dataset Zoo provides datasets that can be directly plugged into a Ludwig model.

The simplest way to use a dataset is to reference it as a URI when specifying the training set:

```bash
ludwig train --dataset ludwig://reuters ...
```

Any Ludwig dataset can be specified as a URI of the form `ludwig://<dataset>`.

Datasets can also be programatically imported and loaded into a Pandas DataFrame using the `.load()` method:

```python
from ludwig.datasets import reuters

# Loads into single dataframe with a 'split' column:
dataset_df = reuters.load()

# Loads into split dataframes:
train_df, test_df, _ = reuters.load(split=True)
```

The `ludwig.datasets` API also provides functions to list, describe, and get datasets.  For example:

```python
import ludwig.datasets

# Gets a list of all available dataset names.
dataset_names = ludwig.datasets.list_datasets()

# Prints the description of the titanic dataset.
print(ludwig.datasets.describe_dataset("titanic"))

titanic = ludwig.datasets.get_dataset("titanic")

# Loads into single dataframe with a 'split' column:
dataset_df = titanic.load()

# Loads into split dataframes:
train_df, test_df, _ = titanic.load(split=True)
```

## Kaggle Datasets

Some datasets are hosted on [Kaggle](https://www.kaggle.com) and require a kaggle account. To use these, you'll need to
[set up Kaggle credentials](https://www.kaggle.com/docs/api) in your environment. If the dataset is part of a Kaggle
competition, you'll need to accept the terms on the competition page.

To check programmatically, datasets have an `.is_kaggle_dataset` property.

## Downloading, Processing, and Exporting

Datasets are first downloaded into `LUDWIG_CACHE`, which may be set as an environment variable and defaults to
`$HOME/.ludwig_cache`.

Datasets are automatically loaded, processed, and re-saved as parquet files in the cache.

To export the processed dataset, including any files it depends on, use the `.export(output_directory)` method. This
is recommended if the dataset contains media files like images or audio files. File paths are relative to the working
directory of the training process.

```python
from ludwig.datasets import twitter_bots

# Exports twitter bots dataset and image files to the current working directory.
twitter_bots.export(".")
```

## End-to-end Example

Here's an end-to-end example of training a model using the MNIST dataset:

```python
from ludwig.api import LudwigModel
from ludwig.datasets import mnist

# Initializes a Ludwig model
config = {
    "input_features": [{"name": "image_path", "type": "image"}],
    "output_features": [{"name": "label", "type": "category"}],
}
model = LudwigModel(config)

# Loads and splits MNIST dataset
training_set, test_set, _ = mnist.load(split=True)

# Exports the mnist image files to the current working directory.
mnist.export(".")

# Runs model training
train_stats, _, _ = model.train(training_set=training_set, test_set=test_set, model_name="mnist_model")
```

## Dataset Splits

All datasets in the dataset zoo are provided with a default train/validation/test split. When loading with
`split=False`, the default split will be returned (and is guaranteed to be the same every time). With `split=True`,
Ludwig will randomly re-split the dataset.

!!! note
    Some benchmark or contest datasets are released with held-out test set labels. In other words, the train and
    validation splits have labels, but the test set does not. Most Kaggle contest datasets have this unlabeled test set.

Splits:

- **train**: Data to train on. Required, must have labels.
- **validation**: Subset of dataset to evaluate while training. Optional, must have labels.
- **test**: Held out from model development, used for later testing. Optional, may not be labeled.

## Zoo Datasets

Here is the list of the currently available datasets:

| Dataset                                   | Hosted On             | Description                                                                                      |
| ----------------------------------------- | --------------------- | ------------------------------------------------------------------------------------------------ |
| adult_census_income                       | archive.ics.uci.edu   | <https://archive.ics.uci.edu/ml/datasets/adult>. Whether a person makes over $50K a year or not. |
| allstate_claims_severity                  | Kaggle                | <https://www.kaggle.com/c/allstate-claims-severity>                                              |
| amazon_employee_access_challenge          | Kaggle                | <https://www.kaggle.com/c/amazon-employee-access-challenge>                                      |
| agnews                                    | Github                | <https://search.r-project.org/CRAN/refmans/textdata/html/dataset_ag_news.html>                   |
| allstate_claims_severity                  | Kaggle                | <https://www.kaggle.com/c/allstate-claims-severity>                                              |
| amazon_employee_access_challenge          | Kaggle                | <https://www.kaggle.com/c/amazon-employee-access-challenge>                                      |
| amazon_review_polarity                    | S3                    | <https://paperswithcode.com/sota/sentiment-analysis-on-amazon-review-polarity>                   |
| amazon_reviews                            | S3                    | <https://s3.amazonaws.com/amazon-reviews-pds/readme.html>                                        |
| ames_housing                              | Kaggle                | <https://www.kaggle.com/c/ames-housing-data>                                                     |
| bbc_news                                  | Kaggle                | <https://www.kaggle.com/c/learn-ai-bbc>                                                          |
| bnp_claims_management                     | Kaggle                | <https://www.kaggle.com/c/bnp-paribas-cardif-claims-management>                                  |
| connect4                                  | Kaggle                | <https://www.kaggle.com/c/connectx/discussion/124397>                                            |
| creditcard_fraud                          | Kaggle                | <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>                                        |
| dbpedia                                   | S3                    | <https://paperswithcode.com/dataset/dbpedia>                                                     |
| electricity                               | S3                    | Predict electricity demand from day of week and outside temperature.                             |
| ethos_binary                              | Github                | <https://github.com/huggingface/datasets/blob/master/datasets/ethos/README.md>                   |
| fever                                     | S3                    | <https://arxiv.org/abs/1803.05355>                                                               |
| flickr8k                                  | Github                | <https://www.kaggle.com/adityajn105/flickr8k>                                                    |
| forest_cover                              | archive.ics.uci.edu   | <https://archive.ics.uci.edu/ml/datasets/covertype>                                              |
| goemotions                                | Github                | <https://arxiv.org/abs/2005.00547>                                                               |
| higgs                                     | archive.ics.uci.edu   | <https://archive.ics.uci.edu/ml/datasets/HIGGS>                                                  |
| ieee_fraud                                | Kaggle                | <https://www.kaggle.com/c/ieee-fraud-detection>                                                  |
| imbalanced_insurance                      | Kaggle                | <https://www.kaggle.com/datasets/arashnic/imbalanced-data-practice>                              |
| imdb                                      | Kaggle                | <https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews>              |
| insurance_lite                            | Kaggle                | <https://www.kaggle.com/infernape/fast-furious-and-insured>                                      |
| iris                                      | archive.ics.uci.edu   | <https://archive.ics.uci.edu/ml/datasets/iris>                                                   |
| irony                                     | Github                | <https://github.com/bwallace/ACL-2014-irony>                                                     |
| kdd_appetency                             | kdd.org               | <https://www.kdd.org/kdd-cup/view/kdd-cup-2009/Data>                                             |
| kdd_churn                                 | kdd.org               | <https://www.kdd.org/kdd-cup/view/kdd-cup-2009/Data>                                             |
| kdd_upselling                             | kdd.org               | <https://www.kdd.org/kdd-cup/view/kdd-cup-2009/Data>                                             |
| mnist                                     | yann.lecun.com        | <http://yann.lecun.com/exdb/mnist/>                                                              |
| mushroom_edibility                        | archive.ics.uci.edu   | <https://archive.ics.uci.edu/ml/datasets/mushroom>                                               |
| naval                                     | archive.ics.uci.edu   | <https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/24098>                 |
| noshow_appointments                       | Kaggle                | <https://www.kaggle.com/datasets/joniarroba/noshowappointments>                                  |
| numerai28pt6                              | Kaggle                | <https://www.kaggle.com/numerai/encrypted-stock-market-data-from-numerai>                        |
| ohsumed_7400                              | Kaggle                | <https://www.kaggle.com/datasets/weipengfei/ohr8r52>                                             |
| ohsumed_cmu                               | boston.lti.cs.cmu.edu | <http://boston.lti.cs.cmu.edu/classes/95-865-K/HW/HW2/>                                          |
| otto_group_product                        | Kaggle                | <https://www.kaggle.com/c/otto-group-product-classification-challenge>                           |
| poker_hand                                | archive.ics.uci.edu   | <https://archive.ics.uci.edu/ml/datasets/Poker+Hand>                                             |
| porto_seguro_safe_driver                  | Kaggle                | <https://www.kaggle.com/c/porto-seguro-safe-driver-prediction>                                   |
| protein                                   | archive.ics.uci.edu   | <https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2932-0>                 |
| reuters_cmu                               | boston.lti.cs.cmu.edu | <http://boston.lti.cs.cmu.edu/classes/95-865-K/HW/HW2/>                                          |
| reuters_r8                                | Kaggle                | Reuters R8 subset of Reuters 21578 dataset from Kaggle.                                          |
| rossmann_store_sales                      | Kaggle                | <https://www.kaggle.com/c/rossmann-store-sales>                                                  |
| santander_customer_satisfaction           | Kaggle                | <https://www.kaggle.com/c/santander-customer-satisfaction>                                       |
| santander_customer_transaction_prediction | Kaggle                | <https://www.kaggle.com/c/santander-customer-transaction-prediction>                             |
| santander_value_prediction                | Kaggle                | <https://www.kaggle.com/c/santander-value-prediction-challenge>                                  |
| sarcos                                    | gaussianprocess.org   | <http://www.gaussianprocess.org/gpml/data/>                                                      |
| sst2                                      | nlp.stanford.edu      | <https://paperswithcode.com/dataset/sst>                                                         |
| sst3                                      | nlp.stanford.edu      | Merging very negative and negative, and very positive and positive classes.                      |
| sst5                                      | nlp.stanford.edu      | <https://paperswithcode.com/dataset/sst>                                                         |
| synthetic_fraud                           | Kaggle                | <https://www.kaggle.com/ealaxi/paysim1>                                                          |
| temperature                               | Kaggle                | <https://www.kaggle.com/selfishgene/historical-hourly-weather-data>                              |
| titanic                                   | Kaggle                | <https://www.kaggle.com/c/titanic>                                                               |
| walmart_recruiting                        | Kaggle                | <https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting>                            |
| wmt15                                     | Kaggle                | <https://www.kaggle.com/dhruvildave/en-fr-translation-dataset>                                   |
| yahoo_answers                             | S3                    | Question classification.                                                                         |
| yelp_review_polarity                      | S3                    | <https://www.yelp.com/dataset>. Predict the polarity or sentiment of a yelp review.              |
| yelp_reviews                              | S3                    | <https://www.yelp.com/dataset>                                                                   |
| yosemite                                  | Github                | <https://github.com/ourownstory/neural_prophet> Yosemite temperatures dataset.                   |

## Adding datasets

To add a dataset to the Ludwig Dataset Zoo, see [Add a Dataset](../../../developer_guide/add_a_dataset).
