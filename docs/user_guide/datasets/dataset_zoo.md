The Ludwig Dataset Zoo provides datasets that can be directly plugged into a Ludwig model.

Each dataset module has `download`, `process` and `load` functions, which
handles downloading, preprocessing, and loading the dataset into a Pandas
DataFrame that Ludwig can use for training.

```python
from ludwig.datasets import reuters

dataset = reuters.load()
```

This is equivalent to performing the steps manually:

```python
from ludwig.datasets.reuters import Reuters

dataset = Reuters()
dataset.download()
dataset.process()
dataset_df = dataset.load()
```

A `cache_dir` parameter can be used to provide a directory where to download the
files (by default it is `~/.ludwig_cache`), and each dataset can have additional
parameters specific to the way they are structured (i.e. a `split` parameter to
return multiple DataFrames for the different splits of the data, if applicable).

When `load()` is called, the existence of a raw dataset directory is determined
and if the data has not yet been downloaded, `download()` is called, then the
existence of a processed dataset directory is determined and if data has not yet
been processed `process()` is called and finally the processed data is loaded in
memory.

Here's an end-to-end example of training a model using the MNIST dataset:

```python
from ludwig.api import LudwigModel
from ludwig.datasets import mnist

# Initialize a Ludwig model
model = LudwigModel(config)

# Load and split MNIST dataset
training_set, test_set, _ = mnist.load(split=True)

# Run model training
train_stats, _, _ = model.train(training_set=training_set, test_set=test_set, model_name="mnist_model")
```

## Zoo Datasets

Here is the list of the currently available datasets:

| Dataset                                   | Hosted On             | Description                                                                                      |
| ----------------------------------------- | --------------------- | ------------------------------------------------------------------------------------------------ |
| adult_census_income                       | archive.ics.uci.edu   | <https://archive.ics.uci.edu/ml/datasets/adult>. Whether a person makes over $50K a year or not. |
| allstate_claims_severity                  | Kaggle                | <https://www.kaggle.com/c/allstate-claims-severity>                                              |
| amazon_employee_access_challenge          | Kaggle                | <https://www.kaggle.com/c/amazon-employee-access-challenge>                                      |
| agnews                                    | Github                | <https://search.r-project.org/CRAN/refmans/textdata/html/dataset_ag_news.html>                   |
| amazon_review_polarity                    | S3                    | <https://paperswithcode.com/sota/sentiment-analysis-on-amazon-review-polarity>                   |
| amazon_reviews                            | S3                    | <https://s3.amazonaws.com/amazon-reviews-pds/readme.html>                                        |
| ames_housing                              | Kaggle                | <https://www.kaggle.com/c/ames-housing-data>                                                     |
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
| ohsumed                                   | boston.lti.cs.cmu.edu | <https://paperswithcode.com/dataset/ohsumed>                                                     |
| otto_group_product                        | Kaggle                | <https://www.kaggle.com/c/otto-group-product-classification-challenge>                           |
| poker_hand                                | archive.ics.uci.edu   | <https://archive.ics.uci.edu/ml/datasets/Poker+Hand>                                             |
| porto_seguro_safe_driver                  | Kaggle                | <https://www.kaggle.com/c/porto-seguro-safe-driver-prediction>                                   |
| protein                                   | archive.ics.uci.edu   | <https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2932-0>                 |
| reuters                                   | boston.lti.cs.cmu.edu | <https://archive.ics.uci.edu/ml/datasets/reuters-21578+text+categorization+collection>           |
| rossmann_store_sales                      | Kaggle                | <https://www.kaggle.com/c/rossmann-store-sales>                                                  |
| santander_customer_satisfaction           | Kaggle                | <https://www.kaggle.com/c/santander-customer-satisfaction>                                       |
| santander_customer_transaction_prediction | Kaggle                | <https://www.kaggle.com/c/santander-customer-transaction-prediction>                             |
| santander-value-prediction-challenge      | Kaggle                | <https://www.kaggle.com/c/santander-value-prediction-challenge>                                  |
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
| yosemite                                  | Github                | <https://www.kaggle.com/balraj98/summer2winter-yosemite>                                         |

## Kaggle

In order to download datasets hosted on Kaggle, please follow [these
instructions](https://github.com/Kaggle/kaggle-api#api-credentials) to obtain
the necessary API credentials. You will also need to "join" the competition from
the Kaggle web UI.

## Adding datasets

To add a dataset to the Ludwig Dataset Zoo, please refer to [this
documentation](../../../developer_guide/add_a_dataset).
