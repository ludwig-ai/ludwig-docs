The Datasets module provides training datasets that can be directly plugged into a Ludwig model.
Datasets can be accessed programmatically by importing the `ludwig.datasets` module.
Each dataset class in the module has a `download`, `process` and `load` function, plus a handy static `load` function to import from the module itself (i.e. `datasets.titanic.load`).
Calling the module `load()` function will handle downloading, preprocessing, and loading the dataset into a Pandas DataFrame that Ludwig can use for training.
A `cache_dir` parameter can be used to provide a directory where to download the files (by default it is `~/.ludwig_cache`), and each dataset can have additional parameters specific to the way they are structured (i.e. a `split` parameter to return multiple DataFrames for the different splits of the data, if applicable).
When `load()` is called, the existence of a raw dataset directory is determined and if the data has not yet been downloaded, `download()` is called, then the existence of a processed dataset directory is determined and if data has not yet been processed `process()` is called and finally the processed data is loaded in memory.

For example:

```python
from ludwig.datasets import reuters
dataset = reuters.load()
```

Is equivalent to performing the steps manually:

```python
from ludwig.datasets.reuters import Reuters
dataset = Reuters()
dataset.download()
dataset.process()
dataset_df = dataset.load()
```

And here is an end-to-end example of training a model using the MNIST dataset:

```python
from ludwig.api import LudwigModel
from ludwig.datasets import mnist

# Initialize a Ludwig model
model = LudwigModel(config)

# Load and split MNIST dataset
training_set, test_set, _ = mnist.load(split=True)

# Run model training
train_stats, _, _ = model.train(
    training_set=training_set,
    test_set=test_set,
    model_name='mnist_model'
)
```

# Currently Available Datasets

Here is the list of the currently available datasets:

- `adult_census_income`
- `agnews`
- `amazon_review_polarity`
- `amazon_reviews`
- `ames_housing` (hosted on Kaggle)
- `dbpedia`
- `electricity`
- `ethos_binary`
- `fever`
- `flickr8k`
- `forest_cover`
- `goemotions`
- `higgs`
- `irony`
- `kdd_appetency`
- `kdd_churn`
- `kdd_upselling`
- `mnist`
- `mushroom_edibility`
- `ohsumed`
- `poker_hand`
- `reuters`
- `rossmann_store_sales` (hosted on Kaggle)
- `sarcos`
- `sst2`, `sst5`, `sst3` (a variant obtained my merging very negative and negative, and very positive and positive classes)
- `temperature` (hosted on Kaggle)
- `titanic` (hosted on Kaggle)
- `yahoo_answers`
- `yelp_review_polarity`
- `yelp_reviews`
- `yosemite`

In order to download the datasets hosted on Kaggle, you can either provide credentials through a `kaggle_username` and `kaggle_key` parameter to the `load()` function, or follow the more secure instructions provided in the [Python Kaggle Client](https://technowhisp.com/kaggle-api-python-documentation/) documentations.
