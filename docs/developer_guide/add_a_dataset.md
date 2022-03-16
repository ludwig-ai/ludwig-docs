The [Ludwig Dataset Zoo](../../user_guide/datasets/dataset_zoo) is a corpus of various datasets from the web
conveniently built into Ludwig.

Ludwig datasets come with several conveniences including: managing credentials for downloading data from sites like
Kaggle, merging multiple files into a single dataset, sharing data parsing code, and loading datasets directly into data
frames that can also be plugged into Ludwig models.

Ludwig datasets are defined under `ludwig/datasets/`.
Dataset mixins are defined under `ludwig/datasets/mixins/`.
For an example of a dataset, see `ludwig/datasets/wmt15/`.

# 1. Create a new directory for your dataset

Create a new directory under `ludwig/datasets/<dataset_name>`, and create two files:

- `__init__.py`
- `config.yaml`

For an example, see `ludwig/datasets/wmt15`.

# 2. Define a class that inherits from `ludwig.datasets.base_dataset.BaseDataset`

We'll want to implement a class that inherits from `ludwig.datasets.base_dataset.BaseDataset` and implement the
following methods:

```python
@abc.abstractmethod
def download_raw_dataset(self):
    """Download the file from config.download_urls and save the file(s) as
    self.raw_dataset_path."""
    raise NotImplementedError()

@abc.abstractmethod
def process_downloaded_dataset(self):
    """Process the dataset into a dataframe and save it at
    self.processed_dataset_path."""
    raise NotImplementedError()

@abc.abstractmethod
def load_processed_dataset(self, split: bool):
    """Loads the processed data from processed_dataset_path into a Pandas
    DataFrame in memory.

    Note: This method is also responsible for splitting the data, returning a
    single dataframe if split=False, and a 3-tuple of train, val, test if
    split=True.

    The split column should always have values 0: train, 1: validation, 2: test.

    :param split: (bool) splits dataset along 'split' column if present.
    """
    raise NotImplementedError()
```

For example:

```python
@register_dataset(name="wmt15")
class WMT15(BaseDataset):
    """French/English parallel texts for training translation models.

    Over 22.5 million sentences in French and English.

    Additional details:
    https://www.kaggle.com/dhruvildave/en-fr-translation-dataset
    """
    pass
```

# 3. Leverage mixins to minimize boilerplate

Ludwig has a set of class mixins that take care of common dataset loading tasks, e.g., extracting zip files, downloading
from Kaggle, etc.).

Plase use Mixins graciously as leveraging mixins could save your dataset subclass from implementing anything.

```python
@register_dataset(name="wmt15")
class WMT15(CSVLoadMixin, IdentityProcessMixin,
            KaggleDownloadMixin, BaseDataset):
    """French/English parallel texts for training translation models.

    Over 22.5 million sentences in French and English.

    Additional details:
    https://www.kaggle.com/dhruvildave/en-fr-translation-dataset
    """

    def __init__(self, cache_dir=DEFAULT_CACHE_LOCATION,
                 kaggle_username=None, kaggle_key=None):
        self.kaggle_username = kaggle_username
        self.kaggle_key = kaggle_key
        self.is_kaggle_competition = False
        super().__init__(dataset_name="wmt15", cache_dir=cache_dir)
```

Check out similar datasets to see which mixins they use, or see `ludwig/datasets/mixins/` for a full list of mixins.

Mixin properites for a specific dataset are configurable within the `config.yaml` file in the new dataset module.
These mixins cover most common functionalities that are available in the subclass you are creating, but new mixins can
also be added.

Before adding a new mixin or writing code for downloading, processing and loading a new dataset, please check if you can
reuse one of the curent mixins.

If not, please consider adding a new mixin if the functionality you need is common among multiple datasets or implement
bespoke code in the implementation of the abstract methods of the `BaseDataset` subclass.

# 4. Test your dataset

Consider adding a unit test in to test that loading data works properly.

Examples of unit tests:

- [Titanic unit test](https://github.com/ludwig-ai/ludwig/tree/master/tests/ludwig/datasets/titanic/test_titanic_workflow.py)
- [MNIST unit test](https://github.com/ludwig-ai/ludwig/blob/master/tests/ludwig/datasets/mnist/test_mnist_workflow.py)

Consider creating and running a simple training script to ensure that the Ludwig training API runs fine with the new
dataset.

```python
from ludwig.api import LudwigModel
from ludwig.datasets import titanic

training_set, test_set, _ = titanic.load(split=True)
model = LudwigModel(config="model_config.yaml", logging_level=logging.INFO)
```

!!! note

    In order to test downloading datasets hosted on Kaggle, please follow [these instructions](https://github.com/Kaggle/kaggle-api#api-credentials) to obtain the necessary API credentials. You may also need to "join" the competition from the Kaggle web UI.

# 5. Add a modeling example

Consider sharing an example for how users can train models using your dataset, for example:

- [Titanic training script](https://github.com/ludwig-ai/ludwig/tree/master/examples/titanic/simple_model_training.py)
- [MNIST training script](https://github.com/ludwig-ai/ludwig/tree/master/examples/mnist/simple_model_training.py)
