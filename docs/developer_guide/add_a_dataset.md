The [Ludwig Dataset Zoo](../../user_guide/datasets/dataset_zoo) is a corpus of various datasets from the web
conveniently built into Ludwig.

Ludwig datasets automate managing credentials for downloading data from sites like Kaggle, merging multiple files into
a single dataset, sharing data parsing code, and loading datasets directly into data frames which can be used to train
Ludwig models.

- The Ludwig Datasets API is contained in `ludwig/datasets/`.
- Dataset configs are defined under `ludwig/datasets/configs/`.
- Custom loaders for specific datasets are in `ludwig/datasets/loaders/`.

Datasets are made available in Ludwig by providing a dataset config .yaml file.  For many datasets, creating this YAML
file is the only necessary step.

# 1. Create a new dataset config

Create a new `.yaml` file under `ludwig/datasets/configs/` with a name matching the name of the dataset. The config file
must have the following required keys:

- `version`: The version of the dataset
- `name`: The name of the dataset. This is the name which will be imported or passed into `get_datasets(datset_name)`.
- `description`: Human-readable description of the dataset. May contain multi-line text with links.
- One of `download_urls`, `kaggle_competition`, or `kaggle_dataset_id`.

Supported compressed archive and data file types will be inferred automatically from the file extension.

For the full set of options, see `ludwig.datasets.dataset_config.DatasetConfig`. If the options provided by
`DatasetConfig` are sufficient to integrate your dataset, skip ahead to step 3. Test your Dataset.

If, however, the dataset requires other processing not provided by the default dataset loader, continue to step 2.

# 2. Define a dataset loader if needed

If the options provided by `DatasetConfig` do not cover the format of your dataset, or if the dataset requires unique
processing before training, you can add python code in a dataset loader.

The loader class should inherit from `ludwig.datasets.loaders.dataset_loader.DatasetLoader`, and its module name should
match the name of the dataset.  For example, AG News has a dataset loader `agnews.AGNewsLoader` in
`ludwig/datasets/loaders/agnews.py`.

To instruct Ludwig to use your loader, add the `loader` property to your dataset config:

```yaml
loader: agnews.AGNewsLoader
```

Datasets are processed in four phases:

1. Download       - The dataset files are downloaded to the cache.
2. Verify         - Hashes of downloaded files are verified.
3. Extract        - The dataset files are extracted from an archive (may be a no-op if data is not archived).
4. Transform      - The dataset is transformed into a format usable for training and is ready to load.
    1. Transform Files      (Files -> Files)
    2. Load Dataframe       (Files -> DataFrame)
    3. Transform Dataframe  (DataFrame -> DataFrame)
    4. Save Processed       (DataFrame -> File)

For each of these phases, there is a corresponding method in `ludwig.datasets.loaders.DatasetLoader` which may be
overridden to provide custom processing.

# 3. Test your dataset

Create a simple training script and ludwig config to ensure that the Ludwig training API runs with the new dataset.
For example:

```python
from ludwig.api import LudwigModel
from ludwig.datasets import titanic

training_set, test_set, _, = titanic.load(split=True)
model = LudwigModel(config="model_config.yaml", logging_level=logging.INFO)
train_stats, _, _ = model.train(training_set=training_set, test_set=test_set, model_name="titanic_model")
```

If you have added a custom loader, please also a unit test to ensure that your loader works with future versions.
Following the examples below, provide a small sample of the data to the unit test so the test will not need to download
the dataset.

Examples of unit tests:

- [Titanic unit test](https://github.com/ludwig-ai/ludwig/tree/master/tests/ludwig/datasets/titanic/test_titanic_workflow.py)
- [MNIST unit test](https://github.com/ludwig-ai/ludwig/blob/master/tests/ludwig/datasets/mnist/test_mnist_workflow.py)

!!! note

    In order to test downloading datasets hosted on Kaggle, please follow
    [these instructions](https://github.com/Kaggle/kaggle-api#api-credentials) to obtain the necessary API credentials.
    You may also need to "join" the competition from the Kaggle web UI.
    For testing, the Titanic example also illustrates how to mock the kaggle client to unit test Kaggle datasets without
    logging in to Kaggle on the test machine.

# 4. Add a modeling example

Consider sharing an example for how users can train models using your dataset, for example:

- [Titanic training script](https://github.com/ludwig-ai/ludwig/tree/master/examples/titanic/simple_model_training.py)
- [MNIST training script](https://github.com/ludwig-ai/ludwig/tree/master/examples/mnist/simple_model_training.py)
