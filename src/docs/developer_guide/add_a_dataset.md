Source code for datasets lives under `ludwig/datasets/`.

Adding a new Dataset
====================

Override `ludwig.datasets.base_dataset.BaseDataset` and implement the following methods:

```python
    @abc.abstractmethod
    def download_raw_dataset(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def process_downloaded_dataset(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def load_processed_dataset(self, split):
        raise NotImplementedError()
```

For common steps (e.g., extracting zip files, downloading from Kaggle, etc.) a set of mixins are available for your subclass:

- Mixin
- Mixin

Mixin properites for a specific dataset are configurable within the `config.yaml` file in the new dataset module.
These mixins cover most common functionalities that are available in the subclass you are creating, but new mixins can also be added.
Before adding a new mixin or writing code for downloading, processing and loading a new dataset, please check if you can reuse one of the curent mixins.
If not, please consider adding a new mixin if the functionality you need is common among multiple datasets or implement bespoke code in the implementation of the abstract methods of the `BaseDataset` subclass.

Please try to mimic the existing unit tests to add new ones for your dataset.
Before submitting a new dataset, please test the functionality locally mimicing the already existing examples to be able to load your dataset, split it and call the Ludwig training API to ensure everything runs fine.


Unit Tests for the Datasets API
===============================

The easiest example of how to extend the Datasets API would be to look at the dataset related unit tests:

- [Titanic unit test](https://github.com/ludwig-ai/ludwig/tree/master/tests/ludwig/datasets/titanic/test_titanic_workflow.py)
- [MNIST unit test](https://github.com/ludwig-ai/ludwig/blob/master/tests/ludwig/datasets/mnist/test_mnist_workflow.py)
