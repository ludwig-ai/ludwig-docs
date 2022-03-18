# Coding Style Guidelines

We expect contributions to mimic existing patterns in the codebase and demonstrate good practices: the code should be
concise, readable, [PEP8-compliant](https://peps.python.org/pep-0008/), and limit each line to 120 characters.

See [codebase structure](../codebase_structure) for guidelines on where new modules should be added.

## pre-commit.ci

The Ludwig repository integrates with [pre-commit.ci](https://pre-commit.ci/), which enforces basic code style
guidelines and automatically fixes minor style issues by adding a commit to pull requests. So, check the results of
pre-commit.ci after creating a new pull request. There may be automatic fixes to pull, or issues which require manual
editing to fix.

To run pre-commit on local branches before pushing, you can install pre-commit locally with pip:

```bash
# Installs pre-commit tool
pip install pre-commit

# Adds pre-commit hooks to local clone of git repository.
pre-commit install

# Runs pre-commit on all files
pre-commit run --all-files

# To disable, simply uninstall pre-commit from the local clone.
pre-commit uninstall
```

## Docstrings

All new files, classes, and methods should have a docstring. Type hints should be used in the function signature
wherever possible, and should use the most specific type accepted by the method.

Example:

```python
def load_processed_dataset(
        self,
        split
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Loads the processed Parquet into a dataframe.

    :param split: Splits along 'split' column if present.
    :returns: The preprocessed dataset, or a tuple of (train, validation, test).
    """
```

Functions with no arguments or return value may have a single-line docstring, ie:

```python
@pytest.fixture()
def csv_filename():
    """Yields a csv filename for holding temporary data."""
```

# Tests

Ludwig uses two types of tests: unit tests and integration tests. Unit tests test a single module, and should
individually be very fast. Integration tests run an end-to-end test of a single Ludwig functionality, like hyperopt or
visualization. Ludwig tests are organized in the following directories:

```
├── ludwig                 - Ludwig library source code
└── tests
    ├── integration_tests  - End-to-end tests of Ludwig workflows
    └── ludwig             - Unit tests. Subdirectories match ludwig/ structure
```

We are using `pytest` as our testing framework. For more information, see the [pytest docs](https://docs.pytest.org).

!!! note

    Ludwig's test coverage is a work in progress, and many modules do not have proper test coverage yet. Contributions
    which get us closer to the goal of 100% test coverage will be welcomed!

## Checklist

Before running tests, make sure:

1. Your python environment is properly setup to run Ludwig.
2. All required dependencies for testing are installed: `pip install ludwig[test]`
3. You have write access on the machine. Some tests require saving temporary files to disk.

## Running tests

To run all tests, execute `python -m pytest` from the ludwig root directory.
Note that you don't need to have ludwig module installed. Running tests from the ludwig source root is useful for
development as the test will import ludwig modules directly from the source tree.

To run all unit tests (will take a few minutes):

```bash
python -m pytest tests/ludwig/
```

Run a single test module:

```bash
python -m pytest tests/ludwig/decoders/test_sequence_decoder.py
```

To run a single test case of a module, you can use `-k` to specify the test case name:

```bash
python -m pytest tests/integration_tests/test_experiment.py \
       -k "test_visual_question_answering"
```

Another useful tool for debugging is the `-vs` flag, which runs the test with eager stdout. This prints log messages to
the console in real time. Also, individual test cases can be specified with the `module::test_case` pattern instead of
`-k`:

```bash
python -m pytest \
    tests/integration_tests/test_api.py::test_api_training_determinism -vs
```
