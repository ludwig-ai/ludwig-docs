# General Guidelines

## Create a unit test for every module

Unit tests are in `tests/ludwig/` which parallels the `ludwig/` source tree. Every source file in `ludwig/` with testable
functionality should have a corresponding unit test file in `test/ludwig/`, with a filename corresponding to the source
file, prefixed by `test_`.

Examples:

| Module                                    | Test                                                 |
| ----------------------------------------- | ---------------------------------------------------- |
| `ludwig/data/dataset_synthesizer.py`      | `tests/ludwig/data/test_dataset_synthesizer.py`      |
| `ludwig/features/audio_feature.py`        | `tests/ludwig/features/test_audio_feature.py`        |
| `ludwig/modules/convolutional_modules.py` | `tests/ludwig/modules/test_convolutional_modules.py` |

!!! note

    At the time of writing, not all modules in Ludwig have proper unit tests or match the guidelines here. These
    guidelines are the goals we aspire to, and we believe incremental improvement is better than demanding perfection.
    Testing of Ludwig is a work in progress, any changes that get us closer to the goal of 100% test coverage are
    welcomed!

## What should be tested

Unit tests should generally test every function or method a module exports, with some exceptions. A good rule is that if
a function or method fulfills a requirement and will be called from outside the module, it should have a test.

- **Test the common cases**. This will provide a notification when something breaks.
- **Test the edge cases of complex methods** if you think they might have errors.
- **Test failure cases** If a method must fail, ex. when input out of range, ensure it does.
- **Bugs** When you find a bug, write a test case to cover it before fixing it.

### Parameterize tests

Use `@pytest.mark.parameterize` to test combinations of parameters that drive through all code paths, to ensure
correct behavior of the function under a variety of situations.

```python
# test combinations of parameters to exercise all code paths
@pytest.mark.parameterize(
    'num_total_blocks, num_shared_blocks',
    [(4, 2), (6, 4), (3, 1)]
)
@pytest.mark.parameterize('virtual_batch_size', [None, 7])
@pytest.mark.parameterize('size', [4, 12])
@pytest.mark.parameterize('input_size', [2, 6])
def test_feature_transformer(
    input_size: int,
    size: int,
    virtual_batch_size: Optional[int],
    num_total_blocks: int,
    num_shared_blocks: int
) -> None:
    feature_transformer = FeatureTransformer(
        input_size,
        size,
        bn_virtual_bs=virtual_batch_size,
        num_total_blocks=num_total_blocks,
        num_shared_blocks=num_shared_blocks
    )
```

### Test edge cases

Test edge cases when possible. For example, if a method takes multiple inputs: test with an empty input, a single input,
and a large number of inputs.

```python
@pytest.mark.parametrize("virtual_batch_size", [None, 7, 64])  # Test with no virtual batch norm, odd size, or large.
@pytest.mark.parametrize("input_size", [1, 8, 256])  # Test with single input feature or many inputs.
```

### Tensor Type and Shape

At minimum, tests related to tensors should confirm no errors are raised when processing tensor(s) and that resulting
tensors are of correct shape and type. This provides minimal assurance that the function is operating as expected.

```python
# pass input features through combiner
combiner_output = combiner(input_features)

# check for required attributes in the generated output
assert hasattr(combiner, 'input_dtype')
assert hasattr(combiner, 'output_shape')

# check for correct data type
assert isinstance(combiner_output, dict)

# required key present
assert 'combiner_output' in combiner_output

# check for correct output shape
assert (combiner_output['combiner_output'].shape
       == (batch_size, *combiner.output_shape))
```

### Trainable Modules

When testing a trainable module (layer, encoder, decoder, combiner or model), make sure that all the variables / weights
get updates after one training step. This will ensure that the computation graph does not contain dangling nodes. This
catches subtle issues which don’t manifest as crashes, which are not caught by looking at the loss scores or by training
the model to convergence (albeit to a usually bad loss), For more details see
[how to unit test machine learning code](https://thenerdstation.medium.com/how-to-unit-test-machine-learning-code-57cf6fd81765).

Add **type checking** based on torch input and outputs. Ensure all module outputs have the expected torch datatype,
dimensionality, and tensor shape.

For trainable modules, we recommend adding at least one **overfitting test**. Ensure that a small ECD model containing
the module is able to overfit on a small dataset. Ensures that models are able to converge on reasonable targets and
catches any unscoped issues that are not captured by shape, type, or weight update tests.

## Best practices

There's lots of great advice on the web for writing good tests. Here are a few highlights from
[Microsoft's recommendations](https://docs.microsoft.com/en-us/dotnet/core/testing/unit-testing-best-practices)
that we ascribe to:

- [Characteristics of a good unit test](https://docs.microsoft.com/en-us/dotnet/core/testing/unit-testing-best-practices#characteristics-of-a-good-unit-test)
- [Arranging unit tests](https://docs.microsoft.com/en-us/dotnet/core/testing/unit-testing-best-practices#arranging-your-tests)
- [Write minimally passing tests](https://docs.microsoft.com/en-us/dotnet/core/testing/unit-testing-best-practices#write-minimally-passing-tests)
- [Avoid logic in tests](https://docs.microsoft.com/en-us/dotnet/core/testing/unit-testing-best-practices#avoid-logic-in-tests)
- [Avoid multiple acts](https://docs.microsoft.com/en-us/dotnet/core/testing/unit-testing-best-practices#avoid-multiple-acts)

## What not to test

It isn't necessary to test every function, a good rule is that if a function or method fulfills a requirement and will
be called from outside the module, it should have a test.

Things that don't need unit tests:

- Constructors or properties. Test them only if they contain validations.
- Configurations like constants, readonly fields, configs, etc.
- Facades or wrappers around other frameworks or libraries.
- Private methods

# Implementation Guidelines

## Use pytest.mark.parameterize

Automates setup for test cases. Be careful to only test case parameter values which are meaningfully different, as the
total number of tests cases grows combinatorially.

```python
@pytest.mark.parameterize('enc_should_embed', [True, False])
@pytest.mark.parameterize('enc_reduce_output', [None, 'sum'])
@pytest.mark.parameterize('enc_norm', [None, 'batch', 'layer'])
@pytest.mark.parameterize('enc_num_layers', [1, 2])
@pytest.mark.parameterize('enc_dropout', [0, 0.2])
@pytest.mark.parameterize('enc_cell_type', ['rnn', 'gru', 'lstm'])
@pytest.mark.parameterize('enc_encoder', ENCODERS + ['passthrough'])
def test_sequence_encoders(
        enc_encoder: str,
        enc_cell_type: str,
        enc_dropout: float,
        enc_num_layers: int,
        enc_norm: Union[None, str],
        enc_reduce_output: Union[None, str],
        enc_should_embed: bool,
        input_sequence: torch.Tensor
):
```

## Use temp_path or tmpdir for generated data

Use temporary directories for any generated data. PyTest provides fixtures for temporary directories, which are
guaranteed unique for each test run and will be cleaned up automatically. We recommend using `tmpdir`, which provides a
`py.path.local` object which is compatible with `os.path` methods. If you are using `pathlib`, PyTest also provides
`tmp_path`, which is a `pathlib.Path`.

For more details, see [Temporary directories and files](https://docs.pytest.org/en/6.2.x/tmpdir.html) in the PyTest
docs.

Example:

```python
@pytest.mark.parametrize("skip_save_processed_input", [True, False])
@pytest.mark.parametrize("in_memory", [True, False])
@pytest.mark.parametrize("image_source", ["file", "tensor"])
@pytest.mark.parametrize("num_channels", [1, 3])
def test_basic_image_feature(
        num_channels, image_source, in_memory, skip_save_processed_input, tmpdir
):
    # Image Inputs
    image_dest_folder = os.path.join(tmpdir, "generated_images")
```

## Consolidate tests which require setup

For example, multiple tests may rely on the same training/test data sets which take time to load. If multiple tests rely
on the same common resources, group these tests into a single module and use appropriately scoped `@pytest.fixture` to
reduce overhead of repeatedly performing setup.

Examples of reusable test fixtures can be found in
`tests/conftest.py`. This module contains reusable `@pytest.fixtures` that have applicability across many tests.

## Deterministic tests

Wherever possible, every test run with the same parameters should produce the same result. When using a random number
generator, always specify a seed. A test will be difficult to debug if it produces different output on different runs.

```python
import torch

RANDOM_SEED = 1919

# setup synthetic tensor, ensure reproducibility by setting seed
torch.manual_seed(RANDOM_SEED)
input_tensor = torch.randn([BATCH_SIZE, input_size], dtype=torch.float32)
```

## Test for parameter updates

The utility function `check_module_parameters_updated()` in the `tests.integration_tests.parameter_utils` module
is available to test whether Ludwig modules, e.,g encoders, combiners, decoders and related sub-components are updating
parameters during the sequence of forward pass-backward pass-optimize step.

Guidelines for implementing parameter updating test:

- Not required for very simple modules, like the fully-connected layer or for well-known pre-trained modules, like the Huggingface text encoders.
- Before implementing the parameter update test ensure that the test does not generate run-time exceptions, the generated output conforms to the expected data structure and the shape of the output is correct.

`check_module_parameters_updated(module, input, target)` function requires three positional arguments:

- `module` is the Ludwig component to be tested, i.e., encoder, combiner or decoder
- `input` is tuple that is the input the Ludwig component's forward method
- `target` is a synthetic tensor representing the target values for computing loss at the end of the forward pass.

The `module` and `input` arguments can be the same as those used in the early part of the test to ensure no run-time exceptions and correct output.

A two-step process is recommended steps for implementing the test:

**Step 1**:

Set the random seed for repeatability.  Partially implement the parameter update test to print the counts of
parameters, e.g.,

```python
    # check for parameter updating
    target = torch.randn(output.shape)
    fpc, tpc, upc, not_updated = check_module_parameters_updated(sequence_rnn_decoder, (combiner_outputs, None), target)
    print(fpc, tpc, upc, not_updated)
```

`target` is a tensor with synthetic data that is used to in computing the loss during the forward pass.  

`fpc` is the count of frozen parameters.  `tpc` is the count of trainable parameters.  `upc` is the updated parameter
count, i.e., the number of parameters that were updated during the cycle of forward pass-backward pass-optimize step.
Inspect the values for correctness, such as

- `fpc` + `tpc` should equal the total number of parameters in the module
- `fpc` should be zero except for pre-trained modules like the Huggingface text encoders.
- `upc` <= `tpc`

In the case that some parameters are not updated, `not_updated` is a Python list containing names of the parameter
that were not updated.

In the ideal case, `upc` == `tpc`, i.e, all of the trainable parameters were updated.  However, there may be
situations where `upc` < `tpc`.  This may occur in situations where dropout is used or with batch normalization
with a single training example or conditional processing in the `forward()` method.  The developer should confirm that
the counts are correct in this situation.

**Step 2**:

Once all the differences between `tpc` and `upc` are accounted for then replace the `print()` statement with the
appropriate set of `assert` statements.  Here are some examples:

```python
    # check for parameter updating
    target = torch.randn(output.shape)
    fpc, tpc, upc, not_updated = check_module_parameters_updated(sequence_rnn_decoder, (combiner_outputs, None), target)
    assert upc == tpc, f"Failed to update parameters.  Parameters not update: {not_updated}"
```

```python
    target = torch.randn(conv1_stack.output_shape)
    fpc, tpc, upc, not_updated = check_module_parameters_updated(conv1_stack, (input,), target)
    if dropout == 0:
        # all trainable parameters should be updated
        assert tpc == upc, (
            f"All parameter not updated. Parameters not updated: {not_updated}" f"\nModule structure:\n{conv1_stack}"
        )
    else:
        # with specified config and random seed, non-zero dropout update parameter count could take different values
        assert (tpc == upc) or (upc == 1), (
            f"All parameter not updated. Parameters not updated: {not_updated}" f"\nModule structure:\n{conv1_stack}"
        )
```

Here is an example of a full test

```python
@pytest.mark.parametrize("cell_type", ["rnn", "gru"])
@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("batch_size", [20, 1])
def test_sequence_rnn_decoder(cell_type, num_layers, batch_size):
    hidden_size = 256
    vocab_size = 50
    max_sequence_length = 10

    # make repeatable
    set_random_seed(RANDOM_SEED)

    combiner_outputs = {HIDDEN: torch.rand([batch_size, hidden_size])}
    sequence_rnn_decoder = SequenceRNNDecoder(
        hidden_size, vocab_size, max_sequence_length, cell_type, num_layers=num_layers
    )

    output = sequence_rnn_decoder(combiner_outputs, target=None)

    assert list(output.size()) == [batch_size, max_sequence_length, vocab_size]

    # check for parameter updating
    target = torch.randn(output.shape)
    fpc, tpc, upc, not_updated = check_module_parameters_updated(sequence_rnn_decoder, (combiner_outputs, None), target)
    assert upc == tpc, f"Failed to update parameters.  Parameters not update: {not_updated}"

```
