# General Guidelines

- Unit test for all ludwig modules.  **Rationale**:  Confirms quality of the code base.

- At minimum, tests related to tensors should confirm no errors are raised when processing tensor(s) and that resulting tensors are of correct shape and type.  **Rationale**:  This provides minimal assurance that the function is operating as expected.  **Illustrative code fragment**:

```
    # pass input features through combiner
    # no exception should be raised in this call
    combiner_output = combiner(input_features)
    
    # check for required attributes in the generated output
    assert hasattr(combiner, 'input_dtype')
    assert hasattr(combiner, 'output_shape')

    # check for correct data type 
    assert isinstance(combiner_output, dict)

    # required key present
    assert 'combiner_output' in combiner_output

    # check for correct output shape
    assert combiner_output['combiner_output'].shape \
           == (batch_size, *combiner.output_shape)

```

- Test combination of parameters that drive through all code paths.  **Rationale**: Ensures correctness of the function under a wide variety of situations.  **Illustrative code fragment**:

```
# test combination of parameters to exercise all code paths
@pytest.mark.parametrize(
    'num_total_blocks, num_shared_blocks',
    [(4, 2), (6, 4), (3, 1)]
)
@pytest.mark.parametrize('virtual_batch_size', [None, 7])
@pytest.mark.parametrize('size', [4, 12])
@pytest.mark.parametrize('input_size', [2, 6])
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

- Test edge cases when possible, e.g., only one or no item; or data structure is at maximum limit.  **Rationale**: Ensures robustness of code.  **Illustrative code fragment**:

```
@pytest.mark.parametrize(
    'feature_list',  # defines parameter for fixture features_to_test()
    [
        [  # single numeric, single categorical
            ('numerical', [BATCH_SIZE, 1]),  # passthrough encoder
            ('category', [BATCH_SIZE, 64])   # dense encoder
        ],
        [  # multiple numeric, multiple categorical
            ('binary', [BATCH_SIZE, 1]),  # passthrough encoder
            ('category', [BATCH_SIZE, 16]),  # dense encoder
            ('numerical', [BATCH_SIZE, 1]),  # passthrough encoder
            ('category', [BATCH_SIZE, 48]),  # dense encoder
            ('numerical', [BATCH_SIZE, 32])  # dense encoder
        ],
        [  # only numeric features
            ('binary', [BATCH_SIZE, 1]),  # passthrough encoder
            ('numerical', [BATCH_SIZE, 1])  # passthrough encoder
        ],
        [  # only category features
            ('category', [BATCH_SIZE, 16]),  # dense encoder
            ('category', [BATCH_SIZE, 8])   # dense encoder
        ],
        [  # only single numeric feature
            ('numerical', [BATCH_SIZE, 1])  # passthrough encoder
        ],
        [  # only single category feature
            ('category', [BATCH_SIZE, 8])   # dense encoder
        ]
    ]
)
```

- When testing a complex layer / module / encoder / combiner / model, make sure that all the variables / weights get updates after one training step.  **Rationale**: Ensures the computation graph doesn’t contain dangling nodes. This catches issues that don’t make the code crash, that are not caught by looking at the loss scores, are they likely will go down, and that are not caught by training the model to convergence (albeit to a usually bad loss), For more details see [this link](https://thenerdstation.medium.com/how-to-unit-test-machine-learning-code-57cf6fd81765).

- Reuse the already established setup for similar tests or establish a new reusable one.  **Rationale**: Ensures consistent test coverage and reduces effort to develop and maintain test.  Examples of reusable test setup can be found in `tests/conftest.py`.  This module contains reusable `pytest.fixtures` that have applicability across many tests.

- TorchTyping tests -- add typechecking based on torch input/outputs. **Rationale**: Allows for stronger typechecking of the form \[batch_size, dim1, dim2, ...\]. We can use the torch typing library to add these tests.

- Overfitting tests — ensure that a small ECD model is able to overfit on a small dataset. **Rationale**: Ensures that models are able to converge on reasonable targets and catches any unscoped issues that aren’t captured by shape/weight update tests.

# Implementation Guidelines

- Use pytest.mark.parametrize for test setup.  **Rationale**:  Automates setup for test cases.  **Illustrative code fragment**:

```
@pytest.mark.parametrize('enc_should_embed', [True, False])
@pytest.mark.parametrize('enc_reduce_output', [None, 'sum'])
@pytest.mark.parametrize('enc_norm', [None, 'batch', 'layer'])
@pytest.mark.parametrize('enc_num_layers', [1, 2])
@pytest.mark.parametrize('enc_dropout', [0, 0.2])
@pytest.mark.parametrize('enc_cell_type', ['rnn', 'gru', 'lstm'])
@pytest.mark.parametrize('enc_encoder', ENCODERS + ['passthrough'])
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

- Use temporary directories for any generated data.  **Rationale**: Avoids polluting the local file system when testing locally.  **Illustrative code fragment**:

```
import tempfile

def test_export_neuropod_cli(csv_filename):
    """Test exporting Ludwig model to neuropod format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_filename = os.path.join(tmpdir,
                                       'config.yaml')
        dataset_filename = _prepare_data(csv_filename,
                                         config_filename)
        _run_ludwig('train',
                    dataset=dataset_filename,
                    config_file=config_filename,
                    output_directory=tmpdir)
        _run_ludwig('export_neuropod',
                    model_path=os.path.join(tmpdir, 'experiment_run', 'model'),
                    output_path=os.path.join(tmpdir, 'neuropod')
                    )
```

- Consolidate tests that require common setup, e.g., training/test data sets, into a single module and use appropriately scoped `@pytest.fixture` to reduce overhead of repeatedly performing setup.  **Rationale**: Reduce test run-time.

- When you use a random function, always specify a seed.  **Rationale**: If a test fails, it would be bad not to be able to replicate it exactly.  **Illustrative code fragment**:

```
import torch 

RANDOM_SEED = 1919 

# setup synthetic tensor, ensure reproducibility by setting seed
torch.manual_seed(RANDOM_SEED)
input_tensor = torch.randn([BATCH_SIZE, input_size], dtype=torch.float32)
```
