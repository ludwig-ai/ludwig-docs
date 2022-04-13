Combiners are responsible for combining the outputs of one or more input features into a single combined representation,
which is usually a vector, but may also be a sequence of vectors or some other higher-dimensional tensor. One or more
output features will use this combined representation to generate predictions.

Users can specify which combiner to use in the `combiner` section of the configuration, if a combiner is not specified
the `concat` combiner will be used.

Recall the ECD (Encoder, Combiner, Decoder) data flow architecture: all input feature outputs flow into the combiner,
and the combiner's output flows into all output features.

```
+-----------+                      +-----------+
|Input      |                      | Output    |
|Feature 1  +-+                  +-+ Feature 1 + ---> Prediction 1
+-----------+ |                  | +-----------+
+-----------+ |   +----------+   | +-----------+
|...        +---> | Combiner +---> |...        +
+-----------+ |   +----------+   | +-----------+
+-----------+ |                  | +-----------+
|Input      +-+                  +-+ Output    |
|Feature N  |                      | Feature N + ---> Prediction N
+-----------+                      +-----------+
```

There is an additional complication to keep in mind: input features may either output vectors, or __sequences__ of
vectors. Thus, a combiner may have to handle a mix of input features whose outputs are of different dimensionality.
`SequenceConcatCombiner`, for example, resolves this by requiring that all input sequences be of the same length. It
will raise a `ValueError` exception if they are not. `SequenceConcatCombiner` tiles non-sequence inputs to the sequence
length before concatenation, processing all input features as sequences of the same length.

New combiners should make it clear in their doc strings if they support sequence inputs, declare any requirements on
sequence length, type, or dimension, and validate their input features.

In this guide we'll outline how to extend Ludwig by adding a new combiner, using the `transformer` combiner as a
template. At a high level, to add a new combiner:

1. Define a dataclass to represent the combiner configuration.
2. Create a new combiner class inheriting from `ludwig.combiners.Combiner` or one of its subclasses.
3. Allocate all layers and state in the `__init__` method.
4. Implement your combiner's forward pass in `def forward(self, inputs: Dict):`.
5. Add tests.
6. Add the new combiner to the combiner registry.

## 1. Define combiner configuration

The combiner configuration is a `dataclass` (that must extend `BaseCombinerConfig`) whose properties are the configuration
parameters of the combiner. All fields should have a type and a default value. The `ludwig.utils.schema_utils.py` module
provides convenience methods for specifying the valid types and ranges of a combiner config. For example, the
`TransformerCombiner` has the following config schema:

```python
import ludwig.marshmallow.marshmallow_schema_utils as schema
from ludwig.combiners.combiners import BaseCombinerConfig

@dataclass
class TransformerCombinerConfig(BaseCombinerConfig):
    num_layers: int = schema.PositiveInteger(default=1)
    hidden_size: int = schema.NonNegativeInteger(default=256)
    num_heads: int = schema.NonNegativeInteger(default=8)
    transformer_output_size: int = schema.NonNegativeInteger(default=256)
    dropout: float = schema.FloatRange(default=0.1, min=0, max=1)
    fc_layers: Optional[List[Dict[str, Any]]] = schema.DictList()
    num_fc_layers: int = schema.NonNegativeInteger(default=0)
    output_size: int = schema.PositiveInteger(default=256)
    use_bias: bool = True
    weights_initializer: Union[str, Dict] = \
        schema.InitializerOrDict(default="xavier_uniform")
    bias_initializer: Union[str, Dict] = \
        schema.InitializerOrDict(default="zeros")
    norm: Optional[str] = schema.StringOptions(["batch", "layer"])
    norm_params: Optional[dict] = schema.Dict()
    fc_activation: str = "relu"
    fc_dropout: float = schema.FloatRange(default=0.0, min=0, max=1)
    fc_residual: bool = False
    reduce_output: Optional[str] = schema.ReductionOptions(default="mean")
```

# 2. Add a new combiner class

Source code for combiners lives in `ludwig/combiners/`. Add a new python module which declares a new combiner class. For
this example, we'll show how to implement a simplified version of `transformer` combiner which would be defined in
`transformer_combiner.py`.

!!! note

    At present, all combiners are defined in `ludwig/combiners/combiners.py`. However, for new combiners we recommend
    creating a new python module with a name corresponding to the new combiner class.

```python

@register_combiner(name="transformer")
class TransformerCombiner(Combiner):
    def __init__(
        self,
        input_features: Dict[str, InputFeature] = None,
        config: TransformerCombinerConfig = None,
        **kwargs
    ):
        super().__init__(input_features)
        self.name = "TransformerCombiner"

    def forward(
        self,
        inputs: Dict,
    ) -> Dict[str: torch.Tensor]:

    @staticmethod
    def get_schema_cls():
        return TransformerCombinerConfig
```

Implement `@staticmethod def get_schema_cls():` and return the class name of your config schema.

# 3. Implement Constructor

The combiner constructor will be initialized with a dictionary of the input features and the combiner config. The
constructor must pass the input features to the superclass constructor, set its `name` property, then create its own
layers and state.

The `input_features` dictionary is passed in to the constructor to make information about the number, size, and type of
the inputs accessible. This may determine what resources the combiner needs to allocate. For example, the `transformer`
combiner treats its input features as a sequence, where the sequence length is the number of features. We can determine
the sequence length here as `self.sequence_size = len(self.input_features)`.

```python
    def __init__(
        self,
        input_features: Dict[str, InputFeature] = None,
        config: TransformerCombinerConfig = None,
        **kwargs
    ):
        super().__init__(input_features)
        self.name = "TransformerCombiner"
        # ...
        self.sequence_size = len(self.input_features)

        self.transformer_stack = TransformerStack(
            input_size=config.hidden_size,
            sequence_size=self.sequence_size,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            output_size=config.transformer_output_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
        # ...
```

# 4. Implement `forward` method

The `forward` method of the combiner should combine the input feature representations into a single output tensor, which
will be passed to output feature decoders. Each key in inputs is an input feature name, and the respective value is a
dictionary of the input feature's outputs. Each feature output dictionary is guaranteed to contain an `encoder_output`
key, and may contain other outputs depending on the encoder.

`forward` returns a dictionary mapping strings to tensors which must contain a `combiner_output` key. It may optionally
return additional values that might be useful for output feature decoding, loss computation, or explanation. For
example, `TabNetCombiner` returns its sparse attention masks (`attention_masks`, and `aggregated_attention_masks`) which
are useful to see which input features were attended to in each prediction step.

For example, the following is a simplified version of `TransformerCombiner`'s forward method:

```python
    def forward(
        self, inputs: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        encoder_outputs = [inputs[k]["encoder_output"] for k in inputs]

        # ================ Flatten ================
        batch_size = encoder_outputs[0].shape[0]
        encoder_outputs = [
            torch.reshape(eo, [batch_size, -1]) for eo in encoder_outputs
        ]

        # ================ Project & Concat ================
        projected = [
            self.projectors[i](eo) for i, eo in enumerate(encoder_outputs)
        ]
        hidden = torch.stack(projected)
        hidden = torch.permute(hidden, (1, 0, 2))

        # ================ Transformer Layers ================
        hidden = self.transformer_stack(hidden)

        # ================ Sequence Reduction ================
        if self.reduce_output is not None:
            hidden = self.reduce_sequence(hidden)
            hidden = self.fc_stack(hidden)

        return_data = {"combiner_output": hidden}
        return return_data
```

__Inputs__

- __inputs__ (`Dict[str, Dict[str, torch.Tensor]]`): A dictionary of input feature outputs, keyed by the input feature
names. Each input feature output dictionary is guaranteed to include `encoder_output`, and may include other key/value
pairs depending on the input feature's encoder.

__Return__

- (`Dict[str, torch.Tensor]`): A dictionary containing the required key `combiner_output` whose value is the combiner
output tensor, and any other optional output key/value pairs.

# 5. Add new class to the registry

Mapping between combiner names in the model config and combiner classes is made by registering the class in the combiner
registry. The combiner registry is defined in `ludwig/combiners/combiners.py`. To register your class, add the
`@register_combiner` decorator on the line above its class definition, specifying the name of the combiner:

```python
@register_combiner(name="transformer")
class TransformerCombiner(Combiner):
```

# 6. Run schema extraction script

Behind the scenes, the `BaseCombinerConfig` is a `marshmallow` object that generates a schema which functions as
intermediary for converting (and validating) a dict into the relevant combiner config. Other Ludwig functionality
requires this schema to be committed as well, so after creating your combiner class finally run the extraction
script found (from the context of the root directory) in `scripts/extract_schema.py`. Afterwards, you should see
a new file named `YourNewCombinerConfig.json` under `ludwig/marshmallow/generated` in your git environment. Track
and commit this file as well.

# 7. Add tests

Add a corresponding unit test module to `tests/ludwig/combiners`, using the name of your combiner module prefixed by
`test_` i.e. `test_transformer_combiner.py`.

At a minimum, the unit test should ensure that:

1. The combiner's forward pass succeeds for all feature types it supports.
2. The combiner fails in expected ways when given unsupported input. (Skip this if the combiner supports all input
feature types.)
3. The combiner produces output of the correct type and dimensionality given a variety of configs.

Use `@pytest.mark.parametrize` to parameterize your test with different configurations, also test edge cases:

```python
@pytest.mark.parametrize("output_size", [8, 16])
@pytest.mark.parametrize("transformer_output_size", [4, 12])
def test_transformer_combiner(
        encoder_outputs: tuple,
        transformer_output_size: int,
        output_size: int) -> None:
    encoder_outputs_dict, input_feature_dict = encoder_outputs
```

For examples of combiner tests, see `tests/ludwig/combiners/test_combiners.py`.

For more detail about unit testing in Ludiwg, see also [Unit Test Design Guidelines](../unit_test_design_guidelines).
