Combiners are responsible for combining the outputs of one or more input features and providing a combined
representation to one or more output features.

There is one additional complexity to be aware of: input features may either output vectors, or __sequences__ of
vectors. Thus, a combiner may have to handle a mix of sequence and scalar input features. SequenceConcatCombiner, for
example, resolves this by requiring that all input sequences be of the same length. It will raise a `ValueError`
exception if they are not. SequenceConcatCombiner tiles scalar inputs to the sequence length before concatenation, so
all input features are treated as sequences of the same length.

New combiners should make it clear in their doc strings whether they support sequence inputs, declare any requirements
on sequence length, type, or dimension, and validate their input features.

Users can specify which combiner to use in the `combiner` section of the configuration, if a combiner is not specified
the `concat` combiner will be used.

To add a new combiner:

1. Create a dataclass to represent the new combiner configuration.
2. Define a new combiner class. Inherit from `ludwig.combiners.Combiner` or one of its subclasses.
3. Create all layers and state in the `__init__` method, after calling `super().__init__(input_features)`.
4. Implement your encoder's forward pass in `def forward(self, inputs: Dict):`.
5. Add the new combiner to the combiner registry.

## 1. Define combiner configuration

The combiner configuration is a `dataclass` whose properties are the configuration parameters of the combiner. All
fields should have a type and a default value. The `ludwig.utils.schema_utils.py` module provides convenience methods
for specifying the valid types and ranges of a combiner config. For example, the `TransformerCombiner` has the following
config schema:

```python
import ludwig.utils.schema_utils as schema

@dataclass
class TransformerCombinerConfig:
    num_layers: int = schema.PositiveInteger(default=1)
    hidden_size: int = schema.NonNegativeInteger(default=256)
    num_heads: int = schema.NonNegativeInteger(default=8)
    transformer_output_size: int = schema.NonNegativeInteger(default=256)
    dropout: float = schema.FloatRange(default=0.1, min=0, max=1)
    fc_layers: Optional[List[Dict[str, Any]]] = schema.DictList()
    num_fc_layers: int = schema.NonNegativeInteger(default=0)
    output_size: int = schema.PositiveInteger(default=256)
    use_bias: bool = True
    weights_initializer: Union[str, Dict] = schema.InitializerOrDict(default="xavier_uniform")
    bias_initializer: Union[str, Dict] = schema.InitializerOrDict(default="zeros")
    norm: Optional[str] = schema.StringOptions(["batch", "layer"])
    norm_params: Optional[dict] = schema.Dict()
    fc_activation: str = "relu"
    fc_dropout: float = schema.FloatRange(default=0.0, min=0, max=1)
    fc_residual: bool = False
    reduce_output: Optional[str] = schema.ReductionOptions(default="mean")
```

# 2. Add a new combiner class

Source code for combiners lives in `ludwig/combiners/`. Add a new python module which declares a new combiner class. For
this example, we'll show how to implement a simplified version of `transformer` combiner.

!!! note

    At present, all combiners are defined in `ludwig/combiners/combiners.py`. However, for new combiners we recommend
    creating a new python module with a name corresponding to the new combiner class.

```python

@register_combiner(name="transformer")
class TransformerCombiner(Combiner):
    def __init__(
        self, input_features: Dict[str, "InputFeature"] = None, config: TransformerCombinerConfig = None, **kwargs
    ):
        super().__init__(input_features)
        self.name = "TransformerCombiner"

    def forward(
        self,
        inputs,  # encoder outputs
    ) -> Dict:

    @staticmethod
    def get_schema_cls():
        return TransformerCombinerConfig
```

Implement `@staticmethod def get_schema_cls():` and return the class name of your config schema.

# 3. Implement Constructor

The combiner constructor will be initialized with a dictionary of the input features and the combiner config. The
constructor must pass the input features to the superclass constructor, set its 'name' property, then create its own
layers and state.

You may want to use the `input_features dictionary` to get information about the size and type of the inputs, which
affect what resources the combiner needs to allocate. For example, the `transformer` combiner treats its input features
as a sequence, where the sequence length is the number of features: `self.sequence_size = len(self.input_features)`.

```python
    def __init__(
        self,
        input_features: Dict[str, "InputFeature"] = None,
        config: TransformerCombinerConfig = None,
        **kwargs
    ):
        super().__init__(input_features)
        self.name = "TransformerCombiner"
        logger.debug(f" {self.name}")

        # ...

        self.sequence_size = len(self.input_features)

        logger.debug("  TransformerStack")
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

# 4. Implement `forward`

The `forward` method of the combiner should combine the input feature embeddings into a single output tensor, which will
be passed to output feature decoders. Each key in inputs is an input feature name, and the corresponding value is a
dictionary of the input feature outputs. Each feature output guaranteed to contain an `encoder_output` key, and may contain other outputs)

`forward` returns a dictionary of torch.Tensors which must contain an `encoder_output` key. It may optionally return
additional value that might be useful for output feature decoding, loss computation, or explanation. For example,
`TabNetCombiner` returns its sparse attention masks (`attention_masks`, which are useful to see which features were
attended to in each prediction step.

For example, the following is a simplified version of `TransformerCombiner`'s forward method:

```python
    def forward(
        self, inputs: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
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

- __inputs__ (Dict[str, Dict[str, torch.Tensor]]): A dictionary of input feature outputs, keyed by the input feature
names. Each input feature output is guaranteed to include `encoder_output`, and may include other outputs as well.

__Return__

- (dict): A dictionary containing the key `combiner_output` whose value is the combiner output tensor.
`{"combiner_output": output_tensor}`.

# 5. Add the new combiner class to the combiner registry

Mapping between combiner names in the model definition and combiner classes is made by registering the class in a
combiner registry. The combiner registry is defined in `ludwig/combiners/combiners.py`. To register your class, add the
`@register_encoder` decorator on the line above its class definition, specifying the name of the combiner:

```python
@register_combiner(name="transformer")
class TransformerCombiner(Combiner):
```
