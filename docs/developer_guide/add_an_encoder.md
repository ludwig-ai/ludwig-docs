# 1. Add a new encoder class

Source code for encoders lives under `ludwig/encoders/`.
Encoders are grouped into modules by their input feature type. For instance, all new sequence encoders should be added
to `ludwig/encoders/sequence_encoders.py`.

!!! note

    An encoder may support multiple types, if so it should be defined in the module corresponding to its most generic
    supported type. If an encoder is generic with respect to input type, add it to `ludwig/encoders/generic_encoders.py`.

To create a new encoder:

1. Define a new encoder class. Inherit from `ludwig.encoders.base.Encoder` or one of its subclasses.
2. Create all layers and state in the `__init__` method, after calling `super().__init__()`.
3. Implement your encoder's forward pass in `def forward(self, inputs, mask=None):`.
4. Define `@property input_shape` and `@property output_shape`.
5. Define a schema class.

Note: `Encoder` inherits from `LudwigModule`, which is itself a [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html),
so all the usual concerns of developing Torch modules apply.

All encoder parameters should be provided as keyword arguments to the constructor, and must have a default value.
For example the `StackedRNN` encoder takes the following list of parameters in its constructor:

```python
from ludwig.constants import AUDIO, SEQUENCE, TEXT, TIMESERIES
from ludwig.encoders.base import Encoder
from ludwig.encoders.registry import register_encoder

@register_encoder("rnn", [AUDIO, SEQUENCE, TEXT, TIMESERIES])
class StackedRNN(Encoder):
    def __init__(
        self,
        should_embed=True,
        vocab=None,
        representation="dense",
        embedding_size=256,
        embeddings_trainable=True,
        pretrained_embeddings=None,
        embeddings_on_cpu=False,
        num_layers=1,
        max_sequence_length=None,
        state_size=256,
        cell_type="rnn",
        bidirectional=False,
        activation="tanh",
        recurrent_activation="sigmoid",
        unit_forget_bias=True,
        recurrent_initializer="orthogonal",
        dropout=0.0,
        recurrent_dropout=0.0,
        fc_layers=None,
        num_fc_layers=0,
        output_size=256,
        use_bias=True,
        weights_initializer="xavier_uniform",
        bias_initializer="zeros",
        norm=None,
        norm_params=None,
        fc_activation="relu",
        fc_dropout=0,
        reduce_output="last",
        **kwargs,
    ):
    super().__init__()
    # Initialize any modules, layers, or variable state
```

# 2. Implement `forward`, `input_shape`, and `output_shape`

Actual computation of activations takes place inside the `forward` method of the encoder.
All encoders should have the following signature:

```python
    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # perform forward pass
        # ...
        # output_tensor = result of forward pass
        return {"encoder_output": output_tensor}
```

__Inputs__

- __inputs__ (torch.Tensor): input tensor.
- __mask__ (torch.Tensor, default: `None`): binary tensor indicating which values in inputs should be masked out. Note:
mask is not required, and is not implemented for most encoder types.

__Return__

- (dict): A dictionary containing the key `encoder_output` whose value is the encoder output tensor.
`{"encoder_output": output_tensor}`

The `input_shape` and `output_shape` properties must return the fully-specified shape of the encoder's expected input
and output, without batch dimension:

```python
    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        return self.recurrent_stack.output_shape
```

# 3. Add the new encoder class to the encoder registry

Mapping between encoder names in the model definition and encoder classes is made by registering the class in an encoder
registry. The encoder registry is defined in `ludwig/encoders/registry.py`. To register your class,
add the `@register_encoder` decorator on the line above its class definition, specifying the name of the encoder and a
list of supported input feature types:

```python
@register_encoder("rnn", [AUDIO, SEQUENCE, TEXT, TIMESERIES])
class StackedRNN(Encoder):
```

# 4. Define a schema class

In order to ensure that user config validation for your custom defined encoder functions as desired, we need to define a
schema class to go along with the newly defined encoder. To do this, we use a marshmallow_dataclass decorator on a class
definition that contains all the inputs to your custom encoder as attributes. For each attribute, we use utility
functions from the `ludwig.schema.utils` directory to validate that input. Lastly, we need to put a reference to this
schema class on the custom encoder class. For example:

```python
from marshmallow_dataclass import dataclass

from ludwig.constants import SEQUENCE, TEXT
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import register_encoder_config
import ludwig.schema.utils as schema_utils

@register_encoder_config("stacked_rnn", [SEQUENCE, TEXT])
@dataclass
class StackedRNNConfig(BaseEncoderConfig):
        type: str = schema_utils.StringOptions(options=["stacked_rnn"], default="stacked_rnn")
        should_embed: bool = schema_utils.Boolean(default=True, description="")
        vocab: list = schema_utils.List(list_type=str, default=None, description="")
        representation: str = schema_utils.StringOptions(options=["sparse", "dense"], default="dense", description="")
        embedding_size: int = schema_utils.Integer(default=256, description="")
        embeddings_trainable: bool = schema_utils.Boolean(default=True, description="")
        pretrained_embeddings: str = schema_utils.String(default=None, description="")
        embeddings_on_cpu: bool = schema_utils.Boolean(default=False, description="")
        num_layers: int = schema_utils.Integer(default=1, description="")
        max_sequence_length: int = schema_utils.Integer(default=None, description="")
        state_size: int = schema_utils.Integer(default=256, description="")
        cell_type: str = schema_utils.StringOptions(
            options=["rnn", "lstm", "lstm_block", "ln", "lstm_cudnn", "gru", "gru_block", "gru_cudnn"], 
            default="rnn", description=""
        )
        bidirectional: bool = schema_utils.Boolean(default=False, description="")
        activation: str = schema_utils.ActivationOptions(default="tanh", description="")
        recurrent_activation: str = schema_utils.activations(default="sigmoid", description="")
        unit_forget_bias: bool = schema_utils.Boolean(default=True, description="")
        recurrent_initializer: str = schema_utils.InitializerOptions(default="orthogonal", description="")
        dropout: float = schema_utils.FloatRange(default=0.0, min=0, max=1, description="")
        recurrent_dropout: float = schema_utils.FloatRange(default=0.0, min=0, max=1, description="")
        fc_layers: list = schema_utils.DictList(default=None, description="")
        num_fc_layers: int = schema_utils.NonNegativeInteger(default=0, description="")
        output_size: int = schema_utils.Integer(default=256, description="")
        use_bias: bool = schema_utils.Boolean(default=True, description="")
        weights_initialize: str = schema_utils.InitializerOptions(default="xavier_uniform", description="")
        bias_initializer: str = schema_utils.InitializerOptions(default="zeros", description="")
        norm: str = schema_utils.StringOptions(options=["batch", "layer"], default=None, description="")
        norm_params: dict = schema_utils.Dict(default=None, description="")
        fc_activation: str = schema_utils.ActivationOptions(default="relu", description="")
        fc_dropout: float = schema_utils.FloatRange(default=0.0, min=0, max=1, description="")
        reduce_output: str = schema_utils.ReductionOptions(default="last", description="")
```

And lastly you should add a reference to the schema class on the custom encoder:

```python
    @staticmethod
    def get_schema_cls():
        return StackedRNNConfig

```
