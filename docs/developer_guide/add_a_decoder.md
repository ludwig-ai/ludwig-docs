# 1. Add a new decoder class

Source code for decoders lives under `ludwig/decoders/`.
Decoders are grouped into modules by their output feature type. For instance, all new sequence decoders should be added
to `ludwig/decoders/sequence_decoders.py`.

!!! note

    A decoder may support multiple output types, if so it should be defined in the module corresponding to its most
    generic supported type. If a decoder is generic with respect to output type, add it to
    `ludwig/decoders/generic_decoders.py`.

To create a new decoder:

1. Define a new decoder class. Inherit from `ludwig.decoders.base.Decoder` or one of its subclasses.
2. Create all layers and state in the `__init__` method, after calling `super().__init__()`.
3. Implement your decoder's forward pass in `def forward(self, combiner_outputs, **kwargs):`.
4. Define a schema class.

Note: `Decoder` inherits from `LudwigModule`, which is itself a [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html),
so all the usual concerns of developing Torch modules apply.

All decoder parameters should be provided as keyword arguments to the constructor, and must have a default value.
For example the `SequenceGeneratorDecoder` decoder takes the following list of parameters in its constructor:

```python
from ludwig.constants import SEQUENCE, TEXT
from ludwig.decoders.base import Decoder
from ludwig.decoders.registry import register_decoder

@register_decoder("generator", [SEQUENCE, TEXT])
class SequenceGeneratorDecoder(Decoder):
    def __init__(
        self,
        vocab_size: int,
        max_sequence_length: int,
        cell_type: str = "gru",
        input_size: int = 256,
        reduce_input: str = "sum",
        num_layers: int = 1,
        **kwargs,
    ):
    super().__init__()
    # Initialize any modules, layers, or variable state
```

# 2. Implement `forward`

Actual computation of activations takes place inside the `forward` method of the decoder.
All decoders should have the following signature:

```python
    def forward(self, combiner_outputs, **kwargs):
        # perform forward pass
        # combiner_hidden_output = combiner_outputs[HIDDEN]
        # ...
        # logits = result of decoder forward pass
        return {LOGITS: logits}
```

__Inputs__

- __combiner_outputs__ (Dict[str, torch.Tensor]): The input tensor, which is the output of a combiner or the combination
of combiner and the activations of any dependent output decoders. The dictionary of combiner outputs includes a tensor of shape `b x h`, where `b` is the batch
size and `h` is the embedding size, or a sequence of embeddings `b x s x h` where `s` is the sequence length.

__Return__

- (Dict[str, torch.Tensor]): A dictionary of decoder output tensors.  Typical decoders will return values for the keys
`LOGITS`, `PREDICTION`, or both (defined in `ludwig.constants`).

# 3. Add the new decoder class to the corresponding decoder registry

Mapping between decoder names in the model definition and decoder classes is made by registering the class in a decoder
registry. The decoder registry is defined in `ludwig/decoders/registry.py`. To register your class,
add the `@register_decoder` decorator on the line above its class definition, specifying the name of the decoder and a
list of supported output feature types:

```python
@register_decoder("generator", [SEQUENCE, TEXT])
class SequenceGeneratorDecoder(Decoder):
```

# 4. Define a schema class

In order to ensure that user config validation for your custom defined decoder functions as desired, we need to define a
schema class to go along with the newly defined decoder. To do this, we use a marshmallow_dataclass decorator on a class
definition that contains all the inputs to your custom decoder as attributes. For each attribute, we use utility
functions to validate that input from the `ludwig.schema.utils` directory. Lastly, we need to put a reference to this
schema class on the custom decoder class. For example:

```python
from marshmallow_dataclass import dataclass

from ludwig.constants import SEQUENCE, TEXT
from ludwig.schema.decoders.base import BaseDecoderConfig
from ludwig.schema.decoders.utils import register_decoder_config
import ludwig.schema.utils as schema_utils

@register_decoder_config("generator", [SEQUENCE, TEXT])
@dataclass
class SequenceGeneratorDecoderConfig(BaseDecoderConfig):

    type: str = schema_utils.StringOptions(options=["generator"], default="generator")
    vocab_size: int = schema_utils.Integer(default=None, description="")
    max_sequence_length: int = schema_utils.Integer(default=None, description="")
    cell_type: str = schema_utils.String(default="gru", description="")
    input_size: int = schema_utils.Integer(default=256, description="")
    reduce_input: str = schema_utils.ReductionOptions(default="sum")
    num_layers: int = schema_utils.Integer(default=1, description="")
```

And lastly you should add a reference to the schema class on the custom decoder:

```python
    @staticmethod
    def get_schema_cls():
        return SequenceGeneratorDecoderConfig
```
