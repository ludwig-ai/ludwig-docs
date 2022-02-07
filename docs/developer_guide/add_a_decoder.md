# 1. Add a new decoder class

Source code for decoders lives under `ludwig/decoders/`.
New decoder objects should be defined in the corresponding files, for example all new sequence decoders should be added to `ludwig/decoders/sequence_decoders.py`.

All the decoder parameters should be provided as arguments in the constructor with their default values set.
For example the `SequenceGeneratorDecoder` decoder takes the following list of arguments in its constructor:

```python
def __init__(
    self,
    num_classes,
    cell_type='rnn',
    state_size=256,
    embedding_size=64,
    beam_width=1,
    num_layers=1,
    attention=None,
    tied_embeddings=None,
    is_timeseries=False,
    max_sequence_length=0,
    use_bias=True,
    weights_initializer='glorot_uniform',
    bias_initializer='zeros',
    weights_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    reduce_input='sum',
    **kwargs
):
```

Decoders are initialized as class member variables in output feature object constructors and called inside `call` methods.

# 2. Add the new decoder class to the corresponding decoder registry

Mapping between decoder names in the model definition and decoder classes in the codebase is done by encoder registries: for example sequence encoder registry is defined in `ludwig/features/sequence_feature.py` inside the `SequenceOutputFeature` as:

```python
sequence_decoder_registry = {
    'generator': Generator,
    'tagger': Tagger
}
```

All you have to do to make you new decoder available as an option in the model definition is to add it to the appropriate registry.
