# 1. Add a new encoder class

Source code for encoders lives under `ludwig/encoders`.
New encoder objects should be defined in the corresponding files, for example all new sequence encoders should be added to `ludwig/encoders/sequence_encoders.py`.

All the encoder parameters should be provided as arguments in the constructor with their default values set.
For example the `StackedRNN` encoder takes the following list of arguments in its constructor:

```python
def __init__(
    self,
    should_embed=True,
    vocab=None,
    representation='dense',
    embedding_size=256,
    embeddings_trainable=True,
    pretrained_embeddings=None,
    embeddings_on_cpu=False,
    num_layers=1,
    state_size=256,
    cell_type='rnn',
    bidirectional=False,
    activation='tanh',
    recurrent_activation='sigmoid',
    unit_forget_bias=True,
    recurrent_initializer='orthogonal',
    recurrent_regularizer=None,
    dropout=0.0,
    recurrent_dropout=0.0,
    fc_layers=None,
    num_fc_layers=0,
    fc_size=256,
    use_bias=True,
    weights_initializer='glorot_uniform',
    bias_initializer='zeros',
    weights_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    norm=None,
    norm_params=None,
    fc_activation='relu',
    fc_dropout=0,
    reduce_output='last',
    **kwargs
):
```

Typically all the modules the encoder relies upon are initialized in the encoder's constructor (in the case of the `StackedRNN` encoder these are `EmbedSequence` and `RecurrentStack` modules) so that at the end of the constructor call all the layers are fully described.

Actual computation of activations takes place inside the `call` method of the encoder.
All encoders should have the following signature:

```python
def call(self, inputs, training=None, mask=None):
```

__Inputs__

- __inputs__ (tf.Tensor): input tensor.
- __training__ (bool, default: `None`): boolean indicating whether we are currently training the model or performing inference for prediction.
- __mask__ (tf.Tensor, default: `None`): binary tensor indicating which of the values in the inputs tensor should be masked out.

__Return__

- __hidden__ (tf.Tensor): feature encodings.

The shape of the input tensor and the expected tape of the output tensor varies across feature types.

Encoders are initialized as class member variables in input features object constructors and called inside their `call` methods.


# 2. Add the new encoder class to the corresponding encoder registry

Mapping between encoder names in the model definition and encoder classes in the codebase is done by encoder registries: for example sequence encoder registry is defined in `ludwig/features/sequence_feature.py` inside the `SequenceInputFeature` as:

```python
sequence_encoder_registry = {
    'stacked_cnn': StackedCNN,
    'parallel_cnn': ParallelCNN,
    'stacked_parallel_cnn': StackedParallelCNN,
    'rnn': StackedRNN,
    ...
}
```

All you have to do to make you new encoder available as an option in the model definition is to add it to the appropriate registry.

