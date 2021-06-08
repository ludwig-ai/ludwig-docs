 1. Add a new feature class
---------------------------

Souce code for feature classes lives under `ludwig/features`.
Input and output feature classes are defined in the same file, for example `CategoryInputFeature` and `CategoryOutputFeature` are defined in `ludwig/features/category_feature.py`.

An input features inherit from the `InputFeature` and corresponding mixin feature classes, for example `CategoryInputFeature` inherits from `CategoryFeatureMixin` and `InputFeature`.

Similarly, output features inherit from the `OutputFeature` and corresponding base feature classes, for example `CategoryOutputFeature` inherits from `CategoryFeatureMixin` and `OutputFeature`.

Feature parameters are provided in a dictionary of key-value pairs as an argument to the input or output feature constructor which contains default parameter values as well.

## Input features

All input features should implement `__init__` and `call` methods with the following signatures:


### `__init__`

```python
def __init__(self, feature, encoder_obj=None):
```

__Inputs__


- __feature__: (dict) contains all feature parameters.
- __encoder_obj__: (*Encoder, default: `None`) is an encoder object of the type supported (a cateory encoder, binary encoder, etc.). It is used only when two input features share the encoder.


### `call`

```python
def call(self, inputs, training=None, mask=None):
```

__Inputs__

- __inputs__ (tf.Tensor): input tensor.
- __training__ (bool, default: `None`): boolean indicating whether we are currently training the model or performing inference for prediction.
- __mask__ (tf.Tensor, default: `None`): binary tensor indicating which of the values in the inputs tensor should be masked out.

__Return__

- __hidden__ (tf.Tensor): feature encodings.


## Output features

All input features should implement `__init__`, `logits` and `predictions` methods with the following signatures:


### `__init__`

```python
def __init__(self, feature, encoder_obj=None):
```

__Inputs__


- __feature__ (dict): contains all feature parameters.
- __decoder_obj__ (*Decoder, default: `None`): is a decoder object of the type supported (a cateory decoder, binary decoder, etc.). It is used only when two output features share the decoder.

### `logits`

```python
def call(self, inputs, **kwargs):
```

__Inputs__

- __inputs__ (dict): input dictionary that is the output of the combiner.

__Return__

- __hidden__ (tf.Tensor): feature logits.

### `predictions`

```python
def call(self, inputs, **kwargs):
```

__Inputs__

- __inputs__ (dict): input dictionary that contains the output of the combiner and the logits function.

__Return__

- __hidden__ (dict): contains predictions, probabilities and logits.


 2. Add the new feature class to the corresponding feature registry
-------------------------------------------------------------------

Input and output feature registries are defined in `ludwig/features/feature_registries.py`.