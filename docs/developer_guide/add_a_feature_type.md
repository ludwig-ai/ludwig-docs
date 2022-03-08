# 1. Add a new feature class

Source code for feature classes lives under `ludwig/features/`.
Input and output feature classes are defined in the same file, for example `CategoryInputFeature` and
`CategoryOutputFeature` are defined in `ludwig/features/category_feature.py`.

Input features inherit from `ludwig.features.base_feature.InputFeature` and corresponding mixin feature classes:
```python
class CategoryInputFeature(CategoryFeatureMixin, InputFeature):
```

Similarly, output features inherit from the `ludwig.features.base_feature.OutputFeature` and corresponding mixin feature
classes:
```python
class CategoryOutputFeature(CategoryFeatureMixin, OutputFeature):
```

Mixin classes provide shared preprocessing/postprocessing state and logic, such as the mapping from categories to
indices, which are shared by input and output feature implementations. Mixin classes are not torch modules, and do not
need to provide a forward method.

Feature base classes (`InputFeature`, `OutputFeature`) do inherit from `LudwigModule` which is itself a
[torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html), so all the usual concerns of
developing Torch modules apply.

# 2. Implement required methods

## Input features

### Constructor

Feature parameters are provided in a dictionary of key-value pairs as an argument to the constructor.  The `feature`
dictionary should usually be passed to the superclass constructor before initialization:

```python
def __init__(self, feature: [str, Any], encoder_obj=None):
    super().__init__(feature)
    # Initialize any modules, layers, or variable state
```

__Inputs__

- __feature__: (dict) contains all feature config parameters.
- __encoder_obj__: (Encoder, default: `None`) is an encoder object of the supported type (category encoder, binary
encoder, etc.). Input features typically create their own encoder, `encoder_obj` is only specified when two input
features share the same encoder.

### forward

All input features must implement the `forward` method with the following signature:

```python
def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    # perform forward pass
    # ...
    # inputs_encoded = result of forward pass
    return inputs_encoded
```

### input_shape


### output_shape


## Output features


### Constructor

```python
    def __init__(self, feature: Dict[str, Any], output_features: Dict[str, OutputFeature]):
        super().__init__(feature, output_features)
        self.overwrite_defaults(feature)
        # Initialize any modules, layers, or variable state
```

__Inputs__

- __feature__ (dict): contains all feature parameters.
- __decoder_obj__ (\*Decoder, default: `None`): is a decoder object of the type supported (a category decoder, binary decoder, etc.). It is used only when two output features share the decoder.

### logits

```python
    def logits(self, inputs: Dict[str, torch.Tensor], target=None):
        return self.decoder_obj(inputs, target=target)
```

__Inputs__

- __inputs__ (dict): input dictionary that is the output of the combiner.

__Return__

- __hidden__ (torch.Tensor): feature logits.

### create_predict_module

```python
    def create_predict_module(self) -> PredictModule:
        return _SequencePredict()
```

__Inputs__

- __inputs__ (dict): input dictionary that contains the output of the combiner and the logits function.

__Return__

- __hidden__ (dict): contains predictions, probabilities and logits.

### input_shape

```python
    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.input_size])

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

```

### output_shape


# 3. Add the new feature class to the corresponding feature registry

Input and output feature registries are defined in `ludwig/features/feature_registries.py`. Import your new feature
classes, and add them to the appropriate registry dictionaries:

```python
base_type_registry = {
    CATEGORY: CategoryFeatureMixin,
...
}
input_type_registry = {
    CATEGORY: CategoryInputFeature,
...
}
output_type_registry = {
    CATEGORY: CategoryOutputFeature,
...
}
```
