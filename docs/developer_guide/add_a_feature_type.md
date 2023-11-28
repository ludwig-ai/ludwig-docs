# 1. Define the new feature type

Feature types are defined as constants in `ludwig/constants.py`.

Add the name of the new feature type as a constant:

```python
BINARY = "binary"
CATEGORY = "category"
...
NEW_FEATURE_TYPE = "new_feature_type_name"
```

# 2. Add feature classes in a new python module

Source code for feature classes lives under `ludwig/features/`. Add the implementation of the new feature into a new
python module `ludwig/feature/<new_name>_feature.py`.

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

Feature base classes (`InputFeature`, `OutputFeature`) inherit from `LudwigModule` which is itself a
[torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html), so all the usual concerns of
developing Torch modules apply.

Mixin classes provide shared preprocessing/postprocessing state and logic, such as the mapping from categories to
indices, which are shared by input and output feature implementations. Mixin classes are not torch modules, and do not
need to provide a forward method.

```python
class CategoryFeatureMixin(BaseFeatureMixin):
```

# 3. Implement required methods

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
    # inputs_encoded = result of encoder forward pass
    return inputs_encoded
```

__Inputs__

- __inputs__ (torch.Tensor): The input tensor.

__Return__

- (torch.Tensor): Input data encoded by the input feature's encoder.

### input_shape

```python
@property
def input_shape(self) -> torch.Size:
```

__Return__

- (torch.Size): The fully-specified size of the feature's expected input, without batch dimension.

## Output features

### Constructor

```python
def __init__(self, feature: Dict[str, Any], output_features: Dict[str, OutputFeature]):
    super().__init__(feature, output_features)
    self.overwrite_defaults(feature)
    # Initialize any decoder modules, layers, metrics, loss objects, etc.
```

__Inputs__

- __feature__ (dict): contains all feature parameters.
- __output_features__ (dict[Str, OutputFeature]): Dictionary of other output features, only used if this output feature
depends on other outputs.

### logits

Computes feature logits from the combiner output (and any features this feature depends on).

```python
def logits(self, inputs: Dict[str, torch.Tensor],  **kwargs):
    hidden = inputs[HIDDEN]
    # logits = results of decoder operation
    return logits
```

__Inputs__

- __inputs__ (dict): input dictionary which contains the `HIDDEN` key, whose value is the output of the combiner. Will
contain other input keys if this feature depends on other output features.

__Return__

- (torch.Tensor): feature logits.

### create_predict_module

Creates and returns a `torch.nn.Module` that converts raw model outputs (logits) to predictions.
This module is required for exporting models to Torchscript.

```python
def create_predict_module(self) -> PredictModule:
```

__Return__

- (PredictModule): A module whose forward method convert feature logits to predictions.

### output_shape

```python
@property
def output_shape(self) -> torch.Size:
```

__Return__

- (torch.Size): The fully-specified size of the feature's output, without batch dimension.

## Feature Mixins

If your new feature can re-use the preprocessing and postprocessing logic of an existing feature type, you do not need
to implement a new mixin class. If your new feature does require unique pre or post-processing, add a new subclass of
`ludwig.features.base_feature.BaseFeatureMixin`. Implement all abstract methods of `BaseFeatureMixin`.

# 4. Add the new feature classes to the corresponding feature registries

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

# 5. Add schema class definitions for new feature types

In order to validate user input against the expected inputs and input types for the new feature type you have defined,
we need to create schema classes that will autogenerate the json schema required for validation.

If the new feature type will just function as an input feature, you only need to define an input feature schema class.
Here is an example of how the category feature schema classes are defined:

### Input Feature Type

```python
from marshmallow_dataclass import dataclass

from ludwig.constants import CATEGORY
from ludwig.schema import utils as schema_utils
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.features.base import BaseInputFeatureConfig
from ludwig.schema.features.preprocessing.base import BasePreprocessingConfig
from ludwig.schema.features.preprocessing.utils import PreprocessingDataclassField
from ludwig.schema.features.utils import input_config_registry, output_config_registry


@input_config_registry.register(CATEGORY)
@dataclass
class CategoryInputFeatureConfig(BaseInputFeatureConfig):
    """CategoryInputFeatureConfig is a dataclass that configures the parameters used for a category input
    feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=CATEGORY)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type=CATEGORY,
        default="dense",
    )

    tied: str = schema_utils.String(
        default=None,
        allow_none=True,
        description="Name of input feature to tie the weights of the encoder with.  It needs to be the name of a "
        "feature of the same type and with the same encoder parameters.",
    )
```

If the new feature type can also be an output feature type, you will need to define an output feature schema class as
well:

### Output Feature Type

```python
from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils
from ludwig.constants import CATEGORY, SOFTMAX_CROSS_ENTROPY
from ludwig.schema.decoders.base import BaseDecoderConfig
from ludwig.schema.decoders.utils import DecoderDataclassField
from ludwig.schema.features.base import BaseOutputFeatureConfig
from ludwig.schema.features.loss.loss import BaseLossConfig
from ludwig.schema.features.loss.utils import LossDataclassField
from ludwig.schema.features.preprocessing.base import BasePreprocessingConfig
from ludwig.schema.features.preprocessing.utils import PreprocessingDataclassField
from ludwig.schema.features.utils import output_config_registry

@output_config_registry.register(CATEGORY)
@dataclass
class CategoryOutputFeatureConfig(BaseOutputFeatureConfig):
    """CategoryOutputFeatureConfig is a dataclass that configures the parameters used for a category output
    feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type="category_output")

    loss: BaseLossConfig = LossDataclassField(
        feature_type=CATEGORY,
        default=SOFTMAX_CROSS_ENTROPY,
    )

    decoder: BaseDecoderConfig = DecoderDataclassField(
        feature_type=CATEGORY,
        default="classifier",
    )

    reduce_input: str = schema_utils.ReductionOptions(
        default="sum",
        description="How to reduce an input that is not a vector, but a matrix or a higher order tensor, on the first "
        "dimension (second if you count the batch dimension)",
    )

    dependencies: list = schema_utils.List(
        default=[],
        description="List of input features that this feature depends on.",
    )

    reduce_dependencies: str = schema_utils.ReductionOptions(
        default="sum",
        description="How to reduce the dependencies of the output feature.",
    )

    top_k: int = schema_utils.NonNegativeInteger(
        default=3,
        description="Determines the parameter k, the number of categories to consider when computing the top_k "
        "measure. It computes accuracy but considering as a match if the true category appears in the "
        "first k predicted categories ranked by decoder's confidence.",
    )

    calibration: bool = schema_utils.Boolean(
        default=False,
        description="Calibrate the model's output probabilities using temperature scaling.",
    )

```

Lastly, you need to add a reference to the schema class definitions on your input feature type definitions. So for
instance, on the `CategoryInputFeature` class, we need to add a `get_schema_cls` method:

```python
class CategoryInputFeature(CategoryFeatureMixin, InputFeature):

...

    @staticmethod
    def get_schema_cls():
        return CategoryInputFeatureConfig
```

Likewise for the output feature class:

```python
class CategoryOutputFeature(CategoryFeatureMixin, OutputFeature):

...

    @staticmethod
    def get_schema_cls():
        return CategoryOutputFeatureConfig
```
