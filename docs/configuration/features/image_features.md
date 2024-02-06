{% from './macros/includes.md' import render_fields, render_yaml %}

Input image features are transformed into a float valued tensors of size `N x C x H x W` (where `N` is the size of the
dataset, `C` is the number of channels, and `H x W` is the height and width of the image (can be specified by the user).
These tensors are added to HDF5 with a key that reflects the name of column in the dataset.

The column name is added to the JSON file, with an associated dictionary containing preprocessing information about the
sizes of the resizing.

# Supported Image Formats

The number of channels in the image is determined by the image format. The following table lists the supported image
formats and the number of channels.

| Format               | Number of channels |
| -------------------- | ------------------ |
| Grayscale            | 1                  |
| Grayscale with Alpha | 2                  |
| RGB                  | 3                  |
| RGB with Alpha       | 4                  |

# Preprocessing

During preprocessing, raw image files are transformed into numpy arrays and saved in the hdf5 format.

!!! note
    Images passed to an image encoder are expected to have the same size. If images are different sizes, by default they
    will be resized to the dimensions of the first image in the dataset. Optionally, a `resize_method` together with a
    target `width` and `height` can be specified in the feature preprocessing parameters, in which case all images will
    be resized to the specified target size.

{% set preprocessing = get_feature_preprocessing_schema("image") %}
{{ render_yaml(preprocessing, parent="preprocessing") }}

Parameters:

{{ render_fields(schema_class_to_fields(preprocessing)) }}

Preprocessing parameters can also be defined once and applied to all image input features using the [Type-Global Preprocessing](../defaults.md#type-global-preprocessing) section.

# Input Features

The encoder parameters specified at the feature level are:

- `tied` (default `null`): name of another input feature to tie the weights of the encoder with. It needs to be the name of
a feature of the same type and with the same encoder parameters.
- `augmentation` (default `False`): specifies image data augmentation operations to generate synthetic training data.  More details on image augmentation can be found [here](#image-augmentation).

Example image feature entry in the input features list:

```yaml
name: image_column_name
type: image
tied: null
encoder: 
    type: stacked_cnn
```

The available encoder parameters are:

- `type` (default `stacked_cnn`): the possible values are `stacked_cnn`, `resnet`, `mlp_mixer`, `vit`, and [TorchVision Pretrained Image Classification models](#torchvision-pretrained-model-encoders).

Encoder type and encoder parameters can also be defined once and applied to all image input features using the [Type-Global Encoder](../defaults.md#type-global-encoder) section.

## Encoders

### Convolutional Stack Encoder (`stacked_cnn`)

Stack of 2D convolutional layers with optional normalization, dropout, and down-sampling pooling layers, followed by an
optional stack of fully connected layers.

Convolutional Stack Encoder takes the following optional parameters:

{% set image_encoder = get_encoder_schema("image", "stacked_cnn") %}
{{ render_yaml(image_encoder, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(image_encoder, exclude=["type"])) }}

### MLP-Mixer Encoder

Encodes images using MLP-Mixer, as described in [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601).
MLP-Mixer divides the image into equal-sized patches, applying fully connected layers to each patch to compute per-patch
representations (tokens) and combining the representations with fully-connected mixer layers.

The MLP-Mixer Encoder takes the following optional parameters:

{% set image_encoder = get_encoder_schema("image", "mlp_mixer") %}
{{ render_yaml(image_encoder, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(image_encoder, exclude=["type"])) }}

### TorchVision Pretrained Model Encoders

Twenty TorchVision pretrained image classification models are available as Ludwig image encoders.  The available models
are:

- `AlexNet`
- `ConvNeXt`
- `DenseNet`
- `EfficientNet`
- `EfficientNetV2`
- `GoogLeNet`
- `Inception V3`
- `MaxVit`
- `MNASNet`
- `MobileNet V2`
- `MobileNet V3`
- `RegNet`
- `ResNet`
- `ResNeXt`
- `ShuffleNet V2`
- `SqueezeNet`
- `SwinTransformer`
- `VGG`
- `VisionTransformer`
- `Wide ResNet`

See [TorchVison documentation](https://pytorch.org/vision/stable/models.html#classification) for more details.

Ludwig encoders parameters for TorchVision pretrained models:

#### AlexNet

{% set image_encoder = get_encoder_schema("image", "alexnet") %}
{{ render_yaml(image_encoder, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(image_encoder, exclude=["type"])) }}

#### ConvNeXt

{% set image_encoder = get_encoder_schema("image", "convnext") %}
{{ render_yaml(image_encoder, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(image_encoder, exclude=["type"])) }}

#### DenseNet

{% set image_encoder = get_encoder_schema("image", "densenet") %}
{{ render_yaml(image_encoder, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(image_encoder, exclude=["type"])) }}

#### EfficientNet

{% set image_encoder = get_encoder_schema("image", "efficientnet") %}
{{ render_yaml(image_encoder, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(image_encoder, exclude=["type"])) }}

#### GoogLeNet

{% set image_encoder = get_encoder_schema("image", "googlenet") %}
{{ render_yaml(image_encoder, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(image_encoder, exclude=["type"])) }}

#### Inception V3

{% set image_encoder = get_encoder_schema("image", "inceptionv3") %}
{{ render_yaml(image_encoder, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(image_encoder, exclude=["type"])) }}

#### MaxVit

{% set image_encoder = get_encoder_schema("image", "maxvit") %}
{{ render_yaml(image_encoder, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(image_encoder, exclude=["type"])) }}

#### MNASNet

{% set image_encoder = get_encoder_schema("image", "mnasnet") %}
{{ render_yaml(image_encoder, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(image_encoder, exclude=["type"])) }}

#### MobileNet V2

{% set image_encoder = get_encoder_schema("image", "mobilenetv2") %}
{{ render_yaml(image_encoder, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(image_encoder, exclude=["type"])) }}

#### MobileNet V3

{% set image_encoder = get_encoder_schema("image", "mobilenetv3") %}
{{ render_yaml(image_encoder, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(image_encoder, exclude=["type"])) }}

#### RegNet

{% set image_encoder = get_encoder_schema("image", "regnet") %}
{{ render_yaml(image_encoder, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(image_encoder, exclude=["type"])) }}

#### ResNet

{% set image_encoder = get_encoder_schema("image", "resnet") %}
{{ render_yaml(image_encoder, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(image_encoder, exclude=["type"])) }}

#### ResNeXt

{% set image_encoder = get_encoder_schema("image", "resnext") %}
{{ render_yaml(image_encoder, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(image_encoder, exclude=["type"])) }}

#### ShuffleNet V2

{% set image_encoder = get_encoder_schema("image", "shufflenet_v2") %}
{{ render_yaml(image_encoder, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(image_encoder, exclude=["type"])) }}

#### SqueezeNet

{% set image_encoder = get_encoder_schema("image", "squeezenet") %}
{{ render_yaml(image_encoder, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(image_encoder, exclude=["type"])) }}

#### SwinTransformer

{% set image_encoder = get_encoder_schema("image", "swin_transformer") %}
{{ render_yaml(image_encoder, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(image_encoder, exclude=["type"])) }}

#### VGG

{% set image_encoder = get_encoder_schema("image", "vgg") %}
{{ render_yaml(image_encoder, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(image_encoder, exclude=["type"])) }}

#### VisionTransformer

{% set image_encoder = get_encoder_schema("image", "vit") %}
{{ render_yaml(image_encoder, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(image_encoder, exclude=["type"])) }}

#### Wide ResNet

{% set image_encoder = get_encoder_schema("image", "wide_resnet") %}
{{ render_yaml(image_encoder, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(image_encoder, exclude=["type"])) }}

**Note**:

- At this time Ludwig supports only the `DEFAULT` pretrained weights, which are the best available weights for a specific
model. More details on `DEFAULT` weights can be found in this
[blog post](https://pytorch.org/blog/introducing-torchvision-new-multi-weight-support-api/).
- Some TorchVision pretrained models consume large amounts of memory.  These `model_variant` required more than
12GB of memory:
  - `efficientnet_torch`: `b7`
  - `regnet_torch`: `y_128gf`
  - `vit_torch`: `h_14`

### U-Net Encoder

The U-Net encoder is based on
[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).
The encoder implements the contracting downsampling path of the U-Net stack.

U-Net Encoder takes the following optional parameters:

{% set image_encoder = get_encoder_schema("image", "unet") %}
{{ render_yaml(image_encoder, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(image_encoder, exclude=["type"])) }}

### Deprecated Encoders (planned to remove in v0.8)

#### Legacy ResNet Encoder

DEPRECATED: This encoder is deprecated and will be removed in a future release. Please use the equivalent
TorchVision [ResNet](#resnet) encoder instead.

Implements ResNet V2 as described in [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027).

The ResNet encoder takes the following optional parameters:

{% set image_encoder = get_encoder_schema("image", "_resnet_legacy") %}
{{ render_yaml(image_encoder, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(image_encoder, exclude=["type"])) }}

#### Legacy Vision Transformer Encoder

DEPRECATED: This encoder is deprecated and will be removed in a future release. Please use the equivalent
TorchVision [VisionTransformer](#visiontransformer) encoder instead.

Encodes images using a Vision Transformer as described in
[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929).

Vision Transformer divides the image into equal-sized patches, uses a linear transformation to encode each flattened
patch, then applies a deep transformer architecture to the sequence of encoded patches.

The Vision Transformer Encoder takes the following optional parameters:

{% set image_encoder = get_encoder_schema("image", "_vit_legacy") %}
{{ render_yaml(image_encoder, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(image_encoder, exclude=["type"])) }}

## Image Augmentation

Image augmentation is a technique used to increase the diversity of a training dataset by applying random
transformations to the images. The goal is to train a model that is robust to the variations in the training data.

Augmentation is specified by the `augmentation` section in the image feature configuration and can be specified in one of the following ways:

**Boolean: `False` (Default)** No augmentation is applied to the images.

```yaml
augmentation: False
```

**Boolean: `True`** The following augmentation methods are applied to the images: `random_horizontal_flip` and `random_rotate`.

```yaml
augmentation: True
```

**List of Augmentation Methods** One or more of the following augmentation methods are applied to the images in the order specified by the user: `random_horizontal_flip`, `random_vertical_flip`, `random_rotate`, `random_blur`, `random_brightness`, and `random_contrast`.  The following is an illustrative example.

```yaml
augmentation:
    - type: random_horizontal_flip
    - type: random_vertical_flip
    - type: random_rotate
      degree: 10
    - type: random_blur
      kernel_size: 3
    - type: random_brightness
      min: 0.5
      max: 2.0
    - type: random_contrast
      min: 0.5
      max: 2.0
```

Augmentation is applied to the batch of images in the training set only.  The validation and test sets are not augmented.

Following illustrates how augmentation affects an image:

![Original Image](augmentation_samples/original.png)

**Horizontal Flip**: Image is randomly flipped horizontally.

{% set image_augmentation = get_augmentation_schema("image", "random_horizontal_flip") %}
{{ render_yaml(image_augmentation) }}

<!-- No parameters for Horizontal Flip
Parameters:

{{ render_fields(schema_class_to_fields(image_augmentation, exclude="type")) }} -->

![Horizontal Flip](augmentation_samples/horizontal_flip.png)

**Vertical Flip**:  Image is randomly flipped vertically.

{% set image_augmentation = get_augmentation_schema("image", "random_vertical_flip") %}
{{ render_yaml(image_augmentation) }}

<!-- No parameters for Vertical Flip
Parameters:

{{ render_fields(schema_class_to_fields(image_augmentation, exclude="type")) }} -->

![Vertical Flip](augmentation_samples/vertical_flip.png)

**Rotate**: Image is randomly rotated by an amount in the range [-degree, +degree].  `degree` must be a positive integer.

{% set image_augmentation = get_augmentation_schema("image", "random_rotate") %}
{{ render_yaml(image_augmentation) }}

Parameters:

{{ render_fields(schema_class_to_fields(image_augmentation, exclude="type")) }}

Following shows the effect of rotating an image:

![Rotate Image](augmentation_samples/rotation.png)

**Blur**:  Image is randomly blurred using a Gaussian filter with kernel size specified by the user.  The `kernel_size` must be a positive, odd integer.

{% set image_augmentation = get_augmentation_schema("image", "random_blur") %}
{{ render_yaml(image_augmentation) }}

Parameters:

{{ render_fields(schema_class_to_fields(image_augmentation, exclude="type")) }}

Following shows the effect of blurring an image with various kernel sizes:

![Blur Image](augmentation_samples/blur.png)

**Adjust Brightness**: Image brightness is adjusted by a factor randomly selected in the range [min, max].   Both `min` and `max` must be a float greater than 0, with `min` less than `max`.

{% set image_augmentation = get_augmentation_schema("image", "random_brightness") %}
{{ render_yaml(image_augmentation) }}

Parameters:

{{ render_fields(schema_class_to_fields(image_augmentation, exclude="type")) }}

Following shows the effect of brightness adjustment with various factors:

![Adjust Brightness](augmentation_samples/brightness.png)

**Adjust Contrast**: Image contrast is adjusted by a factor randomly selected in the range [min, max].  Both `min` and `max` must be a float greater than 0, with `min` less than `max`.

{% set image_augmentation = get_augmentation_schema("image", "random_contrast") %}
{{ render_yaml(image_augmentation) }}

Parameters:

{{ render_fields(schema_class_to_fields(image_augmentation, exclude="type")) }}

Following shows the effect of contrast adjustment with various factors:

![Adjust Contrast](augmentation_samples/contrast.png)

**Illustrative Examples of Image Feature Configuration with Augmentation**

```yaml
name: image_column_name
type: image
encoder: 
    type: resnet
    model_variant: 18
    use_pretrained: true
    pretrained_cache_dir: None
    trainable: true
augmentation: false
```

```yaml
name: image_column_name
type: image
encoder: 
    type: stacked_cnn
augmentation: true
```

```yaml
name: image_column_name
type: image
encoder: 
    type: alexnet
augmentation:
    - type: random_horizontal_flip
    - type: random_rotate
      degree: 10
    - type: random_blur
      kernel_size: 3
    - type: random_brightness
      min: 0.5
      max: 2.0
    - type: random_contrast
      min: 0.5
      max: 2.0
    - type: random_vertical_flip
```

# Output Features

Image features can be used when semantic segmentation needs to be performed.
There is only one decoder available for image features: `unet`.

Example image output feature using default parameters:

```yaml
name: image_column_name
type: image
reduce_input: sum
dependencies: []
reduce_dependencies: sum
loss:
    type: softmax_cross_entropy
decoder:
    type: unet
```

Parameters:

- **`reduce_input`** (default `sum`): defines how to reduce an input that is not a vector, but a matrix or a higher order
tensor, on the first dimension (second if you count the batch dimension). Available values are: `sum`, `mean` or `avg`,
`max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension).
- **`dependencies`** (default `[]`): the output features this one is dependent on. For a detailed explanation refer to
[Output Feature Dependencies](../output_features#output-feature-dependencies).
- **`reduce_dependencies`** (default `sum`): defines how to reduce the output of a dependent feature that is not a vector,
but a matrix or a higher order tensor, on the first dimension (second if you count the batch dimension). Available
values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last
vector of the first dimension).
- **`loss`** (default `{type: softmax_cross_entropy}`): is a dictionary containing a loss `type`. `softmax_cross_entropy` is
the only supported loss type for image output features. See [Loss](#loss) for details.
- **`decoder`** (default: `{"type": "unet"}`): Decoder for the desired task. Options: `unet`. See [Decoder](#decoder) for details.

## Decoders

### U-Net Decoder
The U-Net decoder is based on
[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).
The decoder implements the expansive upsampling path of the U-Net stack.
Semantic segmentation supports one input and one output feature. The `num_fc_layers` in the decoder
and combiner sections must be set to 0 as U-Net does not have any fully connected layers.

U-Net Decoder takes the following optional parameters:

{% set decoder = get_decoder_schema("image", "unet") %}
{{ render_yaml(decoder, parent="decoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(decoder, exclude=["type"]), details=details) }}

Decoder type and decoder parameters can also be defined once and applied to all image output features using the [Type-Global Decoder](../defaults.md#type-global-decoder) section.

## Loss

### Softmax Cross Entropy

{% set loss = get_loss_schema("softmax_cross_entropy") %}
{{ render_yaml(loss, parent="loss") }}

Parameters:

{{ render_fields(schema_class_to_fields(loss, exclude=["type"]), details=details) }}

Loss and loss related parameters can also be defined once and applied to all image output features using the [Type-Global Loss](../defaults.md#type-global-loss) section.

## Metrics

The measures that are calculated every epoch and are available for image features are the `accuracy` and `loss`.
You can set either of them as `validation_metric` in the `training` section of the configuration if you set the
`validation_field` to be the name of a category feature.
