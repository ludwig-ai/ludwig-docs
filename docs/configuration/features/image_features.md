Input image features are transformed into a float valued tensors of size `N x C x H x W` (where `N` is the size of the dataset, `C` is the number of channels, and `H x W` is the height and width of the image (can be specified by the user). These tensors are added to HDF5 with a key that reflects the name of column in the dataset.

The column name is added to the JSON file, with an associated dictionary containing preprocessing information about the sizes of the resizing.

# Supported Image Formats

The number of channels in the image is determined by the image format. The following table lists the supported image formats and the number of channels.

| Format      | Number of channels |
| ----------- | ----------- |
| Grayscale      | 1       |
| Grayscale with Alpha   | 2        |
| RGB   | 3        |
| RGB with Alpha      | 4       |

# Image Features Preprocessing

During preprocessing, raw image files are transformed into numpy ndarrays and saved in the hdf5 format.

!!! note
    Images passed to an image encoder are expected to have the same size. If images are different sizes, by default they will be resized to the dimenions of the first image in the dataset. Optionally, a `resize_method`, together with a target `width` and `height`, can be specified in the feature preprocessing parameters, in which case all images will be resized to the specified target size.

### `missing_value_strategy`

The strategy to follow when there's a missing image.

- Default: `backfill`
- Options:
  - `fill_with_const`: Replace the missing image with a specific value specified with the `fill_value` parameter.
  - `fill_with_mode`: Replaces the missing image with the most frequent value in the column.
  - `fill_with_mean`: Replace the missing image with the mean of the values in the column.
  - `backfill`: Replace the missing image with the next valid value.

### `fill_value`

The value to use when `missing_value_strategy` is `fill_with_const`.

- Default: `null`

### `height`

Image height in pixels. If set, images will be resized to the specified height using the `resize_method` parameter. If unspecified, images will be resized to the size of the first image in the dataset.

- Default: `null`

### `width`

Image width in pixels. If set, images will be resized to the specified width using the `resize_method` parameter. If unspecified, images will be resized to the size of the first image in the dataset.

- Default: `null`

### `num_channels`

Number of channels in the images. If specified, images will be read in the mode specified by the number of channels. If not specified, the number of channels will be inferred from the image format of the first valid image in the dataset.

E.g., if `num_channels = 1`, any RGB images will be converted to Grayscale. If `num_channels = 3`, any images with 1 channel will be converted to RGB by repeating the channel 3 times.

- Default: `null`

### `resize_method`

The method to use for resizing images.

- Default: `crop_or_pad`
- Options:
  - `crop_or_pad`: If image is larger than the specified dimensions, crops images. If image is smaller, pads images using edge padding
  - `interpolate`: Uses interpolation to resize images to the specified `width` and `height`

### `infer_image_dimensions`

If set, then the height and width of images in the dataset will be inferred from a sample of the first `n` images in the dataset. The size of `n` is set by the parameter `infer_image_sample_size`. Each image that doesn't conform to these dimensions will be resized according to `resize_method`. If set to `false`, then the height and width of images in the dataset will be specified by the user.

This parameter will have no effect if `width` and `height` are specified.

- Default: `true`

### `infer_image_max_height`

If `infer_image_dimensions` is set, this is used as the maximum height of the images in the dataset.

- Default: 256

### `infer_image_max_width`

If `infer_image_dimensions` is set, this is used as the maximum width of the images in the dataset.

- Default: 256

### `infer_image_num_channels`

If set, then the number of channels in the dataset is inferred from a sample of the first `n` images in the dataset. The size of `n` is set by the parameter `infer_image_sample_size`.

- Default: `true`

### `infer_image_sample_size`

The sample size used for inferring dimensions of images in `infer_image_dimensions`.

- Default: 100

### `scaling`

The scaling strategy for pixel values in the image.

- Default: `pixel_normalization`
- Options:
  - `pixel_normalization`: Normalizes pixel values to be between 0 and 1 by dividing each pixel value by 255.
  - `pixel_standardization`: Normalizes pixel values based on the mean and standard deviation of images in ImageNet.

### `in_memory`

Whether image dataset will reside in memory during the training process or will be dynamically fetched from disk (useful for large datasets). In the latter case a training batch of input images will be fetched from disk each training iteration.

- Default: `true`

### `num_processes`

Specifies the number of processes to run for preprocessing images.

- Default: 1

!!! note
    Depending on the application, it is preferrable not to exceed a size of `256 x 256`, as bigger sizes will, in most cases, not provide much advantage in terms of performance, while they will considerably slow down training and inference and also make both forward and backward passes consume considerably more memory, leading to memory overflows on machines with limited amounts of RAM or on GPUs with limited amounts of VRAM.

Example of a preprocessing specification:

```yaml
name: image_feature_name
type: image
preprocessing:
  missing_value_strategy: fill_with_const
  fill_value: 0.5
  height: 128
  width: 128
  num_channels: 3
  resize_method: interpolate
  scaling: pixel_normalization
  in_memory: true
  num_processes: 4
```

# Image Input Features and Encoders

The default encoder is `stacked_cnn`.

## Convolutional Stack Encoder (`stacked_cnn`)

Creates an encoder built by stacking multiple 2D convolutional layers, followed by an optional stack of fully connected layers.

### `conv_layers`

A list of dictionaries containing the parameters of all the convolutional layers. The length of the list determines the number of stacked convolutional layers and the content of each dictionary determines the parameters for a specific layer. If a parameter for a layer is not specified in the dictionary, then the default value for the stacked CNN encoder is used.

- Default: `null`
- Parameters for each layer:
  - `kernel_size`: The size of the convolutional kernel.
  - `out_channels`: The number of output channels.
  - `stride`: The stride of the convolutional kernel.
  - `padding`: The padding of the convolutional kernel.
  - `dilation`: The dilation of the convolutional kernel.
  - `groups`: The number of groups for grouped convolution.
  - `bias`: Whether to add a bias term to the convolution.
  - `padding_mode`: The padding mode to use for the convolution.
  - `norm`: The type of normalization to use for the convolution.
  - `norm_params`: Optional parameters for the normalization.
  - `activation`: The type of activation to use for the convolution.
  - `dropout`: The dropout probability to use for the convolution.
  - `pool_function`: The type of pooling function to use for the convolution.
  - `pool_kernel_size`: The size of the pooling kernel.
  - `pool_stride`: The stride of the pooling kernel.
  - `pool_padding`: The padding of the pooling kernel.
  - `pool_dilation`: The dilation of the pooling kernel.

### `num_conv_layers`

If `conv_layers` is `null`, then this parameter determines the number of convolutional layers in the encoder. Each layer will use default parameters for the convolutional layer.

- Default: `null`

!!! note
    If both `conv_layers` and `num_conv_layers` are `null`, `conv_layers` is set to the following default value:

    ```python
    conv_layers = [
      {
        kernel_size: 3,
        out_channels: 32,
        pool_kernel_size: 2,
      },
      {
        kernel_size: 3,
        out_channels: 64,
        pool_kernel_size: 2,
      },
    ]
    ```

### `out_channels`

If the number of output channels is not specified in `conv_layers`, specifies the default number of output channels for each convolutional layer.

- Default: 32

### `kernel_size`

If a `kernel_size` is not already specified in `conv_layers`, specifies the default `kernel_size` of the 2D convolutional kernel that will be used for each layer.

This can be a single integer or a tuple of integers. If single integer, then the kernel has the same dimension in both height and width. If a tuple of integers, then the first integer is the height and the second integer is the width.

- Default: 3

### `stride`

If a `stride` is not already specified in `conv_layers`, specifies the default `stride` of the 2D convolutional kernel that will be used for each layer.

This can be a single integer or a tuple of integers. If single integer, then the kernel has the same dimension in both height and width. If a tuple of integers, then the first integer is the height and the second integer is the width.

- Default: 1

### `padding`

If `padding` is not already specified in `conv_layers`, specifies the default `padding` of the 2D convolutional kernel that will be used for each layer.

- Default: `valid`
- Choices: `valid`, `same`

### `dilation`

If `dilation` is not already specified in `conv_layers`, specifies the default `dilation` of the 2D convolutional kernel that will be used for each layer.

- Default: `(1, 1)`

### `conv_bias`

If `bias` not already specified in `conv_layers`, specifies if the 2D convolutional kernel should have a bias term.

- Default: `true`

### `conv_weights_initializer`

 (default `'glorot_uniform'`): initializer for the weights matrix. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).

### `conv_bias_initializer`

 (default `'zeros'`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).

### `conv_norm`

 (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be used for each layer. It indicates the norm of the output and it can be `null`, `batch` or `layer`.

### `conv_norm_params`

 (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters used with `batch` see [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization) or for `layer` see [Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization).

### `conv_activation`

 (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default `activation` that will be used for each layer. It indicates the activation function applied to the output.

### `conv_dropout`

 (default `0`): dropout rate

### `pool_function`

 (default `max`):  pooling function: `max` will select the maximum value.  Any of these--`average`, `avg` or `mean`--will compute the mean value.

### `pool_size`

 (default `(2, 2)`): if a `pool_size` is not already specified in `conv_layers` this is the default `pool_size` that will be used for each layer. It indicates the size of the max pooling that will be performed along the `s` sequence dimension after the convolution operation.

### `pool_strides`

 (default `null`): factor to scale down

### `fc_layers`

 (default `null`): it is a list of dictionaries containing the parameters of all the fully connected layers. The length of the list determines the number of stacked fully connected layers and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `fc_size`, `norm`, `activation` and `regularize`. If any of those values is missing from the dictionary, the default one specified as a parameter of the encoder will be used instead. If both `fc_layers` and `num_fc_layers` are `null`, a default list will be assigned to `fc_layers` with the value `[{fc_size: 512}, {fc_size: 256}]` (only applies if `reduce_output` is not `null`).

### `num_fc_layers`

 (default `1`): This is the number of stacked fully connected layers.

### `fc_size`

 (default `256`): if a `fc_size` is not already specified in `fc_layers` this is the default `fc_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.

### `fc_use_bias`

 (default `true`): boolean, whether the layer uses a bias vector.

### `fc_weights_initializer`

 (default `'glorot_uniform'`): initializer for the weights matrix. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).

### `fc_bias_initializer`

 (default `'zeros'`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).

### `fc_weights_regularizer`

 (default `null`): regularizer function applied to the weights matrix.  Valid values are `l1`, `l2` or `l1_l2`.

### `fc_bias_regularizer`

 (default `null`): regularizer function applied to the bias vector.  Valid values are `l1`, `l2` or `l1_l2`.

### `fc_activity_regularizer`

 (default `null`): regurlizer function applied to the output of the layer.  Valid values are `l1`, `l2` or `l1_l2`.

### `fc_norm`

 (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be used for each layer. It indicates the norm of the output and it can be `null`, `batch` or `layer`.

### `fc_norm_params`

 (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters used with `batch` see [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization) or for `layer` see [Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization).

### `fc_activation`

 (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default `activation` that will be used for each layer. It indicates the activation function applied to the output.

### `fc_dropout`

 (default `0`): dropout rate

Example image feature entry using a convolutional stack encoder (with default parameters) in the input features list:

```yaml
name: image_column_name
type: image
encoder: stacked_cnn
tied: null
conv_layers: null
num_conv_layers: null
filter_size: 3
num_filters: 256
strides: (1, 1)
padding: valid
dilation_rate: (1, 1)
conv_use_bias: true
conv_weights_initializer: glorot_uniform
conv_bias_initializer: zeros
conv_norm: null
conv_norm_params: null
conv_activation: relu
conv_dropout: 0
pool_function: max
pool_size: (2, 2)
pool_strides: null
fc_layers: null
num_fc_layers: 1
fc_size: 256
fc_use_bias: true
fc_weights_initializer: glorot_uniform
fc_bias_initializer: zeros
fc_norm: null
fc_norm_params: null
fc_activation: relu
fc_dropout: 0
preprocessing:  # example pre-processing
    height: 28
    width: 28
    num_channels: 1

```

## ResNet Encoder

[ResNet](https://arxiv.org/abs/1603.05027) Encoder takes the following optional parameters:

- `resnet_size` (default `50`): A single integer for the size of the ResNet model. If has to be one of the following values: `8`, `14`, `18`, `34`, `50`, `101`, `152`, `200`.
- `num_filters` (default `16`): It indicates the number of filters, and by consequence the output channels of the 2d convolution.
- `kernel_size` (default `3`): The kernel size to use for convolution.
- `conv_stride` (default `1`): Stride size for the initial convolutional layer.
- `first_pool_size` (default `null`): Pool size to be used for the first pooling layer. If none, the first pooling layer is skipped.
- `batch_norm_momentum` (default `0.9`): Momentum of the batch norm running statistics. The suggested parameter in [TensorFlow's implementation](https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py#L36) is `0.997`, but that leads to a big discrepancy between the normalization at training time and test time, so the default value is a more conservative `0.9`.
- `batch_norm_epsilon` (default `0.001`): Epsilon of the batch norm. The suggested parameter in [TensorFlow's implementation](https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py#L37) is `1e-5`, but that leads to a big discrepancy between the normalization at training time and test time, so the default value is a more conservative `0.001`.
- `fc_layers` (default `null`): it is a list of dictionaries containing the parameters of all the fully connected layers. The length of the list determines the number of stacked fully connected layers and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `fc_size`, `norm`, `activation` and `regularize`. If any of those values is missing from the dictionary, the default one specified as a parameter of the encoder will be used instead. If both `fc_layers` and `num_fc_layers` are `null`, a default list will be assigned to `fc_layers` with the value `[{fc_size: 512}, {fc_size: 256}]` (only applies if `reduce_output` is not `null`).
- `num_fc_layers` (default `1`): This is the number of stacked fully connected layers.
- `fc_size` (default `256`): if a `fc_size` is not already specified in `fc_layers` this is the default `fc_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `'glorot_uniform'`): initializer for the weights matrix. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `bias_initializer` (default `'zeros'`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be used for each layer. It indicates the norm of the output and it can be `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters used with `batch` see [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization) or for `layer` see [Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization).
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default `activation` that will be used for each layer. It indicates the activation function applied to the output.
- `dropout` (default `0`): dropout rate

Example image feature entry using a ResNet encoder (with default parameters) in the input features list:

```yaml
name: image_column_name
type: image
encoder: resnet
tied: null
resnet_size: 50
num_filters: 16
kernel_size: 3
conv_stride: 1
first_pool_size: null
batch_norm_momentum: 0.9
batch_norm_epsilon: 0.001
fc_layers: null
num_fc_layers: 1
fc_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
norm: null
norm_params: null
activation: relu
dropout: 0
preprocessing:
    height: 224
    width: 224
    num_channels: 3
```

### MLP-Mixer Encoder

Encodes images using MLP-Mixer, as described in [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601)
MLP-Mixer divides the image into equal-sized patches, applying fully connected layers to each patch to compute per-patch
representations (tokens) and combining the representations with fully-connected mixer layers.

The MLP-Mixer Encoder takes the following optional parameters:

- `patch_size` (default `16`): The image patch size. Each patch is `patch_size`² pixels. Must evenly divide
the image width and height.
- `embed_size` (default `512`): The patch embedding size, the output size of the mixer if `avg_pool` is true.
- `token_size` (default `2048`): The per-patch embedding size.
- `channel_dim` (default `256`): Number of channels in hidden layer.
- `num_layers` (default `8`): The depth of the network (the number of Mixer blocks).
- `dropout` (default `0`): Dropout rate.
- `avg_pool` (default `true`): If true, pools output over patch dimension, outputs a vector of shape `(embed_size)`. If
false, the output tensor is of shape `(n_patches, embed_size)`, where n_patches is `img_height` x `img_width` / `patch_size`².

Example image feature config using an MLP-Mixer encoder:

```yaml
name: image_column_name
type: image
encoder: mlp_mixer
patch_size: 16
embed_size: 512
token_size: 2048
channel_dim: 256
num_layers: 8
dropout: 0.0
avg_pool: True
preprocessing:
    height: 64
    width: 64
    num_channels: 3
```

### Vision Transformer Encoder

Encodes images using a Vision Transformer as described in
[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929).

Vision Transformer divides the image into equal-sized patches, uses a linear transformation to encode each flattened
patch, then applies a deep transformer architecture to the sequence of encoded patches.

The Vision Transformer Encoder takes the following optional parameters:

- `use_pretrained` (default `true`): Use pre-trained model weights from Hugging Face.
- `pretrained_model` (default `google/vit-base-patch16-224`): The pre-trained model to use. See the [model hub](https://huggingface.co/models?search=vit)
for other pretrained vision transformer models.
- `hidden_size` (default `768`): Dimensionality of the encoder layers and the pooling layer.
- `num_hidden_layers` (default `12`): Number of hidden layers in the Transformer encoder.
- `num_attention_heads` (default `12`): Number of attention heads in each attention layer.
- `intermediate_size` (default `3072`): Dimensionality of the intermediate (i.e., feed-forward) layer in the Transformer
encoder.
- `hidden_act` (default `gelu`): Hidden layer activation, one of `gelu`, `relu`, `selu` or `gelu_new`.
- `hidden_dropout_prob` (default `0.1`): The dropout rate for all fully connected layers in the embeddings, encoder, and
pooling.
- `attention_probs_dropout_prob` (default `0.1`): The dropout rate for the attention probabilities.
- `initializer_range` (default `768`): The standard deviation of the truncated_normal_initializer for initializing all
weight matrices.
- `layer_norm_eps` (default `1e-12`): The epsilon used by the layer normalization layers.
- `gradient_checkpointing` (default `false`):
- `patch_size` (default `16`): The image patch size. Each patch is `patch_size`² pixels. Must evenly divide
the image width and height.
- `trainable` (default `true`): Is the encoder trainable.

Example image feature config using an MLP-Mixer encoder:

```yaml
name: image_column_name
type: image
encoder: vit
use_pretrained: true
preprocessing:
    height: 128
    width: 128
    num_channels: 3
```

# Image Output Features and Decoders

There are no image decoders at the moment (WIP), so image cannot be used as output features.

# Image Features Measures

As no image decoders are available at the moment, there are also no image measures.
