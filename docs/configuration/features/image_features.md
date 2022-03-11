## Image Features Preprocessing

Ludwig supports both grayscale and color images.
The number of channels is inferred, but all images in the dataset should have the same number of channels.
During preprocessing, raw image files are transformed into numpy arrays and saved in the hdf5 format.

Ideally, all images in the dataset should have the same size. If there are different image sizes present in the dataset,
a `resize_method` together with a target `width` and `height` must be specified in the feature preprocessing parameters.

- `in_memory` (default `true`): defines whether image dataset will reside in memory during the training process or will
be dynamically fetched from disk (useful for large datasets). In the latter case a training batch of input images will
be fetched from disk each training iteration.
- `num_processes` (default 1): specifies the number of processes to run for preprocessing images.
- `resize_method` (default `crop_or_pad`): `crop_or_pad` crops images larger than the specified `width` and `height` to
the desired size or pads smaller images using edge padding. `interpolate` uses interpolation to resample images to the
specified `width` and `height`.
- `height` (default `null`): image height in pixels, must be set if resizing is required.
- `width` (default `null`): image width in pixels, must be set if resizing is required.
- `infer_image_dimensions` (default: `true`): whether to automatically resize differently-sized images to
inferred dimensions. Target dimensions are inferred from the dimensions of the first `infer_image_sample_size` sample
images, then applying `infer_image_max_height` and `infer_image_max_width`. This parameter has no effect if explicit
`width` and `height` are specified.
- `infer_image_sample_size` (int, default `100`): Number of images to sample to `infer_image_dimensions`.
- `infer_image_max_height` (int, default `256`): maximum height of an image transformed using `infer_image_dimensions`.
- `infer_image_max_width` (int, default `256`): maximum width of an image transformed using `infer_image_dimensions`.
- `num_channels` (default `null`): number of channels in the images. By default, the number of channels of the first
image of the dataset will be used. If there is an image in the dataset with a different number of channels, an error
will be reported. If `num_channels` is specified, images in the dataset will be adapted to the specified number of
channels. If the value is `1`, all images with more than one channel will be converted to grayscale and reduced to one
channel (transparency will be lost). If the value is `3` all images with 1 channel will be repeated 3 times to obtain 3
channels, while images with 4 channels will lose the transparency channel. If the value is `4`, all images with less
than 4 channels will have the remaining channels filled with zeros.
- `scaling` (default `pixel_normalization`): what scaling to perform on images. By default `pixel_normalization` is
performed, which consists in dividing each pixel values by 255. `pixel_standardization` is also available, which
normalizes each image to the same channel-wise mean and variance.

Depending on the application, it is preferable not to exceed a size of `256 x 256` as bigger sizes will seldom provide a
significant performance advantage. Larger images will considerably slow down training and inference and consume more
memory, leading to memory overflows on machines with limited amounts of RAM or OOM (out-of-memory) on GPUs.

Example of a preprocessing specification:

```yaml
name: image_feature_name
type: image
preprocessing:
  height: 128
  width: 128
  resize_method: interpolate
  scaling: pixel_normalization
```

## Image Input Features and Encoders

Image inputs are transformed into float valued tensors of size `N x H x W x C` (where `N` is the size of the dataset,
`H x W` is a specific resizing of the image that can be set, and `C` is the number of channels). Preprocessed images are
added to HDF5 with a key that reflects the name of the column in the dataset.
The column name is added to the JSON file, with an associated dictionary containing preprocessing information including
resizing settings.

Ludwig currently supports the following encoders for images:

- `stacked_cnn`: Convolutional Stack Encoder
- `resnet`: ResNet Encoder
- `mlp_mixer`: MLP-Mixer Encoder
- `vit`: Vision Transformer Encoder

### Convolutional Stack Encoder

Stack of 2D convolutional layers with optional normalization, dropout, and downsampling pooling layers.

Convolutional Stack Encoder takes the following optional parameters:

- `conv_layers` (default `null`): it is a list of dictionaries containing the parameters of all the convolutional
layers. The length of the list determines the number of stacked convolutional layers and the content of each dictionary
determines the parameters for a specific layer. The available parameters for each layer are: `out_channels`,
`kernel_size`, `stride`, `padding`, `dilation`, `groups`, `bias`, `padding_mode`, `norm`, `norm_params`, `activation`,
`dropout`, `pool_function`, `pool_kernel_size`, `pool_stride`, `pool_padding`, and `pool_dilation`. If any of those
values is missing from the dictionary, the default one specified as a parameter of the encoder will be used instead. If
both `conv_layers` and `num_conv_layers` are `null`, a default list will be assigned to `conv_layers` with the value
`[{out_channels: 32}, {out_channels: 64}]`.
- `num_conv_layers` (default `null`): if `conv_layers` is `null`, this is the number of stacked convolutional layers.
- `out_channels` (default `256`): indicates the number of filters, and by consequence the output channels of the 2d
convolution. If `out_channels` is not already specified in `conv_layers` this is the default `out_channels` that will be
used for each layer.
- `kernel_size` (default `3`): An integer or pair of integers specifying the kernel size. A single integer specifies a
square kernel, while a pair of integers specifies the height and width of the kernel in that order (`[h, w]`). If a
`kernel_size` is not specified in `conv_layers` this `kernel_size` that will be used for each layer.
- `stride` (default `1`): An integer or pair of integers specifying the stride of the convolution along the height and
width.
- `padding` (default `valid`): int, pair of ints `[h, w]`, or one of `valid`, `same`.
- `dilation` (default `1`): An integer or pair of integers specifying the dilation rate to use for dilated convolution.
- `groups` (default `1`): groups controls the connectivity between inputs and outputs. in_channels and out_channels must
both be divisible by groups.
- `bias` (default `true`): boolean, whether the convolution layer uses a bias vector.
- `padding_mode` (default `zeros`): one of `zeros`, `reflect`, `replicate` or `circular`. Padding is only added if
the `padding` option is `same` or an integer greater than 0.
- `norm` (default `null`): if a `norm` is not already specified in `conv_layers` this is the default `norm` that will be
used for each layer. It indicates the normalization applied to the activations and can be `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`. For information on parameters
used with `batch` see [Torch's documentation on batch normalization](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)
or for `layer` see [Torch's documentation on layer normalization](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html).
- `activation` (default `relu`): if an `activation` is not already specified in `conv_layers` this is the default
`activation` that will be used for each layer. It indicates the activation function applied to the output.
- `dropout` (default `0`): dropout rate
- `pool_function` (default `max`): `max` will use max pooling. Any of `average`, `avg` or `mean` will use average pooling.
- `pool_kernel_size` (default `2`): An integer or pair of integers specifying the pooling size. If `pool_kernel_size` is
not specified in `conv_layers` this is the default value that will be used for each layer.
- `pool_stride` (default `null`): An integer or pair of integers specifying the pooling stride, which is the factor by
which the pooling layer downsamples the feature map. Defaults to `pool_kernel_size`.
- `pool_padding` (default `0`): An integer or pair of ints specifying pooling padding `(h, w)`.
- `pool_dilation` (default `1`): An integer or pair of ints specifying pooling dilation rate `(h, w)`.
- `fc_layers` (default `null`): a list of dictionaries containing the parameters of all the fully connected
layers. The length of the list determines the number of stacked fully connected layers and the content of each
dictionary determines the parameters for a specific layer. The available parameters for each layer are: `activation`,
`dropout`, `norm`, `norm_params`, `output_size`, `use_bias`, `bias_initializer` and `weights_initializer`. If any of
those values is missing from the dictionary, the default one specified as a parameter of the encoder will be used
instead.
- `num_fc_layers` (default `1`): The number of stacked fully connected layers.
- `output_size` (default `128`): if `output_size` is not already specified in `fc_layers` this is the default
`output_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `fc_use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `fc_weights_initializer` (default `xavier_uniform`): initializer for the weights matrix. Options are: `constant`,
`identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`,
`glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and
other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer,
please refer to [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html).
- `fc_bias_initializer` (default `zeros`): initializer for the bias vector. Options are: `constant`, `identity`,
`zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`,
`glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other
keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please
refer to [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html).
- `fc_norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will
be used for each layer. It indicates the norm of the output and can be `null`, `batch` or `layer`.
- `fc_norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on
parameters used with `batch` see [Torch's documentation on batch normalization](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)
or for `layer` see [Torch's documentation on layer normalization](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html).
- `fc_activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default
`activation` that will be used for each layer. It indicates the activation function applied to the output.
- `fc_dropout` (default `0`): dropout rate

Example image feature config using a convolutional stack encoder:

```yaml
name: image_column_name
type: image
encoder: stacked_cnn
conv_layers: null
num_conv_layers: null
kernel_size: 3
out_channels: 256
padding: valid
conv_use_bias: true
conv_norm: batch
conv_activation: relu
conv_dropout: 0
pool_function: max
pool_size: 2
num_fc_layers: 1
output_size: 128
fc_use_bias: true
fc_weights_initializer: xavier_uniform
fc_bias_initializer: zeros
fc_norm: batch
fc_activation: relu
fc_dropout: 0
preprocessing:  # example pre-processing
    height: 32
    width: 32
    num_channels: 1
```

### ResNet Encoder

Implements ResNet V2 as described in [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027).

The ResNet encoder takes the following optional parameters:

- `resnet_size` (default `50`): The ResNet size, one of: `8`, `14`, `18`, `34`, `50`, `101`, `152`, `200`.
- `out_channels` (default `16`): The number of filters, and by consequence the output channels of the 2d convolution.
- `kernel_size` (default `3`): An integer or pair of integers specifying the convolution kernel size. A single integer
specifies a square kernel, a pair of integers specifies the height and width of the kernel in that order (`[h, w]`).
- `conv_stride` (default `1`): An integer or pair of integers specifying the stride of the initial convolutional layer.
- `first_pool_kernel_size` (default `null`): Pool size to be used for the first pooling layer. If none, the first pooling layer is skipped.
- `first_pool_stride` (default `null`): Stride for first pooling layer. If `null`, defaults to `first_pool_kernel_size`.
- `batch_norm_momentum` (default `0.9`): Momentum of the batch norm running statistics.
- `batch_norm_epsilon` (default `0.001`): Epsilon of the batch norm.
- `fc_layers` (default `null`): a list of dictionaries containing the parameters of all the fully connected
layers. The length of the list determines the number of stacked fully connected layers and the content of each
dictionary determines the parameters for a specific layer. The available parameters for each layer are: `activation`,
`dropout`, `norm`, `norm_params`, `output_size`, `use_bias`, `bias_initializer` and `weights_initializer`. If any of
those values is missing from the dictionary, the default one specified as a parameter of the encoder will be used
instead.
- `num_fc_layers` (default `1`): The number of stacked fully connected layers.
- `output_size` (default `256`): if `output_size` is not already specified in `fc_layers` this is the default
`output_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `xavier_uniform`): initializer for the weights matrix. Options are: `constant`,
`identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`,
`glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other
keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please
refer to [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html).
- `bias_initializer` (default `zeros`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`,
`ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`,
`xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is
possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its
parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to
[torch.nn.init](https://pytorch.org/docs/stable/nn.init.html).
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be
used for each layer. It indicates the norm of the output and it can be `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters
used with `batch` see [Torch's documentation on batch normalization](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)
or for `layer` see [Torch's documentation on layer normalization](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html).
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default
`activation` that will be used for each layer. It indicates the activation function applied to the output.
- `dropout` (default `0`): dropout rate

Example image input feature config using a ResNet encoder:

```yaml
name: image_column_name
type: image
encoder: resnet
resnet_size: 50
num_filters: 16
kernel_size: 3
conv_stride: 1
batch_norm_momentum: 0.9
batch_norm_epsilon: 0.001
num_fc_layers: 1
output_size: 256
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

- `patch_size` (default `16`): The image patch size. Each patch is `patch_size`<sup>2</sup> pixels. Must evenly divide
the image width and height.
- `embed_size` (default `512`): The patch embedding size, the output size of the mixer if `avg_pool` is true.
- `token_size` (default `2048`): The per-patch embedding size.
- `channel_dim` (default `256`): Number of channels in hidden layer.
- `num_layers` (default `8`): The depth of the network (the number of Mixer blocks).
- `dropout` (default `0`): Dropout rate.
- `avg_pool` (default `true`): If true, pools output over patch dimension, outputs a vector of shape `(embed_size)`. If
false, the output tensor is of shape `(n_patches, embed_size)`, where n_patches is `img_height` x `img_width` / `patch_size`<sup>2</sup>.

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
- `patch_size` (default `16`): The image patch size. Each patch is `patch_size`<sup>2</sup> pixels. Must evenly divide
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

## Image Output Features and Decoders

There are no image decoders at the moment (WIP), so image cannot be used as output features.

## Image Features Metrics

As no image decoders are available at the moment, there are also no image metrics.
