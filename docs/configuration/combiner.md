{% from './macros/includes.md' import render_fields %}
{% set norm_details = "See [Normalization](#normalization) for details." %}
{% set details = {"norm": norm_details, "norm_params": norm_details} %}

Combiners take the outputs of all input features encoders and combine them before providing the combined representation
to the output feature decoders.

You can specify which one to use in the `combiner` section of the configuration, and if you don't specify a combiner,
the `concat` combiner will be used.

## Combiner Types

### Concat Combiner

```
+-----------+
|Input      |
|Feature 1  +-+
+-----------+ |            +---------+
+-----------+ | +------+   |Fully    |
|...        +--->Concat+--->Connected+->
+-----------+ | +------+   |Layers   |
+-----------+ |            +---------+
|Input      +-+
|Feature N  |
+-----------+
```

The `concat` combiner assumes all outputs from encoders are tensors of size `b x h` where `b` is the batch size and `h`
is the hidden dimension, which can be different for each input.
If any inputs have more than 2 dimensions, a sequence or set feature for example, set the `flatten_inputs` parameter to `true`.
It concatenates along the `h` dimension, and then (optionally) passes the concatenated tensor through a stack of fully connected layers.
It returns the final `b x h'` tensor where `h'` is the size of the last fully connected layer or the sum of the sizes of
the `h` of all inputs in the case there are no fully connected layers.
If only a single input feature and no fully connected layer is specified, the output of the input feature encoder is
passed through the combiner unchanged.

{% set concat_combiner = get_combiner_schema("concat") %}

```yaml
combiner:
    {% for line in schema_class_to_yaml(concat_combiner).split("\n") %}
    {{- line }}
    {% endfor %}
```

Parameters:

{{ render_fields(schema_class_to_fields(concat_combiner, exclude=["type"]), details=details) }}

### Sequence Concat Combiner

```
Sequence
Feature
Output

+---------+
|emb seq 1|
+---------+
|...      +--+
+---------+  |  +-----------------+
|emb seq n|  |  |emb seq 1|emb oth|   +------+
+---------+  |  +-----------------+   |      |
             +-->...      |...    +-->+Reduce+->
Other        |  +-----------------+   |      |
Feature      |  |emb seq n|emb oth|   +------+
Output       |  +-----------------+
             |
+-------+    |
|emb oth+----+
+-------+
```

The `sequence_concat` combiner assumes at least one output from encoders is a tensors of size `b x s x h` where `b` is
the batch size, `s` is the length of the sequence and `h` is the hidden dimension.
A sequence-like (sequence, text or time series) input feature can be specified with the `main_sequence_feature`
parameter which takes the name of sequence-like input feature as its value.
If no `main_sequence_feature` is specified, the combiner will look through all the features in the order they are
defined in the configuration and will look for a feature with a rank 3 tensor output (sequence, text or time series).
If it cannot find one it will raise an exception, otherwise the output of that feature will be used for concatenating
the other features along the sequence `s` dimension.

If there are other input features with a rank 3 output tensor, the combiner will concatenate them alongside the `s`
dimension, which means that all of them must have identical `s` dimension, otherwise a dimension mismatch error will be
returned thrown during training when a datapoint with two sequential features of different lengths are provided.

Other features that have a `b x h` rank 2 tensor output will be replicated `s` times and concatenated to the `s` dimension.
The final output is a `b x s x h'` tensor where `h'` is the size of the concatenation of the `h` dimensions of all input features.

{% set sequence_concat_combiner = get_combiner_schema("sequence_concat") %}

```yaml
combiner:
    {% for line in schema_class_to_yaml(sequence_concat_combiner).split("\n") %}
    {{- line }}
    {% endfor %}
```

Parameters:

{{ render_fields(schema_class_to_fields(sequence_concat_combiner, exclude=["type"]), details=details) }}

### Sequence Combiner

```
Sequence
Feature
Output

+---------+
|emb seq 1|
+---------+
|...      +--+
+---------+  |  +-----------------+
|emb seq n|  |  |emb seq 1|emb oth|   +--------+
+---------+  |  +-----------------+   |Sequence|
             +-->...      |...    +-->+Encoder +->
Other        |  +-----------------+   |        |
Feature      |  |emb seq n|emb oth|   +--------+
Output       |  +-----------------+
             |
+-------+    |
|emb oth+----+
+-------+
```

The `sequence` combiner stacks a sequence concat combiner with a sequence encoder.
All the considerations about input tensor ranks described for the [sequence concat combiner](#sequence-concat-combiner)
apply also in this case, but the main difference is that this combiner uses the `b x s x h'` output of the sequence
concat combiner, where `b` is the batch size, `s` is the sequence length and `h'` is the sum of the hidden dimensions of
all input features, as input for any of the sequence encoders described in the [sequence features encoders section](../features/sequence_features#sequence-input-features-and-encoders).
Refer to that section for more detailed information about the sequence encoders and their parameters.
All considerations on the shape of the outputs for the sequence encoders also apply to sequence combiner.

{% set sequence_combiner = get_combiner_schema("sequence") %}

```yaml
combiner:
    {% for line in schema_class_to_yaml(sequence_combiner).split("\n") %}
    {{- line }}
    {% endfor %}
```

Parameters:

{{ render_fields(schema_class_to_fields(sequence_combiner, exclude=["type"]), details=details) }}

### TabNet Combiner

```
+-----------+
|Input      |
|Feature 1  +-+
+-----------+ |
+-----------+ | +------+
|...        +--->TabNet+-->
+-----------+ | +------+
+-----------+ |
|Input      +-+
|Feature N  |
+-----------+
```

The `tabnet` combiner implements the [TabNet](https://arxiv.org/abs/1908.07442) model, which uses attention and sparsity
to achieve high performance on tabular data. It assumes all outputs from encoders are tensors of size `b x h` where `b`
is the batch size and `h` is the hidden dimension, which can be different for each input.
If the input tensors have a different shape, it automatically flattens them.
It returns the final `b x h'` tensor where `h'` is the user-specified output size.

These are the available parameters of a `tabnet` combiner:

- `size`: the size of the hidden layers. `N_a` in the paper.
- `output_size`: the size of the output of each step and of the final aggregated representation. `N_d` in the paper.
- `num_steps` (default `1`): number of steps / repetitions of the attentive transformer and feature transformer computations. `N_steps` in the paper.
- `num_total_blocks` (default `4`): total number of feature transformer blocks at each step.
- `num_shared_blocks` (default `2`): number of shared feature transformer blocks across the steps.
- `relaxation_factor` (default `1.5`): Factor that influences how many times a feature should be used across the steps
of computation. a value of `1` implies it each feature should be use once, a higher value allows for multiple usages. `gamma` in the paper.
- `bn_epsilon` (default `0.001`): epsilon to be added to the batch norm denominator.
- `bn_momentum` (default `0.7`): momentum of the batch norm. `m_B` in the paper.
- `bn_virtual_bs` (default `null`): size of the virtual batch size used by ghost batch norm. If `null`, regular batch
norm is used instead. `B_v` from the paper.
- `sparsity` (default `0.00001`): multiplier of the sparsity inducing loss. `lambda_sparse` in the paper.
- `dropout` (default `0`): dropout rate.

Example configuration of a `tabnet` combiner:

```yaml
type: tabnet
size: 32
ooutput_size: 32
num_steps: 5
num_total_blocks: 4
num_shared_blocks: 2
relaxation_factor: 1.5
bn_epsilon: 0.001
bn_momentum: 0.7
bn_virtual_bs: 128
sparsity: 0.00001
dropout: 0
```

### Transformer Combiner

The `transformer` combiner combines input features using a stack of Transformer blocks (from [Attention Is All You Need](https://arxiv.org/abs/1706.03762)).
It assumes all outputs from encoders are tensors of size `b x h` where `b` is the batch size and `h` is the hidden
dimension, which can be different for each input.
If the input tensors have a different shape, it automatically flattens them.
It then projects each input tensor to the same hidden / embedding size and encodes them with a stack of Transformer layers.
Finally, the transformer combiner applies a reduction to the outputs of the Transformer stack, followed by optional
fully connected layers.
The output is a `b x h'` tensor where `h'` is the size of the last fully connected layer or the hidden / embedding
size, or a `b x n x h'` where `n` is the number of input features and `h'` is the hidden / embedding size if no reduction
is applied.

```
+-----------+
|Input      |
|Feature 1  +-+
+-----------+ |
+-----------+ |  +------------+   +------+   +----------+
|...        +--->|Transformer +-->|Reduce+-->|Fully     +->
|           | |  |Stack       |   +------+   |Connected |
+-----------+ |  +------------+              |Layers    |
+-----------+ |                              +----------+
|Input      +-+
|Feature N  |
+-----------+
```

These are the available parameters of a `transformer` combiner:

- `num_layers` (default `1`): number of layers in the stack of transformer blocks.
- `hidden_size` (default `256`): hidden / embedding size of each transformer block.
- `num_heads` (default `8`): number of attention heads of each transformer block.
- `transformer_output_size` (default `256`): size of the fully connected layers inside each transformer block.
- `dropout` (default `0`): dropout rate after the transformer.
- `fc_layers` (default `null`): is a list of dictionaries containing the parameters of all the fully connected layers.
The length of the list determines the number of stacked fully connected layers and the content of each dictionary
determines the parameters for a specific layer. The available parameters for each layer are: `activation`, `dropout`,
`norm`, `norm_params`, `output_size`, `use_bias`, `bias_initializer` and `weights_initializer`. If any of those values
is missing from the dictionary, the default one specified as a parameter of the decoder will be used instead.
- `num_fc_layers` (default 0): this is the number of stacked fully connected layers to apply after reduction of the
transformer output sequence.
- `output_size` (default `256`): if an `output_size` is not already specified in `fc_layers` this is the default
`output_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `'glorot_uniform'`): initializer for the weight matrix. Options are: `constant`,
`identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`,
`glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`,
`lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of
initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each
initializer, please refer to [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html).
- `bias_initializer` (default `'zeros'`):  initializer for the bias vector. Options are: `constant`, `identity`,
`zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`,
`glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other
keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please
refer to [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html).
- `norm` (default `null`): normalization applied at the beginnging of the fully-connected stack. If a `norm` is not already specified for the `fc_layers` this is the default `norm` that will be used for each layer. One of: `null`, `batch`, `layer`, `ghost`. See [Normalization](#normalization) for details.
- `norm_params` (default `null`): parameters passed to the `norm` module. See [Normalization](#normalization) for details.
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default
`activation` that will be used for each layer. It indicates the activation function applied to the output.
- `fc_dropout` (default `0`): dropout rate for the fully connected layers.
- `fc_residual` (default `false`): if `true` adds a residual connection to each fully connected layer block. It is
required that all fully connected layers have the same size for this parameter to work correctly.
- `reduce_output` (default `mean`): describes the strategy to use to aggregate the output of the transformer.
Possible values include `last`, `sum`, `mean`, `concat`, or `none`.

Example configuration of a `transformer` combiner:

```yaml
type: transformer
num_layers: 1
hidden_size: 256
num_heads: 8
transformer_output_size: 256
dropout: 0.1
fc_layers: null
num_fc_layers: 0
output_size: 256
use_bias: True
weights_initializer: glorot_uniform
bias_initializer: zeros
norm: null
norm_params: null
fc_activation: relu
fc_dropout: 0
fc_residual: null
reduce_output: mean
```

Resources to learn more about transformers:

- [CS480/680 Lecture 19: Attention and Transformer Networks](https://www.youtube.com/watch?v=OyFJWRnt_AY&list=PLk4mwFjvagV3vp1JZ3lNohb1LalMp-VOY&index=1) (VIDEO)
- [Attention is all you need - Attentional Neural Network Models Masterclass](https://www.youtube.com/watch?v=rBCqOTEfxvg&list=PLk4mwFjvagV3vp1JZ3lNohb1LalMp-VOY&index=2) (VIDEO)
- [Illustrated: Self-Attention](https://colab.research.google.com/drive/1rPk3ohrmVclqhH7uQ7qys4oznDdAhpzF) (Colab notebook)

### TabTransformer Combiner

The `tabtransformer` combiner combines input features in the following sequence of operations. Except for binary and number features, the combiner projects features to an embedding size. These features are concatenated as if they were a sequence and passed through a transformer. After the transformer, the number and binary features are concatenated (which are of size 1) and then concatenated  with the output of the transformer and is passed to a stack of fully connected layers (from [TabTransformer: Tabular Data Modeling Using Contextual Embeddings](https://arxiv.org/abs/2012.06678)).
It assumes all outputs from encoders are tensors of size `b x h` where `b` is the batch size and `h` is the hidden
dimension, which can be different for each input.
If the input tensors have a different shape, it automatically flattens them.
It then projects each input tensor to the same hidden / embedding size and encodes them with a stack of Transformer layers.
Finally, the transformer combiner applies a reduction to the outputs of the Transformer stack, followed by the above concatenation and optional fully connected layers.
The output is a `b x h'` tensor where `h'` is the size of the last fully connected layer or the hidden / embedding
size, or a `b x n x h'` where `n` is the number of input features and `h'` is the hidden / embedding size if no reduction
is applied.

```
+-----------+
|Input      |
|Feature 1  +-+
+-----------+ |
+-----------+ |  +-------------+  +--------------+    +------ +   +----------+  +----------+
|           +--->| Categoricial+->|TabTransformer +-->|Reduce +-> | Combined +->|Fully     +->
|           | |  | Embeddings  |  |Stack          |   +-------+   | Hidden   |  |Connected |
|           | |  +-------------+  +---------------+               | Layers   |  |Layers    |
|...        | |                                                   +----------+  +----------+
|           | |                          +-----------+                 ^
|           | |                          | Binary &  |                 |
+-----------+ |------------------------->| Numerical |------------------
+-----------+ |                          | Encodings |
|Input      +-+                          +-----------+
|Feature N  |
+-----------+
```

These are the available parameters of a `transformer` combiner:

- `num_layers` (default `1`): number of layers in the stack of transformer blocks.
- `hidden_size` (default `256`): hidden / embedding size of each transformer block.
- `num_heads` (default `8`): number of attention heads of each transformer block.
- `transformer_output_size` (default `256`): size of the fully connected layers inside each transformer block.
- `dropout` (default `0`): dropout rate after the transformer.
- `fc_layers` (default `null`): is a list of dictionaries containing the parameters of all the fully connected layers.
The length of the list determines the number of stacked fully connected layers and the content of each dictionary
determines the parameters for a specific layer. The available parameters for each layer are: `activation`, `dropout`,
`norm`, `norm_params`, `output_size`, `use_bias`, `bias_initializer` and `weights_initializer`. If any of those values
is missing from the dictionary, the default one specified as a parameter of the decoder will be used instead.
- `num_fc_layers` (default 0): this is the number of stacked fully connected layers to apply after reduction of the
transformer output sequence.
- `output_size` (default `256`): if an `output_size` is not already specified in `fc_layers` this is the default
`output_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `'glorot_uniform'`): initializer for the weight matrix. Options are: `constant`,
`identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`,
`glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`,
`lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of
initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each
initializer, please refer to [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html).
- `bias_initializer` (default `'zeros'`):  initializer for the bias vector. Options are: `constant`, `identity`,
`zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`,
`glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other
keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please
refer to [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html).
- `norm` (default `null`): normalization applied at the beginnging of the fully-connected stack. If a `norm` is not already specified for the `fc_layers` this is the default `norm` that will be used for each layer. One of: `null`, `batch`, `layer`, `ghost`. See [Normalization](#normalization) for details.
- `norm_params` (default `null`): parameters passed to the `norm` module. See [Normalization](#normalization) for details.
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default
`activation` that will be used for each layer. It indicates the activation function applied to the output.
- `fc_dropout` (default `0`): dropout rate for the fully connected layers.
- `fc_residual` (default `false`): if `true` adds a residual connection to each fully connected layer block. It is
required that all fully connected layers have the same size for this parameter to work correctly.
- `reduce_output` (default `mean`): describes the strategy to use to aggregate the output of the transformer.
Possible values include `last`, `sum`, `mean`, `concat`, or `none`.
- `embed_input_feature_name` (default `null`) controls the size of the embeddings.  Valid values are `add` which uses the `hidden_size` value or an integer to set a specific value.  In the case of an integer value, it must be smaller than `hidden_size`.

Example configuration of a `tabtransformer` combiner:

```yaml
type: tabtransformer
num_layers: 1
hidden_size: 256
num_heads: 8
transformer_output_size: 256
dropout: 0.1
fc_layers: null
num_fc_layers: 0
output_size: 256
use_bias: True
weights_initializer: glorot_uniform
bias_initializer: zeros
norm: null
norm_params: null
fc_activation: relu
fc_dropout: 0
fc_residual: null
reduce_output: mean
embed_input_fature_name: null
```

### Comparator Combiner

The `comparator` combiner compares the hidden representation of two entities defined by lists of features.
It assumes all outputs from encoders are tensors of size `b x h` where `b` is the batch size and `h` is the hidden
dimension, which can be different for each input.
If the input tensors have a different shape, it automatically flattens them.
It then concatenates the representations of each entity and projects them both to vectors of size `output_size`.
Finally, it compares the two entity representations by dot product, element-wise multiplication, absolute difference and bilinear product.
It returns the final `b x h'` tensor where `h'` is the size of the concatenation of the four comparisons.

```
+-----------+
|Entity 1   |
|Input      |
|Feature 1  +-+
+-----------+ |
+-----------+ |  +-------+   +----------+
|...        +--->|Concat +-->|FC Layers +--+
|           | |  +-------+   +----------+  |
+-----------+ |                            |
+-----------+ |                            |
|Entity 1   +-+                            |
|Input      |                              |
|Feature N  |                              |
+-----------+                              |   +---------+
                                           +-->| Compare +->
+-----------+                              |   +---------+
|Entity 2   |                              |
|Input      |                              |
|Feature 1  +-+                            |
+-----------+ |                            |
+-----------+ |  +-------+   +----------+  |
|...        +--->|Concat +-->|FC Layers +--+
|           | |  +-------+   +----------+
+-----------+ |
+-----------+ |
|Entity 2   +-+
|Input      |
|Feature N  |
+-----------+
```

These are the available parameters of a `comparator` combiner:

- `entity_1`: list of input features that compose the first entity to compare.
- `entity_2`: list of input features that compose the second entity to compare.
- `fc_layers` (default `null`): is a list of dictionaries containing the parameters of all the fully connected layers.
The length of the list determines the number of stacked fully connected layers and the content of each dictionary
determines the parameters for a specific layer. The available parameters for each layer are: `activation`, `dropout`,
`norm`, `norm_params`, `output_size`, `use_bias`, `bias_initializer` and `weights_initializer`. If any of those values
is missing from the dictionary, the default one specified as a parameter of the decoder will be used instead.
- `num_fc_layers` (default 0): this is the number of stacked fully connected layers that the input to the feature passes
through. Their output is projected in the feature's output space.
- `output_size` (default `256`): if `output_size` is not already specified in `fc_layers` this is the default
`output_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `'glorot_uniform'`): initializer for the weight matrix. Options are: `constant`,
`identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`,
`glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other
keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please
refer to [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html).
- `bias_initializer` (default `'zeros'`):  initializer for the bias vector. Options are: `constant`, `identity`,
`zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`,
`glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other
keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please
refer to [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html).
- `norm` (default `null`): normalization applied at the beginnging of the fully-connected stack. If a `norm` is not already specified for the `fc_layers` this is the default `norm` that will be used for each layer. One of: `null`, `batch`, `layer`, `ghost`. See [Normalization](#normalization) for details.
- `norm_params` (default `null`): parameters passed to the `norm` module. See [Normalization](#normalization) for details.
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default
`activation` that will be used for each layer. It indicates the activation function applied to the output.
- `dropout` (default `0`): dropout rate for the fully connected layers.

Example configuration of a `comparator` combiner:

```yaml
type: comparator
entity_1: [feature_1, feature_2]
entity_2: [feature_3, feature_4]
fc_layers: null
num_fc_layers: 0
output_size: 256
use_bias: true
weights_initializer: 'glorot_uniform'
bias_initializer: 'zeros'
norm: null
norm_params: null
activation: relu
dropout: 0
```

## Common Parameters

These parameters are used across multiple combiners (and some encoders / decoders) in similar ways.

### Normalization

Normalization applied at the beginnging of the fully-connected stack. If a `norm` is not already specified for the `fc_layers` this is the default `norm` that will be used for each layer. One of: 

- `null`: no normalization
- `batch`: batch normalization
- `layer`: layer normalization
- `ghost`: ghost batch normalization

#### Batch Normalization

Applies Batch Normalization as described in the paper [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167). See [PyTorch documentation on batch normalization](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html) for more details.

Parameters:

- `eps`: Epsilon to be added to the batch norm denominator. Default: `0.001`.
- `momentum`: The value used for the running_mean and running_var computation. Can be set to None for cumulative moving average (i.e. simple average). Default: `0.1`.
- `affine`: A boolean value that when set to `true`, this module has learnable affine parameters. Default: `true`.
- `track_running_stats`: A boolean value that when set to `true`, this module tracks the running mean and variance, and when set to `false`, this module does not track such statistics, and initializes statistics buffers running_mean and running_var as `null`. When these buffers are `null`, this module always uses batch statistics. in both training and eval modes. Default: `true`.

Example:

```yaml
norm: batch
norm_params:
  eps: 0.001
  momentum: 0.1
  affine: true
  track_running_stats: true
```

#### Layer Normalization

Applies Layer Normalization over a mini-batch of inputs as described in the paper [Layer Normalization](https://arxiv.org/abs/1607.06450). See [PyTorch documentation on layer normalization](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) for more details.

Parameters:

- `eps`: A value added to the denominator for numerical stability. Default: `0.00001`.
- `elementwise_affine`: A boolean value that when set to `true`, this module has learnable per-element affine parameters initialized to ones (for weights) and zeros (for biases). Default: `true`.

Example:

```yaml
norm: layer
norm_params:
  eps: 0.00001
  elementwise_affine: true
```

#### Ghost Batch Normalization

Ghost Batch Norm is a technique designed to address the "generalization gap" whereby the training process breaks down with very large batch sizes.
If you are using a large batch size (typically in the thousands) to maximize GPU utilization, but the model is not converging well, enabling ghost
batch norm can be a useful technique to improve convergence. 

When using ghost batch norm, you specify a `virtual_batch_size` (default `128`) representing the "ideal" batch size to train with (ignoring throughput or GPU utilization). The ghost batch norm will then subdivide each batch into subbatches of size `virtual_batch_size` and apply batch normalization to each.

A notable downside to ghost batch norm is that it is more computationally expensive than traditional batch norm, so it is only recommended to use it
when the batch size that maximizes throughput is significantly higher than the batch size that yields the best convergence (one or more orders of magnitude higher).

The approach was introduced in [Train Longer, Generalize Better: Closing the Generalization Gap in Large Batch Training of Neural Networks](https://arxiv.org/abs/1705.08741) and since popularized by its use in [TabNet](#tabnet-combiner).

Parameters:

- `virtual_batch_size`: Size of the virtual batch size used by ghost batch norm. If null, regular batch norm is used instead. `B_v` from the TabNet paper.
- `epsilon`: Epsilon to be added to the batch norm denominator.
- `momentum`: Momentum of the batch norm. 1 - `m_B` from the TabNet paper.

```yaml
norm: ghost
norm_params:
  virtual_batch_size: 128
  epsilon: 0.001
  momentum: 0.05
```


