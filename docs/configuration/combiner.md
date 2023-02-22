{% from './macros/includes.md' import render_fields, render_yaml %}
{% set norm_details = "See [Normalization](#normalization) for details." %}
{% set details = {"norm": norm_details, "norm_params": norm_details} %}

Combiners take the outputs of all input features encoders and combine them before providing the combined representation
to the output feature decoders.

You can specify which one to use in the `combiner` section of the configuration, and if you don't specify a combiner,
the `concat` combiner will be used.

## Combiner Types

### Concat Combiner

``` mermaid
graph LR
  I1[Encoder Output 1] --> C[Concat];
  IK[...] --> C;
  IN[Encoder Output N] --> C;
  C --> FC[Fully Connected Layers];
  FC --> ...;
  subgraph COMBINER..
  C
  FC
  end
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
{{ render_yaml(concat_combiner, parent="combiner") }}

Parameters:

{{ render_fields(schema_class_to_fields(concat_combiner, exclude=["type"]), details=details) }}

### Sequence Concat Combiner

``` mermaid
graph LR
  I1 --> X1{Tile};
  IK --> X1;
  IN --> X1;
  IO --> X1;

  X1 --> SC1;
  X1 --> SCK;
  X1 --> SCN;

  SC1 --> R[Reduce];
  SCK --> R;
  SCN --> R;
  R --> ...;
  subgraph CONCAT["TENSOR.."]
    direction TB
    SC1["emb seq 1 | emb oth" ];
    SCK[...];
    SCN["emb seq n | emb oth"];
  end
  subgraph COMBINER..
  X1
  CONCAT
  R
  end
  subgraph SF[SEQUENCE FEATS..]
  direction TB
  I1["emb seq 1" ];
  IK[...];
  IN["emb seq n"];
  end
  subgraph OF[OTHER FEATS..]
  direction TB
  IO["emb oth"]
  end
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
{{ render_yaml(sequence_concat_combiner, parent="combiner") }}

Parameters:

{{ render_fields(schema_class_to_fields(sequence_concat_combiner, exclude=["type"]), details=details) }}

### Sequence Combiner

``` mermaid
graph LR
  I1 --> X1{Tile};
  IK --> X1;
  IN --> X1;
  IO --> X1;

  X1 --> SC1;
  X1 --> SCK;
  X1 --> SCN;

  SC1 --> R["Sequence Encoder"];
  SCK --> R;
  SCN --> R;
  R --> ...;
  subgraph CONCAT["TENSOR.."]
    direction TB
    SC1["emb seq 1 | emb oth" ];
    SCK[...];
    SCN["emb seq n | emb oth"];
  end
  subgraph COMBINER..
  X1
  CONCAT
  R
  end
  subgraph SF[SEQUENCE FEATS..]
  direction TB
  I1["emb seq 1" ];
  IK[...];
  IN["emb seq n"];
  end
  subgraph OF[OTHER FEATS..]
  direction TB
  IO["emb oth"]
  end
```

The `sequence` combiner stacks a sequence concat combiner with a sequence encoder.
All the considerations about input tensor ranks described for the [sequence concat combiner](#sequence-concat-combiner)
apply also in this case, but the main difference is that this combiner uses the `b x s x h'` output of the sequence
concat combiner, where `b` is the batch size, `s` is the sequence length and `h'` is the sum of the hidden dimensions of
all input features, as input for any of the sequence encoders described in the [sequence features encoders section](../features/sequence_features#sequence-input-features-and-encoders).
Refer to that section for more detailed information about the sequence encoders and their parameters.
All considerations on the shape of the outputs for the sequence encoders also apply to sequence combiner.

{% set sequence_combiner = get_combiner_schema("sequence") %}
{{ render_yaml(sequence_combiner, parent="combiner") }}

Parameters:

{{ render_fields(schema_class_to_fields(sequence_combiner, exclude=["type"]), details=details) }}

### TabNet Combiner

``` mermaid
graph LR
  I1[Encoder Output 1] --> C[TabNet];
  IK[...] --> C;
  IN[Encoder Output N] --> C;
  C --> ...;
```

The `tabnet` combiner implements the [TabNet](https://arxiv.org/abs/1908.07442) model, which uses attention and sparsity
to achieve high performance on tabular data. It assumes all outputs from encoders are tensors of size `b x h` where `b`
is the batch size and `h` is the hidden dimension, which can be different for each input.
If the input tensors have a different shape, it automatically flattens them.
It returns the final `b x h'` tensor where `h'` is the user-specified output size.

{% set tabnet_combiner = get_combiner_schema("tabnet") %}
{{ render_yaml(tabnet_combiner, parent="combiner") }}

Parameters:

{% set ghost_bn_details = "See [Ghost Batch Normalization](#ghost-batch-normalization) for details." %}
{% set details = merge_dicts({"bn_virtual_bs": ghost_bn_details}, details) %}
{{ render_fields(schema_class_to_fields(tabnet_combiner, exclude=["type"]), details=details) }}

### Transformer Combiner

``` mermaid
graph LR
  I1[Encoder Output 1] --> C["Transformer Stack"];
  IK[...] --> C;
  IN[Encoder Output N] --> C;
  C --> R[Reduce];
  R --> FC[Fully Connected Layers];
  FC --> ...;
  subgraph COMBINER..
  C
  R
  FC
  end
```

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

Resources to learn more about transformers:

- [CS480/680 Lecture 19: Attention and Transformer Networks](https://www.youtube.com/watch?v=OyFJWRnt_AY&list=PLk4mwFjvagV3vp1JZ3lNohb1LalMp-VOY&index=1) (VIDEO)
- [Attention is all you need - Attentional Neural Network Models Masterclass](https://www.youtube.com/watch?v=rBCqOTEfxvg&list=PLk4mwFjvagV3vp1JZ3lNohb1LalMp-VOY&index=2) (VIDEO)
- [Illustrated: Self-Attention](https://colab.research.google.com/drive/1rPk3ohrmVclqhH7uQ7qys4oznDdAhpzF) (Colab notebook)

{% set transformer_combiner = get_combiner_schema("transformer") %}
{{ render_yaml(transformer_combiner, parent="combiner") }}

Parameters:

{{ render_fields(schema_class_to_fields(transformer_combiner, exclude=["type"]), details=details) }}

### TabTransformer Combiner

``` mermaid
graph LR
  I1[Cat Emb 1] --> T1["Concat"];
  IK[...] --> T1;
  IN[Cat Emb N] --> T1;
  N1[Number ...] --> T4["Concat"];
  B1[Binary ...] --> T4;
  T1 --> T2["Transformer"];
  T2 --> T3["Reduce"];
  T3 --> T4;
  T4 --> T5["FC Layers"];
  T5 --> ...;
  subgraph COMBINER..
  CAT
  T4
  T5
  end
  subgraph ENCODER OUT..
  I1
  IK
  IN
  N1
  B1
  end
  subgraph CAT["CATEGORY PIPELINE.."]
  direction TB
  T1
  T2
  T3
  end
```

The `tabtransformer` combiner combines input features in the following sequence of operations. The combiner projects all encoder outputs except binary and number features into an embedding space. These features are concatenated as if they were a sequence and passed through a transformer. After the transformer, the number and binary features are concatenated (which are of size 1) and then concatenated  with the output of the transformer and is passed to a stack of fully connected layers (from [TabTransformer: Tabular Data Modeling Using Contextual Embeddings](https://arxiv.org/abs/2012.06678)).
It assumes all outputs from encoders are tensors of size `b x h` where `b` is the batch size and `h` is the hidden
dimension, which can be different for each input.
If the input tensors have a different shape, it automatically flattens them.
It then projects each input tensor to the same hidden / embedding size and encodes them with a stack of Transformer layers.
Finally, the transformer combiner applies a reduction to the outputs of the Transformer stack, followed by the above concatenation and optional fully connected layers.
The output is a `b x h'` tensor where `h'` is the size of the last fully connected layer or the hidden / embedding
size, or a `b x n x h'` where `n` is the number of input features and `h'` is the hidden / embedding size if no reduction
is applied.

{% set tabtransformer_combiner = get_combiner_schema("tabtransformer") %}
{{ render_yaml(tabtransformer_combiner, parent="combiner") }}

Parameters:

{{ render_fields(schema_class_to_fields(tabtransformer_combiner, exclude=["type"]), details=details) }}

### Comparator Combiner

``` mermaid
graph LR
  I1[Entity 1 Embed 1] --> C1[Concat];
  IK[...] --> C1;
  IN[Entity 1 Embed N] --> C1;
  C1 --> FC1[FC Layers];
  FC1 --> COMP[Compare];

  I2[Entity 2 Embed 1] --> C2[Concat];
  IK2[...] --> C2;
  IN2[Entity 2 Embed N] --> C2;
  C2 --> FC2[FC Layers];
  FC2 --> COMP;

  COMP --> ...;

  subgraph ENTITY1["ENTITY 1.."]
  I1
  IK
  IN
  end

  subgraph ENTITY2["ENTITY 2.."]
  I2
  IK2
  IN2
  end

  subgraph COMBINER..
  C1
  FC1
  C2
  FC2
  COMP
  end
```

The `comparator` combiner compares the hidden representation of two entities defined by lists of features.
It assumes all outputs from encoders are tensors of size `b x h` where `b` is the batch size and `h` is the hidden
dimension, which can be different for each input.
If the input tensors have a different shape, it automatically flattens them.
It then concatenates the representations of each entity and projects them both to vectors of size `output_size`.
Finally, it compares the two entity representations by dot product, element-wise multiplication, absolute difference and bilinear product.
It returns the final `b x h'` tensor where `h'` is the size of the concatenation of the four comparisons.

{% set comparator_combiner = get_combiner_schema("comparator") %}
{{ render_yaml(comparator_combiner, parent="combiner") }}

Parameters:

{{ render_fields(schema_class_to_fields(comparator_combiner, exclude=["type"]), details=details) }}

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

```yaml
norm: batch
norm_params:
  eps: 0.001
  momentum: 0.1
  affine: true
  track_running_stats: true
```

Parameters:

- **`eps`** (default: `0.001`): Epsilon to be added to the batch norm denominator.
- **`momentum`** (default: `0.1`): The value used for the running_mean and running_var computation. Can be set to None for cumulative moving average (i.e. simple average). Default: `0.1`.
- **`affine`** (default: `true`): A boolean value that when set to `true`, this module has learnable affine parameters.
- **`track_running_stats`** (default: `true`): A boolean value that when set to `true`, this module tracks the running mean and variance, and when set to `false`, this module does not track such statistics, and initializes statistics buffers running_mean and running_var as `null`. When these buffers are `null`, this module always uses batch statistics. in both training and eval modes.

#### Layer Normalization

Applies Layer Normalization over a mini-batch of inputs as described in the paper [Layer Normalization](https://arxiv.org/abs/1607.06450). See [PyTorch documentation on layer normalization](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) for more details.

```yaml
norm: layer
norm_params:
  eps: 0.00001
  elementwise_affine: true
```

Parameters:

- **`eps`** (default: `0.00001`): A value added to the denominator for numerical stability.
- **`elementwise_affine`** (default: `true`): A boolean value that when set to `true`, this module has learnable per-element affine parameters initialized to ones (for weights) and zeros (for biases)

#### Ghost Batch Normalization

Ghost Batch Norm is a technique designed to address the "generalization gap" whereby the training process breaks down with very large batch sizes.
If you are using a large batch size (typically in the thousands) to maximize GPU utilization, but the model is not converging well, enabling ghost
batch norm can be a useful technique to improve convergence. 

When using ghost batch norm, you specify a `virtual_batch_size` (default `128`) representing the "ideal" batch size to train with (ignoring throughput or GPU utilization). The ghost batch norm will then subdivide each batch into subbatches of size `virtual_batch_size` and apply batch normalization to each.

A notable downside to ghost batch norm is that it is more computationally expensive than traditional batch norm, so it is only recommended to use it
when the batch size that maximizes throughput is significantly higher than the batch size that yields the best convergence (one or more orders of magnitude higher).

The approach was introduced in [Train Longer, Generalize Better: Closing the Generalization Gap in Large Batch Training of Neural Networks](https://arxiv.org/abs/1705.08741) and since popularized by its use in [TabNet](#tabnet-combiner).

```yaml
norm: ghost
norm_params:
  virtual_batch_size: 128
  epsilon: 0.001
  momentum: 0.05
```

Parameters:

- **`virtual_batch_size`** (default: `128`): Size of the virtual batch size used by ghost batch norm. If null, regular batch norm is used instead. `B_v` from the TabNet paper.
- **`epsilon`** (default: `0.001`): Epsilon to be added to the batch norm denominator.
- **`momentum`** (default: `0.05`): Momentum of the batch norm. 1 - `m_B` from the TabNet paper.
