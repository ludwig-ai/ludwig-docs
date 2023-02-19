Example:

```yaml
defaults:
  text:
    encoder:
      type: bert
      trainable: true
trainer:
  epochs: 5
  batch_size: auto
  learning_rate: 0.00001
  learning_rate_scheduler:
    warmup_fraction: 0.2
    decay: linear
  optimizer:
    type: adamw
  use_mixed_precision: true
```

# Suggested Configurations

## Feature Encoder and Preprocessing

### Text Encoders

### Image Encoders

### Trainable

### Cache Encoder Embeddings

## Trainer

### Epochs

### Batch Size

### Learning Rate

#### Base learning rate

Use a very small larning rate when fine-tuning to avoid
catastrophic forgetting of all the pretrained model's previously learned knowledge.

- When `trainable=true` we recommend starting as low as `learning_rate=0.00001`.
- When `trainable=false` we recommend starting with `learning_rate=0.00002`.

Note that setting `learning_rate=auto` will automatically set the above defaults
on your behalf based on the selected model architecture.

```yaml
trainer:
  learning_rate: auto
```

#### Learning rate schedule

It's important to both warmup the learning rate (particularly when using
[distributed training](./distributed_training.md)) and decay it to avoid catastrophic
forgetting.

As a starting point, we suggest:

```yaml
trainer:
  epochs: 5
  learning_rate_scheduler:
    warmup_fraction: 0.2
    decay: linear
```

A `warmup_fraction` of `0.2` will result in 20% of the total training steps being spent
linearly scaling the learning rate up from 0 to the initial value provided in `trainer.learning_rate`. This
is useful as otherwise the learning process may over-correct the weights of the pretrained model
in the early stages of training.

Using `linear` decay is a very aggressive decay strategy that linearly reduces the
learning rate down to 0 as training approaches the final epoch. The decay will only start
after the learning rate warmup period has finished and the learning rate is set to its initial
value.

Both warmup and decay are affected by the total `epochs` set in the `trainer` config, so it's important to make
sure the `epochs` are set to a sufficiently low value for the warmup and decay to be effective. If `epochs` is left at
the default value of `100`, then too much time will be spent in warmup and the decay will not be noticeable.

#### Learning rate scaling

The base `trainer.learning_rate` will be scaled up as the number of training workers increases for distributed training. By
default the learning rate will scale linearly (`linear`), but this can be relaxed if you notice catastrophic forgetting is
occurring, in which case a softer `learning_rate_scaling=sqrt` setting may be worth considering. 

```yaml
trainer:
  learning_rate_scaling: sqrt
```

### Optimizer

We recommend using `adamw` as the optimizer for fine-tuning.

```yaml
trainer:
  optimizer:
    type: adamw
```

AdamW is typically recommended over more traditional optimizers such as SGD or Adam due to its improved handling of weight decay, which
is of particular importance during fine-tuning to avoid catastrophic forgetting.

### Mixed Precision

It's highly recommended to set `use_mixed_precision=true` when fine-tuning. Empirically, it can speedup training by
about 2.5x witout loss of model quality.

```yaml
trainer:
  use_mixed_precision: true
```

## Backend

Fine-tuning large pretrained models typically benefit from [distributed training](./distributed_training.md) without
requiring a lot of additional hyperparameter tuning. As such, we recommend using the Ray [backend](../configuration/backend.md) 
in order to take advantage of multi-GPU training and to scale to large datasets.

In most cases, the `horovod` or `ddp` distributed [strategy](../configuration/backend.md#trainer) will work well, but if the
model is too large for your GPU type, then you should try model parallelism as described below.

### Model Parallelism for LLMs

Some large language models have billions of parameters and are too large to train even on the most powerful single GPUs. Some examples
include the large variants of:

- [BLOOM](https://huggingface.co/docs/transformers/model_doc/bloom)
- [GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2)
- [GPT-J](https://huggingface.co/docs/transformers/model_doc/gptj)
- [OPT](https://huggingface.co/docs/transformers/model_doc/opt)
- [FLAN-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5)

For these models, it's recommended to enable the `fsdp` distributed strategy in a multi-GPU Ray cluster.

Example:

```yaml
defaults:
  text:
    encoder:
      type: auto_transformer
      pretrained_model_name_or_path: bigscience/bloom-3b
      trainable: true
backend:
  type: ray
  trainer:
    strategy: fsdp
```
