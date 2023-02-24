Fine-tuning is the process of taking a model previously trained one dataset, and
adapting it to a more specialized dataset / task. Typically the original dataset
is very large and very general (for example: a crawl of a large portion of the public
Internet), and consequently the models are very large in order to reason about all this
information (billions of parameters or more).

Libraries like HuggingFace's [transformers](https://huggingface.co/docs/transformers/index) provide
acess to state-of-the-art pretrained models that can be used as input feature [encoders](../../configuration/features/input_features.md#encoders)
in Ludwig, allowing you to take advantage of these large pretrained models and adapt them to solve your specific tasks, combining them
with other domain-specific features like tabular metadata to create powerful multi-modal model architectures.

Ludwig's default configuration is designed to be fast and flexible, and as such, there are a few adjustments to the default
configuration parameters we suggest making when fine-tuning a pretraine model. The sections below show examples of configurations
we've found to give good results, along with the rationale behind each overridden parameter.

# Suggested Configuration

The below partial configuration shows the "full fine-tuning configuration" including
trainable weights, batch size set to maximize throughput, and learning rate warmup / decay:

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

If you're looking to get the best performance you can out of the model, and are insensitive to
training time, this is a good place to start. In the sections below, we'll also cover options that
tradeoff some potential performance in favor of large speedups to the training throughput.

## Feature Encoder and Preprocessing

### Trainable

### Cache Encoder Embeddings

### Text Encoders

All of the [HuggingFace encoders](../../configuration/features/text_features.md#huggingface-encoders) in Ludwig can be used for fine-tuning.
If there is a specific model you want to use, but don't see it listed, you can use the `auto_transformer` encoder in conjunction with
providing the model name in the `pretrained_model_name_or_path` parameter.

### Image Encoders

All of the [Torchvision pretrained models](https://pytorch.org/vision/stable/models.html) in Ludwig can be used for fine-tuning.

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
[distributed training](./index.md)) and decay it to avoid catastrophic
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

Fine-tuning large pretrained models typically benefit from [distributed training](./index.md) without
requiring a lot of additional hyperparameter tuning. As such, we recommend using the Ray [backend](../../configuration/backend.md)
in order to take advantage of multi-GPU training and to scale to large datasets.

In most cases, the `horovod` or `ddp` distributed [strategy](../../configuration/backend.md#trainer) will work well, but if the
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
