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

## Feature Encoder and Preprocessing

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
