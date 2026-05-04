---
description: "Compare optimizers (Adam, AdamW, Lion, Sophia) for Ludwig model training — configuration examples and practical guidance."
---

# Optimizer Comparison

Choosing the right optimizer can meaningfully affect convergence speed and final accuracy.
Ludwig exposes all major optimizers through the `trainer.optimizer` config block.

## Supported optimizers

| Optimizer | Key strength | Typical use case |
|-----------|-------------|-----------------|
| `sgd` | Baseline, well-understood | Small tabular models |
| `adam` | Fast convergence, adaptive LR | Default for most tasks |
| `adamw` | Adam + decoupled weight decay | Fine-tuning pretrained models |
| `rmsprop` | Stable for RNNs | Sequence models |
| `lion` | Memory-efficient (no second moment) | Large models on limited memory |
| `sophia` | Second-order curvature estimate | Transformers, NLP tasks |

## Configurations

### Adam (default)

```yaml
trainer:
  optimizer:
    type: adam
    lr: 1.0e-3
    betas:
      - 0.9
      - 0.999
    eps: 1.0e-8
```

### AdamW

AdamW decouples weight decay from the gradient update, which prevents weight decay from inadvertently acting as
an adaptive learning rate dampener.  Use it whenever training from scratch with regularisation or when fine-tuning.

```yaml
trainer:
  optimizer:
    type: adamw
    lr: 3.0e-4
    weight_decay: 0.01
```

### Lion

[Lion](https://arxiv.org/abs/2302.06675) (EvoLved Sign Momentum) uses only the sign of the gradient update,
requiring no second moment accumulator.  This halves optimizer memory usage vs. Adam/AdamW — a major advantage
when training large models.  Lion typically works best with a learning rate 3–10× smaller than AdamW.

```yaml
trainer:
  optimizer:
    type: lion
    lr: 3.0e-5
    weight_decay: 0.1
    betas:
      - 0.9
      - 0.99
```

### Sophia

[Sophia](https://arxiv.org/abs/2305.14342) estimates the diagonal Hessian of the loss using a Hutchinson estimator
and uses it to precondition the gradient step.  This can accelerate training of transformer-based models by 2× over
AdamW on NLP benchmarks.

```yaml
trainer:
  optimizer:
    type: sophia
    lr: 2.0e-4
    betas:
      - 0.965
      - 0.99
    rho: 0.04
    weight_decay: 0.1
    update_period: 10     # how often to re-estimate the Hessian
```

## Learning rate schedulers

Optimizers are paired with learning rate schedulers.  The most commonly used are:

```yaml
trainer:
  learning_rate_scheduler:
    type: cosine          # cosine annealing — good default for fine-tuning
    # type: reduce_on_plateau  # reduce LR when val loss stagnates
    # type: linear_warmup   # warmup then constant — good for transformers
    warmup_fraction: 0.1  # fraction of training steps used for warm-up
```

## Hyperopt search

Use Ludwig's hyperopt to find the best optimizer and learning rate automatically:

```yaml
hyperopt:
  search_alg: hyperband
  goal: minimize
  metric: validation_loss
  parameters:
    trainer.optimizer.type:
      type: category
      values: [adam, adamw, lion]
    trainer.optimizer.lr:
      type: float
      low: 1.0e-5
      high: 1.0e-2
      scale: log
    trainer.optimizer.weight_decay:
      type: float
      low: 0.0
      high: 0.1
```

## Practical guidance

- **Start with AdamW** for fine-tuning tasks and Adam for training from scratch.
- **Switch to Lion** if GPU memory is tight and you cannot reduce batch size further.
- **Try Sophia** for transformer encoders in NLP tasks when training time matters.
- **Learning rate is the most important hyperparameter** — always tune it first regardless of optimizer.
- **Warm-up** (5–10 % of total steps) is important for all adaptive optimizers when training transformers.
