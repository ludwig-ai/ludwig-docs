{% from './macros/includes.md' import render_fields, render_yaml %}
{% set opt_details = "See [Optimizer parameters](#optimizer-parameters) for details." %}
{% set details = {"optimizer": opt_details} %}

# Overview

The `trainer` section of the configuration lets you specify parameters that
configure the training process, like the number of epochs or the learning rate.
By default, the ECD trainer is used.

=== "ECD"

    {% set ecd_trainer = get_trainer_schema("ecd") %}
    {{ render_yaml(ecd_trainer, parent="trainer") | indent }}

=== "LLM"

    {% set llm_trainer = get_trainer_schema("llm") %}
    {{ render_yaml(llm_trainer, parent="trainer") | indent }}

## Trainer parameters

=== "ECD"

    {{ render_fields(schema_class_to_fields(ecd_trainer), details=details) | indent }}

=== "LLM"

    {{ render_fields(schema_class_to_fields(llm_trainer), details=details) | indent }}

## Optimizer parameters

=== "ECD / LLM"

    The available optimizers wrap the ones available in PyTorch.
    For details about the parameters that can be used to configure different optimizers, please refer to the [PyTorch documentation](https://pytorch.org/docs/stable/optim.html).

    The `learning_rate` parameter used by the optimizer comes from the `trainer` section.
    Other optimizer specific parameters, shown with their Ludwig default settings, follow:

    {% set opt_classes = get_optimizer_schemas() %}
    {% for opt in opt_classes %}
    ### {{ opt.type }}

    {{ render_yaml(opt, parent="optimizer") | indent }}

    {{ render_fields(schema_class_to_fields(opt, exclude=["type"])) | indent }}
    {% endfor %}

    !!! note
        Gradient clipping is also configurable, through optimizers, with the following parameters:

        ```yaml
        clip_global_norm: 0.5
        clipnorm: null
        clip_value: null
        ```

## Optimizer guidance

Ludwig 0.14 adds five optimizers on top of the existing PyTorch family. Quick picks:

- **`radam`** — Rectified Adam (Liu et al., ICLR 2020). Drop-in replacement for `adam` that
  removes the need for manual warmup by adaptively rectifying the variance of the adaptive
  learning rate in the early steps.
- **`adafactor`** — Adafactor (Shazeer & Stern, ICML 2018). Factorizes the second-moment matrix
  to cut optimizer memory roughly in half, which makes it a common choice for fine-tuning
  large transformers. When `relative_step: true` (the default) Adafactor manages its own
  schedule — **do not combine it with a `learning_rate_scheduler`** and leave `learning_rate`
  unset on the trainer.
- **`schedule_free_adamw`** — Schedule-Free AdamW (Defazio et al., 2024). Matches cosine-decay
  AdamW without needing an LR scheduler at all. The optimizer maintains two iterate states —
  Ludwig handles the required `optimizer.train()` / `optimizer.eval()` calls automatically at
  the train/eval boundaries.
- **`muon`** — Muon (Jordan et al., 2024). Uses momentum plus Newton–Schulz orthogonalization
  to produce stable, well-conditioned updates. Competitive with AdamW for pretraining and
  typically uses a **much higher base learning rate than Adam** (default `0.02`).
- **`soap`** — SOAP (Vyas et al., 2024). Shampoo-style preconditioner stacked on AdamW. Strong
  empirical results on large-scale training; requires installing the optional `soap-pytorch`
  package. Registered only when the dependency is available.

!!! note
    The legacy `ftrl` optimizer was removed in 0.14. Configs that set `optimizer.type: ftrl`
    will fail validation — use `adagrad` or `sgd` with momentum as replacements.

## Learning rate schedulers

The `learning_rate_scheduler` section of the trainer controls how the learning rate evolves
during training. Ludwig 0.14 adds four new schedule types on top of `linear`, `exponential`,
and `cosine`:

| `decay` | Best for | Key parameters |
|---------|----------|----------------|
| `one_cycle` | Fast supervised training, "superconvergence" | `max_lr`, `pct_start`, `div_factor`, `final_div_factor` |
| `inverse_sqrt` | Transformer pretraining ("Noam" schedule) | `inverse_sqrt_warmup_steps` |
| `polynomial` | Fine-tuning with a smooth ramp-down | `polynomial_power`, `polynomial_end_lr` |
| `wsd` | Long continued pretraining with annealing | `wsd_warmup_fraction`, `wsd_stable_fraction`, `wsd_decay_fraction` |

### OneCycleLR

Implements Smith's 1-cycle policy: warm up from `initial_lr = max_lr / div_factor` to
`max_lr` over `pct_start` of the total steps, then anneal down to
`min_lr = initial_lr / final_div_factor`.

```yaml
trainer:
  optimizer:
    type: adamw
  learning_rate_scheduler:
    decay: one_cycle
    max_lr: 0.001
    pct_start: 0.3
    div_factor: 25.0
    final_div_factor: 10000.0
```

If `max_lr` is left unset it defaults to the trainer's `learning_rate`.

### Inverse square root (Noam)

After a linear warmup over `inverse_sqrt_warmup_steps`, the learning rate decays as
`1 / sqrt(step)`. This is the schedule used in the original Transformer paper and is a good
default for training language models from scratch.

```yaml
trainer:
  learning_rate: 0.0005
  learning_rate_scheduler:
    decay: inverse_sqrt
    inverse_sqrt_warmup_steps: 4000
```

### Polynomial decay

Smoothly decays from the base learning rate to `polynomial_end_lr` over the full training
run. `polynomial_power: 1.0` is linear decay; `2.0` is quadratic; values `< 1.0` decay
faster early on.

```yaml
trainer:
  learning_rate_scheduler:
    decay: polynomial
    polynomial_power: 1.0
    polynomial_end_lr: 0.0
```

### Warmup-Stable-Decay (WSD)

WSD splits the run into three phases — a short warmup, a long stable phase at the peak
learning rate, and a short cooldown. Because the stable phase dominates, you can stop at
any time and take a clean cooldown snapshot, which is useful for long pretraining runs
where you want to branch continued training off of intermediate checkpoints.

```yaml
trainer:
  learning_rate_scheduler:
    decay: wsd
    wsd_warmup_fraction: 0.1
    wsd_stable_fraction: 0.8
    wsd_decay_fraction: 0.1
```

The three fractions should sum to `1.0`.

### Warmup and plateau parameters

All schedules support the shared warmup and reduce-on-plateau knobs:

- `warmup_fraction` / `warmup_evaluations` — linear warmup of the base learning rate.
- `reduce_on_plateau`, `reduce_on_plateau_patience`, `reduce_on_plateau_rate` — cap the
  number of on-plateau reductions and the reduction factor. Combine with `reduce_eval_metric`
  and `reduce_eval_split` to pick which metric triggers the reduction.

# Training length

The length of the training process is configured by:

=== "ECD / LLM"
    - `epochs` (default: 100): One epoch is one pass through the entire dataset. By
        default, `epochs` is 100 which means that the training process will run for
        a maximum of 100 epochs before terminating.
    - `train_steps` (default: `None`): The maximum number of steps to train for,
        using one mini-batch per step. By default this is unset, and `epochs` will
        be used to determine training length.

!!! tip

    In general, it's a good idea to set up a long training runway, relying on
    early stopping criteria (`early_stop`) to stop training when there
    hasn't been any improvement for a long time.

# Early stopping

Machine learning models, when trained for too long, are often prone to
overfitting. It's generally a good policy to set up some early stopping criteria
as it's not useful to have a model train after it's maximized what it can learn,
as to retain it's ability to generalize to new data.

## How early stopping works in Ludwig

By default, Ludwig sets `trainer.early_stop=5`, which means that if there have
been `5` consecutive rounds of evaluation where there hasn't been any
improvement on the **validation** subset, then training will terminate.

Ludwig runs evaluation once per checkpoint, which by default is once per epoch.
Checkpoint frequency can be configured using `checkpoints_per_epoch` (default:
`1`) or `steps_per_checkpoint` (default: `0`, disabled). See
[this section](#checkpoint-evaluation-frequency) for more details.

## Changing the metric early stopping metrics

The metric that dictates early stopping is
`trainer.validation_field` and `trainer.validation_metric`. By default, early
stopping uses the combined loss on the validation subset.

```yaml
trainer:
    validation_field: combined
    validation_metric: loss
```

However, this can be configured to use other metrics. For example, if we had an
output feature called `recommended`, then we can configure early stopping on the
output feature accuracy like so:

```yaml
trainer:
    validation_field: recommended
    validation_metric: accuracy
```

## Disabling early stopping

`trainer.early_stop` can be set to `-1`, which disables early stopping entirely.

# Checkpoint-evaluation frequency

=== "ECD / LLM"
Evaluation is run every time the model is checkpointed.

By default, checkpoint-evaluation will occur once every epoch.

The frequency of checkpoint-evaluation can be configured using:

* `steps_per_checkpoint` (default: 0): every `n` training steps
* `checkpoints_per_epoch` (default: 0): `n` times per epoch

!!! note

    It is invalid to specify both non-zero `steps_per_checkpoint` and non-zero
    `checkpoints_per_epoch`.

!!! tip

    Running evaluation once per epoch is an appropriate fit for small datasets 
    that fit in memory and train quickly. However, this can be a poor fit for
    unstructured datasets, which tend to be much larger, and train more slowly
    due to larger models.

    Running evaluation too frequently can be wasteful while running evaluation
    not frequently enough can be uninformative. In large-scale training runs,
    it's common for evaluation to be configured to run on a sub-epoch time
    scale, or every few thousand steps.
    
    We recommend configuring evaluation such that new evaluation results are
    available at least several times an hour. In general, it is not necessary
    for models to train over the entirety of a dataset, nor evaluate over the
    entirety of a test set, to produce useful monitoring metrics and signals to
    indicate model performance.

# Increasing throughput on GPUs

## Increase batch size

=== "ECD / LLM"

```yaml
trainer:
    batch_size: auto
```

Users training on GPUs can often increase training throughput by increasing
the `batch_size` so that more examples are computed every training step. Set
`batch_size` to `auto` to use the largest batch size that can fit in memory.

## Use mixed precision

=== "ECD / LLM"

```yaml
trainer:
    use_mixed_precision: true
```

Speeds up training by using float16 parameters where it makes sense. Mixed precision training on GPU can dramatically
speedup training, with some risks to model convergence. In practice, it works particularly well when fine-tuning
a pretrained model like a HuggingFace transformer. See blog [here](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/) for more details.

## Multi-Task Loss Balancing

When training models with multiple output features, the default behavior is to sum each feature's loss
with a static weight. The `loss_balancing` parameter enables adaptive strategies that automatically
balance task losses during training.

=== "ECD"

    ```yaml
    trainer:
        loss_balancing: uncertainty  # none, log_transform, uncertainty, famo, gradnorm
    ```

Available strategies:

- `none` (default): Static weighted sum.
- `log_transform`: Applies `log(1 + loss)` to compress loss scales before weighting. Simple and always beneficial when task losses have very different magnitudes.
- `uncertainty`: Learns a log-variance parameter per task (Kendall et al., CVPR 2018). No hyperparameters needed.
- `famo`: Fast Adaptive Multitask Optimization (Liu et al., NeurIPS 2023). O(1) overhead, competitive with gradient-based methods.
- `gradnorm`: Gradient normalization across tasks (Chen et al., ICML 2018). Dynamically adjusts weights to normalize gradient magnitudes.

## Modality Dropout

During training, randomly replaces input feature encoder outputs with learnable "missing modality" embeddings.
This improves robustness when some inputs may be missing at inference time.

=== "ECD"

    ```yaml
    trainer:
        modality_dropout: 0.1  # probability per feature, 0.0 to disable
    ```

## Model Soup

Averages the weights of the top-K checkpoints saved during training for better generalization at
zero inference cost (Wortsman et al., ICML 2022).

=== "ECD"

    ```yaml
    trainer:
        model_soup: uniform  # uniform, greedy, or null to disable
        model_soup_top_k: 5
    ```

## Quality Presets

Quality presets auto-configure the combiner, trainer, and other settings for different quality/speed tradeoffs.
User-specified config values always take precedence over preset defaults.

```yaml
preset: best_quality  # medium_quality, high_quality, or best_quality
input_features:
  - name: feature1
    type: number
output_features:
  - name: target
    type: category
```

Available presets:

- `medium_quality`: Concat combiner, 50 epochs, batch_size 256. Fast training.
- `high_quality`: Transformer combiner, uncertainty loss balancing, 100 epochs.
- `best_quality`: FT-Transformer combiner, uncertainty loss balancing, model soup, 200 epochs.

## Preference-Based LLM Training

Ludwig supports several preference optimization trainers for LLMs that align model outputs with human
preferences. These trainers are available when using `model_type: llm`.

### DPO Trainer

Direct Preference Optimization (Rafailov et al., NeurIPS 2023) trains a model to prefer chosen completions
over rejected ones without a separate reward model. DPO reformulates the RLHF objective as a simple
classification loss on preference pairs.

Requires data with a prompt column, the output column (containing chosen completions), and a `rejected`
column (containing rejected completions).

```yaml
model_type: llm
trainer:
    type: dpo
    dpo_beta: 0.1
    dpo_loss_type: sigmoid  # sigmoid or ipo
    dpo_label_smoothing: 0.0
    rejected_column: rejected
```

Parameters:

- **`dpo_beta`** (default `0.1`): Temperature parameter controlling how much the policy can deviate from
  the reference model. Lower values keep the policy closer to the reference. Typical range: 0.05 to 0.5.
- **`dpo_loss_type`** (default `sigmoid`): DPO loss variant. `sigmoid` is the standard DPO loss.
  `ipo` is Identity Preference Optimization which uses a squared loss.
- **`dpo_label_smoothing`** (default `0.0`): Label smoothing for DPO preference targets. 0 means no smoothing.
- **`rejected_column`** (default `rejected`): Name of the column containing rejected completions.

### KTO Trainer

Kahneman-Tversky Optimization (Ethayarajh et al., 2024) is a preference optimization method that works
with binary feedback (good/bad) rather than requiring paired preferences. This makes it practical when
paired preference data is unavailable.

```yaml
model_type: llm
trainer:
    type: kto
    kto_beta: 0.1
    rejected_column: rejected
```

### ORPO Trainer

Odds Ratio Preference Optimization (Hong et al., 2024) combines supervised fine-tuning with preference
optimization in a single training step, eliminating the need for a reference model.

```yaml
model_type: llm
trainer:
    type: orpo
    orpo_beta: 0.1
    rejected_column: rejected
```

### GRPO Trainer

Group Relative Policy Optimization (Shao et al., 2024) generates multiple completions per prompt and
uses group-relative rewards to optimize the policy. This is the method used to train DeepSeek-R1.

```yaml
model_type: llm
trainer:
    type: grpo
    grpo_beta: 0.04
    grpo_epsilon: 0.2
    grpo_num_generations: 4
```

Parameters:

- **`grpo_beta`** (default `0.04`): KL penalty coefficient.
- **`grpo_epsilon`** (default `0.2`): PPO clipping parameter.
- **`grpo_num_generations`** (default `4`): Number of completions to generate per prompt.
