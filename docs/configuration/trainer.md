{% from './macros/includes.md' import render_fields %}

# Overview

The `trainer` section of the configuration lets you specify parameters that
configure the training process, like the number of epochs or the learning rate.
By default, the ECD trainer is used.

=== "ECD"

    ```yaml
    {% set ecd_defaults = render_trainer_ecd_defaults_yaml() %}
    trainer:
        {% for line in ecd_defaults.split("\n") %}
        {{- line }}
        {% endfor %}
    ```

=== "GBM"

    ```yaml
    {% set gbm_defaults = render_trainer_gbm_defaults_yaml() %}
    trainer:
        {% for line in gbm_defaults.split("\n") %}
        {{- line }}
        {% endfor %}
    ```

## Trainer parameters

=== "ECD"

    {% set ecd_fields = trainer_ecd_params() %}
    {{ render_fields(ecd_fields) }}

=== "GBM"

    See the [LightGBM documentation](https://lightgbm.readthedocs.io/en/latest/Parameters.html) for more details about the available parameters.

    {% set gbm_fields = trainer_gbm_params() %}
    {{ render_fields(gbm_fields) }}

## Optimizer parameters

=== "ECD"

    The available optimizers wrap the ones available in PyTorch.
    For details about the parameters that can be used to configure different optimizers, please refer to the [PyTorch documentation](https://pytorch.org/docs/stable/optim.html).

    The `learning_rate` parameter used by the optimizer comes from the `trainer` section.
    Other optimizer specific parameters, shown with their Ludwig default settings, follow:

    {% set opt_classes = optimizers() %}
    {% for opt in opt_classes %}
    ### {{ opt.type }}

    ```yaml
    optimizer:
        {% for line in schema_class_to_yaml(opt).split("\n") %}
        {{- line }}
        {% endfor %}
    ```

    {{ render_fields(schema_class_to_fields(opt, exclude=["type"])) }}
    {% endfor %}

    !!! note
        Gradient clipping is also configurable, through optimizers, with the following parameters:

        ```yaml
        clip_global_norm: 0.5
        clipnorm: null
        clip_value: null
        ```

=== "GBM"

    No optimizer parameters are available for the LightGBM trainer.

# Training length

The length of the training process is configured by:

=== "ECD"
    - `epochs` (default: 100): One epoch is one pass through the entire dataset. By
        default, `epochs` is 100 which means that the training process will run for
        a maximum of 100 epochs before terminating.
    - `train_steps` (default: `None`): The maximum number of steps to train for,
        using one mini-batch per step. By default this is unset, and `epochs` will
        be used to determine training length.

=== "GBM"
    - `num_boost_round` (default: 100): The number of boosting iterations. By default,
        `num_boost_round` is 100 which means that the training process will run for
        a maximum of 100 boosting iterations before terminating.

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

=== "ECD"
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

=== "ECD"

Users training on GPUs can often increase training throughput by increasing
the `batch_size` so that more examples are computed every training step. Set
`batch_size` to `auto` to use the largest batch size that can fit in memory.

## Use mixed precision

=== "ECD"

`use_mixed_precision=true`

Speeds up training by using float16 parameters where it makes sense. Mixed precision training on GPU can dramatically
speedup training, with some risks to model convergence. In practice, it works particularly well when fine-tuning
a pretrained model like a HuggingFace transformer. See blog [here](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/) for more details.
