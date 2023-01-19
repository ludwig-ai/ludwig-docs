# Overview

The `trainer` section of the configuration lets you specify parameters that
configure the training process, like the number of epochs or the learning rate.
By default, the ECD trainer is used.

=== "ECD"

    ```yaml
    trainer:
        type: trainer
        epochs: 100
        train_steps: None
        early_stop: 5
        batch_size: 128
        eval_batch_size: null
        evaluate_training_set: True
        checkpoints_per_epoch: 0
        steps_per_checkpoint: 0
        regularization_lambda: 0
        regularization_type: l2
        increase_batch_size_on_plateau: 0
        increase_batch_size_on_plateau_patience: 5
        increase_batch_size_on_plateau_rate: 2
        increase_batch_size_on_plateau_max: 512
        learning_rate: 0.001
        learning_rate_scheduler:
            decay: true
            decay_steps: 10000
            decay_rate: 0.96
            staircase: false
            reduce_on_plateau: 0
            reduce_on_plateau_patience: 5
            reduce_on_plateau_rate: 0.5
        validation_field: combined
        validation_metric: loss
        bucketing_field: null
        learning_rate_warmup_epochs: 1
        optimizer:
            type: adam
            beta1: 0.9
            beta2: 0.999
            epsilon: 1e-08
            clip_global_norm: 0.5
            clipnorm: null
            clip_value: null
    ```

=== "GBM"

    ```yaml
    trainer:
        type: lightgbm_trainer
        boosting_type: gbdt
        num_boost_round: 100
        learning_rate: 0.001
        max_cat_to_onehot: 4
        max_delta_step: 0.0
        lambda_l1: 0.0
        linear_lambda: 0.0
        cat_l2: 10.0
        neg_bagging_fraction: 1.0
        skip_drop: 0.5
        tree_learner: serial
        extra_trees: False
        lambda_l2: 0.0
        min_data_per_group: 100
        min_gain_to_split: 0.0
        validation_metric: loss
        max_cat_threshold: 32
        max_bin: 255
        early_stop: 5
        cegb_penalty_split: 0.0
        cegb_tradeoff: 1.0
        other_rate: 0.1
        path_smooth: 0.0
        evaluate_training_set: True
        num_leaves: 31
        cat_smooth: 10.0
        extra_seed: 6
        bagging_seed: 3
        min_sum_hessian_in_leaf: 0.001
        min_data_in_leaf: 20
        top_rate: 0.2
        feature_fraction_seed: 2
        drop_rate: 0.1
        xgboost_dart_mode: False
        drop_seed: 4
        max_depth: -1
        feature_fraction_bynode: 1.0
        bagging_freq: 0
        pos_bagging_fraction: 1.0
        feature_fraction: 1.0
        eval_batch_size: 128
        bagging_fraction: 1.0
        uniform_drop: False
        validation_field: combined
        max_drop: 50
        verbose: 0
    ```

## Trainer parameters

=== "ECD"

    - `type` (default `trainer`): Trainer to use for training the model. Must be one of ['trainer', 'ray_legacy_trainer'] - corresponds to name in `ludwig.trainers.registry.(ray_)trainers_registry` (default: 'trainer')
    - `epochs` (default `100`): number of epochs the training process will run for.
    - `train_steps` (default `None`): Maximum number of training steps the training process will run for. If unset, then `epochs` is used to determine training length.
    - `early_stop` (default `5`): Number of consecutive rounds of evaluation without any improvement on the `validation_metric` that triggers training to stop. Can be set to -1, which disables early stopping entirely.
    - `batch_size` (default `128`): size of the batch used for training the model.
    - `eval_batch_size` (default `null`): size of the batch used for evaluating the model. If it is `0`, the same value of `batch_size` is used. This is useful to speedup evaluation with a much bigger batch size than training, if enough memory is available.
    - `evaluate_training_set`: Whether to include the entire training set during evaluation (default: True).
    - `checkpoints_per_epoch`: Number of checkpoints per epoch. For example, 2 -> checkpoints are written every half of an epoch. Note that it is invalid to specify both non-zero `steps_per_checkpoint` and non-zero `checkpoints_per_epoch` (default: 0).
    - `steps_per_checkpoint`: How often the model is checkpointed. Also dictates maximum evaluation frequency. If 0 the model is checkpointed after every epoch. (default: 0).
    - `regularization_lambda` (default `0`): the lambda parameter used for adding regularization loss to the overall loss.
    - `regularization_type` (default `l2`): the type of regularization.
    - `learning_rate` (default `0.001`): the learning rate to use.
    - `learning_rate_scheduler` section:
        - `reduce_on_plateau` (default `0`): if theres a validation set, how many times to reduce the learning rate when a plateau of validation measure is reached.
        - `reduce_on_plateau_patience` (default `10`): if theres a validation set, number of epochs of patience without an improvement on the validation measure before reducing the learning rate.
        - `reduce_on_plateau_rate` (default `0.1`): if theres a validation set, the reduction rate of the learning rate.
        - `decay` (default `false`): one of `null`, `linear`, `exponential`. Whether to use one of the aforementioned decay strategies. Specifying `null` deactivates learning rate decay.
        - `decay_rate` (default `0.96`): the rate of the exponential learning rate decay.
        - `decay_steps` (default `10000`): the number of steps of the exponential learning rate decay.
        - `staircase` (default `false`): decays the learning rate at discrete intervals.
        - `warmup_evaluations` (default `0`): Is the number or training epochs where learning rate warmup will be used. It is calculated as described in [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677). In the paper the authors suggest `6` epochs of warmup, that parameter is suggested for large datasets and big batches.
        - `warmup_fraction` (default `0`): Fraction of total training steps to warmup the learning rate for.
        - `reduce_eval_metric` (default: `loss`): Metric used to trigger when we reduce the learning rate when `reduce_on_plateau` > 0.
        - `reduce_eval_split` (default: `training`): Which dataset split to compute `reduce_eval_metric` on for reducing the learning rate when `reduce_on_plateau` > 0.
    - `increase_batch_size_on_plateau` (default `0`): if theres a validation set, how many times to increase the batch size when a plateau of validation measure is reached.
    - `increase_batch_size_on_plateau_patience` (default `5`): if theres a validation set, number of epochs of patience without an improvement on the validation measure before increasing the learning rate.
    - `increase_batch_size_on_plateau_rate` (default `2`): if theres a validation set, the increase rate of the batch size.
    - `increase_batch_size_on_plateau_max` (default `512`): if theres a validation set, the maximum value of batch size.
    - `validation_field` (default `combined`): when there is more than one output feature, which one to use for computing if there was an improvement on validation. The measure to use to determine if there was an improvement can be set with the `validation_measure` parameter. Different data types have different metrics, refer to the datatype-specific section for more details. `combined` indicates the use the combination of all features. For instance the combination of `combined` and `loss` as measure uses a decrease in the combined loss of all output features to check for improvement on validation, while `combined` and `accuracy` considers on how many examples the predictions for all output features were correct (but consider that for some features, for instance `numeric` there is no accuracy measure, so you should use `accuracy` only if all your output features have an accuracy measure).
    - `validation_metric` (default `loss`): the metric to use to determine if there was an improvement. The metric is considered for the output feature specified in `validation_field`. Different data types have different available metrics, refer to the datatype-specific section for more details.
    - `bucketing_field` (default `null`): when not `null`, when creating batches, instead of shuffling randomly, the length along the last dimension of the matrix of the specified input feature is used for bucketing examples and then randomly shuffled examples from the same bin are sampled. Padding is trimmed to the longest example in the batch. The specified feature should be either a `sequence` or `text` feature and the encoder encoding it has to be `rnn`. When used, bucketing improves speed of `rnn` encoding up to 1.5x, depending on the length distribution of the inputs.
    - `optimizer` (default `{type: adam, beta1: 0.9, beta2: 0.999, epsilon: 1e-08}`): which optimizer to use with the relative parameters. The available optimizers are: `sgd` (or `stochastic_gradient_descent`, `gd`, `gradient_descent`, they are all the same), `adam`, `adadelta`, `adagrad`, `adamax`, `ftrl`, `nadam`, `rmsprop`. Check [PyTorch optimizer documentation](https://pytorch.org/docs/stable/optim.html) for a full list of parameters for each optimizer. The optimizer definition can also specify gradient clipping using `clipglobalnorm`, `clipnorm`, and `clipvalue`.

=== "GBM"

    See the [LightGBM documentation](https://lightgbm.readthedocs.io/en/latest/Parameters.html) for more details about the available parameters.

    - `type` (default `lightgbm_trainer`): Trainer to use for training the model. Must be one of ['lightgbm_trainer'] - corresponds to name in `ludwig.trainers.registry.(ray_)trainers_registry`.
    - `boosting_type` (default `gbdt`): Type of boosting algorithm to use. Options: `gbdt` (traditional Gradient Boosting Decision Tree), `rf` (random forest), `dart`, `goss`.
    - `num_boost_round` (default `100`): Number of boosting rounds to perform.
    - `learning_rate` (default `0.001`): Boosting learning rate.
    - `max_cat_to_onehot` (default `4`): Maximum categorical cardinality required before one-hot encoding.
    - `max_delta_step` (default `0.0`): Used to limit the max output of tree leaves. A negative value means no constraint.
    - `lambda_l1` (default `0.0`): L1 regularization factor.
    - `linear_lambda` (default `0.0`): Linear tree regularization.
    - `cat_l2` (default `10.0`): L2 regularization factor for categorical split.
    - `neg_bagging_fraction` (default `1.0`): Fraction of negative data to use for bagging.
    - `skip_drop` (default `0.5`): Probability of skipping the dropout during one boosting iteration. Used only with boosting_type 'dart'.
    - `tree_learner` (default `serial`): Type of tree learner to use.
    - `extra_trees` (default `False`): Whether to use extremely randomized trees.
    - `lambda_l2` (default `0.0`): L2 regularization factor.
    - `min_data_per_group` (default `100`): Minimum number of data points per categorical group.
    - `min_gain_to_split` (default `0.0`): Minimum gain to split a leaf.
    - `validation_metric` (default `loss`): Metric used on `validation_field`, set by default to accuracy.
    - `max_cat_threshold` (default `32`): Number of split points considered for categorical features.
    - `max_bin` (default `255`): Maximum number of bins to use for discretizing features.
    - `early_stop` (default `5`): Number of consecutive rounds of evaluation without any improvement on the `validation_metric` that triggers training to stop. Can be set to -1, which disables early stopping entirely.
    - `cegb_penalty_split` (default `0.0`): Cost-effective gradient boosting penalty for splitting a node.
    - `cegb_tradeoff` (default `1.0`): Cost-effective gradient boosting multiplier for all penalties.
    - `other_rate` (default `0.1`): The retain ratio of small gradient data. Used only with boosting_type 'goss'.
    - `path_smooth` (default `0.0`): Smoothing factor applied to tree nodes.
    - `evaluate_training_set` (default `True`): Whether to include the entire training set during evaluation.
    - `num_leaves` (default `31`): Number of leaves to use in the tree.
    - `cat_smooth` (default `10.0`): Smoothing factor for categorical split.
    - `extra_seed` (default `6`): Random seed for extremely randomized trees.
    - `bagging_seed` (default `3`): Random seed for bagging.
    - `min_sum_hessian_in_leaf` (default `0.001`): Minimum sum of hessians in a leaf.
    - `min_data_in_leaf` (default `20`): Minimum number of data points in a leaf.
    - `top_rate` (default `0.2`): The retain ratio of large gradient data. Used only with boosting_type 'goss'.
    - `feature_fraction_seed` (default `2`): Random seed for feature fraction.
    - `drop_rate` (default `0.1`): Dropout rate. Used only with boosting_type 'dart'.
    - `xgboost_dart_mode` (default `False`): Whether to use xgboost dart mode. Used only with boosting_type 'dart'.
    - `drop_seed` (default `4`): Random seed to choose dropping models. Used only with boosting_type 'dart'.
    - `max_depth` (default `-1`): Maximum depth of a tree. A negative value means no limit.
    - `feature_fraction_bynode` (default `1.0`): Fraction of features to use for each tree node.
    - `bagging_freq` (default `0`): Frequency of bagging.
    - `pos_bagging_fraction` (default `1.0`): Fraction of positive data to use for bagging.
    - `feature_fraction` (default `1.0`): Fraction of features to use.
    - `eval_batch_size` (default `128`): Size of batch to pass to the model for evaluation.
    - `bagging_fraction` (default `1.0`): Fraction of data to use for bagging.
    - `uniform_drop` (default `False`): Whether to use uniform dropout. Used only with boosting_type 'dart'.
    - `validation_field` (default `combined`): First output feature, by default it is set as the same field of the first output feature.
    - `max_drop` (default `50`): Maximum number of dropped trees during one boosting iteration. Used only with boosting_type 'dart'. A negative value means no limit.
    - `verbose` (default `0`): Verbosity level for GBM trainer.

## Optimizer parameters

=== "ECD"

    The available optimizers wrap the ones available in PyTorch.
    For details about the parameters that can be used to configure different optimizers, please refer to the [PyTorch documentation](https://pytorch.org/docs/stable/optim.html).

    The `learning_rate` parameter used by the optimizer comes from the `trainer` section.
    Other optimizer specific parameters, shown with their Ludwig default settings, follow:

    - `sgd` (or `stochastic_gradient_descent`, `gd`, `gradient_descent`)

    ```yaml
    momentum: 0.0,
    nesterov: false
    ```

    - `adam`

    ```yaml
    beta_1: 0.9,
    beta_2: 0.999,
    epsilon: 1e-08
    ```

    - `adadelta`

    ```yaml
    rho: 0.95,
    epsilon: 1e-08
    ```

    - `adagrad`

    ```yaml
    initial_accumulator_value: 0.1,
    epsilon: 1e-07
    ```

    - `adamax`

    ```yaml
    beta_1: 0.9,
    beta_2: 0.999,
    epsilon: 1e-07
    ```

    - `ftrl`

    ```yaml
    learning_rate_power: -0.5,
    initial_accumulator_value: 0.1,
    l1_regularization_strength: 0.0,
    l2_regularization_strength: 0.0,
    ```

    - `nadam`,

    ```yaml
    beta_1: 0.9,
    beta_2: 0.999,
    epsilon: 1e-07
    ```

    - `rmsprop`

    ```yaml
    decay: 0.9,
    momentum: 0.0,
    epsilon: 1e-10,
    centered: false
    ```

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

# Increasing throughput

## Skip evaluation on the training set

Consider setting `evaluate_training_set=False`, which skips evaluation on the
training set.

!!! note

    Sometimes it can be useful to monitor evaluation metrics on the training
    set, as a secondary validation set. However, running evaluation on the full
    training set, when your training set is large, can be a huge computational
    cost. Turning off training set evaluation will lead to significant gains in
    training throughput and efficiency.

## Increase batch size

=== "ECD"

Users training on GPUs can often increase training throughput by increasing
the `batch_size` so that more examples are computed every training step. Set
`batch_size` to `auto` to use the largest batch size that can fit in memory.
