The `trainer` section of the configuration lets you specify parameters that
configure the training process, like the number of epochs or the learning rate.

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
        learning_rate: 0.001
        reduce_learning_rate_on_plateau: 0
        reduce_learning_rate_on_plateau_patience: 5
        reduce_learning_rate_on_plateau_rate: 0.5
        increase_batch_size_on_plateau: 0
        increase_batch_size_on_plateau_patience: 5
        increase_batch_size_on_plateau_rate: 2
        increase_batch_size_on_plateau_max: 512
        decay: false
        decay_steps: 10000
        decay_rate: 0.96
        staircase: false
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

    - `epochs` (default `100`): number of epochs the training process will run for.
    - `train_steps` (default `None`): Maximum number of training steps the training process will run for. If unset, then `epochs` is used to determine training length.
    - `early_stop` (default `5`): if theres a validation set, number of epochs of patience without an improvement on the validation measure before the training is stopped.
    - `batch_size` (default `128`): size of the batch used for training the model.
    - `eval_batch_size` (default `null`): size of the batch used for evaluating the model. If it is `0`, the same value of `batch_size` is used. This is useful to speedup evaluation with a much bigger batch size than training, if enough memory is available.
    - `evaluate_training_set`: Whether to include the entire training set during evaluation (default: True).
    - `checkpoints_per_epoch`: Number of checkpoints per epoch. For example, 2 -> checkpoints are written every half of an epoch. Note that it is invalid to specify both non-zero `steps_per_checkpoint` and non-zero `checkpoints_per_epoch` (default: 0).
    - `steps_per_checkpoint`: How often the model is checkpointed. Also dictates maximum evaluation frequency. If 0 the model is checkpointed after every epoch. (default: 0).
    - `regularization_lambda` (default `0`): the lambda parameter used for adding regularization loss to the overall loss.
    - `regularization_type` (default `l2`): the type of regularization.
    - `learning_rate` (default `0.001`): the learning rate to use.
    - `reduce_learning_rate_on_plateau` (default `0`): if theres a validation set, how many times to reduce the learning rate when a plateau of validation measure is reached.
    - `reduce_learning_rate_on_plateau_patience` (default `5`): if theres a validation set, number of epochs of patience without an improvement on the validation measure before reducing the learning rate.
    - `reduce_learning_rate_on_plateau_rate` (default `0.5`): if theres a validation set, the reduction rate of the learning rate.
    - `increase_batch_size_on_plateau` (default `0`): if theres a validation set, how many times to increase the batch size when a plateau of validation measure is reached.
    - `increase_batch_size_on_plateau_patience` (default `5`): if theres a validation set, number of epochs of patience without an improvement on the validation measure before increasing the learning rate.
    - `increase_batch_size_on_plateau_rate` (default `2`): if theres a validation set, the increase rate of the batch size.
    - `increase_batch_size_on_plateau_max` (default `512`): if theres a validation set, the maximum value of batch size.
    - `decay` (default `false`): if to use exponential decay of the learning rate or not.
    - `decay_rate` (default `0.96`): the rate of the exponential learning rate decay.
    - `decay_steps` (default `10000`): the number of steps of the exponential learning rate decay.
    - `staircase` (default `false`): decays the learning rate at discrete intervals.
    - `validation_field` (default `combined`): when there is more than one output feature, which one to use for computing if there was an improvement on validation. The measure to use to determine if there was an improvement can be set with the `validation_measure` parameter. Different data types have different metrics, refer to the datatype-specific section for more details. `combined` indicates the use the combination of all features. For instance the combination of `combined` and `loss` as measure uses a decrease in the combined loss of all output features to check for improvement on validation, while `combined` and `accuracy` considers on how many examples the predictions for all output features were correct (but consider that for some features, for instance `numeric` there is no accuracy measure, so you should use `accuracy` only if all your output features have an accuracy measure).
    - `validation_metric` (default `loss`): the metric to use to determine if there was an improvement. The metric is considered for the output feature specified in `validation_field`. Different data types have different available metrics, refer to the datatype-specific section for more details.
    - `bucketing_field` (default `null`): when not `null`, when creating batches, instead of shuffling randomly, the length along the last dimension of the matrix of the specified input feature is used for bucketing examples and then randomly shuffled examples from the same bin are sampled. Padding is trimmed to the longest example in the batch. The specified feature should be either a `sequence` or `text` feature and the encoder encoding it has to be `rnn`. When used, bucketing improves speed of `rnn` encoding up to 1.5x, depending on the length distribution of the inputs.
    - `learning_rate_warmup_epochs` (default `1`): Its the number or training epochs where learning rate warmup will be used. It is calculated as described in [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677). In the paper the authors suggest `6` epochs of warmup, that parameter is suggested for large datasets and big batches.
    - `optimizer` (default `{type: adam, beta1: 0.9, beta2: 0.999, epsilon: 1e-08}`): which optimizer to use with the relative parameters. The available optimizers are: `sgd` (or `stochastic_gradient_descent`, `gd`, `gradient_descent`, they are all the same), `adam`, `adadelta`, `adagrad`, `adamax`, `ftrl`, `nadam`, `rmsprop`. Check [PyTorch optimizer documentation](https://pytorch.org/docs/stable/optim.html) for a full list of parameters for each optimizer. The optimizer definition can also specify gradient clipping using `clipglobalnorm`, `clipnorm`, and `clipvalue`.

=== "GBM"

    See the [LightGBM documentation](https://lightgbm.readthedocs.io/en/latest/Parameters.html) for more details about the available parameters.

    - `type` (default `lightgbm_trainer`): Trainer to use for training the model. Must be one of ['lightgbm_trainer'] - corresponds to name in `ludwig.trainers.registry.(ray_)trainers_registry`.
    - `boosting_type` (default `gbdt`): Type of boosting algorithm to use.
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

    The `learning_rate` parameter the optimizer will use come from the `trainer` section.
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