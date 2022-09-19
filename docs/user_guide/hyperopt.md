Ludwig supports hyperparameter optimization using [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) or a local
executor.

The hyperparameter optimization strategy is specified as part of the Ludwig configuration and run using
the `ludwig hyperopt` command. Every parameter within the config can be tuned using hyperopt.

# Hyperopt Configuration

Most parameters or nested parameters of a Ludwig configuration may be optimized, including `input_features`,
`output_features`, `combiner`, `preprocessing`, `trainer` and `defaults`.  Supported types are `float`, `int` and `category`.

To enable hyperparameter optimization, add the `hyperopt` dictionary at the top level of your config.yaml. The
`hyperopt` section declares which parameters to optimize, the search strategy, and the optimization goal.

```yaml title="config.yaml"
hyperopt:
  parameters:
    title.num_filters:
      space: choice
      categories: [128, 256, 512]
    training.learning_rate:
      space: loguniform
      lower: 0.0001
      upper: 0.1
    combiner.num_fc_layers:
      space: randint
      lower: 2
      upper: 6
  goal: minimize
  metric: loss
```

## Default Hyperopt Parameters

In addition to defining hyperopt parameters for individual input or output features (like the `title` feature
in the example above), default parameters can be specified for entire feature types (for example, the encoder
to use for all text features in your dataset). These parameters will follow the same convention as the
[defaults section](../configuration/defaults.md) of the Ludwig config. This is particularly helpful in cases where a dataset has
a large number of features and you don't want to define parameters for each feature individually.

Default hyperopt parameters are defined using the following keywords in order separated by the `.` delimiter:

- `defaults`: The defaults keyword used to indicate a feature-level parameter
- `feature_type`: Any input or output feature type that Ludwig supports. This can be one of text, numeric, category,
etc. See the full list of support feature types [here](../configuration/features/supported_data_types.md)
- `subsection`: One of `preprocessing`, `encoder`, `decoder` or `loss`, the 4 sections that can be modified via the
Ludwig defaults section
- `parameter`: A valid parameter belonging to the `subsection`. For e.g., `most_common` is a valid parameter for the
`preprocessing` sub-section for `text` feature types

For each hyperopt trial, a value will be sampled from the parameter space and applied to either input features
(`preprocessing` or `encoder` related parameters) or output features (`decoder` or `loss` related parameters) of
that feature type. Additionally, parameters defined for individual features (like `title.preprocessing.most_common`)
will take precedence over default parameters (like `defaults.text.preprocessing.most_common`) if they share the
same feature type and parameter and both parameters are defined in the Ludwig hyperopt config.

```yaml title="config.yaml"
...
hyperopt:
  parameters:
    title.num_filters:
      space: choice
      categories: [128, 256, 512]
    defaults.text.preprocessing.most_common:
      space: choice
      categories: [100, 500, 1000]
  goal: minimize
  metric: loss
...
```

In this example, `defaults.text.preprocessing.most_common` is a default parameter. Here:

- `defaults` helps denote a default hyperopt parameter
- `text` refers to the group of text input features since it is a `preprocessing` releated parameter
- `preprocessing` refers to the text preprocessing sub-section within Ludwig's default section. This means that
this parameter will modify preprocessing for all text input features
- `most_common` is the parameter within `preprocessing` that we want to modify for all text input features

## Nested Ludwig Config Parameters

Ludwig also extends the range of hyperopt parameters to support parameter choices that consist of partial or
complete blocks of nested Ludwig config sections. This allows users to search over a set of Ludwig configs, as
opposed to needing to specify config params individually and search over all combinations.

To provide a parameter that represents a subsection of the Ludwig config, the `.` key name can be used.

For example, one can define a hyperopt search space like the one below and sample partial Ludwig configs:

```yaml
hyperopt:
    parameters:
        .:  space: choice
            categories: 
                -   combiner:
                        type: tabnet
                    trainer:
                        learning_rate: 0.001
                        batch_size: 64
                -   combiner:
                        type: concat
                    trainer:
                        batch_size: 256
        trainer.decay_rate:
            space: loguniform
            lower: 0.001
            upper: 0.1
```

The `.` parameter defines the nested hyperopt parameter with two choices. These will be sampled and
used to update the Ludwig config for each trial based on which of the two choices are sampled.

This config above will create hyperopt samples that look like the following:

```yaml
# Trial 1
  combiner:
    type: tabnet
  trainer: 
  learning_rate: 0.001
  batch_size: 64
  decay_rate: 0.02

# Trial 2
  combiner:
    type: tabnet
  trainer: 
    learning_rate: 0.001
    batch_size: 64
    decay_rate: 0.001

# Trial 3
  combiner:
    type: concat
  trainer: 
    batch_size: 64
    decay_rate: 0.001
```

# Running Hyperparameter Optimization

Use the `ludwig hyperopt` command to run hyperparameter optimization.

```
ludwig hyperopt --dataset reuters-allcats.csv --config hyperopt_config.yaml
```

For a complete reference of hyperparameter search and execution options, see the
[Hyperopt](../configuration/hyperparameter_optimization.md) page of the Ludwig configuration guide.
