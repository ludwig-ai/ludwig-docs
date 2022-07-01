Ludwig supports hyperparameter optimization using [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) or a local
executor.

The hyperparameter optimization strategy is specified as part of the Ludwig configuration and run using
the `ludwig hyperopt` command. Every parameter within the config can be tuned using hyperopt.

# Hyperopt Configuration

Most parameters or nested parameters of a Ludwig configuration may be optimized, including `input_features`,
`output_features`, `combiner`, `preprocessing`, and `trainer`.  Supported types are `float`, `int` and `category`.

To enable hyperparameter optimization, add the `hyperopt` dictionary at the top level of your config.yaml. The
`hyperopt` section declares which parameters to optimize, the search strategy, and the optimization goal.

```yaml
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

## Feature Level Parameters

In addition to defining hyperopt parameters for individual input or output features (like the `title` feature
in the example above), feature level parameters can be specified for entire feature types. This is particularly
helpful in cases where a dataset has a large number of features.

For each hyperopt trial, a value will be sampled from the feature level parameter space and applied to all input and
output features of that feature type as long as they use the default encoder (for input features) or default decoder
(for output features). Additionally, parameters defined for individual features (like `title.num_filters`) will take
precedence over feature level parameters (like `defaults.text.num_filters`) if they share the same type and parameter.

Feature level parameters are defined using the `default` keyword, followed by the `feature_type` and `parameter`.

```yaml title="config.yaml"
hyperopt:
  parameters:
    title.num_filters:
      space: choice
      categories: [128, 256, 512]
    defaults.text.num_filters:
      space: choice
      categories: [128, 256, 512]
    defaults.category.embedding_size:
      space: choice
      categories: [64, 128, 256]
  goal: minimize
  metric: loss
```

In this example, there are two feature level parameters defined:

- `defaults.text.num_filters`: This will apply the sampled `num_filters` value and apply it to all text features
using either the default text encoder (for input features) or text decoder (for output features) for a trial.
- `defaults.category.embedding_size`: This will apply the sampled `embedding_size` value and apply it to all category
features using either the default category encoder (for input features) or category decoder (for output features)
for a trial.

# Running Hyperparameter Optimization

Use the `ludwig hyperopt` command to run hyperparameter optimization.

```
ludwig hyperopt --dataset reuters-allcats.csv --config hyperopt_config.yaml
```

For a complete reference of hyperparameter search and execution options, see the
[Hyperopt](../configuration/hyperparameter_optimization.md) page of the Ludwig configuration guide.
