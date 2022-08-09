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

## Default Feature - Level Parameters

In addition to defining hyperopt parameters for individual input or output features (like the `title` feature
in the example above), feature level parameters can be specified for entire feature types. These parameters will
follow the same convention as the `defaults` section of the Ludwig config. This is particularly helpful in cases
where a dataset has a large number of features and you don't want to define parameters for each feature individually.

Feature level parameters are defined using the following keywords in order separated by the `.` delimiter:

- `defaults`: The defaults keyword used to indicate a feature-level parameter.
- `feature_type`: Any input or output feature type in the current Ludwig config.
- `subsection`: One of `preprocessing`, `encoder`, `decoder` or `loss`.
- `parameter`: A valid parameter belonging to the `subsection`. For e.g., `most_common` is a valid parameter for the
`preprocessing` section for `text` feature type.

For each hyperopt trial, a value will be sampled from the feature level parameter space and applied to either input features
(`preprocessing` or `encoder` related parameters) or output features (`decoder` or `loss` related parameters) of that feature type
as long as they use the default encoder (for input features) or default decoder (for output features). Additionally, parameters
defined for individual features (like `title.num_filters`) will take precedence over feature level parameters
(like `defaults.text.encoder.num_filters`) if they share the same type and parameter.

```yaml title="config.yaml"
hyperopt:
  parameters:
    title.num_filters:
      space: choice
      categories: [128, 256, 512]
    defaults.text.encoder.num_filters:
      space: choice
      categories: [128, 256, 512]
    defaults.category.decoder.reduce_input:
      space: choice
      categories: ['mean', 'sum', 'max']
  goal: minimize
  metric: loss
```

In this example, there are two feature level parameters defined:

- `defaults.text.encoder.num_filters`: This will apply the sampled `num_filters` value to all input text features
using the default text encoder for that particular trial.
- `defaults.category.decoder.reduce_input`: This will apply the sampled `reduce_input` value to all output category
features using the default category decoder for that particular trial.

# Running Hyperparameter Optimization

Use the `ludwig hyperopt` command to run hyperparameter optimization.

```
ludwig hyperopt --dataset reuters-allcats.csv --config hyperopt_config.yaml
```

For a complete reference of hyperparameter search and execution options, see the
[Hyperopt](../configuration/hyperparameter_optimization.md) page of the Ludwig configuration guide.
