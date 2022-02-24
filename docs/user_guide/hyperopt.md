Ludwig supports hyperparameter optimization using [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) or a local
executor.

The hyperparameter optimization strategy is specified as part of the Ludwig configuration and run using
the `ludwig hyperopt` command. Every parameter within the config can be tuned using hyperopt.

# Hyperopt Configuration

Most parameters or nested parameters of a ludwig configuration may be optimized, including `input_features`,
`output_features`, `combiner`, `preprocessing`, and `trainer`.  Supported types are `float`, `int` and `category`.

To enable hyperparameter optimization, add the `hyperopt` dictionary at the top level of your config.yaml.  The
`hyperopt` section declares which parameters to optimize, the search strategy, and the optimization goal.

```yaml
hyperopt:
  parameters:
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

# Running Hyperparameter Optimization

Use the `ludwig hyperopt` command to run hyperparameter optimization.

```
ludwig hyperopt --dataset reuters-allcats.csv --config hyperopt_config.yaml
```


For a complete reference of hyperparameter search and execution options, see the
[Hyperopt](../configuration/hyperparameter_optimization.md) page of the Ludwig configuration guide.
