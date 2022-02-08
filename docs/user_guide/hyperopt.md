Ludwig supports hyperparameter optimization using [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) or a local executor.

The hyperparameter optimization strategy is specified as part of the Ludwig configuration and run using
the `ludwig hyperopt` command. Every parameter within the config can be tuned using hyperopt.

The full configuration specification can be found [here](../configuration/hyperparameter_optimization.md).
