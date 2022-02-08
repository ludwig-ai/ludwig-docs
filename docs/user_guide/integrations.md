# Using Integrations

Ludwig provides an extendable interface to integrate with third-party
systems.
To activate a particular integration, simply insert its flag
into the command line. Each integration may have specific requirements
and use.

# Available integrations

- `--comet` - logs training metrics, environment details, test results, visualizations, and more to [Comet.ML](https://comet.ml). Requires a freely available account. For more details, see Comet's [Running Ludwig with Comet](https://www.comet.ml/docs/python-sdk/ludwig/#running-ludwig-with-comet).

- `--wandb` - logs training metrics, configuration parameters, environment details, and trained model to [Weights & Biases](https://www.wandb.com/). For more details, refer to [W&B Quickstart](https://docs.wandb.com/quickstart).

- `--mlflow` - logs training metrics, hyperopt parameters, output artifacts, and train models to [MLflow](https://mlflow.org/). Set the environment variable `MLFLOW_TRACKING_URI` to log results to a remote tracking server.

For more information about integration contributions, please see the [Developer Guide](/ludwig-docs/developer_guide).
