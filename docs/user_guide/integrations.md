# Using Integrations

Ludwig provides an extendable interface to integrate with third-party systems. To activate a particular integration,
simply insert its flag into the command line. Each integration may have specific requirements and use case.

# Available integrations

## Comet ML

`--comet` - logs training metrics, environment details, test results, visualizations, and more to
[Comet.ML](https://comet.ml). Requires a free account. For more details, see Comet's
[Running Ludwig with Comet](https://www.comet.ml/docs/python-sdk/ludwig/#running-ludwig-with-comet).

## Aim

`--aim` - complete experimentation trackings with configuration, metadata, hyperparametrs, losses and terminal logs.
In order to see and end to end Aim-Ludwig training and tracking example please refer to our [demo](https://github.com/aimhubio/aim-ludwig-demo).
For more details about [Aim](https://aimstack.io/) refere to the [documentation](https://aimstack.readthedocs.io/en/latest/).

## Weights & Biases

`--wandb` - logs training metrics, configuration parameters, environment details, and trained model to
[Weights & Biases](https://www.wandb.com/). For more details, refer to
[W&B Quickstart](https://docs.wandb.com/quickstart).

## ML Flow

`--mlflow` - logs training metrics, hyperopt parameters, output artifacts, and trained models to
[MLflow](https://mlflow.org/). Set the environment variable `MLFLOW_TRACKING_URI` to log results to a remote tracking
server.

## Add more integrations

For more information about integration contributions, please see the [Developer Guide](../developer_guide/index.md).
