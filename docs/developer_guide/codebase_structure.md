```
├── docker                 - Ludwig Docker images
├── examples               - Configs demonstrating Ludwig on various tasks
├── ludwig                 - Ludwig library source code
│   ├── automl             - Configurations, defaults, and utilities for AutoML
│   ├── backend            - Execution backends (local, horovod, ray)
│   ├── combiners
│   ├── contribs           - 3rd-party integrations (MLFlow, WandB, Comet)
│   ├── data               - Data loading, pre/postprocessing, sampling
│   ├── datasets           - Datasets provided by the ludwig.datasets API
│   ├── decoders           - Output feature decoders
│   ├── encoders           - Input feature encoders
│   ├── features           - Implementations of feature types
│   ├── hyperopt
│   ├── models             - Implementations of ECG, trainer, predictor.
│   ├── modules            - Torch modules including layers, metrics, and losses
│   ├── utils
│   ├── api.py             - Entry point for python API. Declares LudwigModel.
│   └── cli.py             - ludwig command-line tool
└── tests
    ├── integration_tests  - End-to-end tests of Ludwig workflows
    └── ludwig             - Unit tests. Subdirectories match ludwig/ structure
```

The codebase is organized in a modular, datatype / feature centric way. Adding a feature for a new datatype can be done
with minimal edits to existing code:

1. Add a module implementing the new feature
1. Import it in the appropriate registry file i.e. `ludwig/features/feature_registries.py`
1. Add the new module to the intended registries i.e. `input_type_registry`

All datatype-specific logic lives in the corresponding feature module, all of which are under `ludwig/features/`.

## Features

Feature classes provide raw data preprocessing logic specific to each data type in datatype mixin classes, i.e.
`BinaryFeatureMixin`, `NumericalFeatureMixin`, `CategoryFeatureMixin`.
Feature mixins contain data preprocessing functions to obtain feature metadata (`get_feature_meta`, one-time
dataset-wide operations to collect things like min, max, average, vocabulary, etc.) and to transform raw data into
tensors using the previously calculated metadata (`add_feature_data`, which usually work on a dataset row basis).

Output features also contain datatype-specific logic to compute data postprocessing, to transform model predictions back
into data space, and output metrics such as loss, accuracy, etc...

## Model Architectures

Encoders and decoders are modularized as well (they are under `ludwig/encoders/` and `ludwig/decoders/` respectively) so
that they can be used by multiple features. For example sequence encoders are shared by text, sequence, and timeseries
features.

Various model architecture components which can be reused are also split into dedicated modules (i.e. convolutional
modules, fully connected layers, attention, etc...) which are available in `ludwig/modules/`.

## Training and Inference

The training logic resides in `ludwig/models/trainer.py` which initializes a training session, feeds the data, and
executes the training loop. Prediction logic including batch prediction and evaluation resides in
`ludwig/models/predictor.py`.

## Ludwig CLI

The command line interface is managed by the `ludwig/cli.py` script, which imports the other scripts in the `ludwig/`
top-level directory which perform various sub-commands (experiment, evaluate, export, visualize, etc...).

The programmatic interface (which is also used by the CLI commands) is available in the `ludwig/api.py`.

## Testing

All test code is in the `tests/` directory. The `tests/integration_tests/` subdirectory contains test cases which aim
to provide end-to-end test coverage of all workflows provided by Ludwig.

The `tests/ludwig/` directory contains unit tests, organized in a subdirectory tree parallel to the `ludwig/` source
tree. For more details on testing, see [Style Guidelines and Tests](../style_guidelines_and_tests).

## Misc

Hyper-parameter optimization logic is implemented in the scripts in the `ludwig/hyperopt/` package.

The `ludwig/utils/` package contains various utilities used by all other packages.

Finally the `ludwig/contrib/` packages contains user contributed code that integrates with external libraries.
