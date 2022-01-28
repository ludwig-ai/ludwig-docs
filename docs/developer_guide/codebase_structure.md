The codebase is organized in a modular, datatype / feature centric way so that adding a feature for a new datatype is pretty straightforward and requires isolated code changes.
All the datatype specific logic lives in the corresponding feature module, all of which are under `ludwig/features/`.

Feature classes contain raw data preprocessing logic specific to each data type in datatype mixin clases, like `BinaryFeatureMixin`, `NumericalFeatureMixin`, `CategoryFeatureMixin`, and so on.
Those classes contain data preprocessing functions to obtain feature metadata (`get_feature_meta`, one-time dataset-wide operation to collect things like min, max, average, vocabularies, etc.) and to transform raw data into tensors using the previously calculated metadata (`add_feature_data`, which usually work on a a dataset row basis).
Output features also contain datatype-specific logic to compute data postprocessing, to transform model predictions back into data space, and output metrics such as loss, accuracy, etc..

Encoders and decoders are modularized as well (they are under `ludwig/encoders` and `ludwig/decoders` respectively) so that they can be used by multiple features.
For example sequence encoders are shared among text, sequence, and timeseries features.

Various model architecture components which can be reused are also split into dedicated modules, for example convolutional modules, fully connected modules, etc.. which are available in `ludwig/modules`.

The training logic resides in `ludwig/models/trainer.py` which initializes a training session, feeds the data, and executes training.
The prediction logic resides in `ludwig/models/predictior.py` instead.

The command line interface is managet by the `ludwig/cli.py` script, which imports the various scripts in `ludwig/` that perform the different command line commands.

The programmatic interface (which is also used by the CLI commands) is available in the `ludwig/api.py` script.

Hyper-parameter optimization logic is implemented in the scripts in the `ludwig/hyperopt` package. 

The `ludwig/utils` package contains various utilities used by all other packages.

Finally the `ludwig/contrib` packages contains user contributed code that integrates with external libraries.