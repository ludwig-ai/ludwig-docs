{% from './macros/includes.md' import render_config_yaml %}

## Configuration Structure

Ludwig models are configured by a single config with the following parameters:

```yaml
model_type: ecd
input_features: []
output_features: []
combiner: {}
preprocessing: {}
defaults: {}
trainer: {}
hyperopt: {}
backend: {}
```

The config specifies input features, output features, preprocessing, model architecture, training loop, hyperparameter
search, and backend infrastructure -- everything that's needed to build, train, and evaluate a model:

- [model_type](./model_type.md): the model variant used for training. Defaults to ECD, which is a neural network based architecture. Also supports
GBM, a gradient-boosted machine (tree based model).
- [input_features](./features/input_features.md): which columns from your training dataset will be used as inputs to the model, what their
data types are, how they should be preprocessed, and how they should be encoded.
- [output_features](./features/output_features.md): the targets we want the model to learn to predict. The data type of the output feature defines
the task (`number` is a regression task, `category` is a multi-class classification task, etc.).
- [combiner](./combiner.md): the backbone model architecture that takes as input all encoded input features and transforms them into a single
embedding vector. The combiner effectively combines individual feature-level models into a model that can accept any number of inputs. GBM models do not make use of the combiner.
- [preprocessing](./preprocessing.md): global preprocessing options including how to split the dataset and how to sample the data.
- [defaults](./defaults.md) default feature configuration. Useful when you have many input features of the same type, and want to apply the same
preprocessing, encoders, etc. to all of them. Overridden by the feature-level configuration if provided.
- [trainer](./trainer.md): hyperparameters used to control the training process, including batch size, learning rate, number of training epochs, etc.
- [hyperopt](./hyperopt/index.md): hyperparameter optimization options. Any param from the previous sections can be treated as a
hyperparameter and explored in combination with other config params.
- [backend](./backend.md): infrastructure and runtime options, including what libraries and distribution strategies will be used during training, how
many cluster resources to use per training worker, how many total workers, whether to use GPUs, etc.

The Ludwig configuration mixes ease of use, by means of reasonable defaults, with flexibility, by means of detailed
control over the parameters of your model. Only `input_features` and `output_features` are required while all other
fields use reasonable defaults, but can be optionally set or modified manually.

The config can be expressed as a python dictionary (`--config_str` for
[Ludwig's CLI](./../user_guide/command_line_interface)), or as a YAML file (`--config`).

=== "YAML"

    ```yaml
    input_features:
        -
            name: Pclass
            type: category
        -
            name: Sex
            type: category
        -
            name: Age
            type: number
            preprocessing:
                missing_value_strategy: fill_with_mean
        -
            name: SibSp
            type: number
        -
            name: Parch
            type: number
        -
            name: Fare
            type: number
            preprocessing:
                missing_value_strategy: fill_with_mean
        -
            name: Embarked
            type: category

    output_features:
        -
            name: Survived
            type: binary
    ```

=== "Python Dict"

    ```python
    {
        "input_features": [
            {
                "name": "Pclass",
                "type": "category"
            },
            {
                "name": "Sex",
                "type": "category"
            },
            {
                "name": "Age",
                "type": "number",
                "preprocessing": {
                    "missing_value_strategy": "fill_with_mean"
                }
            },
            {
                "name": "SibSp",
                "type": "number"
            },
            {
                "name": "Parch",
                "type": "number"
            },
            {
                "name": "Fare",
                "type": "number",
                "preprocessing": {
                    "missing_value_strategy": "fill_with_mean"
                }
            },
            {
                "name": "Embarked",
                "type": "category"
            }
        ],
        "output_features": [
            {
                "name": "Survived",
                "type": "binary"
            }
        ]
    }
    ```

## Rendered Defaults

Ludwig has many parameter options, but with the exception of input and output feature
names and types, all of the other parameters are optional. When a parameter is unspecified, Ludwig
assigns it a *reasonable default* value. Ludwig defines "reasonable" to mean that it is unlikely
to produce bad results, and will train in a reasonable amount of time on commodity hardware. In other
words, Ludwig defaults are intended to be good **baseline** configs upon which more advanced options
can be layed on top.

Here's an example of a minimal config generated from the following command:

```bash
ludwig init_config --dataset ludwig://sst2 --target label --output sst2.yaml
```

```yaml
input_features:
- name: sentence
  type: text
output_features:
- name: label
  type: binary
```

And here is the fully rendered config generated from the following command:

```bash
ludwig render_config --config sst2.yaml --output sst2_rendered.yaml
```

{{ render_config_yaml({"input_features": [{"name": "sentence", "type": "text"}], "output_features": [{"name": "label", "type": "binary"}]}) }}
