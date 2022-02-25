Ludwig models are configured by a single config with the following keys:

```yaml
input_features: []
combiner: {}
output_features: []
training: {}
preprocessing: {}
hyperopt: {}
```

The config specifies input features, output features, preprocessing, model architecture, training loop, hyperparameter
search, and backend infrastructure -- everything that's needed to build, train, and evaluate a model.

The Ludwig configuration mixes ease of use, by means of reasonable defaults, with flexibility, by means of detailed
control over the parameters of your model. Only `input_features` and `output_features` are required while all other
fields use reasonable defaults, but can be optionally set or modified manually.

The config can be expressed as a python dictionary (`--config` for
[Ludwig's CLI](../../docs/user_guide/command_line_interface)), or as a YAML file (`--config_file`).

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
