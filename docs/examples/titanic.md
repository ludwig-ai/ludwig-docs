This example describes how to use Ludwig to train a model for the
[kaggle competition](https://www.kaggle.com/c/titanic/), on predicting a passenger's probability of surviving the Titanic disaster.
Here's a sample of the data:

| Pclass | Sex    | Age | SibSp | Parch | Fare    | Survived | Embarked |
| ------ | ------ | --- | ----- | ----- | ------- | -------- | -------- |
| 3      | male   | 22  | 1     | 0     | 7.2500  | 0        | S        |
| 1      | female | 38  | 1     | 0     | 71.2833 | 1        | C        |
| 3      | female | 26  | 0     | 0     | 7.9250  | 0        | S        |
| 3      | male   | 35  | 0     | 0     | 8.0500  | 0        | S        |

The full data and the column descriptions can be found [here](https://www.kaggle.com/c/titanic/data).

After downloading the data, to train a model on this dataset using Ludwig,

```
ludwig experiment \
  --dataset <PATH_TO_TITANIC_CSV> \
  --config_file config.yaml
```

With `config.yaml`:

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
        type: numerical
        preprocessing:
          missing_value_strategy: fill_with_mean
    -
        name: SibSp
        type: numerical
    -
        name: Parch
        type: numerical
    -
        name: Fare
        type: numerical
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

Better results can be obtained with morerefined feature transformations and preprocessing, but this example has the only aim to show how this type do tasks and data can be used in Ludwig.
