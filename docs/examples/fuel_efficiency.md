This example replicates the Keras example at <https://www.tensorflow.org/tutorials/keras/basic_regression> to predict the miles per gallon of a car given its characteristics in the [Auto MPG](https://archive.ics.uci.edu/ml/datasets/auto+mpg) dataset.

| MPG  | Cylinders | Displacement | Horsepower | Weight | Acceleration | ModelYear | Origin |
| ---- | --------- | ------------ | ---------- | ------ | ------------ | --------- | ------ |
| 18.0 | 8         | 307.0        | 130.0      | 3504.0 | 12.0         | 70        | 1      |
| 15.0 | 8         | 350.0        | 165.0      | 3693.0 | 11.5         | 70        | 1      |
| 18.0 | 8         | 318.0        | 150.0      | 3436.0 | 11.0         | 70        | 1      |
| 16.0 | 8         | 304.0        | 150.0      | 3433.0 | 12.0         | 70        | 1      |

```
ludwig experiment \
--dataset auto_mpg.csv \
  --config_file config.yaml
```

With `config.yaml`:

```yaml
training:
    batch_size: 32
    epochs: 1000
    early_stop: 50
    learning_rate: 0.001
    optimizer:
        type: rmsprop
input_features:
    -
        name: Cylinders
        type: number
    -
        name: Displacement
        type: number
    -
        name: Horsepower
        type: number
    -
        name: Weight
        type: number
    -
        name: Acceleration
        type: number
    -
        name: ModelYear
        type: number
    -
        name: Origin
        type: category
output_features:
    -
        name: MPG
        type: number
        optimizer:
            type: mean_squared_error
        num_fc_layers: 2
        fc_size: 64

```
