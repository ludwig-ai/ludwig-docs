# Introduction

Ludwig is an open-source, [declarative machine learning framework](#why-declarative-machine-learning-systems)
that makes it easy to define deep learning pipelines with a simple and flexible
data-driven configuration system. Ludwig is suitable for a wide variety of AI
tasks, and is hosted by the [Linux Foundation AI & Data](https://lfaidata.foundation/).

Ludwig enables you to apply state-of-the-art tabular, natural language
processing, and computer vision models to your existing data and put them into
production with just a [few short commands](../command_line_interface).

=== "CLI"

    ```sh
    ludwig train --config config.yaml --dataset data.csv
    ludwig predict --model_path results/experiment_run/model --dataset test.csv
    ```

=== "Python"

    ```python
    from ludwig.api import LudwigModel
    import pandas as pd

    # train a model
    config = {
        "input_features": [
            {
                "name": "sepal_length_cm",
                "type": "number"
            },
            {
                "name": "sepal_width_cm",
                "type": "number"
            },
            {
                "name": "petal_length_cm",
                "type": "number"
            },
            {
                "name": "petal_width_cm",
                "type": "number"
            }
        ],
        "output_features": [
            {
                "name": "class",
                "type": "category"
            }
        ]
    }
    model = LudwigModel(config)
    data = pd.read_csv("data.csv")
    train_stats, _, model_dir = model.train(data)

    # or load a model
    model = LudwigModel.load(model_dir)

    # obtain predictions
    predictions = model.predict(data)
    ```

=== "data.csv"

    ```
    sepal_length_cm,sepal_width_cm,petal_length_cm,petal_width_cm
    4.9,3.0,1.4,0.2
    4.7,3.2,1.3,0.2
    4.6,3.1,1.5,0.2
    5.0,3.6,1.4,0.2
    5.4,3.9,1.7,0.4
    4.6,3.4,1.4,0.3
    5.0,3.4,1.5,0.2
    4.4,2.9,1.4,0.2
    4.9,3.1,1.5,0.1
    ```

=== "config.yaml"

    ```yaml
    input_features:
    - name: sepal_length_cm
      type: number
    - name: sepal_width_cm
      type: number
    - name: petal_length_cm
      type: number
    - name: petal_width_cm
      type: number
    output_features:
    - name: class
      type: category
    ```

Ludwig makes this possible through its **declarative** approach to structuring
machine learning pipelines. Instead of writing code for your model, training
loop, preprocessing, postprocessing, evaluation and hyperparameter optimization,
you only need to declare the schema of your data with a simple YAML
configuration:

```yaml
input_features:
-
  name: title
  type: text
-
  name: author
  type: category
-
  name: description
  type: text
-
  name: cover
  type: image

output_features:
-
  name: genre
  type: set
-
  name: price
  type: number
```

Starting from a simple config like the one above, any and all aspects of the
model architecture, training loop, hyperparameter search, and backend
infrastructure can be modified as additional fields in the declarative
configuration to customize the pipeline to meet your requirements:

```yaml
input_features:
-
  name: title
  type: text
  encoder: rnn
  cell: lstm
  num_layers: 2
  state_size: 128
  preprocessing:
    tokenizer: space_punct
-
  name: author
  type: category
  embedding_size: 128
  preprocessing:
    most_common: 10000
-
  name: description
  type: text
  encoder: bert
-
  name: cover
  type: image
  encoder: resnet
  num_layers: 18

output_features:
-
  name: genre
  type: set
-
  name: price
  type: number
  preprocessing:
    normalization: zscore

trainer:
  epochs: 50
  batch_size: 256
  optimizer:
    type: adam
    beat1: 0.9
  learning_rate: 0.001

backend:
  type: local
  cache_format: parquet

hyperopt:
  metric: f1
  sampler: random
  parameters:
    title.num_layers:
      lower: 1
      upper: 5
    training.learning_rate:
      values: [0.01, 0.003, 0.001]
```

Ludwig is a single framework that guides you through machine learning
end-to-end; from experimenting with different training recipes, exploring
state-of-the-art model architectures, to scaling up to large out-of-memory
datasets and multi-node clusters, and finally serving the best model in
production.

# Why Declarative Machine Learning Systems

![img](../images/why_declarative.png)

Ludwig’s declarative approach to machine learning provides the simplicity of an
AutoML solution with the flexibility of writing your own PyTorch code. This is
achieved by creating an extensible, declarative configuration with optional
parameters for every aspect of the pipeline.

## Multi-modal, multi-task learning out-of-the-box

Mix and match tabular data, text, images, and even audio into complex model
configurations without writing code.

## Fully customizable and extensible

Every part of the model and training process can be controlled through a
simple configuration interface.

## Minimal machine learning boilerplate

Engineering complexity of deep learning is handled out of the box, enabling
research scientists to focus on building models at the highest level of
abstraction.

Data preprocessing, hyperparameter optimization, device management, and
distributed training for newly registered `torch.nn.Module` models come
completely free.

# Why Ludwig

Ludwig’s declarative programming model allows for key features such as:

## Highly configurable data preprocessing, modeling, and metrics

Any and all aspects of the model architecture, training loop, hyperparameter
search, and backend infrastructure can be modified as additional fields in the
declarative configuration to customize the pipeline to meet your requirements.

For details on what can be configured, check out
[Ludwig Configuration](https://ludwig-ai.github.io/ludwig-docs/latest/configuration/) docs.

## Integration with any structured data source

If it can be read into a SQL table or Pandas DataFrame, Ludwig can train a model
on it.

## Hyperparameter optimization

Perform a variety of hyperparameter search algorithms locally or across many
workers in parallel using Ray Tune.

## Rich model exporting and tracking

Automatically track all trials and metrics with tools like Tensorboard, Comet
ML, Weights & Biases, and MLflow.

## Automatically scale training to multi-GPU, multi-node clusters

Go from training on your local machine to the cloud without code or config
changes.

## Easily build your benchmarks

Creating a state-of-the-art baseline and comapring it with a new model is a simple config change.

## Easily apply new architectures to multiple problems and datasets

Apply new models across the extensive set of tasks and datasets that Ludwig supports.
Ludwig includes a [full benchmarking toolkit](https://arxiv.org/abs/2111.04260)
accessible to any user, for running experiments with multiple models across
multiple datasets with just a simple configuration.

## Low-code interface for state-of-the-art models, including pre-trained Huggingface Transformers

Ludwig also natively integrates with pre-trained models, such as the ones
available in [Huggingface Transformers](https://huggingface.co/docs/transformers/index).
Users can choose from a vast collection of state-of-the-art pre-trained PyTorch
models to use without needing to write any code at all. For example, training a
BERT-based sentiment analysis model with Ludwig is as simple as:

```shell
ludwig train --dataset sst5 -–config_str “{input_features: [{name: sentence, type: text, encoder: bert}], output_features: [{name: label, type: category}]}”
```

## Low-code interface for AutoML

[Ludwig AutoML](https://ludwig-ai.github.io/ludwig-docs/latest/user_guide/automl/)
allows users to obtain trained models by providing just a dataset, the target column, and a time budget.

```python
auto_train_results = ludwig.automl.auto_train(dataset=my_dataset_df, target=target_column_name, time_limit_s=7200)
```

## Easy productionisation

Ludwig makes it easy to serve deep learning models, including on GPUs. Launch a
REST API for your trained Ludwig model.

```shell
ludwig serve --model_path=/path/to/model
```

Ludwig supports exporting models to efficient Torschscript bundles.

```shell
ludwig export_torchscript -–model_path=/path/to/model
```
