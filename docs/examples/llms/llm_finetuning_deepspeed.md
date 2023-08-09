# LLM Fine-tuning

These examples show you how to fine-tune Large Language Models by taking advantage of model parallelism
with [DeepSpeed](https://www.deepspeed.ai/), allowing Ludwig to scale to very large models with billions of
parameters.

The task here will be to fine-tune a large billion+ LLM to classify the sentiment of [IMDB movie reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). As such, we'll be taking a pretrained LLM, attaching a classification head,
and fine-tuning the weights to improve performance of the LLM on the task. Ludwig will do this for you without no machine learning
code, just configuration.

## Prerequisites

- Installed Ludwig with `ludwig[distributed]` dependencies
- Have a CUDA-enabled version of PyTorch installed
- Have access to a machine or cluster of machines with multiple GPUs
- The IMDB dataset used in these examples comes from Kaggle, so make sure you have your credentials set (e.g., `$HOME/.kaggle.kaggle.json`)

## Source files

=== "train_imdb_ray.py"

    ```python
    import logging
    import os

    import yaml

    from ludwig.api import LudwigModel

    config = yaml.safe_load(
        """
    input_features:
    - name: review
        type: text

        encoder:
        type: auto_transformer
        pretrained_model_name_or_path: bigscience/bloom-3b
        trainable: true
        adapter:
            type: lora

    output_features:
    - name: sentiment
        type: category

    trainer:
    batch_size: 4
    epochs: 3

    backend:
    type: ray
    trainer:
        use_gpu: true
        strategy:
        type: deepspeed
        zero_optimization:
            stage: 3
            offload_optimizer:
            device: cpu
            pin_memory: true
    """
    )

    # Define Ludwig model object that drive model training
    model = LudwigModel(config=config, logging_level=logging.INFO)

    # initiate model training
    (
        train_stats,  # dictionary containing training statistics
        preprocessed_data,  # tuple Ludwig Dataset objects of pre-processed training data
        output_directory,  # location of training results stored on disk
    ) = model.train(
        dataset="ludwig://imdb",
        experiment_name="imdb_sentiment",
        model_name="bloom3b",
    )

    # list contents of output directory
    print("contents of output directory:", output_directory)
    for item in os.listdir(output_directory):
        print("\t", item)

    ```

=== "imdb_deepspeed_zero3.yaml"

    ```yaml
    input_features:
    - name: review
        type: text
        encoder:
        type: auto_transformer
        pretrained_model_name_or_path: bigscience/bloom-3b
        trainable: true
        adapter: lora

    output_features:
    - name: sentiment
        type: category

    trainer:
    batch_size: 4
    epochs: 3
    gradient_accumulation_steps: 8

    backend:
    type: deepspeed
    zero_optimization:
        stage: 3
        offload_optimizer:
        device: cpu
        pin_memory: true
    ```

=== "imdb_deepspeed_zero3_ray.yaml"

    ```yaml
    input_features:
    - name: review
        type: text
        encoder:
        type: auto_transformer
        pretrained_model_name_or_path: bigscience/bloom-3b
        trainable: true
        adapter: lora

    output_features:
    - name: sentiment
        type: category

    trainer:
    batch_size: 4
    epochs: 3
    gradient_accumulation_steps: 8

    backend:
    type: ray
    trainer:
        use_gpu: true
        strategy:
        type: deepspeed
        zero_optimization:
            stage: 3
            offload_optimizer:
            device: cpu
            pin_memory: true
    ```

=== "run_train_dsz3.sh"

    ```sh
    #!/usr/bin/env bash

    # Fail fast if an error occurs
    set -e

    # Get the directory of this script, which contains the config file
    SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

    # Train
    deepspeed --no_python --no_local_rank --num_gpus 4 ludwig train --config ${SCRIPT_DIR}/imdb_deepspeed_zero3.yaml --dataset ludwig://imdb
    ```

=== "run_train_dsz3_ray.sh"

    ```sh
    #!/usr/bin/env bash

    # Fail fast if an error occurs
    set -e

    # Get the directory of this script, which contains the config file
    SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

    # Train
    ludwig train --config ${SCRIPT_DIR}/imdb_deepspeed_zero3_ray.yaml --dataset ludwig://imdb
    ```


## Running DeepSpeed on Ray

This is the recommended way to use DeepSpeed, which supports auto-batch size tuning and distributed data processing.
There is some overhead from using Ray with small datasets (\<100MB), but in most cases performance should be comparable
to using native DeepSpeed.

From the head node of your Ray cluster:

```bash
./run_train_dsz3_ray.sh
```

### Python API

If you want to run Ludwig programatically (from a notebook or as part of a larger workflow), you can run the following
Python script using the Ray cluster launcher from your local machine.

```bash
ray submit cluster.yaml train_imdb_ray.py
```

If running directly on the Ray head node, you can omit the `ray submit` portion and run like an ordinary Python script:

```bash
python train_imdb_ray.py
```

## Running DeepSpeed Native

This mode is suitable for datasets small enough to fit in memory on a single machine, as it doesn't make use of
distributed data processing (requires use of the Ray backend).

The following example assumes you have 4 GPUs available, but can easily be modified to support your preferred
setup.

From a terminal on your machine:

```bash
./run_train_dsz3.sh
```