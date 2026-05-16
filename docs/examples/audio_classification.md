This example shows how to build an audio classifier with Ludwig, mapping raw audio recordings to discrete categories such as emotion, sound event, or spoken intent.

Audio classification covers a wide range of real-world applications: detecting the emotional state of a speaker, recognizing environmental sounds (glass breaking, dog barking, sirens), identifying music genres, or classifying spoken commands in a voice interface. Ludwig handles the full pipeline — loading audio files, computing spectral features, training an encoder, and predicting the output category — from a single config file.

We'll use the **EMO-DB** (Berlin Database of Emotional Speech) dataset. EMO-DB contains 535 German-language utterances recorded by 10 professional actors under controlled studio conditions. Each utterance is labeled with one of seven emotions: anger, boredom, disgust, fear, happiness, neutral, or sadness. Despite its small size, EMO-DB is a widely used benchmark for speech emotion recognition and is large enough to demonstrate the full Ludwig workflow.

## Dataset

The dataset contains the following columns:

| column  | type     | description                                                             |
| ------- | -------- | ----------------------------------------------------------------------- |
| audio   | audio    | Path to the WAV audio file of the utterance                             |
| emotion | category | Target emotion: anger, boredom, disgust, fear, happiness, neutral, sadness |
| gender  | category | Speaker gender: male or female                                          |
| age     | number   | Speaker age in years                                                    |

Sample rows:

| audio                  | emotion   | gender | age |
| ---------------------- | --------- | ------ | --- |
| 03a01Wa.wav            | anger     | male   | 31  |
| 08b10Lb.wav            | boredom   | female | 34  |
| 09a07Eb.wav            | disgust   | female | 21  |
| 14b06Aa.wav            | anger     | male   | 26  |
| 12a01Fb.wav            | happiness | female | 32  |

## Download Dataset

=== "cli"

    Downloads the dataset and writes `emodb.csv` plus the audio files to the current directory.

    ```shell
    ludwig datasets download emodb
    ```

=== "python"

    Downloads the EMO-DB dataset into a pandas DataFrame.

    ```python
    from ludwig.datasets import emodb

    # Loads the dataset as a pandas.DataFrame
    train_df, test_df, val_df = emodb.load()
    ```

The loaded DataFrame contains all columns listed above plus a `split` column (0 = train, 1 = test, 2 = validation).

## Train

### Define Ludwig Config

The Ludwig config declares the machine learning task. It tells Ludwig which columns to use as input and what to predict.

For this example we predict **emotion** from the raw **audio** signal using a stacked CNN encoder, which applies a series of 1-D convolutional layers over the mel-spectrogram representation of the audio.

=== "cli"

    With `config.yaml`:

    ```yaml
    input_features:
      - name: audio
        type: audio
        encoder:
          type: stacked_cnn
    output_features:
      - name: emotion
        type: category
    trainer:
      epochs: 20
    ```

=== "python"

    With config defined in a Python dict:

    ```python
    config = {
        "input_features": [
            {
                "name": "audio",
                "type": "audio",
                "encoder": {
                    "type": "stacked_cnn",  # 1-D CNN over mel-spectrogram frames
                },
            }
        ],
        "output_features": [
            {
                "name": "emotion",
                "type": "category",  # 7-class classification
            }
        ],
        "trainer": {
            "epochs": 20,
        },
    }
    ```

### Create and Train a Model

=== "cli"

    ```shell
    ludwig train --config config.yaml --dataset "ludwig://emodb"
    ```

=== "python"

    ```python
    import logging
    from ludwig.api import LudwigModel
    from ludwig.datasets import emodb

    train_df, test_df, val_df = emodb.load()

    # Construct Ludwig model from the config dictionary
    model = LudwigModel(config, logging_level=logging.INFO)

    # Train the model
    results = model.train(
        training_set=train_df,
        validation_set=val_df,
        test_set=test_df,
    )
    train_stats = results.train_stats
    output_directory = results.output_directory
    ```

Training produces a model checkpoint directory under `results/`. Ludwig automatically computes a mel-spectrogram from each WAV file, runs the stacked CNN encoder, and optimizes the cross-entropy loss for the 7-way emotion classification.

## Evaluate

Generates predictions and performance statistics for the held-out test set.

=== "cli"

    ```shell
    ludwig evaluate \
        --model_path results/experiment_run/model \
        --dataset "ludwig://emodb" \
        --split test \
        --output_directory test_results
    ```

=== "python"

    ```python
    # Generates predictions and performance statistics for the test set
    test_stats, predictions, output_directory = model.evaluate(
        test_df,
        collect_predictions=True,
        collect_overall_stats=True,
    )
    ```

## Visualize Metrics

Visualize the confusion matrix to see which emotions are most often confused with each other.

=== "cli"

    ```shell
    ludwig visualize \
        --visualization confusion_matrix \
        --ground_truth_metadata results/experiment_run/model/training_set_metadata.json \
        --test_statistics test_results/test_statistics.json \
        --output_directory visualizations \
        --file_format png
    ```

=== "python"

    ```python
    from ludwig.visualize import confusion_matrix

    confusion_matrix(
        [test_stats],
        model.training_set_metadata,
        "emotion",
        top_n_classes=[7],
        model_names=[""],
        normalize=True,
    )
    ```

Visualize learning curves to see how loss and accuracy evolved during training.

=== "cli"

    ```shell
    ludwig visualize \
        --visualization learning_curves \
        --ground_truth_metadata results/experiment_run/model/training_set_metadata.json \
        --training_statistics results/experiment_run/training_statistics.json \
        --file_format png \
        --output_directory visualizations
    ```

=== "python"

    ```python
    from ludwig.visualize import learning_curves

    learning_curves(train_stats, output_feature_name="emotion")
    ```

## Make Predictions on New Audio

Once the model is trained, generate predictions for new audio files.

=== "cli"

    Create a `new_audio.csv` file containing the paths to the audio files you want to classify:

    ```
    audio
    /path/to/recording1.wav
    /path/to/recording2.wav
    /path/to/recording3.wav
    ```

    ```shell
    ludwig predict \
        --model_path results/experiment_run/model \
        --dataset new_audio.csv \
        --output_directory predictions
    ```

=== "python"

    ```python
    import pandas as pd

    new_audio = pd.DataFrame(
        {
            "audio": [
                "/path/to/recording1.wav",
                "/path/to/recording2.wav",
                "/path/to/recording3.wav",
            ]
        }
    )

    predictions, output_directory = model.predict(new_audio)
    print(predictions[["emotion_predictions", "emotion_probability"]])
    ```

Prediction outputs are written to `predictions/predictions.parquet`. For each audio file Ludwig outputs the predicted emotion label and the probability for each of the 7 classes.

## Tips

### Audio Preprocessing

Ludwig's audio feature automatically computes spectrogram features before passing them to the encoder. You can control the feature type and parameters explicitly:

```yaml
input_features:
  - name: audio
    type: audio
    encoder:
      type: stacked_cnn
    preprocessing:
      audio_feature:
        type: fbank         # mel-filterbank energies
        num_filter_bands: 80
        window_length_in_s: 0.025
        window_shift_in_s: 0.010
      audio_file_length_limit_in_s: 5.0  # clip/pad all files to 5 seconds
      norm: per_file
```

Available `audio_feature` types:
- `fbank` — mel-filterbank energies (recommended for speech)
- `stft` — short-time Fourier transform magnitude
- `stft_phase` — STFT magnitude + phase
- `group_delay` — group delay features

### Encoder Options

| Encoder | Description | When to use |
| ------- | ----------- | ----------- |
| `stacked_cnn` | 1-D CNNs over time frames | Fast baseline, works well with mel-spectrogram |
| `rnn` | Bidirectional LSTM | Good for variable-length sequences |
| `transformer` | Self-attention over frames | Best accuracy when data is sufficient |
| `parallel_cnn` | Multiple CNN filter sizes in parallel | Captures multi-scale temporal patterns |

For small datasets like EMO-DB, `stacked_cnn` is a practical choice. For larger datasets, consider the `transformer` encoder:

```yaml
input_features:
  - name: audio
    type: audio
    encoder:
      type: transformer
      num_heads: 4
      num_layers: 2
      hidden_size: 256
```

### Using Additional Speaker Features

EMO-DB also includes speaker metadata. Adding `gender` and `age` as additional inputs can improve accuracy on small datasets:

```yaml
input_features:
  - name: audio
    type: audio
    encoder:
      type: stacked_cnn
  - name: gender
    type: category
  - name: age
    type: number
output_features:
  - name: emotion
    type: category
trainer:
  epochs: 20
```

### Hyperparameters to Tune

- `preprocessing.audio_file_length_limit_in_s` — set to the 95th-percentile duration in your dataset to avoid excessive zero-padding
- `trainer.learning_rate` — try `1e-3` (default) down to `1e-4` for deeper encoders
- `trainer.batch_size` — audio tensors are large; reduce to 16 or 8 if you run out of GPU memory
- `encoder.num_filters` in CNN layers — increase for larger datasets (e.g., 64, 128, 256)
- `trainer.epochs` with `trainer.early_stop` — set `early_stop: 5` to stop when validation accuracy plateaus

## Other Ludwig Datasets for Audio Classification

Ludwig provides several additional audio classification datasets you can use directly:

| Dataset | Ludwig name | Description | Size |
| ------- | ----------- | ----------- | ---- |
| ESC-50 | `esc50` | Environmental sound classification, 50 classes (sirens, animals, nature, etc.) | 2,000 clips |
| MINDS-14 | `minds14` | Multilingual banking intent recognition from spoken audio, 14 intents × 14 languages | ~14,000 clips |
| Speech MASSIVE | `speech_massive` | Spoken intent and slot detection across 51 languages, 60 intent classes | ~115,000 clips |

### ESC-50: Environmental Sound Classification

ESC-50 is a benchmark dataset for environmental sound classification containing 2,000 5-second recordings organized into 50 semantic classes (dog, rain, clock, chainsaw, etc.).

```shell
ludwig datasets download esc50
```

The ESC-50 config predicts `target` (50-class category) from `audio_path`:

```yaml
input_features:
  - name: audio_path
    type: audio
    encoder:
      type: stacked_cnn
    preprocessing:
      audio_file_length_limit_in_s: 5.0
output_features:
  - name: target
    type: category
trainer:
  epochs: 50
```

### MINDS-14: Multilingual Intent from Speech

MINDS-14 tests whether a model can recognize banking-domain intents (e.g., "transfer funds", "check balance") from spoken audio across 14 languages and dialects.

```shell
ludwig datasets download minds14
```
