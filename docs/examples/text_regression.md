This example shows how to predict a continuous numeric score from text using Ludwig — a task known as text regression or rating prediction.

Many real-world problems require mapping a piece of text to a number rather than a discrete category: predicting the star rating of a product review, estimating the reading difficulty of an article, scoring the quality of a generated code snippet, or forecasting the engagement of a social-media post. Ludwig supports this directly by pairing a `text` input feature with a `number` output feature.

We'll use the **App Reviews** dataset, which contains 288,065 user reviews of mobile apps sourced from the Google Play Store. Each review is paired with a star rating between 1 and 5. This makes it an ideal benchmark for learning to predict a numeric quality score from raw text.

## Dataset

The dataset contains the following columns:

| column | type   | description                                               |
| ------ | ------ | --------------------------------------------------------- |
| review | text   | The full text of the user review                          |
| star   | number | Star rating assigned by the reviewer, integer from 1 to 5 |

Sample rows:

| review                                                               | star |
| -------------------------------------------------------------------- | ---- |
| Total trash, waste of time. Do not download!                         | 1    |
| It's decent. Gets the job done but the UI could use some polish.     | 3    |
| Works great and the updates keep making it better. Highly recommend! | 5    |
| Used to be good, now crashes constantly after the last update.       | 2    |
| Simple, clean, does exactly what I need. No complaints.              | 4    |

## Download Dataset

=== "cli"

    Downloads the dataset and writes `app_reviews.csv` to the current directory.

    ```shell
    ludwig datasets download app_reviews
    ```

=== "python"

    Downloads the App Reviews dataset into a pandas DataFrame.

    ```python
    from ludwig.datasets import app_reviews

    # Loads the dataset as a pandas.DataFrame
    train_df, test_df, val_df = app_reviews.load()
    ```

The loaded DataFrame contains the above columns plus a `split` column (0 = train, 1 = test, 2 = validation).

## Train

### Define Ludwig Config

The Ludwig config declares the machine learning task. For text regression we use a BERT-based encoder (pre-trained on a large text corpus) fine-tuned to minimize mean squared error on the star rating.

Using a pretrained transformer encoder like BERT allows the model to leverage knowledge of language learned from billions of words, dramatically improving accuracy compared to training from scratch — especially relevant when predicting subtle numeric differences between 3-star and 4-star reviews.

=== "cli"

    With `config.yaml`:

    ```yaml
    input_features:
      - name: review
        type: text
        encoder:
          type: bert
          trainable: true
    output_features:
      - name: star
        type: number
    trainer:
      epochs: 5
      learning_rate: 1.0e-5
      batch_size: 32
    ```

=== "python"

    With config defined in a Python dict:

    ```python
    config = {
        "input_features": [
            {
                "name": "review",
                "type": "text",
                "encoder": {
                    "type": "bert",  # Use BERT for text encoding
                    "trainable": True,  # Fine-tune BERT weights
                },
            }
        ],
        "output_features": [
            {
                "name": "star",
                "type": "number",  # Predict a continuous number (regression)
            }
        ],
        "trainer": {
            "epochs": 5,
            "learning_rate": 1e-5,  # Small LR for fine-tuning pretrained weights
            "batch_size": 32,
        },
    }
    ```

### Create and Train a Model

=== "cli"

    ```shell
    ludwig train --config config.yaml --dataset "ludwig://app_reviews"
    ```

=== "python"

    ```python
    import logging
    from ludwig.api import LudwigModel
    from ludwig.datasets import app_reviews

    train_df, test_df, val_df = app_reviews.load()

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

Ludwig fine-tunes the BERT encoder end-to-end with mean squared error loss on the star rating. It automatically handles tokenization, padding, and attention masks.

## Evaluate

Generates predictions and regression metrics (MAE, MSE, R²) for the held-out test set.

=== "cli"

    ```shell
    ludwig evaluate \
        --model_path results/experiment_run/model \
        --dataset "ludwig://app_reviews" \
        --split test \
        --output_directory test_results
    ```

=== "python"

    ```python
    # Generates predictions and regression performance statistics for the test set
    test_stats, predictions, output_directory = model.evaluate(
        test_df,
        collect_predictions=True,
        collect_overall_stats=True,
    )
    ```

For number outputs Ludwig reports:
- **mean absolute error (MAE)** — average absolute difference between predicted and true star ratings
- **mean squared error (MSE)**
- **R²** — coefficient of determination (1.0 is perfect)

## Visualize Metrics

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

    learning_curves(train_stats, output_feature_name="star")
    ```

## Make Predictions on New Reviews

=== "cli"

    Create `reviews_to_rate.csv`:

    ```
    review
    The app crashes every time I open it. Terrible.
    Really useful for tracking workouts. Love the interface.
    It's okay. Nothing special but it works.
    ```

    ```shell
    ludwig predict \
        --model_path results/experiment_run/model \
        --dataset reviews_to_rate.csv \
        --output_directory predictions
    ```

=== "python"

    ```python
    import pandas as pd

    reviews_to_rate = pd.DataFrame(
        {
            "review": [
                "The app crashes every time I open it. Terrible.",
                "Really useful for tracking workouts. Love the interface.",
                "It's okay. Nothing special but it works.",
            ]
        }
    )

    predictions, output_directory = model.predict(reviews_to_rate)
    print(predictions[["star_predictions"]])
    ```

Prediction outputs include `star_predictions` (the predicted numeric star rating) in `predictions/predictions.parquet`.

## Tips

### Loss Function

By default Ludwig uses mean squared error for `number` outputs. You can switch to mean absolute error if you want to be less sensitive to outlier ratings:

```yaml
output_features:
  - name: star
    type: number
    loss:
      type: mean_absolute_error
```

### Clipping Predictions

Star ratings are bounded between 1 and 5. Adding a clip constraint prevents the model from predicting values outside this range:

```yaml
output_features:
  - name: star
    type: number
    clip: [1, 5]
```

### Treating the Task as Classification

An alternative approach is to treat the star rating as a 5-class category rather than a continuous number. This often performs better when the labels are integers with limited range and the differences between adjacent classes are meaningful categorical distinctions rather than a continuous gradient.

```yaml
output_features:
  - name: star
    type: category
trainer:
  epochs: 5
  learning_rate: 1.0e-5
  batch_size: 32
```

The trade-off: classification gives you per-class probabilities but loses ordinal information (it treats 1-star and 5-star as equally distant from 3-star as 2-star).

### Encoder Options

| Encoder | `type` value | Notes |
| ------- | ------------ | ----- |
| BERT | `bert` | Strong general-purpose baseline |
| RoBERTa | `auto_transformer` with `pretrained_model_name_or_path: roberta-base` | Often slightly better than BERT |
| DistilBERT | `auto_transformer` with `pretrained_model_name_or_path: distilbert-base-uncased` | 40% faster, ~97% of BERT quality |
| Parallel CNN | `parallel_cnn` | Much faster, no pretrained weights, lower accuracy |

To use a different pretrained model:

```yaml
input_features:
  - name: review
    type: text
    encoder:
      type: auto_transformer
      pretrained_model_name_or_path: roberta-base
      trainable: true
```

### Hyperparameters to Tune

- `trainer.learning_rate` — fine-tuning a pretrained model works best with small learning rates (`1e-5` to `5e-5`); training from scratch needs larger rates (`1e-3` to `1e-4`)
- `trainer.batch_size` — larger batches stabilize fine-tuning; try 32 or 64 with gradient accumulation if GPU memory is limited
- `encoder.max_sequence_length` — default is 256 tokens; reviews longer than this are truncated. Set to 512 for BERT-compatible models at the cost of more memory
- `trainer.epochs` with `trainer.early_stop: 3` — stop fine-tuning once validation MAE stops improving

## Other Ludwig Datasets for Text Regression

| Dataset | Ludwig name | Input | Output | Size |
| ------- | ----------- | ----- | ------ | ---- |
| Amazon Reviews 2023 | `amazon_reviews_2023` | `title` + `text` (review title and body) | `rating` (number, 1–5) | varies by category |

### Amazon Reviews 2023

The Amazon Reviews 2023 dataset spans multiple product categories (electronics, books, clothing, etc.) and provides both the review title and full review text. Using both as inputs often improves accuracy:

```yaml
input_features:
  - name: title
    type: text
    encoder:
      type: bert
      trainable: true
  - name: text
    type: text
    encoder:
      type: bert
      trainable: true
output_features:
  - name: rating
    type: number
trainer:
  epochs: 3
  learning_rate: 1.0e-5
  batch_size: 32
```

```shell
ludwig datasets download amazon_reviews_2023
```
