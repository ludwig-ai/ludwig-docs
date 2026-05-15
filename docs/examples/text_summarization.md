This example shows how to train a text summarization model with Ludwig, mapping long documents to short abstractive summaries.

Text summarization is the task of condensing a long document into a concise summary that preserves its key information. It is a core NLP task with wide applications: summarizing news articles for busy readers, generating executive summaries of legal or scientific documents, distilling customer feedback, and compressing long chat histories for LLM context windows.

We'll use the **CNN / Daily Mail** dataset, one of the most widely cited benchmarks for abstractive summarization. It contains 287,113 news articles paired with multi-sentence highlights (bullet-point summaries) written by professional editors. Articles average about 800 words; highlights average about 55 words.

## Dataset

| column     | type | description                                               |
| ---------- | ---- | --------------------------------------------------------- |
| article    | text | Full text of the news article                             |
| highlights | text | Multi-sentence bullet-point summary written by an editor  |

Sample rows:

| article (truncated)                                                                 | highlights (truncated)                                        |
| ----------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| LONDON, England (Reuters) — Harry Potter star Daniel Radcliffe gains access...      | Harry Potter star Daniel Radcliffe gets $20 million fortune...  |
| Editor's note: In our Behind the Scenes series, CNN correspondents share...         | Mentally ill inmates in Miami are housed on the 15th floor...   |
| MINNEAPOLIS, Minnesota (CNN) — Drivers who were not wearing seat belts...           | New car-safety study uses data from 20 million vehicles...      |

## Download Dataset

=== "cli"

    Downloads the dataset and writes `cnn_dailymail.csv` to the current directory.

    ```shell
    ludwig datasets download cnn_dailymail
    ```

=== "python"

    Downloads the CNN / Daily Mail dataset into a pandas DataFrame.

    ```python
    from ludwig.datasets import cnn_dailymail

    # Loads the dataset as a pandas.DataFrame
    train_df, test_df, val_df = cnn_dailymail.load()
    ```

The loaded DataFrame contains the above columns plus a `split` column (0 = train, 1 = test, 2 = validation).

## Train

### Define Ludwig Config

For text summarization we use an `auto_transformer` encoder initialized from a pretrained sequence-to-sequence model (BART), paired with a `generator` decoder. This lets Ludwig fine-tune the full encoder-decoder stack on the CNN / Daily Mail data.

The `auto_transformer` encoder type loads any Hugging Face model, so you can swap `facebook/bart-base` for `t5-small`, `google/pegasus-xsum`, or any other seq2seq checkpoint.

We use gradient accumulation (`gradient_accumulation_steps: 8`) with a small batch size to keep GPU memory requirements reasonable while still training on effectively large batches.

=== "cli"

    With `config.yaml`:

    ```yaml
    input_features:
      - name: article
        type: text
        encoder:
          type: auto_transformer
          pretrained_model_name_or_path: facebook/bart-base
          trainable: true
          max_sequence_length: 512
    output_features:
      - name: highlights
        type: text
        decoder:
          type: generator
          max_new_tokens: 128
    trainer:
      epochs: 3
      batch_size: 4
      gradient_accumulation_steps: 8
      learning_rate: 5.0e-5
    ```

=== "python"

    With config defined in a Python dict:

    ```python
    config = {
      "input_features": [
        {
          "name": "article",
          "type": "text",
          "encoder": {
            "type": "auto_transformer",
            "pretrained_model_name_or_path": "facebook/bart-base",
            "trainable": True,            # Fine-tune the full encoder
            "max_sequence_length": 512,   # Truncate articles longer than 512 tokens
          }
        }
      ],
      "output_features": [
        {
          "name": "highlights",
          "type": "text",
          "decoder": {
            "type": "generator",
            "max_new_tokens": 128,  # Maximum length of generated summary
          }
        }
      ],
      "trainer": {
        "epochs": 3,
        "batch_size": 4,
        "gradient_accumulation_steps": 8,  # Effective batch size = 32
        "learning_rate": 5e-5,
      }
    }
    ```

### Create and Train a Model

=== "cli"

    ```shell
    ludwig train --config config.yaml --dataset "ludwig://cnn_dailymail"
    ```

=== "python"

    ```python
    import logging
    from ludwig.api import LudwigModel
    from ludwig.datasets import cnn_dailymail

    train_df, test_df, val_df = cnn_dailymail.load()

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

!!! note

    Fine-tuning BART on the full 287K training set can take several hours on a single GPU. For a quick experiment, consider subsampling the training set to 10,000–20,000 examples or using the smaller `t5-small` checkpoint.

## Evaluate

Text generation models are typically evaluated with ROUGE scores, which measure n-gram overlap between generated and reference summaries.

=== "cli"

    ```shell
    ludwig evaluate \
        --model_path results/experiment_run/model \
        --dataset "ludwig://cnn_dailymail" \
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

Ludwig reports ROUGE-1, ROUGE-2, and ROUGE-L for text generation outputs. These measure unigram precision/recall, bigram precision/recall, and longest common subsequence overlap respectively.

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

    learning_curves(train_stats, output_feature_name="highlights")
    ```

## Make Predictions on New Articles

=== "cli"

    Create `articles_to_summarize.csv`:

    ```
    article
    "Scientists at MIT have developed a new battery technology that could store ten times more energy than conventional lithium-ion batteries. The breakthrough, published in Nature Energy, uses a sulfur-based electrolyte..."
    "The Federal Reserve raised interest rates by 25 basis points on Wednesday, the ninth consecutive increase since March 2022. Chair Jerome Powell signaled that further increases may be necessary..."
    ```

    ```shell
    ludwig predict \
        --model_path results/experiment_run/model \
        --dataset articles_to_summarize.csv \
        --output_directory predictions
    ```

=== "python"

    ```python
    import pandas as pd

    articles_to_summarize = pd.DataFrame({
        "article": [
            "Scientists at MIT have developed a new battery technology that could store ten times more energy...",
            "The Federal Reserve raised interest rates by 25 basis points on Wednesday...",
        ]
    })

    predictions, output_directory = model.predict(articles_to_summarize)
    print(predictions[["highlights_predictions"]])
    ```

The `highlights_predictions` column in `predictions/predictions.parquet` contains the generated summary text for each article.

## Tips

### Choosing the Right Pretrained Model

Different pretrained seq2seq models have different trade-offs:

| Model | HF identifier | Strengths |
| ----- | ------------- | --------- |
| BART-base | `facebook/bart-base` | Strong summarization baseline, 140M params |
| BART-large-CNN | `facebook/bart-large-cnn` | Fine-tuned on CNN/DM, best out-of-the-box accuracy |
| T5-small | `t5-small` | Very fast, good for experimentation |
| T5-base | `t5-base` | Good balance of speed and quality |
| Pegasus-XSUM | `google/pegasus-xsum` | Excellent for single-sentence summaries |

To use a different model, change `pretrained_model_name_or_path` in the encoder:

```yaml
encoder:
  type: auto_transformer
  pretrained_model_name_or_path: facebook/bart-large-cnn
  trainable: true
  max_sequence_length: 1024
```

### Input Truncation

News articles often exceed 512 or even 1,024 tokens. Content beyond `max_sequence_length` is silently truncated. Since the most important information tends to appear at the beginning of news articles, leading-sentence truncation works reasonably well. For patent or scientific paper summarization, consider splitting the document into chunks and summarizing each chunk separately.

### Gradient Accumulation

When `batch_size` is small due to GPU memory constraints, `gradient_accumulation_steps` lets you simulate a larger effective batch:

```
effective_batch_size = batch_size × gradient_accumulation_steps
```

For fine-tuning BART-large, `batch_size: 2` with `gradient_accumulation_steps: 16` gives an effective batch of 32.

### Generation Parameters

Control how summaries are generated by adjusting decoder parameters:

```yaml
output_features:
  - name: highlights
    type: text
    decoder:
      type: generator
      max_new_tokens: 128   # Maximum summary length in tokens
      min_new_tokens: 20    # Force at least 20 tokens
      num_beams: 4          # Beam search with 4 beams (higher quality, slower)
      length_penalty: 2.0   # Penalize short summaries
      no_repeat_ngram_size: 3  # Avoid repeated 3-grams
```

### Hyperparameters to Tune

- `encoder.max_sequence_length` — 512 for BART-base/T5; 1024 for BART-large; balance quality vs. memory
- `trainer.learning_rate` — `1e-5` to `5e-5` for fine-tuning; too high causes catastrophic forgetting
- `trainer.batch_size` × `trainer.gradient_accumulation_steps` — target an effective batch of 16–64
- `decoder.max_new_tokens` — match to your desired summary length; 64–256 for news highlights
- `decoder.num_beams` — beam search (`num_beams >= 2`) consistently outperforms greedy decoding for summarization

## Other Ludwig Datasets for Text Summarization

| Dataset | Ludwig name | Input | Output | Size |
| ------- | ----------- | ----- | ------ | ---- |
| ArXiv Summarization | `arxiv_summarization` | `article` (full paper body) | `abstract` | 203,037 papers |
| Big Patent | `big_patent` | `description` (patent description) | `abstract` (patent abstract) | 1.3M patents |

### ArXiv Summarization

The ArXiv dataset pairs the full body of scientific papers with their author-written abstracts. It is significantly more abstractive than CNN / Daily Mail — the model must generalize beyond copying sentences.

```shell
ludwig datasets download arxiv_summarization
```

```yaml
input_features:
  - name: article
    type: text
    encoder:
      type: auto_transformer
      pretrained_model_name_or_path: facebook/bart-base
      trainable: true
      max_sequence_length: 512
output_features:
  - name: abstract
    type: text
    decoder:
      type: generator
      max_new_tokens: 256
trainer:
  epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 8
  learning_rate: 3.0e-5
```

### Big Patent

Big Patent contains 1.3 million U.S. patents from 9 technology sections. Each patent description is paired with its abstract. Patent language is highly technical and domain-specific, making this a challenging long-document summarization task.

```shell
ludwig datasets download big_patent
```
