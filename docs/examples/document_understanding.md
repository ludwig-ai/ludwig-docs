This example shows how to build a document visual question answering (DocVQA) model with Ludwig, combining document image understanding with natural language question answering.

Document understanding goes beyond standard image classification or text QA. When a model must read text embedded in document images — scanned contracts, receipts, invoices, ID cards, scientific figures — it needs both visual understanding (layout, fonts, tables) and language understanding (entity recognition, reasoning). DocVQA is the standard benchmark for this capability: given an image of a document and a question, the model must locate and extract the correct answer from the image.

We'll use the **DocVQA** dataset, which contains document images (scanned forms, technical reports, letters, advertisements) paired with human-written questions and answers. The questions test span extraction (reading a number from a table), compositional reasoning (comparing two values), and layout understanding (finding a header or footnote).

## Dataset

| column     | type  | description                                                                   |
| ---------- | ----- | ----------------------------------------------------------------------------- |
| image_path | image | Path to the document image (scanned page, PNG or JPEG)                        |
| question   | text  | Natural language question about the content of the document                   |
| answer     | text  | Correct answer extracted from or inferred from the document                   |

Sample rows:

| image_path                 | question                                          | answer              |
| -------------------------- | ------------------------------------------------- | ------------------- |
| documents/img_001.png      | What is the total amount on the invoice?          | $2,345.00           |
| documents/img_002.png      | Who is the report addressed to?                   | Dr. Sarah Mitchell  |
| documents/img_003.png      | What year was the contract signed?                | 1998                |
| documents/img_004.png      | How many items are listed in the table?           | 7                   |
| documents/img_005.png      | What is the phone number in the letterhead?       | (415) 555-0192      |

## Download Dataset

=== "cli"

    Downloads the dataset and writes `docvqa.csv` plus the document images to the current directory.

    ```shell
    ludwig datasets download docvqa
    ```

=== "python"

    Downloads the DocVQA dataset into a pandas DataFrame.

    ```python
    from ludwig.datasets import docvqa

    # Loads the dataset as a pandas.DataFrame
    train_df, test_df, val_df = docvqa.load()
    ```

The loaded DataFrame contains the above columns plus a `split` column (0 = train, 1 = test, 2 = validation).

## Train

### Define Ludwig Config

This task requires two input features — the document image and the question text — combined by a `concat` combiner, with a generator decoder producing the answer text.

We use a Vision Transformer (`vit`) encoder for the image, which divides the document into patches and learns patch embeddings with self-attention. This allows the model to attend to specific regions of the document. For the text encoder we use BERT, which encodes the question. Both encoders are fine-tuned end-to-end.

=== "cli"

    With `config.yaml`:

    ```yaml
    input_features:
      - name: image_path
        type: image
        encoder:
          type: vit
          trainable: true
      - name: question
        type: text
        encoder:
          type: bert
          trainable: true
    output_features:
      - name: answer
        type: text
        decoder:
          type: generator
          max_new_tokens: 64
    combiner:
      type: concat
    trainer:
      epochs: 5
      batch_size: 8
      learning_rate: 1.0e-5
    ```

=== "python"

    With config defined in a Python dict:

    ```python
    config = {
      "input_features": [
        {
          "name": "image_path",
          "type": "image",
          "encoder": {
            "type": "vit",       # Vision Transformer: reads document patches
            "trainable": True,
          }
        },
        {
          "name": "question",
          "type": "text",
          "encoder": {
            "type": "bert",      # BERT encodes the question
            "trainable": True,
          }
        }
      ],
      "output_features": [
        {
          "name": "answer",
          "type": "text",
          "decoder": {
            "type": "generator",
            "max_new_tokens": 64,  # Most DocVQA answers are short (< 10 words)
          }
        }
      ],
      "combiner": {
        "type": "concat",  # Concatenate image and text representations
      },
      "trainer": {
        "epochs": 5,
        "batch_size": 8,
        "learning_rate": 1e-5,
      }
    }
    ```

### Create and Train a Model

=== "cli"

    ```shell
    ludwig train --config config.yaml --dataset "ludwig://docvqa"
    ```

=== "python"

    ```python
    import logging
    from ludwig.api import LudwigModel
    from ludwig.datasets import docvqa

    train_df, test_df, val_df = docvqa.load()

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

Ludwig resizes each document image to the ViT input resolution, computes patch embeddings, encodes the question with BERT, concatenates the two representations, and trains the generator decoder to produce the correct answer text.

## Evaluate

Generates predictions and text generation metrics (ROUGE, exact match) for the test set.

=== "cli"

    ```shell
    ludwig evaluate \
        --model_path results/experiment_run/model \
        --dataset "ludwig://docvqa" \
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

DocVQA is traditionally evaluated with Average Normalized Levenshtein Similarity (ANLS), which is tolerant of small OCR or spelling errors. Ludwig reports ROUGE scores for text outputs, which provide a useful approximation.

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

    learning_curves(train_stats, output_feature_name="answer")
    ```

## Make Predictions on New Documents

=== "cli"

    Create `documents_to_query.csv`:

    ```
    image_path,question
    /path/to/invoice.png,What is the invoice number?
    /path/to/contract.png,What is the effective date of the agreement?
    /path/to/report.png,Who are the authors listed on the title page?
    ```

    ```shell
    ludwig predict \
        --model_path results/experiment_run/model \
        --dataset documents_to_query.csv \
        --output_directory predictions
    ```

=== "python"

    ```python
    import pandas as pd

    documents_to_query = pd.DataFrame({
        "image_path": [
            "/path/to/invoice.png",
            "/path/to/contract.png",
            "/path/to/report.png",
        ],
        "question": [
            "What is the invoice number?",
            "What is the effective date of the agreement?",
            "Who are the authors listed on the title page?",
        ]
    })

    predictions, output_directory = model.predict(documents_to_query)
    print(predictions[["answer_predictions"]])
    ```

Predictions are written to `predictions/predictions.parquet`. The `answer_predictions` column contains the generated answer text for each document-question pair.

## Tips

### Image Resolution and Preprocessing

Document images are often high-resolution (300 DPI scans). Ludwig resizes images to the encoder's expected resolution:

- ViT-base/16 expects `224×224` or `384×384` pixels
- Higher resolution preserves more text detail but uses more GPU memory

Configure image preprocessing explicitly:

```yaml
input_features:
  - name: image_path
    type: image
    encoder:
      type: vit
      trainable: true
    preprocessing:
      height: 384
      width: 384
      num_channels: 3
```

For documents with very small text, using a higher resolution (384×384) can significantly improve accuracy.

### Document-Specialized Models

General-purpose ViT models are pre-trained on natural images, not document scans. Document-specialized models trained on IIT-CDIP, DocBank, or PubLayNet perform much better on document understanding tasks. You can use these via the `auto_transformer` encoder for image inputs or by fine-tuning from a document-aware checkpoint:

```yaml
input_features:
  - name: image_path
    type: image
    encoder:
      type: auto_transformer
      pretrained_model_name_or_path: microsoft/dit-base
      trainable: true
```

### Treating DocVQA as Extractive QA

If your answers are always spans that appear verbatim in the document, you can simplify the task by treating it as span extraction (multiple-choice over candidate spans) rather than free-form generation. This can improve accuracy when the answer vocabulary is constrained.

For extractive-style QA, change the output feature to `category` (selecting from candidate answers) or use a span-prediction decoder.

### Multi-Modal Combiner Options

The `concat` combiner simply concatenates image and text embeddings. For richer cross-modal interactions, try:

```yaml
combiner:
  type: transformer
  num_heads: 8
  num_layers: 2
```

The `transformer` combiner applies cross-attention between the image and text representations, allowing the model to dynamically weight which parts of the image are relevant for each word in the question.

### Hyperparameters to Tune

- `trainer.learning_rate` — `1e-5` is a good starting point for fine-tuning both ViT and BERT; try `5e-6` to `2e-5`
- `trainer.batch_size` — multimodal models with two large encoders are memory-intensive; start with 8 and reduce if needed
- `preprocessing.height` / `preprocessing.width` — higher resolution = better text legibility but more GPU memory
- `decoder.max_new_tokens` — DocVQA answers are typically 1–10 words; 64 tokens is more than sufficient
- `trainer.gradient_accumulation_steps` — use if batch size must be small due to memory constraints

## Other Ludwig Datasets for Document Understanding

| Dataset | Ludwig name | Task | Input | Output |
| ------- | ----------- | ---- | ----- | ------ |
| TextVQA | `textvqa` | QA on text in natural images | image + question | answer text |
| DocVQA | `docvqa` | QA on scanned documents | document image + question | answer text |

### TextVQA

TextVQA tests whether a model can read and reason about text that appears in natural photographs: street signs, product labels, menu boards, whiteboards. Unlike DocVQA where documents are structured (tables, forms), TextVQA images are unstructured and varied.

```shell
ludwig datasets download textvqa
```

The TextVQA config is identical in structure to DocVQA — replace the dataset name and column names as appropriate for the TextVQA schema.

### Receipt and Invoice Understanding

For structured document extraction tasks (parsing receipts, invoices, or forms into structured JSON), the output modality shifts from free-text answers to structured fields. You can model each field as a separate output feature:

```yaml
input_features:
  - name: image_path
    type: image
    encoder:
      type: vit
      trainable: true
output_features:
  - name: total_amount
    type: text
    decoder:
      type: generator
      max_new_tokens: 16
  - name: vendor_name
    type: text
    decoder:
      type: generator
      max_new_tokens: 32
  - name: invoice_date
    type: text
    decoder:
      type: generator
      max_new_tokens: 16
combiner:
  type: concat
trainer:
  epochs: 10
  batch_size: 16
  learning_rate: 1.0e-5
```
