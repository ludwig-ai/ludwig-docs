This example shows how to apply Ludwig to code intelligence tasks — using source code as text input to train classifiers, defect detectors, and documentation generators.

Code intelligence is a broad family of tasks that treat programs as structured text: detecting bugs before they reach production, generating docstrings from function bodies, searching a codebase by natural language description, predicting whether a code change introduces a vulnerability, or classifying code snippets by programming language. Because code has rich syntactic and semantic structure, pretrained code language models (CodeBERT, GraphCodeBERT, StarCoder) provide a large advantage over general-purpose text encoders.

We'll use the **Code Defect Detection** dataset, which contains 21,854 C/C++ functions labeled as either clean or defective. Functions were extracted from real open-source projects and the labels come from git history: functions that were later fixed in a security-related commit are labeled defective. Detecting such functions automatically can dramatically reduce the cost of code review.

## Dataset

| column | type   | description                                                         |
| ------ | ------ | ------------------------------------------------------------------- |
| func   | text   | Source code of the C/C++ function (one function per row)            |
| target | binary | 1 = function contains a defect, 0 = function is clean              |

Sample rows:

| func (truncated)                                                                                    | target |
| --------------------------------------------------------------------------------------------------- | ------ |
| `static int foo(char *buf, int len) { if (len > MAX) { memcpy(dst, buf, len); } }`                 | 1      |
| `int clamp(int val, int lo, int hi) { return val < lo ? lo : val > hi ? hi : val; }`               | 0      |
| `void process(uint8_t *data, size_t n) { for (size_t i = 0; i <= n; i++) { handle(data[i]); } }`  | 1      |

## Download Dataset

=== "cli"

    Downloads the dataset and writes `code_defect_detection.csv` to the current directory.

    ```shell
    ludwig datasets download code_defect_detection
    ```

=== "python"

    Downloads the Code Defect Detection dataset into a pandas DataFrame.

    ```python
    from ludwig.datasets import code_defect_detection

    # Loads the dataset as a pandas.DataFrame
    train_df, test_df, val_df = code_defect_detection.load()
    ```

The loaded DataFrame contains the above columns plus a `split` column (0 = train, 1 = test, 2 = validation).

## Train

### Define Ludwig Config

We use CodeBERT (`microsoft/codebert-base`), a pretrained transformer model trained on both natural language and programming language pairs. CodeBERT understands code structure and semantics significantly better than general-purpose language models for code-related tasks.

=== "cli"

    With `config.yaml`:

    ```yaml
    input_features:
      - name: func
        type: text
        encoder:
          type: auto_transformer
          pretrained_model_name_or_path: microsoft/codebert-base
          trainable: true
    output_features:
      - name: target
        type: binary
    trainer:
      epochs: 10
      learning_rate: 2.0e-5
      batch_size: 32
    ```

=== "python"

    With config defined in a Python dict:

    ```python
    config = {
      "input_features": [
        {
          "name": "func",
          "type": "text",
          "encoder": {
            "type": "auto_transformer",
            "pretrained_model_name_or_path": "microsoft/codebert-base",
            "trainable": True,  # Fine-tune CodeBERT weights
          }
        }
      ],
      "output_features": [
        {
          "name": "target",
          "type": "binary",  # Binary classification: defective vs. clean
        }
      ],
      "trainer": {
        "epochs": 10,
        "learning_rate": 2e-5,
        "batch_size": 32,
      }
    }
    ```

### Create and Train a Model

=== "cli"

    ```shell
    ludwig train --config config.yaml --dataset "ludwig://code_defect_detection"
    ```

=== "python"

    ```python
    import logging
    from ludwig.api import LudwigModel
    from ludwig.datasets import code_defect_detection

    train_df, test_df, val_df = code_defect_detection.load()

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

Ludwig tokenizes the source code using CodeBERT's byte-pair encoding (which handles identifiers, keywords, and operators correctly), passes token embeddings through the transformer, and trains a binary classifier on top of the `[CLS]` representation.

## Evaluate

Generates predictions and binary classification metrics (accuracy, F1, ROC-AUC) for the test set.

=== "cli"

    ```shell
    ludwig evaluate \
        --model_path results/experiment_run/model \
        --dataset "ludwig://code_defect_detection" \
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

!!! note

    Defect detection datasets are typically class-imbalanced (clean functions outnumber defective ones). Pay attention to F1 and ROC-AUC rather than raw accuracy. Ludwig reports both.

## Visualize Metrics

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
        "target",
        top_n_classes=[2],
        model_names=[""],
        normalize=True,
    )
    ```

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

    learning_curves(train_stats, output_feature_name="target")
    ```

## Make Predictions on New Functions

=== "cli"

    Create `functions_to_check.csv`:

    ```
    func
    "int read_buffer(char *buf, int size) { return fread(buf, 1, size, stdin); }"
    "void safe_copy(char *dst, const char *src, size_t n) { strncpy(dst, src, n); dst[n-1] = '\0'; }"
    ```

    ```shell
    ludwig predict \
        --model_path results/experiment_run/model \
        --dataset functions_to_check.csv \
        --output_directory predictions
    ```

=== "python"

    ```python
    import pandas as pd

    functions_to_check = pd.DataFrame({
        "func": [
            "int read_buffer(char *buf, int size) { return fread(buf, 1, size, stdin); }",
            "void safe_copy(char *dst, const char *src, size_t n) { strncpy(dst, src, n); dst[n-1] = '\\0'; }",
        ]
    })

    predictions, output_directory = model.predict(functions_to_check)
    print(predictions[["target_predictions", "target_probability"]])
    ```

Outputs include `target_predictions` (True = defective, False = clean) and `target_probability` (confidence score) in `predictions/predictions.parquet`.

## Tips

### Handling Long Functions

CodeBERT's maximum sequence length is 512 tokens. Functions longer than this are truncated. Since defects often appear in the middle or tail of a function, truncation can hurt recall. Options:

1. **Increase `max_sequence_length`** up to 512 (CodeBERT's hard limit):
   ```yaml
   encoder:
     type: auto_transformer
     pretrained_model_name_or_path: microsoft/codebert-base
     trainable: true
     max_sequence_length: 512
   ```

2. **Use a long-context model** such as `microsoft/graphcodebert-base` or `Salesforce/codet5-base` which have similar token limits but may handle code structure differently.

3. **Chunk and aggregate**: split long functions into overlapping windows, run the model on each chunk, and aggregate predictions (e.g., take the maximum defect probability across chunks).

### Class Imbalance

If defective functions are rare in your dataset, adjust the loss weight:

```yaml
output_features:
  - name: target
    type: binary
    loss:
      positive_class_weight: 3.0  # Weight defective examples 3× more
```

Or use threshold tuning on the validation set to find the probability cutoff that maximizes F1.

### Choosing a Code Language Model

| Model | HF identifier | Languages | Notes |
| ----- | ------------- | --------- | ----- |
| CodeBERT | `microsoft/codebert-base` | 6 PL + NL | Best for classification tasks |
| GraphCodeBERT | `microsoft/graphcodebert-base` | 6 PL + NL | Uses data flow graphs, better for code search |
| CodeT5 | `Salesforce/codet5-base` | 8 PL | Good for generation tasks (code → docstring) |
| StarCoder | `bigcode/starcoderbase` | 80+ PL | Largest coverage, requires more GPU memory |

### Code Summarization (Code → Docstring)

For generating docstrings from Python functions, switch the output feature to `text` with a generator decoder:

```yaml
input_features:
  - name: func
    type: text
    encoder:
      type: auto_transformer
      pretrained_model_name_or_path: Salesforce/codet5-base
      trainable: true
output_features:
  - name: docstring
    type: text
    decoder:
      type: generator
      max_new_tokens: 64
trainer:
  epochs: 5
  learning_rate: 2.0e-5
  batch_size: 16
```

### Hyperparameters to Tune

- `trainer.learning_rate` — `1e-5` to `5e-5` for fine-tuning; too large causes instability
- `trainer.batch_size` — 32 is a good default; reduce to 16 with gradient accumulation if memory is tight
- `trainer.epochs` — 5–15 epochs; use `trainer.early_stop: 3` on validation F1
- `encoder.max_sequence_length` — set to the 95th-percentile function length in tokens (measure with CodeBERT tokenizer)
- `output_features[0].loss.positive_class_weight` — tune on a held-out set if the dataset is imbalanced

## Other Ludwig Datasets for Code Intelligence

| Dataset | Ludwig name | Task | Input | Output |
| ------- | ----------- | ---- | ----- | ------ |
| Code Defect Detection | `code_defect_detection` | Bug detection | C/C++ function | binary (defective/clean) |
| Code Contests | `code_contests` | Competitive programming | problem description | solution code |
| Codex Thinking | `codex_thinking` | Chain-of-thought code reasoning | problem + code | reasoning trace |

### Code Contests

The Code Contests dataset contains competitive programming problems paired with correct Python/C++/Java solutions. You can use it to train a model that generates code from a problem description:

```shell
ludwig datasets download code_contests
```

```yaml
input_features:
  - name: description
    type: text
    encoder:
      type: auto_transformer
      pretrained_model_name_or_path: Salesforce/codet5-base
      trainable: true
      max_sequence_length: 512
output_features:
  - name: solution
    type: text
    decoder:
      type: generator
      max_new_tokens: 512
trainer:
  epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 8
  learning_rate: 3.0e-5
```
