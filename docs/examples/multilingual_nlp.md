This example shows how to train a single Ludwig model on multilingual data — text from dozens of languages — using multilingual pre-trained encoders.

## Why Multilingual NLP?

Many real-world applications must handle user input from multiple languages simultaneously.  A multilingual model:

- Eliminates the need to train and maintain a separate model per language
- Benefits from cross-lingual transfer — training signal from high-resource languages helps lower-resource ones
- Scales easily as new languages are added to a dataset

Ludwig supports multilingual NLP out of the box through its `auto_transformer` encoder, which can load any HuggingFace model including multilingual ones like `bert-base-multilingual-cased` and `xlm-roberta-base`.

## Datasets

Ludwig ships with multilingual NLP datasets spanning dozens of languages:

| Dataset | Languages | Task | Size |
| ------- | --------- | ---- | ---- |
| `amazon_massive_intent` | 51 | Intent classification (60 classes) | 587K train |
| `amazon_massive_scenario` | 51 | Scenario classification (18 classes) | 587K train |
| `amazon_science_massive` | 51 | Intent classification (60 classes) | 587K train |
| `belebele` | 122 | Reading comprehension (multiple choice) | 900 per language |
| `belebele_fr` | 1 (French) | Reading comprehension | 900 test |
| `wikiann_en` | English | NER (IOB2) | 20K train |
| `wikiann_de` | German | NER (IOB2) | 20K train |
| `multinerd` | 10 | Fine-grained NER (31 entity types) | 134K train |
| `mls_german` | German | Speech recognition (ASR) | 469K train |
| `bornholm_bitext` | Danish/Bornholmish | Translation | 19K pairs |
| `europarl_bg_cs` | Bulgarian/Czech | Translation | 400K pairs |
| `clue_afqmc` | Chinese | Sentence similarity | 34K train |
| `cmrc2018` | Chinese | Reading comprehension | 10K train |
| `alpaca_gpt4_zh` | Chinese | Instruction tuning | 42K train |

This tutorial uses the **Amazon MASSIVE** multilingual intent classification dataset, which spans 51 languages and 60 intent classes — one of the most comprehensive multilingual NLP benchmarks available.

A sample of the dataset:

| locale | utt | intent |
| ------ | --- | ------ |
| en-US | set an alarm for seven am | alarm_set |
| de-DE | stelle einen alarm für sieben uhr ein | alarm_set |
| zh-CN | 设置七点的闹钟 | alarm_set |
| ar-SA | اضبط منبهاً للساعة السابعة صباحاً | alarm_set |
| ... | ... | ... |

## Download the Dataset

=== "cli"

    Downloads the dataset and writes `amazon_massive_intent.csv` in the current directory.

    ```shell
    ludwig datasets download amazon_massive_intent
    ```

=== "python"

    ```python
    from ludwig.datasets import amazon_massive_intent

    train_df, val_df, test_df = amazon_massive_intent.load(split=True)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(train_df.head())
    ```

## Define the Ludwig Config

The key to good multilingual performance is using a multilingual pre-trained encoder.  Two strong options:

- **`bert-base-multilingual-cased`**: Covers 104 languages, good all-around baseline
- **`xlm-roberta-base`**: Stronger than mBERT, covers 100 languages, generally better for low-resource languages

=== "mBERT (104 languages)"

    ```yaml
    # config.yaml
    input_features:
      - name: utt
        type: text
        encoder:
          type: auto_transformer
          pretrained_model_name_or_path: google-bert/bert-base-multilingual-cased
          trainable: true
          max_sequence_length: 128

    output_features:
      - name: intent
        type: category

    trainer:
      epochs: 10
      learning_rate: 2.0e-5
      batch_size: 64
      learning_rate_scheduler:
        warmup_fraction: 0.06
    ```

=== "XLM-RoBERTa (100 languages, stronger)"

    ```yaml
    # config_xlmr.yaml
    input_features:
      - name: utt
        type: text
        encoder:
          type: auto_transformer
          pretrained_model_name_or_path: FacebookAI/xlm-roberta-base
          trainable: true
          max_sequence_length: 128

    output_features:
      - name: intent
        type: category

    trainer:
      epochs: 10
      learning_rate: 2.0e-5
      batch_size: 64
      learning_rate_scheduler:
        warmup_fraction: 0.06
    ```

=== "Zero-shot cross-lingual"

    Train only on English, evaluate on all languages.  This tests zero-shot transfer.

    ```yaml
    # config_zeroshot.yaml
    # First filter amazon_massive_intent to en-US locale, then train:
    input_features:
      - name: utt
        type: text
        encoder:
          type: auto_transformer
          pretrained_model_name_or_path: FacebookAI/xlm-roberta-base
          trainable: true

    output_features:
      - name: intent
        type: category

    trainer:
      epochs: 15
      learning_rate: 2.0e-5
      batch_size: 32
    ```

## Train

=== "cli"

    ```shell
    ludwig train --config config.yaml --dataset "ludwig://amazon_massive_intent"
    ```

=== "python"

    ```python
    import yaml
    from ludwig.api import LudwigModel
    from ludwig.datasets import amazon_massive_intent

    config = yaml.safe_load(open("config.yaml"))
    model = LudwigModel(config)

    train_df, val_df, test_df = amazon_massive_intent.load(split=True)

    results = model.train(
        training_set=train_df,
        validation_set=val_df,
        test_set=test_df,
    )
    ```

## Evaluate

=== "cli"

    Overall evaluation across all 51 languages:

    ```shell
    ludwig evaluate \
      --model_path results/experiment_run/model \
      --dataset "ludwig://amazon_massive_intent" \
      --split test \
      --output_directory eval_results
    ```

=== "python"

    Evaluate per language by filtering the test set:

    ```python
    import pandas as pd

    train_df, val_df, test_df = amazon_massive_intent.load(split=True)

    # Overall
    eval_stats, predictions, _ = model.evaluate(test_df, collect_predictions=True)
    print("Overall accuracy:", eval_stats["intent"]["accuracy"])

    # Per language
    for locale in test_df["locale"].unique():
        subset = test_df[test_df["locale"] == locale]
        stats, _, _ = model.evaluate(subset)
        print(f"{locale}: {stats['intent']['accuracy']:.3f}")
    ```

## Predict on New Text

=== "cli"

    ```shell
    ludwig predict \
      --model_path results/experiment_run/model \
      --dataset my_utterances.csv
    ```

=== "python"

    ```python
    import pandas as pd

    new_data = pd.DataFrame(
        {
            "utt": [
                "set an alarm for 8am",  # English
                "stell einen wecker für 8 uhr",  # German
                "设置八点的闹钟",  # Chinese
                "fija una alarma para las 8",  # Spanish
            ]
        }
    )

    predictions, _ = model.predict(new_data)
    print(predictions[["utt_predictions", "utt_probabilities"]])
    ```

## Tips

### Choosing a multilingual encoder

| Model | Languages | Notes |
| ----- | --------- | ----- |
| `google-bert/bert-base-multilingual-cased` | 104 | Fast, well-tested baseline |
| `FacebookAI/xlm-roberta-base` | 100 | Generally stronger, especially low-resource |
| `FacebookAI/xlm-roberta-large` | 100 | Larger, better quality, slower |
| `microsoft/mdeberta-v3-base` | 100 | Strong multilingual model based on DeBERTa |
| `intfloat/multilingual-e5-base` | 100 | Optimised for retrieval/similarity tasks |

### Language-balanced training

The MASSIVE dataset contains the same number of examples per language, so the model sees equal training signal for every locale.  If your dataset has imbalanced language counts, consider upsampling under-represented languages:

```python
# Oversample minority locales to 10K examples each
target_count = 10_000
balanced_dfs = []
for locale, group in train_df.groupby("locale"):
    if len(group) < target_count:
        group = group.sample(target_count, replace=True, random_state=42)
    balanced_dfs.append(group)
balanced_train = pd.concat(balanced_dfs).sample(frac=1, random_state=42)
```

### Using language as an extra feature

If your dataset has a language/locale column, you can provide it as an additional categorical input:

```yaml
input_features:
  - name: utt
    type: text
    encoder:
      type: auto_transformer
      pretrained_model_name_or_path: FacebookAI/xlm-roberta-base
      trainable: true
  - name: locale
    type: category
output_features:
  - name: intent
    type: category
combiner:
  type: concat
```

### Named Entity Recognition across languages

For multilingual NER, use the `multinerd` or `wikiann_*` datasets with a `sequence` output feature:

```yaml
# config_multilingual_ner.yaml
input_features:
  - name: sentence
    type: text
    encoder:
      type: auto_transformer
      pretrained_model_name_or_path: FacebookAI/xlm-roberta-base
      trainable: true

output_features:
  - name: ner_tags
    type: sequence

trainer:
  epochs: 10
  learning_rate: 2.0e-5
  batch_size: 32
```

Train with:

```shell
ludwig train --config config_multilingual_ner.yaml --dataset "ludwig://multinerd"
```

### Cross-lingual transfer experiment

A common research pattern is to train on one high-resource language and evaluate zero-shot on others:

```python
# Train on English only
en_train = train_df[train_df["locale"] == "en-US"]
en_val = val_df[val_df["locale"] == "en-US"]

model.train(training_set=en_train, validation_set=en_val)

# Evaluate on every language
for locale in test_df["locale"].unique():
    subset = test_df[test_df["locale"] == locale]
    stats, _, _ = model.evaluate(subset)
    print(f"{locale}: {stats['intent']['accuracy']:.3f}")
```

## Related Ludwig Datasets

| Dataset | Languages | Task | Ludwig name |
| ------- | --------- | ---- | ----------- |
| Amazon MASSIVE (all languages) | 51 | Intent classification | `amazon_massive_intent` |
| Amazon MASSIVE (scenario) | 51 | Scenario classification | `amazon_massive_scenario` |
| Amazon MASSIVE (Science split) | 51 | Intent classification | `amazon_science_massive` |
| Belebele | 122 | Reading comprehension | `belebele` |
| MultiNERD | 10 | Fine-grained NER | `multinerd` |
| WikiANN English | English | NER | `wikiann_en` |
| WikiANN German | German | NER | `wikiann_de` |
| CLUE AFQMC | Chinese | Sentence similarity | `clue_afqmc` |
| CMRC 2018 | Chinese | Reading comprehension | `cmrc2018` |
| Alpaca GPT-4 Chinese | Chinese | Instruction tuning | `alpaca_gpt4_zh` |
| Bornholm Bitext | Danish/Bornholmish | Translation | `bornholm_bitext` |
| Europarl Bulgarian-Czech | Bulgarian/Czech | Translation | `europarl_bg_cs` |

## See Also

- [Text Classification](./text_classification.md) — single-language text classification
- [Named Entity Recognition](./ner_tagging.md) — sequence tagging
- [Machine Translation](./machine_translation.md) — translating between languages
- [Natural Language Understanding](./nlu.md) — intent and slot classification
