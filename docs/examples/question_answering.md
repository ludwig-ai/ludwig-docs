This example shows how to build a question answering model with Ludwig — given a passage of text and a question, the model generates an answer span or free-form text response.

## What is Question Answering?

Question answering (QA) is the task of automatically generating an answer to a natural language question, typically conditioned on a supporting passage or knowledge source.  There are two main flavours:

- **Extractive QA**: the model selects a span from the passage as its answer.
- **Generative QA / open-domain QA**: the model generates the answer text freely, drawing on parametric knowledge or retrieved passages.

Ludwig treats both as text-generation problems — the output is a `text` feature decoded with a generator.

## Datasets

Ludwig ships with several QA datasets that can be loaded with one command:

| Dataset | Description | Size |
| ------- | ----------- | ---- |
| `drop` | Discrete Reasoning Over Paragraphs — reading comprehension requiring arithmetic | 77K train |
| `ambig_qa` | AmbigQA — naturally ambiguous open-domain questions with multiple valid answers | 14K train |
| `nq_open` | Natural Questions Open — open-domain QA with Wikipedia answers | 88K train |
| `boolq` | BoolQ — naturally occurring yes/no questions with supporting passage | 9K train |
| `arc_challenge` | ARC Challenge — science exam questions requiring reasoning | 1.1K train |
| `arc_easy` | ARC Easy — science exam questions, easier split | 2.3K train |
| `cmrc2018` | CMRC 2018 — Chinese machine reading comprehension | 10K train |
| `aqua_rat` | AQuA-RAT — algebraic word problems with rationales | 97K train |

This tutorial uses the **DROP** dataset, which requires multi-step reasoning to answer questions about a passage.  A sample from DROP looks like this:

| passage | question | answers_spans |
| ------- | -------- | ------------- |
| In the 1950s, rock and roll was born... | How many decades after the birth of rock and roll was disco popular? | 2 |
| The Broncos scored 14 points in Q1 and 7 in Q2... | How many points did the Broncos score in the first half? | 21 |
| ... | ... | ... |

## Download the Dataset

=== "cli"

    ```shell
    ludwig datasets download drop
    ```

    This writes `drop.csv` to the current directory.

=== "python"

    ```python
    from ludwig.datasets import drop

    train_df, val_df, test_df = drop.load()
    ```

## Define the Ludwig Config

The following config fine-tunes a pre-trained BERT encoder on both the passage and question, then concatenates the representations and feeds them to a text generator that produces the answer.

=== "Extractive / short-answer"

    ```yaml
    # config.yaml
    input_features:
      - name: passage
        type: text
        encoder:
          type: auto_transformer
          pretrained_model_name_or_path: google-bert/bert-base-uncased
          trainable: true
          max_sequence_length: 384
      - name: question
        type: text
        encoder:
          type: auto_transformer
          pretrained_model_name_or_path: google-bert/bert-base-uncased
          trainable: true
          max_sequence_length: 128

    output_features:
      - name: answers_spans
        type: text
        decoder:
          type: generator
          max_new_tokens: 32

    combiner:
      type: concat

    trainer:
      epochs: 5
      batch_size: 16
      learning_rate: 2.0e-5
      learning_rate_scheduler:
        warmup_fraction: 0.06
    ```

=== "Open-domain (no passage)"

    For open-domain QA datasets like `nq_open` where no supporting passage is provided, simply use the question as the sole input:

    ```yaml
    # config_open_domain.yaml
    input_features:
      - name: question
        type: text
        encoder:
          type: auto_transformer
          pretrained_model_name_or_path: google/flan-t5-base
          trainable: true

    output_features:
      - name: answer
        type: text
        decoder:
          type: generator
          max_new_tokens: 64

    trainer:
      epochs: 5
      batch_size: 16
      learning_rate: 3.0e-5
    ```

=== "Boolean QA (yes/no)"

    For yes/no QA datasets like `boolq`, the answer is a `binary` output which is simpler and faster to train:

    ```yaml
    # config_boolq.yaml
    input_features:
      - name: passage
        type: text
        encoder:
          type: auto_transformer
          pretrained_model_name_or_path: google-bert/bert-base-uncased
          trainable: true
      - name: question
        type: text
        encoder:
          type: auto_transformer
          pretrained_model_name_or_path: google-bert/bert-base-uncased
          trainable: true

    output_features:
      - name: answer
        type: binary

    combiner:
      type: concat

    trainer:
      epochs: 5
      learning_rate: 2.0e-5
      batch_size: 32
    ```

## Train

=== "cli"

    ```shell
    ludwig train --config config.yaml --dataset "ludwig://drop"
    ```

=== "python"

    ```python
    from ludwig.api import LudwigModel
    from ludwig.datasets import drop
    import yaml

    config = yaml.safe_load(open("config.yaml"))
    model = LudwigModel(config)

    train_df, val_df, test_df = drop.load()
    results = model.train(
        training_set=train_df,
        validation_set=val_df,
        test_set=test_df,
    )
    ```

## Evaluate

=== "cli"

    ```shell
    ludwig evaluate \
      --model_path results/experiment_run/model \
      --dataset "ludwig://drop" \
      --split test \
      --output_directory eval_results
    ```

=== "python"

    ```python
    eval_stats, predictions, _ = model.evaluate(
        dataset=test_df,
        collect_predictions=True,
    )
    print(eval_stats)
    ```

## Predict on New Examples

=== "cli"

    ```shell
    ludwig predict \
      --model_path results/experiment_run/model \
      --dataset my_questions.csv
    ```

=== "python"

    ```python
    import pandas as pd

    new_questions = pd.DataFrame({
        "passage": [
            "The Eiffel Tower is located in Paris, France. It was built in 1889."
        ],
        "question": [
            "In what year was the Eiffel Tower built?"
        ],
    })

    predictions, _ = model.predict(dataset=new_questions)
    print(predictions["answers_spans_predictions"])
    ```

## Tips

### Choosing the right encoder

| Encoder | Best for |
| ------- | -------- |
| `google-bert/bert-base-uncased` | Short passages, extractive QA |
| `deepset/roberta-base-squad2` | Pre-trained on SQuAD — strong starting point for extractive QA |
| `google/flan-t5-base` | Open-domain generative QA |
| `facebook/bart-base` | Generative answers requiring paraphrasing |

### Long passage handling

Many QA datasets have passages longer than a single encoder window.  Cap the passage encoder:

```yaml
- name: passage
  type: text
  encoder:
    type: auto_transformer
    pretrained_model_name_or_path: google-bert/bert-base-uncased
    max_sequence_length: 512   # adjust to GPU memory
    trainable: true
```

Or use a sliding-window encoder like `longformer` for very long documents:

```yaml
- name: passage
  type: text
  encoder:
    type: auto_transformer
    pretrained_model_name_or_path: allenai/longformer-base-4096
    trainable: true
```

### Controlling answer length

Tune `max_new_tokens` in the generator decoder to cap how long generated answers can be:

```yaml
output_features:
  - name: answers_spans
    type: text
    decoder:
      type: generator
      max_new_tokens: 32     # short factoid answers
      # max_new_tokens: 256  # longer explanations
```

### Using an LLM for zero-shot QA

For open-domain QA without training, use Ludwig's LLM backend:

```yaml
model_type: llm
base_model: meta-llama/Llama-3.1-8B

quantization:
  bits: 4

prompt:
  template: |
    Answer the following question based on the passage.

    Passage: {passage}

    Question: {question}

    Answer:

input_features:
  - name: prompt
    type: text

output_features:
  - name: answers_spans
    type: text

trainer:
  type: none   # zero-shot — no fine-tuning
```

## Related Ludwig Datasets

| Dataset | Task | Ludwig name |
| ------- | ---- | ----------- |
| DROP | Discrete reasoning reading comprehension | `drop` |
| AmbigQA | Open-domain QA with ambiguous questions | `ambig_qa` |
| Natural Questions Open | Open-domain Wikipedia QA | `nq_open` |
| BoolQ | Yes/no QA | `boolq` |
| BoolQ Standalone | Yes/no QA without passage | `boolq_standalone` |
| ARC Challenge | Science exam QA | `arc_challenge` |
| ARC Easy | Science exam QA (easy) | `arc_easy` |
| CMRC 2018 | Chinese machine reading comprehension | `cmrc2018` |
| AQuA-RAT | Math word problems with rationales | `aqua_rat` |
| BigBench | Diverse reasoning tasks | `bigbench` |

## See Also

- [Text Summarization](./text_summarization.md) — generating text from text
- [Natural Language Understanding](./nlu.md) — intent and slot filling
- [Named Entity Recognition](./ner_tagging.md) — span extraction
- [Multilingual NLP](./multilingual_nlp.md) — multilingual QA (CMRC, Belebele)
