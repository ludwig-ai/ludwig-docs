---
description: Ludwig is an open-source declarative deep learning framework. Train, fine-tune, and deploy models for text, image, tabular, audio, and LLMs using a simple YAML config file — no custom code required.
---

![Ludwig logo](images/ludwig_hero_transparent.png#only-light)
![Ludwig logo](images/ludwig_hero_transparent_dark.png#only-dark)

<p align="center">

    <em>Declarative deep learning framework built for scale and efficiency.</em>

</p>
<p align="center">

<a href="https://badge.fury.io/py/ludwig" target="_blank" style="text-decoration: none;">
    <img src="https://badge.fury.io/py/ludwig.svg" alt="pypi package">
</a>
<a href="https://pepy.tech/project/ludwig" target="_blank" style="text-decoration: none;">
    <img src="https://pepy.tech/badge/ludwig" alt="downloads">
</a>
<a href="https://github.com/ludwig-ai/ludwig" alt="Activity" target="_blank" style="text-decoration: none;">
        <img src="https://img.shields.io/github/commit-activity/m/ludwig-ai/ludwig" /></a>
<a href="https://github.com/ludwig-ai/ludwig/blob/main/LICENSE" target="_blank" style="text-decoration: none;">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="license">
</a>
<a href="https://discord.gg/CBgdrGnZjy" target="_blank" style="text-decoration: none;">
    <img src="https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white" alt="Discord">
</a>
</p>

# 📖 What is Ludwig?

Ludwig is a **low-code** framework for building **custom** AI models like **LLMs** and other deep neural networks.

Key features:

- 🛠 **Build custom models with ease:** a declarative YAML configuration file is all you need to train a state-of-the-art LLM on your data. Support for multi-task and multi-modality learning. Comprehensive config validation detects invalid parameter combinations and prevents runtime failures.
- ⚡ **Optimized for scale and efficiency:** automatic batch size selection, distributed training ([DDP](https://pytorch.org/tutorials/beginner/ddp_series_theory.html), [DeepSpeed](https://github.com/microsoft/DeepSpeed)), parameter efficient fine-tuning ([PEFT](https://github.com/huggingface/peft)), 4-bit quantization (QLoRA), torchao QAT, multi-adapter PEFT, GRPO reward-based alignment, and larger-than-memory datasets.
- 📐 **Expert level control:** retain full control of your models down to the activation functions. Support for hyperparameter optimization, explainability, and rich metric visualizations.
- 🧱 **Modular and extensible:** experiment with different model architectures, tasks, features, and modalities with just a few parameter changes in the config. Think building blocks for deep learning.
- 🚢 **Engineered for production:** prebuilt [Docker](https://hub.docker.com/u/ludwigai) containers, native support for running with [Ray](https://www.ray.io/) on [Kubernetes](https://github.com/ray-project/kuberay), export models to [Torchscript](https://pytorch.org/docs/stable/jit.html) and [Triton](https://developer.nvidia.com/triton-inference-server), upload to [HuggingFace](https://huggingface.co/models) with one command.

Ludwig is hosted by the
[Linux Foundation AI & Data](https://lfaidata.foundation/).

**Tech stack:** Python 3.12 | PyTorch 2.6 | Pydantic 2 | Transformers 5 | Ray 2.54

![img](https://raw.githubusercontent.com/ludwig-ai/ludwig-docs/main/docs/images/ludwig_legos_unanimated.gif)

# 💾 Installation

Install from PyPI. Be aware that Ludwig requires Python 3.12+.

```shell
pip install ludwig
```

Or install with all optional dependencies:

```shell
pip install ludwig[full]
```

# 🏃 Quick Start

For a full tutorial, check out the official [getting started guide](./getting_started/index.md),
or take a look at end-to-end [Examples](./examples/index.md).

## Large Language Model Fine-Tuning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1c3AO8l_H6V_x37RwQ8V7M6A-RmcBf2tG?usp=sharing)

Let's fine-tune a pretrained LLM to follow instructions like a chatbot ("instruction tuning").

### Prerequisites

- [HuggingFace API Token](https://huggingface.co/docs/hub/security-tokens)
- Access approval to your chosen base model (e.g., [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B))
- GPU with at least 12 GiB of VRAM (in our tests, we used an Nvidia T4)

### Running

We'll use the [Stanford Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) dataset, which will be formatted as a table-like file that looks like this:

|                    instruction                    |      input       |                      output                       |
| :-----------------------------------------------: | :--------------: | :-----------------------------------------------: |
|       Give three tips for staying healthy.        |                  | 1.Eat a balanced diet and make sure to include... |
| Arrange the items given below in the order to ... | cake, me, eating |                  I eating cake.                   |
| Write an introductory paragraph about a famous... |  Michelle Obama  | Michelle Obama is an inspirational woman who r... |
|                        ...                        |       ...        |                        ...                        |

Create a YAML config file named `model.yaml` with the following:

```yaml
model_type: llm
base_model: meta-llama/Llama-3.1-8B

quantization:
  bits: 4

adapter:
  type: lora

prompt:
  template: |
    Below is an instruction that describes a task, paired with an input that may provide further context.
    Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Input:
    {input}

    ### Response:

input_features:
  - name: prompt
    type: text

output_features:
  - name: output
    type: text

trainer:
  type: finetune
  learning_rate: 0.0001
  batch_size: 1
  gradient_accumulation_steps: 16
  epochs: 3
  learning_rate_scheduler:
    warmup_fraction: 0.01

preprocessing:
  sample_ratio: 0.1
```

And now let's train the model:

```bash
ludwig train --config model.yaml --dataset "ludwig://alpaca"
```

## Supervised ML

Let's build a neural network that predicts whether a given movie critic's review on [Rotten Tomatoes](https://www.kaggle.com/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset) was positive or negative.

Our dataset will be a CSV file that looks like this:

|     movie_title      | content_rating |              genres              | runtime | top_critic | review_content                                                                                                                                                                                                   | recommended |
| :------------------: | :------------: | :------------------------------: | :-----: | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| Deliver Us from Evil |       R        |    Action & Adventure, Horror    |  117.0  | TRUE       | Director Scott Derrickson and his co-writer, Paul Harris Boardman, deliver a routine procedural with unremarkable frights.                                                                                       | 0           |
|       Barbara        |     PG-13      | Art House & International, Drama |  105.0  | FALSE      | Somehow, in this stirring narrative, Barbara manages to keep hold of her principles, and her humanity and courage, and battles to save a dissident teenage girl whose life the Communists are trying to destroy. | 1           |
|   Horrible Bosses    |       R        |              Comedy              |  98.0   | FALSE      | These bosses cannot justify either murder or lasting comic memories, fatally compromising a farce that could have been great but ends up merely mediocre.                                                        | 0           |
|         ...          |      ...       |               ...                |   ...   | ...        | ...                                                                                                                                                                                                              | ...         |

Download a sample of the dataset from [here](https://ludwig.ai/latest/data/rotten_tomatoes.csv).

```bash
wget https://ludwig.ai/latest/data/rotten_tomatoes.csv
```

Next create a YAML config file named `model.yaml` with the following:

```yaml
input_features:
  - name: genres
    type: set
    preprocessing:
      tokenizer: comma
  - name: content_rating
    type: category
  - name: top_critic
    type: binary
  - name: runtime
    type: number
  - name: review_content
    type: text
    encoder:
      type: embed
output_features:
  - name: recommended
    type: binary
```

That's it! Now let's train the model:

```bash
ludwig train --config model.yaml --dataset rotten_tomatoes.csv
```

**Happy modeling**

Try applying Ludwig to your data. [Reach out](https://discord.gg/CBgdrGnZjy)
if you have any questions.

# ❓ Why you should use Ludwig

- **Minimal machine learning boilerplate**

  Ludwig takes care of the engineering complexity of machine learning out of
  the box, enabling research scientists to focus on building models at the
  highest level of abstraction. Data preprocessing, hyperparameter
  optimization, device management, and distributed training for
  `torch.nn.Module` models come completely free.

- **Easily build your benchmarks**

  Creating a state-of-the-art baseline and comparing it with a new model is a
  simple config change.

- **Easily apply new architectures to multiple problems and datasets**

  Apply new models across the extensive set of tasks and datasets that Ludwig
  supports. Ludwig includes a
  [full benchmarking toolkit](https://arxiv.org/abs/2111.04260) accessible to
  any user, for running experiments with multiple models across multiple
  datasets with just a simple configuration.

- **Highly configurable data preprocessing, modeling, and metrics**

  Any and all aspects of the model architecture, training loop, hyperparameter
  search, and backend infrastructure can be modified as additional fields in
  the declarative configuration to customize the pipeline to meet your
  requirements. For details on what can be configured, check out
  [Ludwig Configuration](./configuration/index.md)
  docs.

- **Multi-modal, multi-task learning out-of-the-box**

  Mix and match tabular data, text, images, and even audio into complex model
  configurations without writing code.

- **Rich model exporting and tracking**

  Automatically track all trials and metrics with tools like Tensorboard,
  Comet ML, Weights & Biases, MLFlow, and Aim Stack.

- **Automatically scale training to multi-GPU, multi-node clusters**

  Go from training on your local machine to the cloud without code changes.

- **Low-code interface for state-of-the-art models, including pre-trained Huggingface Transformers**

  Ludwig also natively integrates with pre-trained models, such as the ones
  available in [Huggingface Transformers](https://huggingface.co/docs/transformers/index).
  Users can choose from a vast collection of state-of-the-art pre-trained
  PyTorch models to use without needing to write any code at all. For example,
  training a ModernBERT-based sentiment analysis model with Ludwig is as simple as:

  ```shell
  ludwig train --dataset sst5 --config_str "{input_features: [{name: sentence, type: text, encoder: {type: bert, pretrained_model_name_or_path: answerdotai/ModernBERT-base}}], output_features: [{name: label, type: category}]}"
  ```

- **Low-code interface for AutoML**

  [Ludwig AutoML](./user_guide/automl.md)
  allows users to obtain trained models by providing just a dataset, the
  target column, and a time budget.

  ```python
  auto_train_results = ludwig.automl.auto_train(dataset=my_dataset_df, target=target_column_name, time_limit_s=7200)
  ```

- **Easy productionisation**

  Ludwig makes it easy to serve deep learning models, including on GPUs.
  Launch a REST API for your trained Ludwig model.

  ```shell
  ludwig serve --model_path=/path/to/model
  ```

  Ludwig supports exporting models to efficient Torchscript bundles.

  ```shell
  ludwig export_torchscript -–model_path=/path/to/model
  ```

# 🗂️ Task Gallery

Ludwig ships with **500+ built-in datasets** covering every major ML task.  Each dataset can be
loaded with a single command: `ludwig datasets download <name>` or `from ludwig.datasets import <name>`.
Pick a task below to see the config.

=== "Text Classification"

    Classify text into categories — topics, intents, sentiments, or arbitrary labels.

    ```yaml
    # Dataset: amazon_massive_intent — multilingual intent classification (51 languages, 60 intents)
    # ludwig datasets download amazon_massive_intent
    input_features:
      - name: utt
        type: text
    output_features:
      - name: intent
        type: category
    ```

    Other datasets: `agnews`, `clinc_oos`, `banking77`, `aegis_safety`, `go_emotions`

=== "Text Regression"

    Predict a continuous numeric score from text — star ratings, quality scores, relevance.

    ```yaml
    # Dataset: app_reviews — predict 1-5 star rating from mobile app review text (288K examples)
    # ludwig datasets download app_reviews
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
    ```

    Other datasets: `amazon_reviews_2023`, `civil_comments` (toxicity), `bookprice_prediction`

=== "Text Summarization"

    Generate a concise summary from a long document or article.

    ```yaml
    # Dataset: cnn_dailymail — news article → bullet-point highlights (287K examples)
    # ludwig datasets download cnn_dailymail
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

    Other datasets: `arxiv_summarization`, `big_patent`, `aeslc`, `billsum`, `dialogsum`, `xsum`

=== "Named Entity Recognition"

    Tag each token in a sentence with its entity type (person, organisation, location, …).

    ```yaml
    # Dataset: conll2003 — CoNLL-2003 NER; English newspaper text with PER/ORG/LOC/MISC tags
    # ludwig datasets download conll2003
    input_features:
      - name: sentence
        type: text
        encoder:
          type: bert
          pretrained_model_name_or_path: answerdotai/ModernBERT-base
          trainable: true
          reduce_output: null
    output_features:
      - name: ner_tags
        type: sequence
        decoder:
          type: tagger
    trainer:
      epochs: 10
      learning_rate: 2.0e-5
      batch_size: 32
    ```

    Other datasets: `wikiann_en`, `wikiann_de`, `multinerd`, `few_nerd`, `wnut17`, `ncbi_disease`

=== "Question Answering"

    Given a passage and a question, extract or generate the answer.

    ```yaml
    # Dataset: squad — Stanford QA Dataset; 100K passage + question → answer span examples
    # ludwig datasets download squad
    input_features:
      - name: context
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
      - name: answers
        type: text
        decoder:
          type: generator
          max_new_tokens: 32
    combiner:
      type: concat
    trainer:
      epochs: 5
      learning_rate: 2.0e-5
    ```

    Other datasets: `squad_v2`, `natural_questions`, `hotpot_qa`, `drop`, `ambig_qa`, `nq_open`, `arc_challenge`

=== "Machine Translation"

    Translate text from one language to another using sequence-to-sequence transformer models.

    ```yaml
    # Dataset: opus100_en_de — OPUS-100 English-German parallel corpus (1M sentence pairs)
    # ludwig datasets download opus100_en_de
    input_features:
      - name: en
        type: text
        encoder:
          type: auto_transformer
          pretrained_model_name_or_path: facebook/mbart-large-cc25
          trainable: true
          max_sequence_length: 128
    output_features:
      - name: de
        type: text
        decoder:
          type: generator
          max_new_tokens: 128
    trainer:
      epochs: 3
      batch_size: 16
      gradient_accumulation_steps: 4
      learning_rate: 3.0e-5
    ```

    Other datasets: `opus100_en_fr`, `opus100_en_es`, `wmt14_de_en`, `bornholm_bitext`

=== "Natural Language Inference"

    Classify whether a hypothesis entails, contradicts, or is neutral given a premise.

    ```yaml
    # Dataset: mnli — Multi-Genre NLI; premise + hypothesis → entailment/neutral/contradiction (393K examples)
    # ludwig datasets download mnli
    input_features:
      - name: premise
        type: text
        encoder:
          type: bert
          pretrained_model_name_or_path: answerdotai/ModernBERT-base
          trainable: true
      - name: hypothesis
        type: text
        encoder:
          type: bert
          pretrained_model_name_or_path: answerdotai/ModernBERT-base
          trainable: true
    output_features:
      - name: label
        type: category
    combiner:
      type: concat
    trainer:
      epochs: 5
      learning_rate: 2.0e-5
      batch_size: 32
    ```

    Other datasets: `qnli`, `rte`, `snli`, `wnli`, `anli`, `belebele`

=== "Sentence Similarity"

    Score how semantically similar two sentences are on a continuous scale.

    ```yaml
    # Dataset: stsb — Semantic Textual Similarity Benchmark; sentence pairs rated 0–5 (8.6K examples)
    # ludwig datasets download stsb
    input_features:
      - name: sentence1
        type: text
        encoder:
          type: bert
          pretrained_model_name_or_path: answerdotai/ModernBERT-base
          trainable: true
      - name: sentence2
        type: text
        encoder:
          type: bert
          pretrained_model_name_or_path: answerdotai/ModernBERT-base
          trainable: true
    output_features:
      - name: score
        type: number
    combiner:
      type: concat
    trainer:
      epochs: 5
      learning_rate: 2.0e-5
      batch_size: 32
    ```

    Other datasets: `qqp` (duplicate questions), `paws`, `sick`, `biosses` (biomedical)

=== "Code Intelligence"

    Work with source code as input — detect bugs, generate docstrings, search code.

    ```yaml
    # Dataset: code_defect_detection — binary bug classification for C/C++ functions (21K examples)
    # ludwig datasets download code_defect_detection
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

    Other datasets: `codexglue_code_to_text`, `code_search_net`, `mbpp`, `humaneval`, `code_alpaca`

=== "Audio Classification"

    Classify audio clips by emotion, sound type, intent, or speaker characteristics.

    ```yaml
    # Dataset: fsd50k — FSD50K; 200-class audio event classification (51K clips)
    # ludwig datasets download fsd50k
    input_features:
      - name: audio
        type: audio
        encoder:
          type: auto_transformer
          pretrained_model_name_or_path: facebook/wav2vec2-base
          trainable: true
    output_features:
      - name: labels
        type: set
    trainer:
      epochs: 20
      batch_size: 16
      learning_rate: 1.0e-5
    ```

    Other datasets: `emodb` (emotions), `esc50` (50 environmental sounds), `audioset`, `birdset`, `minds14`

=== "Speech Recognition"

    Transcribe spoken audio to text.

    ```yaml
    # Dataset: librispeech — English speech from audiobooks; clean 100h split
    # ludwig datasets download librispeech
    input_features:
      - name: audio
        type: audio
        encoder:
          type: auto_transformer
          pretrained_model_name_or_path: facebook/wav2vec2-base
          trainable: true
    output_features:
      - name: text
        type: text
        decoder:
          type: generator
    trainer:
      epochs: 10
      learning_rate: 1.0e-5
    ```

    Other datasets: `fleurs_en`, `multilingual_librispeech`, `voxpopuli`, `peoples_speech`, `cantonese_asr`

=== "Image Classification"

    Assign a category label to an image.

    ```yaml
    # Dataset: eurosat_rgb — Sentinel-2 satellite image land-use classification (10 classes, 27K examples)
    # ludwig datasets download eurosat_rgb
    input_features:
      - name: image
        type: image
        encoder:
          type: vit
          use_pretrained: true
    output_features:
      - name: label
        type: category
    trainer:
      epochs: 10
      learning_rate: 1.0e-4
    ```

    Other datasets: `cifar10`, `cifar100`, `food101`, `fashion_mnist`, `gtsrb`, `beans`, `resisc45`

=== "Semantic Segmentation"

    Assign a class label to every pixel in an image — land use, building footprints, medical tissue.

    ```yaml
    # Dataset: satellite_building_segmentation — satellite image building footprint detection
    # ludwig datasets download satellite_building_segmentation
    input_features:
      - name: image
        type: image
        encoder:
          type: convnextv2
          model_name: convnextv2_tiny
          use_pretrained: true
          trainable: true

    output_features:
      - name: mask
        type: image
        decoder:
          type: unet
          num_classes: 2
        loss:
          type: softmax_cross_entropy

    preprocessing:
      image:
        height: 256
        width: 256

    trainer:
      epochs: 30
      batch_size: 16
      optimizer:
        type: adamw
        lr: 3.0e-4
      use_mixed_precision: true
    ```

    Other datasets: oxford-iiit-pet (3-class trimap), ADE20K (150 classes), Cityscapes (street scenes)

=== "Document Understanding"

    Answer questions about a document image — invoices, receipts, forms, research papers.

    ```yaml
    # Dataset: docvqa — DocVQA: document image + question → answer
    # ludwig datasets download docvqa
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

    Other datasets: `cord_v2`, `invoice_data`, `textvqa`, `merit`, `vqa_rad`

=== "Visual Question Answering"

    Answer natural-language questions about an image by fusing vision and language understanding.

    ```yaml
    # Dataset: scienceqa — multimodal science questions with images and answer choices (21K examples)
    # ludwig datasets download scienceqa
    input_features:
      - name: image
        type: image
        encoder:
          type: vit
          use_pretrained: true
          trainable: true
      - name: question
        type: text
        encoder:
          type: bert
          pretrained_model_name_or_path: answerdotai/ModernBERT-base
          trainable: true
    output_features:
      - name: answer
        type: category
    combiner:
      type: concat
    trainer:
      epochs: 10
      batch_size: 16
      learning_rate: 1.0e-5
    ```

    Other datasets: `mmmu`, `mathvista`, `vqa_rad`, `textvqa`

=== "VLM Fine-Tuning"

    Fine-tune a Vision-Language Model for open-ended image understanding using QLoRA on a single GPU.

    ```yaml
    # Fine-tune Qwen2-VL on any VQA CSV with columns: image_path, question, answer
    model_type: llm
    base_model: Qwen/Qwen2-VL-7B-Instruct

    is_multimodal: true
    trust_remote_code: true

    adapter:
      type: lora
      r: 16
      alpha: 32
      target_modules: ["q_proj", "v_proj"]

    quantization:
      bits: 4

    input_features:
      - name: image_path
        type: image
      - name: question
        type: text

    output_features:
      - name: answer
        type: text

    trainer:
      type: finetune
      epochs: 3
      batch_size: 4
      gradient_accumulation_steps: 8
      learning_rate: 2.0e-5
      learning_rate_scheduler:
        decay: cosine
        warmup_fraction: 0.03
    ```

    Other datasets: `docvqa`, `textvqa`, `mmmu`, `vqa_rad`, `cord_v2`, `scienceqa`

=== "Content Safety"

    Detect harmful, toxic, or unsafe content in text or model outputs.

    ```yaml
    # Dataset: aegis_safety — NVIDIA Aegis 2.0; 30K prompt+response safety labels
    # ludwig datasets download aegis_safety
    input_features:
      - name: prompt
        type: text
        encoder:
          type: bert
          pretrained_model_name_or_path: answerdotai/ModernBERT-base
          trainable: true
      - name: response
        type: text
        encoder:
          type: bert
          pretrained_model_name_or_path: answerdotai/ModernBERT-base
          trainable: true
    output_features:
      - name: safety_label
        type: category
    trainer:
      epochs: 5
      learning_rate: 2.0e-5
      batch_size: 32
    ```

    Other datasets: `civil_comments` (toxicity), `beavertails`, `jigsaw_toxicity_pred`, `factcheck`

=== "Multilingual NLP"

    Train a single model on data from dozens of languages simultaneously.

    ```yaml
    # Dataset: amazon_massive_intent — 60-intent classification across 51 languages (106K train examples)
    # ludwig datasets download amazon_massive_intent
    input_features:
      - name: utt
        type: text
        encoder:
          type: auto_transformer
          pretrained_model_name_or_path: google-bert/bert-base-multilingual-cased
          trainable: true
    output_features:
      - name: intent
        type: category
    trainer:
      epochs: 10
      learning_rate: 2.0e-5
      batch_size: 64
    ```

    Other datasets: `amazon_massive_scenario`, `wikiann_de`, `belebele`, `bornholm_bitext`, `multilingual_librispeech`

=== "Tabular Classification"

    Predict a category from structured numeric and categorical columns.

    ```yaml
    # Dataset: adult_census_income — predict annual income >$50K from census features (48K examples)
    # ludwig datasets download adult_census_income
    input_features:
      - name: age
        type: number
      - name: workclass
        type: category
      - name: education
        type: category
      - name: occupation
        type: category
      - name: hours-per-week
        type: number
      - name: native-country
        type: category
    output_features:
      - name: income
        type: binary
    trainer:
      epochs: 20
      batch_size: 128
    ```

    Other datasets: `heart_failure`, `breast_cancer`, `forest_cover`, `mushroom_edibility`, `otto_group_product`

=== "Tabular Regression"

    Predict a continuous value from structured data.

    ```yaml
    # Dataset: ames_housing — predict house sale price from 80 features (1460 examples)
    # ludwig datasets download ames_housing
    input_features:
      - name: GrLivArea
        type: number
      - name: BedroomAbvGr
        type: number
      - name: FullBath
        type: number
      - name: YearBuilt
        type: number
      - name: Neighborhood
        type: category
    output_features:
      - name: SalePrice
        type: number
    ```

    Other datasets: `california_housing`, `diabetes_regression`, `wine_quality`, `allstate_claims_severity`

=== "Time Series Forecasting"

    Predict future values of a numeric series from historical observations.

    ```yaml
    # Dataset: chronos_electricity — hourly electricity demand forecasting
    # ludwig datasets download chronos_electricity
    input_features:
      - name: demand
        type: timeseries
        preprocessing:
          window_size: 96
        encoder:
          type: patchtst
          patch_size: 16
          patch_stride: 8
          d_model: 128
          num_heads: 8
          num_layers: 3
          output_size: 256

    output_features:
      - name: demand
        type: timeseries
        preprocessing:
          horizon: 24
        loss:
          type: huber

    trainer:
      epochs: 50
      batch_size: 64
      optimizer:
        type: adamw
        lr: 1.0e-4
    ```

    Other datasets: `gift_eval_pretrain`, ETTh1/ETTm2 (electricity transformer temperature benchmarks)

=== "Fraud Detection"

    Flag anomalous or fraudulent transactions from structured financial and behavioural features.

    ```yaml
    # Dataset: creditcard_fraud — 284K transactions; 28 anonymized PCA features + Amount
    input_features:
      - name: V1
        type: number
      - name: V2
        type: number
      - name: V3
        type: number
      - name: V4
        type: number
      - name: V14
        type: number
      - name: V17
        type: number
      - name: Amount
        type: number

    output_features:
      - name: Class
        type: binary

    combiner:
      type: concat
      num_fc_layers: 2
      output_size: 64

    trainer:
      epochs: 20
      batch_size: 256
    ```

    Other datasets: `tabular_benchmark_clf`, `bank_marketing`, `adult_income_hf`

=== "Multi-label Classification"

    Assign multiple labels simultaneously to a single input.

    ```yaml
    # Dataset: go_emotions — 28-emotion multi-label classification from Reddit comments (58K examples)
    # ludwig datasets download go_emotions
    input_features:
      - name: text
        type: text
        encoder:
          type: bert
          trainable: true
    output_features:
      - name: labels
        type: set
    trainer:
      epochs: 10
      learning_rate: 2.0e-5
    ```

    Other datasets: `audioset` (audio events), `lex_glue_ecthr`, `lex_glue_eurlex`

=== "Multi-Task Learning"

    Share a single encoder across multiple output tasks — reduce training cost while improving each head.

    ```yaml
    # Dataset: conll2003 — NER + POS tagging + chunking simultaneously from one model (22K examples)
    # ludwig datasets download conll2003
    input_features:
      - name: sentence
        type: text
        encoder:
          type: bert
          pretrained_model_name_or_path: answerdotai/ModernBERT-base
          trainable: true
          reduce_output: null

    output_features:
      - name: ner_tags
        type: sequence
        decoder:
          type: tagger
      - name: pos_tags
        type: sequence
        decoder:
          type: tagger
      - name: chunk_tags
        type: sequence
        decoder:
          type: tagger

    trainer:
      epochs: 10
      learning_rate: 2.0e-5
      batch_size: 32
    ```

    Other datasets: any dataset with multiple label columns — e.g. NLU datasets with intent + slots

=== "Multimodal Classification"

    Mix images, text, numbers, and categories as inputs in one model — no custom code required.

    ```yaml
    # Example: Twitter bot detection — profile image + bio text + engagement stats → bot or human
    input_features:
      - name: profile_image_path
        type: image
        encoder:
          type: vit
          use_pretrained: true
      - name: description
        type: text
        encoder:
          type: bert
          trainable: true
      - name: followers_count
        type: number
      - name: statuses_count
        type: number
      - name: verified
        type: binary

    output_features:
      - name: account_type
        type: binary

    combiner:
      type: concat
      num_fc_layers: 2
      output_size: 128

    trainer:
      epochs: 15
      learning_rate: 1.0e-4
    ```

    Other datasets: hateful-memes (image + caption), product listing classification, medical multimodal

=== "Relation Extraction"

    Identify semantic relations between entity mentions within and across sentences.

    ```yaml
    # Dataset: docred — DocRED; document-level relation extraction from Wikipedia (5053 documents)
    # ludwig datasets download docred
    input_features:
      - name: title
        type: text
        encoder:
          type: bert
          pretrained_model_name_or_path: answerdotai/ModernBERT-base
          trainable: true
      - name: sents
        type: text
        encoder:
          type: bert
          pretrained_model_name_or_path: answerdotai/ModernBERT-base
          trainable: true
    output_features:
      - name: relation
        type: category
    combiner:
      type: concat
    trainer:
      epochs: 10
      learning_rate: 2.0e-5
      batch_size: 16
    ```

    Other datasets: `tacred`, `re-tacred`, `semeval_re`, `few_rel`

=== "Speaker Verification"

    Decide whether two audio samples belong to the same speaker.

    ```yaml
    # Dataset: voxceleb — speaker identification from celebrity interviews (100K+ utterances)
    # ludwig datasets download voxceleb
    input_features:
      - name: audio
        type: audio
        preprocessing:
          audio_file_length_limit_in_s: 5.0
        encoder:
          type: auto_transformer
          pretrained_model_name_or_path: microsoft/wavlm-base-plus
          trainable: true

    output_features:
      - name: speaker_id
        type: category

    trainer:
      epochs: 20
      batch_size: 32
      learning_rate: 1.0e-5
    ```

    Other datasets: `librispeech` (speaker splits), `fleurs_en` (per-speaker IDs)

=== "Instruction Tuning"

    Fine-tune an LLM to follow natural language instructions.

    ```yaml
    # Dataset: alpaca_gpt4 — 52K GPT-4-generated instruction-following examples
    # ludwig datasets download alpaca_gpt4
    model_type: llm
    base_model: meta-llama/Llama-3.1-8B
    quantization:
      bits: 4
    adapter:
      type: lora
    prompt:
      template: |
        ### Instruction:
        {instruction}

        ### Input:
        {input}

        ### Response:
    input_features:
      - name: prompt
        type: text
    output_features:
      - name: output
        type: text
    trainer:
      type: finetune
      epochs: 3
      learning_rate: 1.0e-4
      batch_size: 1
      gradient_accumulation_steps: 16
    ```

    Other datasets: `alpaca`, `alpaca_cleaned`, `databricks_dolly_15k`, `code_alpaca`, `coig_cqia`

# 📚 Tutorials

- [Text Classification](./examples/text_classification.md)
- [Tabular Data Classification](./examples/adult_census_income.md)
- [Image Classification](./examples/mnist.md)
- [Multimodal Classification](./examples/multimodal_classification.md)

# 🔬 Example Use Cases

- [Named Entity Recognition Tagging](./examples/ner_tagging.md)
- [Natural Language Understanding](./examples/nlu.md)
- [Machine Translation](./examples/machine_translation.md)
- [Text Summarization](./examples/text_summarization.md)
- [Text Regression: Rating Prediction](./examples/text_regression.md)
- [Chit-Chat Dialogue Modeling through seq2seq](./examples/seq2seq.md)
- [Question Answering](./examples/question_answering.md)
- [Sentiment Analysis](./examples/sentiment_analysis.md)
- [Content Safety & Toxicity Detection](./examples/content_safety.md)
- [Multilingual NLP](./examples/multilingual_nlp.md)
- [Code Intelligence](./examples/code_intelligence.md)
- [One-shot Learning with Siamese Networks](./examples/oneshot.md)
- [Visual Question Answering](./examples/visual_qa.md)
- [Document Understanding](./examples/document_understanding.md)
- [Spoken Digit Speech Recognition](./examples/speech_recognition.md)
- [Speaker Verification](./examples/speaker_verification.md)
- [Audio Classification](./examples/audio_classification.md)
- [Binary Classification (Titanic)](./examples/titanic.md)
- [Timeseries forecasting](./examples/forecasting.md)
- [Timeseries forecasting (Weather)](./examples/weather.md)
- [Movie rating prediction](./examples/movie_ratings.md)
- [Multi-label classification](./examples/multi_label.md)
- [Multi-Task Learning](./examples/multi_task.md)
- [Simple Regression: Fuel Efficiency Prediction](./examples/fuel_efficiency.md)
- [Fraud Detection](./examples/fraud.md)

# 💡 More Information

Read our publications on [Ludwig](https://arxiv.org/abs/1909.07930), [declarative ML](https://arxiv.org/abs/2107.08148), and [Ludwig’s SoTA benchmarks](https://openreview.net/forum?id=hwjnu6qW7E4).

Learn more about [how Ludwig works](./user_guide/how_ludwig_works.md), [how to get started](./getting_started/index.md), and work through more [examples](./examples/index.md).

If you are interested in [contributing](./developer_guide/contributing.md), have questions, comments, or thoughts to share, or if you just want to be in the
know, please consider [joining our Community Discord](https://discord.gg/CBgdrGnZjy) and follow us on [X](https://twitter.com/ludwig_ai)!

# 🤝 Join the community to build Ludwig with us

Ludwig is an actively managed open-source project that relies on contributions from folks just like
you. Consider joining the active group of Ludwig contributors to make Ludwig an even
more accessible and feature rich framework for everyone to use!

<a href="https://github.com/ludwig-ai/ludwig/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ludwig-ai/ludwig" />
</a><br/>

# 👋 Getting Involved

- [Discord](https://discord.gg/CBgdrGnZjy)
- [X](https://twitter.com/ludwig_ai)
- [Medium](https://medium.com/ludwig-ai)
- [GitHub Issues](https://github.com/ludwig-ai/ludwig/issues)
