# Full fine-tuning

Ludwig supports full fine-tuning with any LLM on Huggingface.

```yaml
input_features:
- name: title
  type: text
  encoder:
    type: auto_transformer
    pretrained_model_name_or_path: bigscience/bloom-3b
    trainable: true
output_features:
- name: class
  type: category
trainer:
  learning_rate: 1.0e-05
  epochs: 3
backend:
  type: ray
  trainer:
    strategy: fsdp
```

See a demonstration using Ludwig Python API: [![Text Classification using LLMs on Ludwig](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ludwig-ai/ludwig-docs/blob/master/docs/examples/llm/LLM_fine_tuning_for_Text_Classification_with_Ludwig_Python_API.ipynb)

# Decoder-only fine-tuning with cached encoder embeddings

Ludwig currently supports two variations for fine-tuning encoders, configured via the `trainable` encoder parameter:

1. Modifying the weights of the pretrained encoder to adapt them to the downstream task (`trainable=true`).
2. Keeping the pretrained encoder weights fixed and training a stack of dense layers that sit downstream as the combiner and decoder modules (`trainable=false`). This is sometimes distinguished as transfer learning.

Training can be over 50x faster when `trainable=false` with the following additional configuration adjustments:

- Automatic mixed precision (AMP) training, which is available when both `trainable=true` and `trainable=false`.
- Cached encoder embeddings, which is only available when `trainable=false`.
- Approximate training set evaluation (`evaluate_training_set=false`), which computes the reported training set metrics at the end of each epoch as a running aggregation of the metrics during training, rather than a separate pass over the training set at the end of each epoch of training. Though this makes training metrics appear “noisy” in the early epochs of training, it generally results in a 33% speedup in training time.

```yaml
input_features:
 - name: review
   type: text
   encoder:
     type: auto_transformer
     pretrained_model_name_or_path: bert-base-uncased
     trainable: false
   preprocessing:
     cache_encoder_embeddings: true

output_features:
 - name: sentiment
   type: category
```

# Adapter-based fine-tuning

One of the biggest barriers to cost effective fine-tuning for LLMs is the need to update billions of parameters each training step. Parameter efficient fine-tuning (PEFT) describes a collection of techniques that reduce the number of trainable parameters during fine-tuning to speed up training, and decrease the memory and disk space required to train large language models.

[PEFT](https://github.com/huggingface/peft) is a popular library from HuggingFace that implements a number of popular parameter efficient fine-tuning strategies, and now in Ludwig v0.8, we provide native integration with PEFT, allowing you to leverage any number of techniques to more efficiently fine-tune LLMs with a single parameter change in the configuration.

One of the most commonly used PEFT adapters is low-rank adaptation (LoRA), which can now be enabled for any large language model in Ludwig with the “adapter” parameter:

```yaml
adapter: lora
```

Additionally, any of the LoRA hyperparameters can be configured explicitly to override Ludwig’s defaults:

```yaml
adapter:
 type: lora
 r: 16
 alpha: 32
 dropout: 0.1
```

Adapters can be added to any LLM model type, or any pretrained auto_transformer text encoder in Ludwig with the same parameter options.

In Ludwig v0.8, we’ve added native support for the following PEFT techniques with more to come. Read more about PEFT in Ludwig [here](../../../configuration/large_language_model#adapter).

- LoRA
- AdaLoRA
- Adaptation Prompt (aka, LLaMA Adapter)


In most frameworks, it would be a lot of work to take an LLM that generates text and adapt it to do classification or regression, but in Ludwig it’s as simple as changing a few lines in the YAML config:

```yaml
input_features:
 - name: review
   type: text
   encoder:
     type: auto_transformer
     pretrained_model_name_or_path: meta-llama/Llama-2-7b-hf
     trainable: true
     adapter: lora


output_features:
 - name: sentiment
   type: category
```
