{% from './macros/includes.md' import render_fields, render_yaml %}

Large Language Models (LLMs) are a kind of neural network used for text generation
tasks like chatbots, coding assistants, etc. Unlike ECD models, which are primarily
designed for *predictive* tasks, LLMs are a fundamentally *generative* model type.

The *backbone* of an LLM (without the language model head used for next token
generation) can be used as a **text encoder** in ECD models when using the
[auto_transformer](./features/text_features.md#huggingface-encoders) encoder. If you wish
to use LLMs for predictive tasks like classification and regression, try ECD. For generative
tasks, read on!

Example config for fine-tuning [LLaMA-2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf):

```yaml
model_type: llm
base_model: meta-llama/Llama-2-7b-hf
input_features:
    - name: input
      type: text
output_features:
    - name: response
      type: text
prompt:
    template: |
        <<SYS>>
        You are a helpful, detailed, and polite artificial 
        intelligence assistant. Your answers are clear and 
        suitable for a professional environment.
        
        If context is provided, answer using only the provided 
        contextual information.
        <</SYS>>
        
        [INST] {__sample__} [/INST]
adapter:
    type: lora
quantization:
    bits: 4
trainer:
    type: finetune
    learning_rate: 0.0001
    batch_size: 1
    gradient_accumulation_steps: 8
    epochs: 3
```

# Base Model

The `base_model` parameter specifies the pretrained large language model to serve
as the foundation of your custom LLM.

Currently, any pretrained HuggingFace Causal LM model from the [HuggingFace Hub](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads) is supported as a `base_model`.

Example:

```yaml
base_model: meta-llama/Llama-2-7b-hf
```

!!! attention

    Some models on the HuggingFace Hub require executing untrusted code. For security reasons,
    these models are currently unsupported. If you have interest in using one of these models,
    please file a GitHub issue with your use case.

## HuggingFace Access Token

Some base models like Llama-2 require authorization from HuggingFace to download, 
which in turn requires obtaining a HuggingFace [User Access Token](https://huggingface.co/docs/hub/security-tokens).

Once you have obtained permission to download your preferred base model and have a user access token,
you only need to ensure that your token is exposes as an environment variable in order for Ludwig to be
able to use it:

```bash
export HUGGING_FACE_HUB_TOKEN="<api_token>"
ludwig train ...
```

# Features

## Input Features

Currently, the LLM model type only supports a single input feature of type `text`.

If no `prompt` template is provided, this feature must correspond to a column
in the input dataset. If a prompt template is provided, the rendered prompt
will be used as the input feature value during training and inference.

```yaml
input_features:
    - name: input
      type: text
```

See [Text Features](./features/text_features.md) for
configuration options.

## Output Features

Currently, the LLM model type only supports a single output feature.

When fine-tuning (`trainer.type: finetune`), the output feature type must be
`text`.

For in-context learning (`trainer.type: none`), the output feature type can be
one of `text` or `category`.

```yaml
output_features:
    - name: response
      type: text
```

See [Text Output Features](./features/text_features.md#output-features) for
configuration options.

# Prompt

One of the unique properties of large language models as compared to more conventional deep learning models is their ability to incorporate context inserted into the “prompt” to generate more specific and accurate responses.

The `prompt` parameter can be used to:

- Provide necessary boilerplate needed to make the LLM respond in the correct way (for example, with a response to a question rather than a continuation of the input sequence).
- Combine multiple columns from a dataset into a single text input feature (see [TabLLM](https://arxiv.org/abs/2210.10723)).
- Provide additional context to the model that can help it understand the task, or provide restrictions to prevent hallucinations.

To make use of prompting, one of `prompt.template` or `prompt.task` must be provided. Otherwise the input feature value is passed into
the LLM as-is. Use `template` for fine-grained control over every aspect of the prompt, and use `task` to specify the nature of the
task the LLM is to perform while delegating the exact prompt template to Ludwig's defaults.

!!! attention

    Some models that have already been instruction tuned will have been trained
    to expect a specific prompt template structure. Unfortunately, this isn't
    provided in any model metadata, and as such, you may need to dig around or
    experiment with different prompt templates to find what works best when
    performing [in-context learning](#in-context-learning).

{% set prompt = get_prompt_schema() %}
{{ render_yaml(prompt, parent="prompt") }}

{{ render_fields(schema_class_to_fields(prompt)) }}

## Retrieval

{% set retrieval = get_retrieval_schema() %}
{{ render_yaml(retrieval, parent="retrieval") }}

{{ render_fields(schema_class_to_fields(retrieval)) }}

# Adapter

{% set adapter_classes = get_adapter_schemas() %}
{% for adapter in adapter_classes %}
### {{ adapter.name() }}

{{ adapter.description() }}

{{ render_yaml(adapter, parent="adapter") }}

{{ render_fields(schema_class_to_fields(adapter, exclude=["type"])) }}
{% endfor %}

# Quantization

{% set quantization = get_quantization_schema() %}
{{ render_yaml(quantization, parent="quantization") }}

{{ render_fields(schema_class_to_fields(quantization)) }}

# Model Parameters

```yaml
model_parameters:
    rope_scaling: {}
```

## RoPE Scaling

{% set rs = get_rope_scaling_schema() %}
{{ render_yaml(rs, parent="rope_scaling", updates={"type": "linear", "factor": 2.0}) }}

{{ render_fields(schema_class_to_fields(rs)) }}

# Trainer

LLMs support multiple different training objectives:

- **Fine-Tuning** (`type: finetune`): update the weights of a pretrained LLM with supervised learning.
- **In-Context Learning** (`type: none`): evaluate model performance and predict using only context provided in the prompt.

## Fine-Tuning

For fine-tuning, see the [Trainer](./trainer.md) section for configuration
options.

```yaml
trainer:
    type: finetune
```

## In-Context Learning

For in-context learning, the `none` trainer is specified to denote that no
model parameters will be updated and the "training" step will essentially be
a no-op, except for the purpose of computing metrics on the test set.

```yaml
trainer:
    type: none
```

# Generation

{% set gen = get_generation_schema() %}
{{ render_yaml(gen, parent="generation") }}

{{ render_fields(schema_class_to_fields(gen)) }}
