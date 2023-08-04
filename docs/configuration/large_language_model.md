{% from './macros/includes.md' import render_fields, render_yaml %}

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

## HuggingFace Access Token

# Features

## Input Features

Currently, the LLM model type only supports a single input feature of type `text`.

If no `prompt` template is provided, this feature must correspond to a column
in the input dataset. If a prompt template is provided, the rendered prompt
will be used as the input feature value during training and inference.

See [Text Features](./features/text_features.md) for
configuration options.

## Output Features

Currently, the LLM model type only supports a single output feature.

When fine-tuning (`trainer.type: finetune`), the output feature type must be
`text`.

For in-context learning (`trainer.type: none`), the output feature type can be
one of `text` or `category`.

See [Text Output Features](./features/text_features.md#output-features) for
configuration options.

# Prompt

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
### {{ adapter.type }}

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
