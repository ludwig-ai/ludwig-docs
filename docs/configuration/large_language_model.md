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
```

# Prompt

{% set prompt = get_prompt_schema() %}
{{ render_yaml(prompt, parent="prompt") }}

{{ render_fields(schema_class_to_fields(prompt)) }}

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

{% set mp = get_model_parameters_schema() %}
{{ render_yaml(mp, parent="model_parameters") }}

{{ render_fields(schema_class_to_fields(mp)) }}

# Generation

{% set gen = get_generation_schema() %}
{{ render_yaml(gen, parent="generation") }}

{{ render_fields(schema_class_to_fields(gen)) }}
