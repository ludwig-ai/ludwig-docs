Pretrained LLMs are available as text encoders for general text features, and can be included in ECD models for binary or multi-class text classification tasks.

The LLM encoder shares most of its features with the LLM model type, including base model selection, adapters, quantization, and initialization parameters like RoPE scaling. Unlike the LLM model type, the LLM encoder is part of an ECD architecture and does not generate text directly. Instead the input text is processed by the LLM and the final hidden state is passed forward to the combiner and decoder(s), allowing it to be used for predictive tasks directly.

## Example LLM encoder config

The `agnews` dataset contains the examples of news article titles and descriptions, and the task is to classify the examples into one of four section categories. A config to use LLMs to classify article titles may look like the following:

```yaml
model_type: ecd
input_features:
  - name: title
    type: text
    encoder:
      type: llm
      adapter:
        type: lora
      base_model: meta-llama/Llama-2-7b-hf
      quantization:
        bits: 4
    column: title
output_features:
  - name: class
    type: category
    column: class
trainer:
  epochs: 3
  optimizer:
    type: paged_adam
```

This will fine-tune a 4-bit quantized LoRA adapter for `llama-2-7b` model and simultaneously train a classification head. The adapter weights, combiner parameters, and decoder parameters will be saved in the results after fine-tuning/training.

To learn more about configuring LLMs for text classification, see the [LLM Encoder Reference](../../configuration/features/text_features.md#llm-encoders).
