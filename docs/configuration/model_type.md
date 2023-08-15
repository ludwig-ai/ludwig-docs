The top-level `model_type` parameter specifies the type of model to use.

The following model types are supported:

- `ecd` (default): Encoder-Combiner-Decoder neural network model.
- `llm`: Large Language Model for text generation.
- `gbm`: Gradient Boosting Machine tree-based model.

```yaml
model_type: ecd
```

Every model type has trainers associated with it. See the [Trainer](../trainer) section for details about the supported training algorithms per model type.

## Model Type: ECD

See the [ECD documentation](../../user_guide/how_ludwig_works/#ecd-architecture) for details about the Encoder-Combiner-Decoder deep learning architecture.

!!! check

    The full breadth of Ludwig functionality is available for the `ecd` model type.

## Model Type: LLM

The LLM model type is a large language model for text generation. Large language models like Llama-2 can also be used as [text encoders](./features/text_features.md#huggingface-encoders) in the ECD model type, but for ECD the language model head is removed such that the hidden state (embeddings)
are used as output. The LLM model type, by contrast, retains the LM head, which is then used for token generation.

The LLM model type supports all pretrained HuggingFace Causal LM models from the [HuggingFace Hub](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads).

!!! attention

    Selecting the `llm` model type introduces the following limitations:

    - only a single text input and text output feature is supported (for now)
    - the `combiner` section is ignored

## Model Type: GBM

The GBM model type is a gradient boosting machine (GBM) tree model. It is a tree model that is trained using a supported tree learner. Currently, the only supported tree learner is LightGBM.

!!! attention

    Selecting the `gbm` model type introduces the following limitations:

    - only binary, category, and number features are supported
    - only a single output feature is supported
    - the `combiner` section is ignored
