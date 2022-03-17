For text and images, there exists a wide selection of pre-trained models from
libraries like [huggingface](https://huggingface.co/) that can be useful to
leverage in a Ludwig model, for instance as an encoder.

Any pre-trained model implemented as a `torch.nn.Module` can be used within any
`LudwigModule`, which is itself a [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html).

For demonstration purposes, we'll walk through how to implement huggingface's
pre-trained BERT model as Ludwig text encoder. We recommend reading how to
[add an encoder](../add_an_encoder) as a first step.

# 1. Import/load the pretrained model

Load the pre-trained model in the `LudwigModule`'s constructor.

```python
@register_encoder("bert", TEXT)
class BERTEncoder(Encoder):
    fixed_preprocessing_parameters = {
        "word_tokenizer": "hf_tokenizer",
        "pretrained_model_name_or_path": "feature.pretrained_model_name_or_path",
    }

    default_params = {
        "pretrained_model_name_or_path": "bert-base-uncased",
    }

    def __init__(
        self,
        max_sequence_length: int,
        use_pretrained: bool = True,
        pretrained_model_name_or_path: str = "bert-base-uncased",
        trainable: bool = True,
        reduce_output: str = "cls_pooled",
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: Union[str, Callable] = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        gradient_checkpointing: bool = False,
        position_embedding_type: str = "absolute",
        classifier_dropout: float = None,
        pretrained_kwargs: Dict = None,
        **kwargs
    ):
        super().__init__()
        try:
            from transformers import BertConfig, BertModel
        except ModuleNotFoundError:
            logger.error(
                "The transformers library is not installed. "
                "To use the huggingface pretrained models as a Ludwig text "
                "encoders, please run pip install ludwig[text]."
            )
            sys.exit(-1)

        if use_pretrained:
            pretrained_kwargs = pretrained_kwargs or {}
            self.transformer = BertModel.from_pretrained(pretrained_model_name_or_path, **pretrained_kwargs)
        else:
            config = BertConfig(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                hidden_act=hidden_act,
                hidden_dropout_prob=hidden_dropout_prob,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                max_position_embeddings=max_position_embeddings,
                type_vocab_size=type_vocab_size,
                initializer_range=initializer_range,
                layer_norm_eps=layer_norm_eps,
                pad_token_id=pad_token_id,
                gradient_checkpointing=gradient_checkpointing,
                position_embedding_type=position_embedding_type,
                classifier_dropout=classifier_dropout,
            )
            self.transformer = BertModel(config)

        self.reduce_output = reduce_output
        if not self.reduce_output == "cls_pooled":
            self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        if trainable:
            self.transformer.train()
        self.transformer.resize_token_embeddings(vocab_size)
        self.max_sequence_length = max_sequence_length
```

# 2. Call the pre-trained model in the `LudwigModule`'s forward pass

```python
def forward(self, inputs: torch.Tensor,
            mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    if mask is not None:
        mask = mask.to(torch.int32)
    transformer_outputs = self.transformer(
        input_ids=inputs,
        attention_mask=mask,
        token_type_ids=torch.zeros_like(inputs),
    )

    # Optionally reduce output.
    if self.reduce_output == "cls_pooled":
        hidden = transformer_outputs[1]
    else:
        hidden = transformer_outputs[0][:, 1:-1, :]
        hidden = self.reduce_sequence(hidden, self.reduce_output)

    return {"encoder_output": hidden}
```

# 3. Use pre-trained models

Once the encoder has been registered, users can use the encoder right away in
their Ludwig config.

```yaml
input_features:
    - name: description
      type: text
      encoder: bert
      trainable: false
      max_sequence_length: 128
      num_attention_heads: 3
```
