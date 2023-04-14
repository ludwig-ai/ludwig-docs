Ludwig provides a declarative interface to support fine-tuning using a variety
of Huggigngface and TorchText LLMs.

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

Read the [v0.7 blog post](https://predibase.com/blog/ludwig-v0-7-fine-tuning-pretrained-image-and-text-models-50x-faster-and) to learn about 50X optimizations.

See a demonstration using Ludwig Python API: [![Text Classification using LLMs on Ludwig](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ludwig-ai/ludwig-docs/blob/master/docs/examples/text_classification/Text_Classification_with_Ludwig_Python_API.ipynb)
