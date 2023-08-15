# Llama2-7b Fine-Tuning 4bit (QLoRA)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1c3AO8l_H6V_x37RwQ8V7M6A-RmcBf2tG?usp=sharing)

This example shows how to fine-tune [Llama2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf) to follow instructions.
Instruction tuning is the first step in adapting a general purpose Large Language Model into a chatbot.

This example uses no distributed training or big data functionality. It is designed to run locally on any machine
with GPU availability.

## Prerequisites

- [HuggingFace API Token](https://huggingface.co/docs/hub/security-tokens)
- Access approval to [Llama2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- GPU with at least 12 GiB of VRAM (in our tests, we used an Nvidia T4)

## Installation

```sh
pip install ludwig ludwig[llm]
```

## Running

We'll use the [Stanford Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) dataset, which will be formatted as a table-like file that looks like this:

|               instruction            |  input   | output |
| :----------------------------------: | :------: | :------: |
|  Give three tips for staying healthy.|  | 1.Eat a balanced diet and make sure to include... |
| Arrange the items given below in the order to ... |  cake, me, eating   | I eating cake. |
|  Write an introductory paragraph about a famous... |  Michelle Obama   | Michelle Obama is an inspirational woman who r... |
|                 ...                  |   ...    | ... |

Create a YAML config file named `model.yaml` with the following:

```yaml
model_type: llm
base_model: meta-llama/Llama-2-7b-hf

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
