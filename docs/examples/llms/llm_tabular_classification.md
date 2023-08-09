Prompt templates in Ludwig use Python-style placeholder notation, where every placeholder corresponds to a column in the input dataset:

```yaml
prompt:
 template: "The {color} {animal} jumped over the {size} {object}"
```

When a prompt template like the above is provided, the prompt with all placeholders filled will be used as the text input feature value for the LLM.

```
Dataset:
| color | animal | size | object |
| ----- | ------ | ---- | ------ |
| brown | fox    | big  | dog    |
| white | cat    | huge | rock   |

Prompts:
"The brown fox jumped over the big dog"
"The white cat jumped over the huge rock"
```

Tabular data can be used directly within an LLM fine-tuning setup by
constructing a prompt using the columns of the dataset.

For example, here's a configuration that fine-tunes BERT with a binary
classifcation head on tabular data with the following column names:

- `Recency -- months since last donation`
- `Frequency -- total number of donations`
- `Monetary -- total blood donated in c.c.`
- `Time -- months since first donation`

## Config

```yaml
input_features:
  - name: Recency -- months since last donation
    type: text
    prompt:
      template: >-
        The Recency -- months since last donation is {Recency -- months since
        last donation}. The Frequency -- total number of donations is {Frequency
        -- total number of donations}. The Monetary -- total blood donated in
        c.c. is {Monetary -- total blood donated in c.c.}. The Time -- months
        since first donation is {Time -- months since first donation}.
output_features:
  - name: 'y'
    type: binary
    column: 'y'
preprocessing:
  split:
    type: fixed
    column: split
defaults:
  text:
    encoder:
      type: bert
      trainable: true
combiner:
  type: concat
trainer:
  epochs: 50
  optimizer:
    type: adamw
  learning_rate: 0.00002
  use_mixed_precision: true
  learning_rate_scheduler:
    decay: linear
    warmup_fraction: 0.1
ludwig_version: 0.8.dev
```
