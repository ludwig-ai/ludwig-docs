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

Tabular datasets can be fed directly to an LLM by using brackets to refer to
column names. Ludwig takes care of formatting row values automatically.

For example, here's a configuration that can be used to perform binary classification 
in a zero-shot setting for a tabular dataset with the following column names:

- `Recency -- months since last donation`
- `Frequency -- total number of donations`
- `Monetary -- total blood donated in c.c.`
- `Time -- months since first donation`


## Config

```yaml
model_type: llm
base_model: facebook/opt-350m
generation:
    temperature: 0.1
    top_p: 0.75
    top_k: 40
    num_beams: 4
    max_new_tokens: 64
prompt:
    template: >-
        The Recency -- months since last donation is {Recency -- months since
        last donation}. The Frequency -- total number of donations is {Frequency
        -- total number of donations}. The Monetary -- total blood donated in
        c.c. is {Monetary -- total blood donated in c.c.}. The Time -- months
        since first donation is {Time -- months since first donation}.
input_features:
-
    name: review
    type: text
output_features:
-
    name: label
    type: category
    preprocessing:
        fallback_label: "neutral"
    decoder:
        type: category_extractor
        match:
            "negative":
                type: contains
                value: "positive"
            "neutral":
                type: contains
                value: "neutral"
            "positive":
                type: contains
                value: "positive"
```