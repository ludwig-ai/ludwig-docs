# Structured and Constrained LLM Output

Large language models produce fluent text, but without additional constraints they are free to output anything — extra prose, invalid JSON, or label names that don't match your schema. **Constrained decoding** solves this by modifying token sampling at inference time: a constraint is compiled into per-step logit masks so that only tokens consistent with the constraint can ever be sampled.

Ludwig supports three types of output constraints, all configured in the output feature's `decoder` block:

| Constraint | Config key | Use case |
|-----------|------------|----------|
| JSON schema | `decoder.json_schema` | Structured extraction, tool-call responses |
| Regular expression | `decoder.regex` | Classification, fixed-format fields |
| EBNF grammar | `decoder.grammar` | Complex structured formats |

## JSON schema constraints

A JSON schema constraint guarantees that the model's output is valid JSON matching a specific structure. Ludwig compiles the schema into token masks that are applied at every sampling step, so the model cannot produce malformed JSON, extra keys, or values outside an `enum`.

### Configuration

```yaml
model_type: llm
base_model: microsoft/phi-2

prompt:
  task: >
    Extract the named entities from the input text and return them as a JSON
    object with this structure:
    {"entities": [{"text": "...", "type": "PERSON|ORG|LOC|DATE"}]}.
    Return only valid JSON, nothing else.

input_features:
  - name: text
    type: text

output_features:
  - name: output
    type: text
    decoder:
      type: text_parser
      json_schema:
        type: object
        properties:
          entities:
            type: array
            items:
              type: object
              properties:
                text:
                  type: string
                type:
                  type: string
                  enum: [PERSON, ORG, LOC, DATE]
              required: [text, type]
        required: [entities]
        additionalProperties: false

generation:
  max_new_tokens: 200
  temperature: 0.1
  do_sample: false

backend:
  type: local
```

### Example

Given the input:

```
Apple Inc. was founded by Steve Jobs in Cupertino, California on April 1, 1976.
```

The model produces exactly:

```json
{"entities": [
  {"text": "Apple Inc.", "type": "ORG"},
  {"text": "Steve Jobs", "type": "PERSON"},
  {"text": "Cupertino", "type": "LOC"},
  {"text": "California", "type": "LOC"},
  {"text": "April 1, 1976", "type": "DATE"}
]}
```

The output is always valid JSON. It cannot contain extra keys or type values outside the `enum`.

## Regex constraints

A regex constraint restricts the output to strings that match the pattern. This is ideal for classification tasks where the valid outputs form a small, known set.

### Configuration

```yaml
model_type: llm
base_model: Qwen/Qwen2-0.5B-Instruct

prompt:
  task: >
    Classify the sentiment of the following text.
    Respond with exactly one word: positive, negative, or neutral.

input_features:
  - name: text
    type: text

output_features:
  - name: sentiment
    type: text
    decoder:
      type: text_parser
      regex: "(positive|negative|neutral)"

generation:
  max_new_tokens: 10
  temperature: 0.0
  do_sample: false

backend:
  type: local
```

### Example

| Input | Unconstrained output | Constrained output |
|-------|---------------------|-------------------|
| `"I loved this product!"` | `"The sentiment is positive."` | `positive` |
| `"The service was terrible."` | `"Negative sentiment."` | `negative` |
| `"The movie was okay."` | `"The text expresses neutral or slightly negative..."` | `neutral` |

Without a constraint the model often prepends or appends prose. The regex guarantees a single clean label.

## Grammar constraints

For more complex formats that can't be captured by a single regex, Ludwig supports EBNF grammars:

```yaml
output_features:
  - name: output
    type: text
    decoder:
      type: text_parser
      grammar: |
        root   ::= object
        object ::= "{" members "}"
        members ::= pair ("," pair)*
        pair   ::= string ":" value
        value  ::= string | number | "true" | "false" | "null"
        string ::= "\"" [^"]* "\""
        number ::= [0-9]+
```

Grammar constraints are more expressive than regexes but compile more slowly. Use them when you need recursive structures that can't be expressed as a regular language.

## Logits extraction

Ludwig can return the raw logits (pre-softmax vocabulary scores) for each generated token alongside the prediction. Logits are useful for:

- Token-level confidence scoring
- Calibration and uncertainty quantification
- Downstream reranking or ensemble methods

Enable logits collection with `output_logits: true` in the output feature:

```yaml
output_features:
  - name: response
    type: text
    output_logits: true
```

Then call `predict` with `collect_predictions=True`:

```python
from ludwig.api import LudwigModel
import pandas as pd

model = LudwigModel(config="config.yaml")
preds, output_df, _ = model.predict(
    dataset=pd.DataFrame({"text": ["Is the sky blue?"]}),
    collect_predictions=True,
)

print(preds["response_predictions"].iloc[0])
# -> "Yes"

logits = output_df["response_logits"].iloc[0]
# logits is a 2D array of shape (num_generated_tokens, vocab_size)
```

## Combining with fine-tuning

Constrained decoding works with both zero-shot inference and fine-tuned models. For a fine-tuned model, add the constraint to the output feature decoder in the same config used for prediction:

```python
model = LudwigModel.load("/path/to/finetuned_model")
# Override the decoder config at prediction time if needed
preds, _, _ = model.predict(dataset=df)
```

## Performance considerations

- **JSON schema** constraints carry a small overhead per token (~5–10 ms on CPU) because the schema automaton must be advanced at each step. The overhead is negligible for GPU-based inference.
- **Regex** constraints are fast — typically under 1 ms per token.
- **Grammar** constraints have a higher compile cost at startup. Cache the compiled grammar across requests.
- Constrained decoding is compatible with vLLM serving (see [Serving Ludwig LLMs with vLLM](../serving.md)).

## Interactive notebook

An interactive walkthrough of all examples above is available in the Ludwig examples repository:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ludwig-ai/ludwig/blob/main/examples/llm_structured_output/structured_output.ipynb)

The notebook includes a side-by-side comparison of constrained vs unconstrained outputs and works on a free Colab T4 GPU.

## See also

- [Serving Ludwig LLMs with vLLM](../serving.md)
- [LLM configuration reference](../../configuration/large_language_model.md)
- [In-context learning](in_context_learning.md)
- [Fine-tuning LLMs](finetuning.md)
