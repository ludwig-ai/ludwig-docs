| english                   | italian                   |
| ------------------------- | ------------------------- |
| Hello! How are you doing? | Ciao, come stai?          |
| I got promoted today      | Oggi sono stato promosso! |
| Not doing well today      | Oggi non mi sento bene    |

```
ludwig experiment \
  --dataset translation.csv \
  --config config.yaml
```

With `config.yaml`:

```yaml
input_features:
    -
        name: english
        type: text
        encoder: 
            type: rnn
            cell_type: lstm
            reduce_output: null
        preprocessing:
          tokenizer: english_tokenize

output_features:
    -
        name: italian
        type: text
        decoder: 
            type: generator
            cell_type: lstm
            attention: bahdanau
            reduce_input: null
        loss:
            type: softmax_cross_entropy
        preprocessing:
          tokenizer: italian_tokenize

training:
    batch_size: 96
```
