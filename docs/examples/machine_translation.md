| english                   | italian                   |
| ------------------------- | ------------------------- |
| Hello! How are you doing? | Ciao, come stai?          |
| I got promoted today      | Oggi sono stato promosso! |
| Not doing well today      | Oggi non mi sento bene    |

```
ludwig experiment \
  --dataset translation.csv \
  --config_file config.yaml
```

With `config.yaml`:

```yaml
input_features:
    -
        name: english
        type: text
        level: word
        encoder: rnn
        cell_type: lstm
        reduce_output: null
        preprocessing:
          word_tokenizer: english_tokenize

output_features:
    -
        name: italian
        type: text
        level: word
        decoder: generator
        cell_type: lstm
        attention: bahdanau
        reduce_input: null
        loss:
            type: sampled_softmax_cross_entropy
        preprocessing:
          word_tokenizer: italian_tokenize

training:
    batch_size: 96
```
