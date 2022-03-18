| utterance                      | intent      | slots                             |
| ------------------------------ | ----------- | --------------------------------- |
| I want a pizza                 | order_food  | O O O B-Food_type                 |
| Book a flight to Boston        | book_flight | O O O O B-City                    |
| Book a flight at 7pm to London | book_flight | O O O O B-Departure_time O B-City |

```
ludwig experiment \
  --dataset nlu.csv \
  --config config.yaml
```

With `config.yaml`:

```yaml
input_features:
    -
        name: utterance
        type: text
        level: word
        encoder: rnn
        cell_type: lstm
        bidirectional: true
        num_layers: 2
        reduce_output: null
        preprocessing:
          word_tokenizer: space

output_features:
    -
        name: intent
        type: category
        reduce_input: sum
        num_fc_layers: 1
        output_size: 64
    -
        name: slots
        type: sequence
        decoder: tagger
```
