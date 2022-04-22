| utterance                                                                        | tag                                                             |
| -------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| Blade Runner is a 1982 neo-noir science fiction film directed by Ridley Scott    | Movie Movie O O Date O O O O O O Person Person                  |
| Harrison Ford and Rutger Hauer starred in it                                     | Person Person O Person person O O O                             |
| Philip Dick 's novel Do Androids Dream of Electric Sheep ? was published in 1968 | Person Person O O Book Book Book Book Book Book Book O O O Date |

```
ludwig experiment \
  --dataset sequence_tags.csv \
  --config config.yaml
```

With `config.yaml`:

```yaml
input_features:
    -
        name: utterance
        type: text
        encoder: rnn
        cell_type: lstm
        reduce_output: null
        preprocessing:
          tokenizer: space

output_features:
    -
        name: tag
        type: sequence
        decoder: tagger
```
