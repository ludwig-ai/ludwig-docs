| review                          | sentiment |
| ------------------------------- | --------- |
| The movie was fantastic!        | positive  |
| Great acting and cinematography | positive  |
| The acting was terrible!        | negative  |

```
ludwig experiment \
  --dataset sentiment.csv \
  --config_file config.yaml
```

With `config.yaml`:

```yaml
input_features:
    -
        name: review
        type: text
        level: word
        encoder: parallel_cnn

output_features:
    -
        name: sentiment
        type: category
```
