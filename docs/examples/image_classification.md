| image_path              | class |
| ----------------------- | ----- |
| images/image_000001.jpg | car   |
| images/image_000002.jpg | dog   |
| images/image_000003.jpg | boat  |

```
ludwig experiment \
  --dataset image_classification.csv \
  --config_file config.yaml
```

With `config.yaml`:

```yaml
input_features:
    -
        name: image_path
        type: image
        encoder: stacked_cnn

output_features:
    -
        name: class
        type: category
```
