| image_path              | tags          |
| ----------------------- | ------------- |
| images/image_000001.jpg | car man       |
| images/image_000002.jpg | happy dog tie |
| images/image_000003.jpg | boat water    |

```
ludwig experiment \
--dataset image_data.csv \
  --config config.yaml
```

With `config.yaml`:

```yaml
input_features:
    -
        name: image_path
        type: image
        encoder: 
            type: stacked_cnn

output_features:
    -
        name: tags
        type: set
```
