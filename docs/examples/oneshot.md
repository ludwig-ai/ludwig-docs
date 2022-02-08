This example can be considered a simple baseline for one-shot learning on the [Omniglot](https://github.com/brendenlake/omniglot) dataset.
The task is, given two images of two handwritten characters, recognize if they are two instances of the same character or not.

| image_path_1                     | image_path_2                     | similarity |
| -------------------------------- | -------------------------------- | ---------- |
| balinese/character01/0108_13.png | balinese/character01/0108_18.png | 1          |
| balinese/character01/0108_13.png | balinese/character08/0115_12.png | 0          |
| balinese/character01/0108_04.png | balinese/character01/0108_08.png | 1          |
| balinese/character01/0108_11.png | balinese/character05/0112_02.png | 0          |

```
ludwig experiment \
--dataset balinese_characters.csv \
  --config_file config.yaml
```

With `config.yaml`:

```yaml
input_features:
    -
        name: image_path_1
        type: image
        encoder: stacked_cnn
        preprocessing:
          width: 28
          height: 28
          resize_image: true
    -
        name: image_path_2
        type: image
        encoder: stacked_cnn
        preprocessing:
          width: 28
          height: 28
          resize_image: true
        tied_weights: image_path_1

combiner:
    type: concat
    num_fc_layers: 2
    fc_size: 256

output_features:
    -
        name: similarity
        type: binary
```