| image_path              | question                                  | answer |
| ----------------------- | ----------------------------------------- | ------ |
| imdata/image_000001.jpg | Is there snow on the mountains?           | yes    |
| imdata/image_000002.jpg | What color are the wheels                 | blue   |
| imdata/image_000003.jpg | What kind of utensil is in the glass bowl | knife  |

```
ludwig experiment \
--dataset vqa.csv \
  --config config.yaml
```

With `config.yaml`:

```yaml
input_features:
    -
        name: image_path
        type: image
        encoder: stacked_cnn
    -
        name: question
        type: text
        encoder: parallel_cnn

output_features:
    -
        name: answer
        type: text
        decoder: generator
        cell_type: lstm
        loss:
            type: sampled_softmax_cross_entropy
```
