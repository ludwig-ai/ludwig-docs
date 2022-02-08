This is a complete example of training an image classification model on the MNIST
dataset.

## Download the MNIST dataset

```
git clone https://github.com/myleott/mnist_png.git
cd mnist_png/
tar -xf mnist_png.tar.gz
cd mnist_png/
```

## Create train and test CSVs

Open python shell in the same directory and run this:

```
import os
for name in ['training', 'testing']:
    with open('mnist_dataset_{}.csv'.format(name), 'w') as output_file:
        print('=== creating {} dataset ==='.format(name))
        output_file.write('image_path,label\n')
        for i in range(10):
            path = '{}/{}'.format(name, i)
            for file in os.listdir(path):
                if file.endswith(".png"):
                    output_file.write('{},{}\n'.format(os.path.join(path, file), str(i)))

```

Now you should have `mnist_dataset_training.csv` and `mnist_dataset_testing.csv`
containing 60000 and 10000 examples correspondingly and having the following format

| image_path           | label |
| -------------------- | ----- |
| training/0/16585.png | 0     |
| training/0/24537.png | 0     |
| training/0/25629.png | 0     |

## Train a model

From the directory where you have virtual environment with ludwig installed:

```
ludwig train \
  --training_set <PATH_TO_MNIST_DATASET_TRAINING_CSV> \
  --test_set <PATH_TO_MNIST_DATASET_TEST_CSV> \
  --config_file config.yaml
```

With `config.yaml`:

```yaml
input_features:
    -
        name: image_path
        type: image
        encoder: stacked_cnn
        conv_layers:
            -
                num_filters: 32
                filter_size: 3
                pool_size: 2
                pool_stride: 2
            -
                num_filters: 64
                filter_size: 3
                pool_size: 2
                pool_stride: 2
                dropout: 0.4
        fc_layers:
            -
                fc_size: 128
                dropout: 0.4

output_features:
    -
        name: label
        type: category

training:
    early_stop: 5
```

# Image Captioning

| image_path                | caption                   |
| ------------------------- | ------------------------- |
| imagenet/image_000001.jpg | car driving on the street |
| imagenet/image_000002.jpg | dog barking at a cat      |
| imagenet/image_000003.jpg | boat sailing in the ocean |

```
ludwig experiment \
--dataset image captioning.csv \
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
        name: caption
        type: text
        level: word
        decoder: generator
        cell_type: lstm
```
