This is a complete example of training an image classification model on the MNIST
dataset.

These interactive notebooks follow the steps of this example:

**TODO: point notebook URL to ludwig-ai/ludwig-docs repo before PR merge**

- Ludwig CLI: [![Image Classification with Ludwig CLI](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jimthompson5802/ludwig-docs/blob/expanded-mnist-example-with-colab/docs/examples/mnist_colab_notebooks/MNIST_Classification_with_Ludwig_CLI.ipynb)

## Download the MNIST dataset
MNIST is a collection of gray-scale images of hand-written digits. This collection is made up of 60,000 images for training and 10,000 images for testing model performance.  Each image is 28 X 28 pixels in gray-scale.

```
ludwig datasets download mnist
```
This command will create a dataset `mnist_dataset.csv` in the current directory.  

The columns in the dataset are

|column| description |
|------|-------------|
|image_path|file path string for the image|
|label|single digit 0 to 9 indicating what digit is shown in the image|
|split|integer value indicating a training example (0) or test example (2)|

## Train
The Ludwig configuration file describes the machine learning task.  There is a vast array of options to control the learning process.  This example only covers a small fraction of the options.  Only the options used in this example are described.  Please refer to the [Configuration Section](../../configuration) for all the details. 

First it defines the `input_features`.  For the image feature, the configuration specifies the type of neural network architecture to process the image.  In this example it is a two layer [Stacked Convolutional Neural Network](../../configuration/features/image_features/#convolutional-stack-encoder-stacked_cnn) followed by a fully connected layer with dropout regularization.  

Next the `output_features` are defined.  In this case, there is only one output feature called `label`.  This is a [categorical feature](../../configuration/features/category_features/) that indicates the digit the image represents, 0, 1, 2, ..., 9. 

The last section in this configuration file describes options for how the the [`trainer`](../../configuration/trainer/) will operate.  In this example the `trainer` will process the training data for 5 epochs.

With `config.yaml`:

```yaml
input_features:
- name: image_path
  type: image
  encoder: stacked_cnn
  conv_layers:
    - num_filters: 32
      filter_size: 3
      pool_size: 2
      pool_stride: 2
    - num_filters: 64
      filter_size: 3
      pool_size: 2
      pool_stride: 2
      dropout: 0.4
  fc_layers:
    - output_size: 128
      dropout: 0.4

output_features:
 - name: label
   type: category

trainer:
  epochs: 5
```

```
ludwig train \
  --dataset mnist_dataset.csv \
  --config config.yaml
```
