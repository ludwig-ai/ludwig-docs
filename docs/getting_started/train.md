To train a model with Ludwig, we first need to create a [configuration](/ludwig-docs/user_guide/configuration) file. This file provides at a minimum the input and output features of your model, but you can also expand upon it to include model architectures, training parameters, hyperparameter optimization, and more.

For now, let's use a basic config that just specifies the inputs and output and leaves the rest to Ludwig:

``` yaml title="iris.yaml"
input_features:
    - name: sepal_length_cm
      type: numerical
    - name: sepal_width_cm
      type: numerical
    - name: petal_length_cm
      type: numerical
    - name: petal_width_cm
      type: numerical
output_features:
    - name: class
      type: category
```

This config file tells Ludwig that we want to train a model the uses the *length* and *width* measurements as numerical **input features** to our model and the *class* as the single categorical **output feature**. 

Once you've created `iris.yaml` with the contents above, you're ready to train your first model:

=== "CLI"

    ``` sh
    ludwig train --config iris.yaml --dataset iris.csv
    ```

=== "Python"

    ``` python
    from ludwig.api import LudwigModel
    from ludwig.datasets import iris

    model = LudwigModel(config='iris.yaml')
    model.train(dataset=iris.load())
    ```
