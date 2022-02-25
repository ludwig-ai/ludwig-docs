To train a model with Ludwig, we first need to create a [configuration](/ludwig-docs/user_guide/configuration) file. This file provides at a minimum the input and output features of your model, but you can also expand upon it to include model architectures, training parameters, hyperparameter optimization, and more.

For now, let's use a basic config that just specifies the inputs and output and leaves the rest to Ludwig:

``` yaml title="rotten_tomatoes.yaml"
input_features:
    - name: genres
      type: set
    - name: content_rating
      type: category
    - name: top_critic
      type: binary
    - name: runtime
      type: numerical
    - name: review_content
      type: text
      encoder: embed
output_features:
    - name: recommended
      type: binary
```

This config file tells Ludwig that we want to train a model that uses the following **input features**:
- The *genres* associated with the movie will be used as a **set feature** 
- The movie's *content rating* will be used as a **category feature**
- Whether the review was done by a *top critic* or not will be used as a **binary feature**
- The movie's *runtime* will be used as a **number feature**
- The *review content* will be used as **text feature**

This config file also tells Ludwig that we want our model to have the following **output features**:

- The recommendation of whether to watch the movie or not will be output as a **binary feature**


Once you've created the `rotten_tomatoes.yaml` file with the contents above, you're ready to train your first model:

=== "CLI"

    ``` sh
    ludwig train --config rotten_tomatoes.yaml --dataset rotten_tomatoes.csv
    ```

=== "Python"

    ``` python
    from ludwig.api import LudwigModel

    model = LudwigModel(config='rotten_tomatoes.yaml')
    results = model.train(dataset=df)
    ```
