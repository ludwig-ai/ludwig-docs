To train a model with Ludwig, we first need to create a [Ludwig configuration](./../configuration/index.md). The config specifies input features, output features, preprocessing, model architecture, training loop, hyperparameter search, and backend infrastructure -- everything that's needed to build, train, and evaluate a model.

At a minimum, the config must specify the model's input and output features.

For now, let's use a basic config that just specifies the inputs and output and leaves the rest to Ludwig:

``` yaml title="rotten_tomatoes.yaml"
input_features:
    - name: genres
      type: set
      preprocessing:
          tokenizer: comma
    - name: content_rating
      type: category
    - name: top_critic
      type: binary
    - name: runtime
      type: number
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
    import pandas

    df = pandas.read_csv('rotten_tomatoes.csv')
    model = LudwigModel(config='rotten_tomatoes.yaml')
    results = model.train(dataset=df)
    ```

=== "Docker CLI"

    ``` sh
    mkdir rotten_tomatoes_data
    mv rotten_tomatoes.yaml ./rotten_tomatoes_data
    mv rotten_tomatoes.csv ./rotten_tomatoes_data
    docker run -t -i --mount type=bind,source={absolute/path/to/rotten_tomatoes_data},target=/rotten_tomatoes_data ludwigai/ludwig train --config /rotten_tomatoes_data/rotten_tomatoes.yaml --dataset /rotten_tomatoes_data/rotten_tomatoes.csv --output_directory /rotten_tomatoes_data
    ```

!!! note

    In this example, we encoded the text feature with an `embed` encoder, which assigns an embedding for each word and sums
    them. Ludwig provides many options for [tokenizing](../../configuration/preprocessing#tokenizers) and [embedding](../../configuration/features/sequence_features#sequence-input-features-and-encoders) text like with CNNs, RNNs, Transformers, and pretrained models such as BERT or GPT-2 (provided through [huggingface](https://huggingface.co/docs/transformers/index)). Using a different text encoder is simple as changing encoder option in the config from `embed` to `bert`. Give it a try!

    ```yaml
    input_features:
        - name: genres
          type: set
          preprocessing:
              tokenizer: comma
        - name: content_rating
          type: category
        - name: top_critic
          type: binary
        - name: runtime
          type: number
        - name: review_content
          type: text
          encoder: bert
    output_features:
        - name: recommended
          type: binary
    ```

Ludwig is very flexible. Users can configure just about any parameter in their models including training parameters, preprocessing parameters, and more, directly from the configuration. Check out the [config documentation](./../configuration/index.md) for the full list of parameters available in the configuration.
