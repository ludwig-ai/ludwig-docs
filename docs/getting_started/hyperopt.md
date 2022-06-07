After training our first model and using it to predict new data with reasonable accuracy, how can we make the model
better?

Ludwig can perform hyperparameter optimization by simply adding `hyperopt` to the Ludwig config.

```yaml title="rotten_tomatoes.yaml"
input_features:
    - name: genres
      type: set
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
hyperopt:
  goal: maximize
  output_feature: recommended
  metric: accuracy
  split: validation
  parameters:
    training.learning_rate:
      space: loguniform
      lower: 0.0001
      upper: 0.1
    training.optimizer.type:
      space: choice
      categories: [sgd, adam, adagrad]
    review_content.embedding_size:
      space: choice
      categories: [128, 256]
  search_alg:
    type: variant_generator
  executor:
    num_samples: 10
```

In this example we have specified a basic hyperopt config with the following specifications:

* We have set the `goal` to maximize the **accuracy** metric on the **validation** split
* The parameters we are optimizing are the **learning rate**, the **optimizer type**, and the **embedding_size** of text representation to use.
  * When optimizing **learning rate** we are randomly selecting values on a *log* scale between 0.0001 and 0.1.
  * When optimizing the **optimizer type**, we randomly select the optimizer from *sgd*, *adam*, and *adagrad* optimizers.
  * When optimizing the **embedding_size** of text representation we randomly chose between 128 or 256.
* We set hyperopt `executor` to use Ray Tune's `variant_generator` search algorithm and generates 10 random hyperparameter combinations from the search space we defined.  The execution will locally run trials in parallel.
  * Ludwig supports advanced hyperparameter sampling algorithms like Bayesian optimization and genetical algorithms. See [this guide](../../configuration/hyperparameter_optimization/#hyperopt-configuration-parameters) for details.

The hyperparameter optimization strategy is run using the ludwig hyperopt command:

=== "CLI"

    ```sh
    ludwig hyperopt --config rotten_tomatoes.yaml --dataset rotten_tomatoes.csv
    ```

=== "Python"

    ```python
    from ludwig.hyperopt.run import hyperopt
    import pandas

    df = pandas.read_csv('rotten_tomatoes.csv')
    results = hyperopt(config='rotten_tomatoes.yaml', dataset=df)
    ```

=== "Docker CLI"

    ```sh
    docker run -t -i --mount type=bind,source={absolute/path/to/rotten_tomatoes_data},target=/rotten_tomatoes_data ludwigai/ludwig hyperopt --config /rotten_tomatoes_data/rotten_tomatoes.yaml --dataset /rotten_tomatoes_data/rotten_tomatoes.csv
    ```

Every parameter within the config can be tuned using hyperopt. Refer to the full [hyperopt guide](../../configuration/hyperparameter_optimization) to learn more.
