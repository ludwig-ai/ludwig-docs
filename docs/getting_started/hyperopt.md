Ludwig supports hyperparameter optimization using the hyperopt module.

Let's update our config file in order to utilize hyperopt:

```yaml title="rotten_tomatoes.yaml"
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
hyperopt:
  goal: maximize
  output_feature: recommended
  metric: accuracy
  split: validation
  parameters:
    training.learning_rate:
      type: float
      low: 0.0001
      high: 0.1
      steps: 4
      scale: log
    training.optimizer.type:
      type: category
      values: [sgd, adam, adagrad]
    review_content.level:
      type: category
      values: [word, char]
  sampler:
    type: random
  executor:
    type: parallel
```

In this example we have specified a basic hyperopt config with the following specifications:

- We have set the **goal** to maximize the **accuracy** metric on the **validation** split
- The parameters we are optimizing are the **learning rate**, the **optimizer type**, and the **level** of text representation to use.
  - When optimizing **learning rate** we are testing *four* values on a *log* scale between 0.0001 and 0.1 ([0.0001, 0.001, 0.01, 0.1]).
  - When optimizing the **optimizer type**, we are testing the *sgd*, *adam*, and *adagrad* optimizers
  - When optimizing the **level** of text representation to use, we are testing representations at the *word* level and the *char* level 
- We set the hyperopt **sampler** to use the random sampler. This selects 10 random hyperparameter combinations from the search space by default.
- We set the hyperopt **executor** to use the parallel executor which will use 2 workers by default to optimize in parallel.

The hyperparameter optimization strategy is run using the ludwig hyperopt command:

=== "CLI"

    ``` sh
    ludwig hyperopt --config rotten_tomatoes.yaml --dataset rotten_tomatoes.csv
    ```

=== "Python"

    ``` python
    from ludwig.hyperopt.run import hyperopt

    results = hyperopt.train(config='rotten_tomatoes.yaml', dataset=df)
    ```

Every parameter within the config can be tuned using hyperopt. If you wish to tune other parameters as well, the full configuration specification can be found [here](docs/configuration/hyperparameter_optimization.md).