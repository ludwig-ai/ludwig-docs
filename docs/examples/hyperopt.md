This is a complete example of Ludwig's hyperparameter optimization capability.

These interactive notebooks follow the steps of this example:

- Ludwig CLI: [![Hyperparameter Optimization CLI](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ludwig-ai/ludwig-docs/blob/hyperopt-tutorial/docs/examples/hyperopt_notebooks/hyperopt_notebook_cli.ipynb)
- Ludwig Python API: [![Hyperparameter Optimization API](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ludwig-ai/ludwig-docs/blob/hyperopt-tutorial/docs/examples/hyperopt_notebooks/hyperopt_notebook_api.ipynb)

## Download the Adult Census Income dataset

[Adult Census Income](https://archive.ics.uci.edu/ml/datasets/adult) is an extract of 1994 Census data for predicting whether a person's income exceeds $50K per year.  The data set consists of over 49K records with 14 attributes with missing data.

```shell
ludwig datasets download adult_census_income
```

This command will create a dataset `adult_census_income.csv` in the current directory.

The columns in the dataset are

| column         | description                                                  |
| -------------- | ------------------------------------------------------------ |
| age            | numeric variable, age of person                              |
| workclass      | categorical variable, Type of empolyment                     |
| fnlwgt         | numeric variable, no defintion                               |
| education      | categorical variable, education level                        |
| education-num  | nmeric variable, no definition                               |
| marital-status | categorical variable, marital status                         |
| occupation     | categorical variable, occupation                             |
| relationship   | categorical variable, Relationship to household              |
| race           | categorical variable, race                                   |
| sex            | categorical variable, gender                                 |
| capital-gain   | numeric variable, no definition                              |
| capital-loss   | numeric variable, no definition                              |
| hours-per-week | numeric variable, hours worked per week                      |
| native-country | categorical variable, Country of origin                      |
| income         | binary variable, " <=50K" or " >50K"                         |
| split          | numeric variable, indicating data split training(0), test(2) |


## Setup for hyperparameter optimization run

Hyperparameter optimization is defined with the [`hyperopt` section of the Ludwig configuration specification](../../configuration/hyperparameter_optimization/).  


=== "cli"

    [ludwig hyperopt](../../user_guide/)

    ```yaml

    ```

=== "python"

    ```python
    # define model configuration
    config = {
        'combiner': ... ,
        'input_features': ... ,
        'output_features': ... ,
        'preprocessing': ...,
        'trainer':... ,
    
        # hyperopt specification 
        'hyperopt':  {
            # specify parameters for the Ray Tune to executor to run the hyperparameter optimization
            'executor': {'type': 'ray', ... },
            # specify Ray Tune search algorithm to use
            'search_alg': {... },
            # hyperparameter search space for the optimization
            'parameters': {...},
            # minimize or maximize the metric score
            'goal': ...,
            # metric score to optimize
            'metric': ...,
            # name of the output feature
            'output_feature': ...,
        }
    }
    ```

## Hyperparameter Search Space Specification

For this example, we want to determine the effect of Ludwig's Trainer's `learning_rate` and `num_fc_layers` of the `income` output feature on model's `roc_auc` metric.  To do this we will use two different hyperparameter optimization approaches: Random Search and Grid Search.

### Random Search

=== "cli"


    ```yaml
    ludwig hyperopt ...
    ```

=== "python"


    ```python
     'hyperopt': {
        'executor': {'type': 'ray', 'num_samples': 16, },
        'goal': 'maximize',
        'metric': 'roc_auc',
        'output_feature': 'income',
        'parameters': {
            'income.num_fc_layers': {
                'space': 'randint',
                'lower': 2,
                'upper': 9
            },
            'trainer.learning_rate': {
                'space': 'loguniform',
                'lower': 0.001,
                'upper': 0.1}
            },
        'search_alg': {'type': 'variant_generator', 'random_state': 1919, }
    },
    ```

### Grid Search


=== "cli"

    ```yaml
    
    ```

=== "python"

    ```python
    'hyperopt': {
        'executor': {'num_samples': 1, 'type': 'ray'},
        'goal': 'maximize',
        'metric': 'roc_auc',
        'output_feature': 'income',
        'parameters': {
            'income.num_fc_layers': {'space': 'grid_search', 'values': [2, 4, 6, 8]},
            'trainer.learning_rate': {'space': 'grid_search', 'values': [0.001, 0.003, 0.007, 0.01]}},
        'search_alg': {'type': 'variant_generator', 'random_state': 1919, }
    },
    ```

## Run Hyperparameter Optimization

=== "cli"

    [`ludwig hyperopt` command](../../user_guide/command_line_interface/#hyperopt)

    ```shell
    ludwig hyperopt ...
    ```

=== "python"

    [hyperopt() method](../../user_guide/api/LudwigModel/#hyperopt)

    ```python
    model.hyperopt()
    ```



## Visualize Hyperparameter Optimization Results

TO BE  UPDATED

=== "cli"

    [`ludwig visualize confusion_matrix` command](../../user_guide/visualizations/#confusion-matrix)

    ```shell
    ludwig visualize --visualization confusion_matrix \
                      --ground_truth_metadata results/experiment_run/model/training_set_metadata.json \
                      --test_statistics test_results/test_statistics.json \
                      --output_directory visualizations \
                      --file_format png
    ```

=== "python"

    [`visualize.confusion_matrix()` function](../../user_guide/api/visualization/#confusion_matrix)

    ```python
    # Visualizes confusion matrix, which gives an overview of classifier performance
    # for each class.
    from ludwig.visualize import confusion_matrix

    confusion_matrix(
      [test_stats],
      model.training_set_metadata,
      'label',
      top_n_classes=[5],
      model_names=[''],
      normalize=True,
    )
    ```

![confusion matrix and entropy](mnist_colab_notebooks/images/mnist_confusion_matrix_and_entropy.png)

Display Learning Curves plots.

=== "cli"

    [`ludwig visualize learning_curves` command](../../user_guide/visualizations/#learning-curves)

    ```shell
    ludwig visualize --visualization learning_curves \
                      --ground_truth_metadata results/experiment_run/model/training_set_metadata.json \
                      --training_statistics results/experiment_run/training_statistics.json \
                      --file_format png \
                      --output_directory visualizations
    ```

=== "python"

    [`visualize.learning_curves()` function](../../ser_guide/api/visualization/#learning_curves)

    ```python
    # Visualizes learning curves, which show how performance metrics changed over
    # time during training.
    from ludwig.visualize import learning_curves

    learning_curves(train_stats, output_feature_name='label')
    ```

![confusion learning curves](mnist_colab_notebooks/images/mnist_learning_curves.png)

## Predictions

Generate predictions from test dataset.

=== "cli"

    [`ludwig predict` command](../../user_guide/command_line_interface/#predict)

    ```shell
    ludwig predict --model_path results/experiment_run/model \
                    --dataset mnist_dataset.csv \
                    --split test \
                    --output_directory predictions
    ```

=== "python"

    [`predict()` method](../../user_guide/api/LudwigModel/#predict)

    ```python
    predictions, output_directory = model.predict(test_df)
    ```

Sample test images displaying true("label") and predicted("pred") labels.
![mnist sample predictions](mnist_colab_notebooks/images/mnist_sample_predictions.png)
