After the model has been trained, it can be used to predict the target output features on new data.

We've created a small test dataset containing input features for 10 movie reviews that we can use for testing. Download the test dataset [here](https://github.com/ludwig-ai/ludwig-docs/raw/getting-started/docs/data/rotten_tomatoes_test.csv).

Let's make some predictions on the test dataset!

=== "CLI"

    ``` sh
    ludwig predict --model_path results/experiment_run/model --dataset rotten_tomatoes_test.csv
    ```

=== "Python"

    ``` python
    # This step can be skipped if you are working in a notebook, and you can simply 
    # re-use the model created in the training section.
    model = LudwigModel.load('results/experiment_run/model')

    predictions, _ = model.predict(dataset='rotten_tomatoes_test.csv')
    predictions.head()
    ```

=== "Docker CLI"

    ``` sh
    docker run -t -i --mount type=bind,source={absolute/path/to/rotten_tomatoes_data},target=/rotten_tomatoes_data ludwigai/ludwig predict --model_path /rotten_tomatoes_data/results/experiment_run/model --dataset /rotten_tomatoes_data/rotten_tomatoes.csv
    ```

Running this command will return model predictions. Your results should look something like this:

| Index |          recommended_probabilities          | recommended_predictions | recommended_probabilities_False | recommended_probabilities_True | recommended_probability |
| :---: | :-----------------------------------------: | :---------------------: | :-----------------------------: | :----------------------------: | :---------------------: |
|   0   |  [0.09741002321243286, 0.9025899767875671]  |          True           |            0.097410             |            0.902590            |        0.902590         |
|   1   |  [0.6842662990093231, 0.3157337009906769]   |          False          |            0.684266             |            0.315734            |        0.684266         |
|   2   | [0.026504933834075928, 0.973495066165 9241] |          True           |            0.026505             |            0.973495            |        0.973495         |
|   3   | [0.022977590560913086, 0.9770224094390869]  |          True           |            0.022978             |            0.977022            |        0.977022         |
|   4   | [0.9472369104623795, 0.052763089537620544]  |          False          |            0.947237             |            0.052763            |        0.947237         |

A handy [`ludwig experiment`](../../user_guide/api/LudwigModel/#experiment) CLI command is also available. This one command performs training and then prediction using the checkpoint with the best validation metric.

In addition to predictions, Ludwig also computes a suite of evaluation metrics, depending on the output feature's type.
The exact metrics that are computed for each output feature type can be found [here](../../configuration/features/supported_data_types).

!!! note

    Non-loss evaluation metrics, like accuracy, require ground truth values of the target outputs.

=== "CLI"

    ``` sh
    ludwig evaluate --dataset path/to/data.csv --model_path /path/to/model
    ```

=== "Python"

    ``` python
    eval_stats, _, _ = model.evaluate(dataset='rotten_tomatoes_test.csv')
    ```

=== "Docker CLI"

    ``` sh
    cp rotten_tomatoes_test.csv ./rotten_tomatoes_data
    docker run -t -i --mount type=bind,source={absolute/path/to/rotten_tomatoes_data},target=/rotten_tomatoes_data ludwigai/ludwig evaluate --dataset /rotten_tomatoes_data/rotten_tomatoes_test.csv --model_path /rotten_tomatoes_data/results/experiment_run/model
    ```

Evaluation performance can be visualized using [`ludwig visualize`](../../user_guide/api/visualization/). This enables us to visualize metrics like for omparing performances and predictions across different models. For instance, if you have two models which you want to compare evaluation statistics for, you could use the following commands:

=== "CLI"

    ``` sh
    ludwig visualize --visualization compare_performance --test_statistics path/to/test_statistics_model_1.json path/to/test_statistics_model_2.json
    ```

=== "Python"

    ``` python
    from ludwig.visualize import compare_performance

    compare_performance([eval_stats_model_1, eval_stats_model_2])
    ```

=== "Docker CLI"

    ``` sh
    docker run -t -i --mount type=bind,source={absolute/path/to/rotten_tomatoes_data},target=/rotten_tomatoes_data ludwigai/ludwig visualize --visualization compare_performance --test_statistics /rotten_tomatoes_data/path/to/test_statistics_model_1.json /rotten_tomatoes_data/path/to/test_statistics_model_2.json
    ```

This will return a bar plot comparing the performance of each model on different metrics like the example below.

![Performance Comparison](https://github.com/ludwig-ai/ludwig-docs/blob/master/docs/images/compare_performance.png?raw=true)
