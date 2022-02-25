After the model has been trained, it can be used to predict the target output features on new data.

We've created a small test dataset containing input features for 10 movie reviews that we can use for testing. Download this dataset <a id="raw-url" href="https://raw.githubusercontent.com/connor-mccorm/ludwig-ai/ludwig-docs/tree/master/docs/rotten_tomatoes_test.csv">here</a>.

Now, let's make some predictions on the test dataset:

=== "CLI"

    ``` sh
    ludwig predict --model_path results/experiment_run/model --dataset rotten_tomatoes_test.csv
    ```

=== "Python"

    ``` python
    from ludwig.api import LudwigModel

    model = LudwigModel.load('results/experiment_run/model')
    results = model.predict(dataset='rotten_tomatoes_test.csv')
    ```

Running this command will return model predictions.

If your dataset also contains ground truth values of the target outputs, you can compare them to the predictions obtained from the model to evaluate the model performance.

```
ludwig evaluate --dataset path/to/data.csv --model_path /path/to/model
```

This will produce evaluation performance statistics that can be visualized by the `visualize` tool, which can also be used to compare performances and predictions of different models, for instance:

```
ludwig visualize --visualization compare_performance --test_statistics path/to/test_statistics_model_1.json path/to/test_statistics_model_2.json
```

will return a bar plot comparing the models on different metrics:

![Performance Comparison](images/compare_performance.png "Performance Comparison")

A handy `ludwig experiment` command that performs training and prediction one after the other is also available.