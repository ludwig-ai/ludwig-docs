After the model has been trained, it can be used to predict the target output features on new data.

Let's make a small test dataset `test.csv` containing only input features for the model:

``` title="test.csv"
sepal_length_cm,sepal_width_cm,petal_length_cm,petal_width_cm
4.9,3.0,1.4,0.2
4.7,3.2,1.3,0.2
4.6,3.1,1.5,0.2
5.0,3.6,1.4,0.2
5.4,3.9,1.7,0.4
4.6,3.4,1.4,0.3
5.0,3.4,1.5,0.2
4.4,2.9,1.4,0.2
4.9,3.1,1.5,0.1
```

=== "CLI"

    ``` sh
    ludwig predict --model_path results/experiment_run/model --dataset test.csv
    ```

=== "Python"

    ``` python
    from ludwig.api import LudwigModel

    model = LudwigModel.load('results/experiment_run/model')
    results = model.predict(dataset='test.csv')
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