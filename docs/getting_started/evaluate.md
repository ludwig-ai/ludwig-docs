After the model has been trained, it can be used to predict the target output features on new data.

We've created a small test dataset containing input features for 10 movie reviews that we can use for testing. Click this [link](https://github.com/ludwig-ai/ludwig-docs/raw/getting-started/docs/data/rotten_tomatoes_test.csv) to download the test dataset.

Now, let's make some predictions on the test dataset (Note: if you are working in a notebook, you can use the model created in the training section):

=== "CLI"

    ``` sh
    ludwig predict --model_path results/experiment_run/model --dataset rotten_tomatoes_test.csv
    ```

=== "Python"

    ``` python
    model = LudwigModel.load('results/experiment_run/model')
    
    predictions, _ = model.predict(dataset='rotten_tomatoes_test.csv')
    predictions.head()
    ```

Running this command will return model predictions. Your results should look something like this:


| Index |          recommended_probabilities          | recommended_predictions	 | recommended_probabilities_False | 	 recommended_probabilities_True | recommended_probability | 
|:-----:|:-------------------------------------------:|:------------------------:|:-------------------------------:|:--------------------------------:|:-----------------------:|
|  0	   |  [0.09741002321243286, 0.9025899767875671]  |           True           |            	0.097410            |            	0.902590             |        	0.902590        |  
|  1	   |  [0.6842662990093231, 0.3157337009906769]   |          False           |            0.684266             |            	0.315734             |        0.684266         |   
|  2	   | [0.026504933834075928, 0.973495066165 9241] |           True           |            0.026505             |            	0.973495             |        	0.973495        | 
|  3	   | [0.022977590560913086, 0.9770224094390869]  |           True           |            	0.022978            |            	0.977022             |        	0.977022        | 
|  4	   | [0.9472369104623795, 0.052763089537620544]  |          False           |            	0.947237            |            	0.052763             |        	0.947237        |

A handy [`ludwig experiment`](https://ludwig-ai.github.io/ludwig-docs/0.4/user_guide/api/LudwigModel/#experiment) command that performs training and prediction one after the other is also available.

If your dataset also contains ground truth values of the target outputs, you can compare them to the predictions obtained from the model to evaluate the model performance.


=== "CLI"

    ```
    ludwig evaluate --dataset path/to/data.csv --model_path /path/to/model
    ```

=== "Python"
    
    ``` python
    eval_stats, _, _ = model.evaluate(dataset='rotten_tomatoes_test.csv')
    ```

This will produce evaluation performance statistics that can be visualized by using [`ludwig visualize`](https://ludwig-ai.github.io/ludwig-docs/0.4/user_guide/api/visualization/), which can also be used to compare performances and predictions of different models. For instance, if you have two models which you want to compare evaluation statistics for, you could use the following commands:

=== "CLI"

    ```
    ludwig visualize --visualization compare_performance --test_statistics path/to/test_statistics_model_1.json path/to/test_statistics_model_2.json
    ```

=== "Python"

    ``` python
    from ludwig.visualize import compare_performance
    
    compare_performance([eval_stats_model_1, eval_stats_model_2])
    ```

This will return a bar plot comparing the performance of each model on different metrics like the example below.

![Performance Comparison](https://github.com/ludwig-ai/ludwig-docs/blob/master/docs/images/compare_performance.png?raw=true)
