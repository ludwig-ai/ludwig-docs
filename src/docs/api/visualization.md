# Module functions

----

## learning_curves


```python
ludwig.visualize.learning_curves(
  train_stats_per_model,
  output_feature_name,
  model_names=None,
  output_directory=None,
  file_format='pdf'
)
```


Show how model metrics change over training and validation data epochs.

For each model and for each output feature and metric of the model,
it produces a line plot showing how that metric changed over the course
of the epochs of training on the training and validation sets.

__Inputs__


- __train_stats_per_model__ (List[dict]): list containing dictionary of
training statistics per model.
- __output_feature_name__ (Union[str, `None`]): name of the output feature
to use for the visualization.  If `None`, use all output features.
- __model_names__ (Union[str, List[str]], default: `None`): model name or
list of the model names to use as labels.
- __output_directory__ (str, default: `None`): directory where to save
plots. If not specified, plots will be displayed in a window
- __file_format__ (str, default: `'pdf'`): file format of output plots -
`'pdf'` or `'png'`.

__Return__

- __return__ (None):
 
----

## compare_performance


```python
ludwig.visualize.compare_performance(
  test_stats_per_model,
  output_feature_name,
  model_names=None,
  output_directory=None,
  file_format='pdf'
)
```


Produces model comparison barplot visualization for each overall metric

For each model (in the aligned lists of test_statistics and model_names)
it produces bars in a bar plot, one for each overall metric available
in the test_statistics file for the specified output_feature_name.

__Inputs__


- __test_stats_per_model__ (List[dict]): dictionary containing evaluation
performance statistics.
- __output_feature_name__ (Union[str, `None`]): name of the output feature
to use for the visualization.  If `None`, use all output features.
- __model_names__ (Union[str, List[str]], default: `None`): model name or
list of the model names to use as labels.
- __output_directory__ (str, default: `None`): directory where to save
plots. If not specified, plots will be displayed in a window
- __file_format__ (str, default: `'pdf'`): file format of output plots -
`'pdf'` or `'png'`.

__Return__


- __return__ (None):
 
----

## compare_classifiers_performance_from_prob


```python
ludwig.visualize.compare_classifiers_performance_from_prob(
  probabilities_per_model,
  ground_truth,
  top_n_classes,
  labels_limit,
  model_names=None,
  output_directory=None,
  file_format='pdf'
)
```


Produces model comparison barplot visualization from probabilities.

For each model it produces bars in a bar plot, one for each overall metric
computed on the fly from the probabilities of predictions for the specified
`model_names`.

__Inputs__


- __probabilities_per_model__ (List[numpy.array]): list of model
probabilities.
- __ground_truth__ (numpy.array): numpy.array containing ground truth data,
which are the numeric encoded values the category.
- __top_n_classes__ (List[int]): list containing the number of classes
to plot.
- __labels_limit__ (int): upper limit on the numeric encoded label value.
Encoded numeric label values in dataset that are higher than
`label_limit` are considered to be "rare" labels.
- __model_names__ (Union[str, List[str]], default: `None`): model name or
list of the model names to use as labels.
- __output_directory__ (str, default: `None`): directory where to save
plots. If not specified, plots will be displayed in a window
- __file_format__ (str, default: `'pdf'`): file format of output plots -
`'pdf'` or `'png'`.

__Return__


- __return__ (None):
 
----

## compare_classifiers_performance_from_pred


```python
ludwig.visualize.compare_classifiers_performance_from_pred(
  predictions_per_model,
  ground_truth,
  metadata,
  output_feature_name,
  labels_limit,
  model_names=None,
  output_directory=None,
  file_format='pdf'
)
```


Produces model comparison barplot visualization from predictions.

For each model it produces bars in a bar plot, one for each overall metric
computed on the fly from the predictions for the specified
`model_names`.

__Inputs__


- __predictions_per_model__ (List[list]): list containing the model
predictions for the specified output_feature_name.
- __ground_truth__ (numpy.array): numpy.array containing ground truth data,
which are the numeric encoded values the category.
- __metadata__ (dict): intermediate preprocess structure created during
training containing the mappings of the input dataset.
- __output_feature_name__ (str): name of the output feature to use
for the visualization.
- __labels_limit__ (int): upper limit on the numeric encoded label value.
Encoded numeric label values in dataset that are higher than
`label_limit` are considered to be "rare" labels.
- __model_names__ (Union[str, List[str]], default: `None`): model name or
list of the model names to use as labels.
- __output_directory__ (str, default: `None`): directory where to save
plots. If not specified, plots will be displayed in a window
- __file_format__ (str, default: `'pdf'`): file format of output plots -
`'pdf'` or `'png'`.

__Return__


- __return__ (None):
 
----

## compare_classifiers_performance_subset


```python
ludwig.visualize.compare_classifiers_performance_subset(
  probabilities_per_model,
  ground_truth,
  top_n_classes,
  labels_limit,
  subset,
  model_names=None,
  output_directory=None,
  file_format='pdf'
)
```


Produces model comparison barplot visualization from train subset.

For each model  it produces bars in a bar plot, one for each overall metric
computed on the fly from the probabilities predictions for the
specified `model_names`, considering only a subset of the full training set.
The way the subset is obtained is using the `top_n_classes` and
`subset` parameters.

__Inputs__


- __probabilities_per_model__ (List[numpy.array]): list of model
   probabilities.
- __ground_truth__ (numpy.array): numpy.array containing ground truth data,
   which are the numeric encoded values the category.
- __top_n_classes__ (List[int]): list containing the number of classes
   to plot.
- __labels_limit__ (int): upper limit on the numeric encoded label value.
   Encoded numeric label values in dataset that are higher than
   `label_limit` are considered to be "rare" labels.
- __subset__ (str): string specifying type of subset filtering.  Valid
   values are `ground_truth` or `predictions`.
- __model_names__ (Union[str, List[str]], default: `None`): model name or
   list of the model names to use as labels.
- __output_directory__ (str, default: `None`): directory where to save
   plots. If not specified, plots will be displayed in a window
- __file_format__ (str, default: `'pdf'`): file format of output plots -
   `'pdf'` or `'png'`.

__Return__


- __return__ (None):
 
----

## compare_classifiers_performance_changing_k


```python
ludwig.visualize.compare_classifiers_performance_changing_k(
  probabilities_per_model,
  ground_truth,
  top_k,
  labels_limit,
  model_names=None,
  output_directory=None,
  file_format='pdf'
)
```


Produce lineplot that show Hits@K metric while k goes from 1 to `top_k`.

For each model it produces a line plot that shows the Hits@K metric
(that counts a prediction as correct if the model produces it among the
first k) while changing k from 1 to top_k for the specified
`output_feature_name`.

__Inputs__


- __probabilities_per_model__ (List[numpy.array]): list of model
probabilities.
- __ground_truth__ (numpy.array): numpy.array containing ground truth data,
which are the numeric encoded values the category.
- __top_k__ (int): number of elements in the ranklist to consider.
- __labels_limit__ (int): upper limit on the numeric encoded label value.
Encoded numeric label values in dataset that are higher than
`label_limit` are considered to be "rare" labels.
- __model_names__ (Union[str, List[str]], default: `None`): model name or
list of the model names to use as labels.
- __output_directory__ (str, default: `None`): directory where to save
plots. If not specified, plots will be displayed in a window
- __file_format__ (str, default: `'pdf'`): file format of output plots -
`'pdf'` or `'png'`.

__Return__


- __return__ (None):
 
----

## compare_classifiers_multiclass_multimetric


```python
ludwig.visualize.compare_classifiers_multiclass_multimetric(
  test_stats_per_model,
  metadata,
  output_feature_name,
  top_n_classes,
  model_names=None,
  output_directory=None,
  file_format='pdf'
)
```


Show the precision, recall and F1 of the model for the specified output_feature_name.

For each model it produces four plots that show the precision,
recall and F1 of the model on several classes for the specified output_feature_name.

__Inputs__


- __test_stats_per_model__ (List[dict]): list containing dictionary of
evaluation performance statistics
- __metadata__ (dict): intermediate preprocess structure created during
training containing the mappings of the input dataset.
- __output_feature_name__ (Union[str, `None`]): name of the output feature
to use for the visualization.  If `None`, use all output features.
- __top_n_classes__ (List[int]): list containing the number of classes
to plot.
- __model_names__ (Union[str, List[str]], default: `None`): model name or
list of the model names to use as labels.
- __output_directory__ (str, default: `None`): directory where to save
plots. If not specified, plots will be displayed in a window
- __file_format__ (str, default: `'pdf'`): file format of output plots -
`'pdf'` or `'png'`.

__Return__

- __return__ (None):
 
----

## compare_classifiers_predictions


```python
ludwig.visualize.compare_classifiers_predictions(
  predictions_per_model,
  ground_truth,
  labels_limit,
  model_names=None,
  output_directory=None,
  file_format='pdf'
)
```


Show two models comparison of their output_feature_name predictions.

__Inputs__


- __predictions_per_model__ (List[list]): list containing the model
predictions for the specified output_feature_name.
- __ground_truth__ (numpy.array): numpy.array containing ground truth data,
which are the numeric encoded values the category.
- __labels_limit__ (int): upper limit on the numeric encoded label value.
Encoded numeric label values in dataset that are higher than
`label_limit` are considered to be "rare" labels.
- __model_names__ (Union[str, List[str]], default: `None`): model name or
list of the model names to use as labels.
- __output_directory__ (str, default: `None`): directory where to save
plots. If not specified, plots will be displayed in a window
- __file_format__ (str, default: `'pdf'`): file format of output plots -
`'pdf'` or `'png'`.

__Return__


- __return__ (None):
 
----

## confidence_thresholding_2thresholds_2d


```python
ludwig.visualize.confidence_thresholding_2thresholds_2d(
  probabilities_per_model,
  ground_truths,
  threshold_output_feature_names,
  labels_limit,
  model_names=None,
  output_directory=None,
  file_format='pdf'
)
```


Show confidence threshold data vs accuracy for two output feature names.

The first plot shows several semi transparent lines. They summarize the
3d surfaces displayed by confidence_thresholding_2thresholds_3d that have
thresholds on the confidence of the predictions of the two
`threshold_output_feature_names`  as x and y axes and either the data
coverage percentage or
the accuracy as z axis. Each line represents a slice of the data
coverage  surface projected onto the accuracy surface.

__Inputs__


- __probabilities_per_model__ (List[numpy.array]): list of model
probabilities.
- __ground_truth__ (numpy.array): numpy.array containing ground truth data,
which are the numeric encoded values the category.
- __threshold_output_feature_names__ (List[str]): List containing two output
feature names for visualization.
- __labels_limit__ (int): upper limit on the numeric encoded label value.
Encoded numeric label values in dataset that are higher than
`label_limit` are considered to be "rare" labels.
- __model_names__ (Union[str, List[str]], default: `None`): model name or
list of the model names to use as labels.
- __output_directory__ (str, default: `None`): directory where to save
plots. If not specified, plots will be displayed in a window
- __file_format__ (str, default: `'pdf'`): file format of output plots -
`'pdf'` or `'png'`.

__Return__


- __return__ (None):
 
----

## confidence_thresholding_2thresholds_3d


```python
ludwig.visualize.confidence_thresholding_2thresholds_3d(
  probabilities_per_model,
  ground_truths,
  threshold_output_feature_names,
  labels_limit,
  output_directory=None,
  file_format='pdf'
)
```


Show 3d confidence threshold data vs accuracy for two output feature names.

The plot shows the 3d surfaces displayed by
confidence_thresholding_2thresholds_3d that have thresholds on the
confidence of the predictions of the two `threshold_output_feature_names`
as x and y axes and either the data coverage percentage or the accuracy
as z axis.

__Inputs__


- __probabilities_per_model__ (List[numpy.array]): list of model
probabilities.
- __ground_truth__ (numpy.array): numpy.array containing ground truth data,
which are the numeric encoded values the category.
- __threshold_output_feature_names__ (List[str]): List containing two output
feature names for visualization.
- __labels_limit__ (int): upper limit on the numeric encoded label value.
Encoded numeric label values in dataset that are higher than
`label_limit` are considered to be "rare" labels.
- __output_directory__ (str, default: `None`): directory where to save
plots. If not specified, plots will be displayed in a window
- __file_format__ (str, default: `'pdf'`): file format of output plots -
`'pdf'` or `'png'`.

__Return__


- __return__ (None):
 
----

## confidence_thresholding


```python
ludwig.visualize.confidence_thresholding(
  probabilities_per_model,
  ground_truth,
  labels_limit,
  model_names=None,
  output_directory=None,
  file_format='pdf'
)
```


Show models accuracy and data coverage while increasing treshold

For each model it produces a pair of lines indicating the accuracy of
the model and the data coverage while increasing a threshold (x axis) on
the probabilities of predictions for the specified output_feature_name.

__Inputs__


- __probabilities_per_model__ (List[numpy.array]): list of model
probabilities.
- __ground_truth__ (numpy.array): numpy.array containing ground truth data,
which are the numeric encoded values the category.
- __labels_limit__ (int): upper limit on the numeric encoded label value.
Encoded numeric label values in dataset that are higher than
`label_limit` are considered to be "rare" labels.
- __model_names__ (Union[str, List[str]], default: `None`): model name or
list of the model names to use as labels.
- __output_directory__ (str, default: `None`): directory where to save
plots. If not specified, plots will be displayed in a window
- __file_format__ (str, default: `'pdf'`): file format of output plots -
`'pdf'` or `'png'`.

__Return__


- __return__ (None):
 
----

## confidence_thresholding_data_vs_acc


```python
ludwig.visualize.confidence_thresholding_data_vs_acc(
  probabilities_per_model,
  ground_truth,
  labels_limit,
  model_names=None,
  output_directory=None,
  file_format='pdf'
)
```


Show models comparison of confidence threshold data vs accuracy.

For each model it produces a line indicating the accuracy of the model
and the data coverage while increasing a threshold on the probabilities
of predictions for the specified output_feature_name. The difference with
confidence_thresholding is that it uses two axes instead of three,
not visualizing the threshold and having coverage as x axis instead of
the threshold.

__Inputs__


- __probabilities_per_model__ (List[numpy.array]): list of model
probabilities.
- __ground_truth__ (numpy.array): numpy.array containing ground truth data,
which are the numeric encoded values the category.
- __labels_limit__ (int): upper limit on the numeric encoded label value.
Encoded numeric label values in dataset that are higher than
`label_limit` are considered to be "rare" labels.
- __model_names__ (Union[str, List[str]], default: `None`): model name or
list of the model names to use as labels.
- __output_directory__ (str, default: `None`): directory where to save
plots. If not specified, plots will be displayed in a window
- __file_format__ (str, default: `'pdf'`): file format of output plots -
`'pdf'` or `'png'`.

__Return__

- __return__ (None):
 
----

## confidence_thresholding_data_vs_acc_subset


```python
ludwig.visualize.confidence_thresholding_data_vs_acc_subset(
  probabilities_per_model,
  ground_truth,
  top_n_classes,
  labels_limit,
  subset,
  model_names=None,
  output_directory=None,
  file_format='pdf'
)
```


Show models comparison of confidence threshold data vs accuracy on a
subset of data.

For each model it produces a line indicating the accuracy of the model
and the data coverage while increasing a threshold on the probabilities
of predictions for the specified output_feature_name, considering only a subset of the
full training set. The way the subset is obtained is using the `top_n_classes`
and subset parameters.
The difference with confidence_thresholding is that it uses two axes
instead of three, not visualizing the threshold and having coverage as
x axis instead of the threshold.

If the values of subset is `ground_truth`, then only datapoints where the
ground truth class is within the top n most frequent ones will be
considered  as test set, and the percentage of datapoints that have been
kept  from the original set will be displayed. If the values of subset is
`predictions`, then only datapoints where the the model predicts a class
that is within the top n most frequent ones will be considered as test set,
and the percentage of datapoints that have been kept from the original set
will be displayed for each model.

__Inputs__


- __probabilities_per_model__ (List[numpy.array]): list of model
probabilities.
- __ground_truth__ (numpy.array): numpy.array containing ground truth data,
which are the numeric encoded values the category.
- __top_n_classes__ (List[int]): list containing the number of classes
to plot.
- __labels_limit__ (int): upper limit on the numeric encoded label value.
Encoded numeric label values in dataset that are higher than
`label_limit` are considered to be "rare" labels.
- __subset__ (str): string specifying type of subset filtering.  Valid
values are `ground_truth` or `predictions`.
- __model_names__ (Union[str, List[str]], default: `None`): model name or
list of the model names to use as labels.
- __output_directory__ (str, default: `None`): directory where to save
plots. If not specified, plots will be displayed in a window
- __file_format__ (str, default: `'pdf'`): file format of output plots -
`'pdf'` or `'png'`.

__Return__


- __return__ (None):
 
----

## binary_threshold_vs_metric


```python
ludwig.visualize.binary_threshold_vs_metric(
  probabilities_per_model,
  ground_truth,
  metrics,
  positive_label=1,
  model_names=None,
  output_directory=None,
  file_format='pdf'
)
```


Show confidence of the model against metric for the specified output_feature_name.

For each metric specified in metrics (options are `f1`, `precision`, `recall`,
`accuracy`), this visualization produces a line chart plotting a threshold
on  the confidence of the model against the metric for the specified
output_feature_name.  If output_feature_name is a category feature,
positive_label, which is specified as the numeric encoded value, indicates
the class to be considered positive class and all others will be
considered negative. To figure out the
association between classes and numeric encoded values check the
ground_truth_metadata JSON file.

__Inputs__


- __probabilities_per_model__ (List[numpy.array]): list of model
probabilities.
- __ground_truth__ (numpy.array): numpy.array containing ground truth data,
which are the numeric encoded values the category.
- __metrics__ (List[str]): metrics to display (`'f1'`, `'precision'`,
`'recall'`, `'accuracy'`).
- __positive_label__ (int, default: `1`): numeric encoded value for the
positive class.
- __model_names__ (List[str], default: `None`): list of the names of the
models to use as labels.
- __output_directory__ (str, default: `None`): directory where to save
plots. If not specified, plots will be displayed in a window
- __file_format__ (str, default: `'pdf'`): file format of output plots -
`'pdf'` or `'png'`.

__Return__


- __return__ (`None`):
 
----

## roc_curves


```python
ludwig.visualize.roc_curves(
  probabilities_per_model,
  ground_truth,
  positive_label=1,
  model_names=None,
  output_directory=None,
  file_format='pdf'
)
```


Show the roc curves for output features in the specified models.

This visualization produces a line chart plotting the roc curves for the
specified output feature name. If output feature name is a category feature,
`positive_label` indicates which is the class to be considered positive
class and all the others will be considered negative. `positive_label` is
the encoded numeric value for category classes. The numeric value can be
determined by association between classes and integers captured in the
training metadata JSON file.

__Inputs__


- __probabilities_per_model__ (List[numpy.array]): list of model
probabilities.
- __ground_truth__ (numpy.array): numpy.array containing ground truth data,
which are the numeric encoded values the category.
- __positive_label__ (int, default: `1`): numeric encoded value for the
positive class.
- __model_names__ (Union[str, List[str]], default: `None`): model name or
list of the model names to use as labels.
- __output_directory__ (str, default: `None`): directory where to save
plots. If not specified, plots will be displayed in a window
- __file_format__ (str, default: `'pdf'`): file format of output plots -
`'pdf'` or `'png'`.

__Return__


- __return__ (None):
 
----

## roc_curves_from_test_statistics


```python
ludwig.visualize.roc_curves_from_test_statistics(
  test_stats_per_model,
  output_feature_name,
  model_names=None,
  output_directory=None,
  file_format='pdf'
)
```


Show the roc curves for the specified models output binary
`output_feature_name`.

This visualization uses `output_feature_name`, `test_stats_per_model` and
`model_names` parameters. `output_feature_name` needs to be binary feature.
This visualization produces a line chart plotting the roc curves for the
specified `output_feature_name`.

__Inputs__


- __test_stats_per_model__ (List[dict]): dictionary containing evaluation
performance statistics.
- __output_feature_name__ (str): name of the output feature to use
for the visualization.
- __model_names__ (Union[str, List[str]], default: `None`): model name or
list of the model names to use as labels.
- __output_directory__ (str, default: `None`): directory where to save
plots. If not specified, plots will be displayed in a window
- __file_format__ (str, default: `'pdf'`): file format of output plots -
`'pdf'` or `'png'`.

__Return__


- __return__ (None):
 
----

## calibration_1_vs_all


```python
ludwig.visualize.calibration_1_vs_all(
  probabilities_per_model,
  ground_truth,
  top_n_classes,
  labels_limit,
  model_names=None,
  output_directory=None,
  file_format='pdf'
)
```


Show models probability of predictions for the specified output_feature_name.

For each class or each of the k most frequent classes if top_k is
specified,  it produces two plots computed on the fly from the
probabilities  of predictions for the specified output_feature_name.

The first plot is a calibration curve that shows the calibration of the
predictions considering the current class to be the true one and all
others  to be a false one, drawing one line for each model (in the
aligned  lists of probabilities and model_names).

The second plot shows the distributions of the predictions considering
the  current class to be the true one and all others to be a false one,
drawing the distribution for each model (in the aligned lists of
probabilities and model_names).

__Inputs__


- __probabilities_per_model__ (List[numpy.array]): list of model
probabilities.
- __ground_truth__ (numpy.array): numpy.array containing ground truth data,
which are the numeric encoded values the category.
- __top_n_classes__ (list): List containing the number of classes to plot.
- __labels_limit__ (int): upper limit on the numeric encoded label value.
Encoded numeric label values in dataset that are higher than
`label_limit` are considered to be "rare" labels.
- __model_names__ (List[str], default: `None`): list of the names of the
models to use as labels.
- __output_directory__ (str, default: `None`): directory where to save
plots. If not specified, plots will be displayed in a window
- __file_format__ (str, default: `'pdf'`): file format of output plots -
`'pdf'` or `'png'`.

__String__


- __return__ (None):
 
----

## calibration_multiclass


```python
ludwig.visualize.calibration_multiclass(
  probabilities_per_model,
  ground_truth,
  labels_limit,
  model_names=None,
  output_directory=None,
  file_format='pdf'
)
```


Show models probability of predictions for each class of the
specified output_feature_name.

__Inputs__


- __probabilities_per_model__ (List[numpy.array]): list of model
probabilities.
- __ground_truth__ (numpy.array): numpy.array containing ground truth data,
which are the numeric encoded values the category.
- __labels_limit__ (int): upper limit on the numeric encoded label value.
Encoded numeric label values in dataset that are higher than
`label_limit` are considered to be "rare" labels.
- __model_names__ (List[str], default: `None`): list of the names of the
models to use as labels.
- __output_directory__ (str, default: `None`): directory where to save
plots. If not specified, plots will be displayed in a window
- __file_format__ (str, default: `'pdf'`): file format of output plots -
`'pdf'` or `'png'`.

__Return__


- __return__ (None):
 
----

## confusion_matrix


```python
ludwig.visualize.confusion_matrix(
  test_stats_per_model,
  metadata,
  output_feature_name,
  top_n_classes,
  normalize,
  model_names=None,
  output_directory=None,
  file_format='pdf'
)
```


Show confision matrix in the models predictions for each
`output_feature_name`.

For each model (in the aligned lists of test_statistics and model_names)
it  produces a heatmap of the confusion matrix in the predictions for
each  output_feature_name that has a confusion matrix in test_statistics.
The value of `top_n_classes` limits the heatmap to the n most frequent
classes.

__Inputs__


- __test_stats_per_model__ (List[dict]): dictionary containing evaluation
  performance statistics.
- __metadata__ (dict): intermediate preprocess structure created during
training containing the mappings of the input dataset.
- __output_feature_name__ (Union[str, `None`]): name of the output feature
to use for the visualization.  If `None`, use all output features.
- __top_n_classes__ (List[int]): number of top classes or list
containing the number of top classes to plot.
- __normalize__ (bool): flag to normalize rows in confusion matrix.
- __model_names__ (Union[str, List[str]], default: `None`): model name or
list of the model names to use as labels.
- __output_directory__ (str, default: `None`): directory where to save
plots. If not specified, plots will be displayed in a window
- __file_format__ (str, default: `'pdf'`): file format of output plots -
`'pdf'` or `'png'`.

__Return__


- __return__ (None):
 
----

## frequency_vs_f1


```python
ludwig.visualize.frequency_vs_f1(
  test_stats_per_model,
  metadata,
  output_feature_name,
  top_n_classes,
  model_names=None,
  output_directory=None,
  file_format='pdf'
)
```


Show prediction statistics for the specified `output_feature_name` for
each model.

For each model (in the aligned lists of `test_stats_per_model` and
`model_names`), produces two plots statistics of predictions for the
specified `output_feature_name`.

The first plot is a line plot with one x axis representing the different
classes and two vertical axes colored in orange and blue respectively.
The orange one is the frequency of the class and an orange line is plotted
to show the trend. The blue one is the F1 score for that class and a blue
line is plotted to show the trend. The classes on the x axis are sorted by
f1 score.

The second plot has the same structure of the first one,
but the axes are flipped and the classes on the x axis are sorted by
frequency.

__Inputs__


- __test_stats_per_model__ (List[dict]): dictionary containing evaluation
performance statistics.
- __metadata__ (dict): intermediate preprocess structure created during
training containing the mappings of the input dataset.
- __output_feature_name__ (Union[str, `None`]): name of the output feature
to use for the visualization.  If `None`, use all output features.
- __top_n_classes__ (List[int]): number of top classes or list
containing the number of top classes to plot.
- __model_names__ (Union[str, List[str]], default: `None`): model name or
list of the model names to use as labels.
- __output_directory__ (str, default: `None`): directory where to save
plots. If not specified, plots will be displayed in a window
- __file_format__ (str, default: `'pdf'`): file format of output plots -
`'pdf'` or `'png'`.

__Return__


- __return__ (None):
 
----

## hyperopt_report


```python
ludwig.visualize.hyperopt_report(
  hyperopt_stats_path,
  output_directory=None,
  file_format='pdf'
)
```



Produces a report about hyperparameter optimization
creating one graph per hyperparameter to show the distribution of results
and one additional graph of pairwise hyperparameters interactions.

__Inputs__


- __hyperopt_stats_path__ (str): path to the hyperopt results JSON file.
- __output_directory__ (str, default: `None`): directory where to save
plots. If not specified, plots will be displayed in a window.
- __file_format__ (str, default: `'pdf'`): file format of output plots -
`'pdf'` or `'png'`.

__Return__


- __return__ (None):
 
----

## hyperopt_hiplot


```python
ludwig.visualize.hyperopt_hiplot(
  hyperopt_stats_path,
  output_directory=None
)
```



Produces a parallel coordinate plot about hyperparameter optimization
creating one HTML file and optionally a CSV file to be read by hiplot

__Inputs__


- __hyperopt_stats_path__ (str): path to the hyperopt results JSON file.
- __output_directory__ (str, default: `None`): directory where to save
plots. If not specified, plots will be displayed in a window.

__Return__


- __return__ (None):
 