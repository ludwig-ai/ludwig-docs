Ludwig supports AutoML, which returns a tuned Deep Learning model, given a dataset, the target column, and a time budget.

Ludwig AutoML is currently experimental and is focused on tabular datasets.  A blog describing its development, evaluation,
and use is [here](https://medium.com/ludwig-ai/ludwig-automl-for-deep-learning-cf64de9d49c8).

Ludwig AutoML infers the types of the input and output features, chooses the model architecture, and launches a Ray Tune
Async HyperBand search job across a set of hyperparameters and ranges, limited by the specified time budget.  It returns
the set of models produced by the trials in the search sorted from best to worst, along with a hyperparameter search report,
which can be inspected manually or post-processed by various Ludwig visualization tools.

The basic API for Ludwig AutoML is `auto_train`.  A simple example of its invocation can be found
[here](https://github.com/ludwig-ai/experiments/blob/main/automl/validation/mushroom_edibility/run_auto_train_2hr.py).

Users can audit and interact with Ludwig AutoML in various ways.

The Ludwig AutoML `create_auto_config` API outputs `auto_train`’s hyperparameter search configuration without running the search.
This API is useful for examining AutoML's chosen input and output feature types, model architecture, and hyperparameters and ranges.
A simple example of its invocation can be found
[here](https://github.com/ludwig-ai/experiments/blob/main/automl/validation/mushroom_edibility/get_auto_train_config.py).
The API is also useful for manual refinement of the AutoML-generated search; the output of this API can be edited and then
directly used as the input configuration for a Ludwig hyperparameter search job.

The `user_config` parameter can be provided to the `auto_train` or `create_auto_config` APIs to override specified parts of the
configuration produced.  For example, [this](https://github.com/ludwig-ai/experiments/blob/main/automl/heuristics/walmart_recruiting/run_auto_train_1hr.py)
`auto_train` invocation for the Walmart Recruiting dataset specifies that the TripType output feature be set to type category,
to override the Ludwig AutoML type detection system’s characterization of the feature as numerical.  And for example,
[this](https://github.com/ludwig-ai/experiments/blob/main/automl/validation/mushroom_edibility/run_auto_train_1hr_max_accuracy.py)
`auto_train` invocation specifies that the hyperparameter search job optimize for maximum accuracy of the specified
output feature rather than minimal loss of all combined output features, which is the default.

