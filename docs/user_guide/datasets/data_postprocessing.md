The JSON metadata file obtained during preprocessing is also used for postprocessing: Ludwig models return output
predictions and, depending on their datatype they are mapped back into raw data.

Number and timeseries do not require additional transformations and are returned as they are, directly from the
model.

Category, set, sequence, and text features are represented in the model as integers. These predictions are mapped back
into the original tokens / names using the `idx2str` in the JSON file.

Users running `experiment` or `predict` will find multiple prediction results files: 1) a CSV file for each output
containing the mapped predictions, 2) a probability CSV file containing the probability of that prediction, 3) a
probabilities CSV file containing the probabilities for all alternatives (for instance, the probabilities of all the
categories in case of a categorical feature).

Users will also get the raw unmapped predictions from the model as NPY files. If you don't need them users can use the
`--skip_save_unprocessed_output` argument.
