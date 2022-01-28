Postprocessing Outputs
===

The JSON file obtained from preprocessing is used also for postprocessing: Ludwig models return output predictions and, depending on their datatype they are mapped back into the original space.
Numerical and timeseries are returned as they are, while category, set, sequence, and text features output integers, those integers are mapped back into the original tokens / names using the `idx2str` in the JSON file.
When you run `experiment` or `predict` you will find both a CSV file for each output containing the mapped predictions, a probability CSV file containing the probability of that prediction, a probabilities CSV file containing the probabilities for all alternatives (for instance, the probabilities of all the categories in case of a categorical feature).  You will also find the unmapped NPY files.
If you don't need them you can use the `--skip_save_unprocessed_output` argument.