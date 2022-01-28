Serving Ludwig Model Pipelines
==============================

Model pipelines trained with Ludwig can be served using the [serve command](command_line_interface.md#serve).
The command will spawn a Rest API using the FastAPI library.
Such API has two endpoints: `predict` and `predict_batch`.
`predict` should be used to obtain predictions for a single datapoints, while `predict_batch` should be used to obtain predictions for an entire DataFrame / for multiple datapoints.

In Ludwig model pipelines are defined based on their input, their outputs and their data  types.
Models can have multiple inputs and multiple outputs of arbitrary data types.
For instance a text classification model can be defined by a text input and a category outputs, while a regression model can be defined as several numerical, binary and category inputs and one numerical outputs.

The structure of the input to the REST API and the structure of the output that will be returned depends on the data types of the input and outputs of the Ludwig model pipeline.

REST Endpoints
==============

## predict

### Input format

For each input of the model, the predict endpoint expects a field with a name.
For instance, a model trained with an input text field named `english_text` would expect a POST like:
```
curl http://0.0.0.0:8000/predict -X POST -F 'english_text=words to be translated'
```

If the model was trained with an input image field, it will instead expects a POST with a file, like:

```
curl http://0.0.0.0:8000/predict -X POST -F 'image=@path_to_image/example.png'
```

A model with both a text and an image field will expect a POST like:

```
curl http://0.0.0.0:8000/predict -X POST -F 'text=mixed together with' -F 'image=@path_to_image/example.png'
```

### Output format

The output format is  a JSON that is independent of the number of inputs and their data types, it only depends on the number of outputs the model pipeline was trained to predict and their data types.

At the moment, Ludwig can predict binary, numerical, categorical, set, sequence and text outputs.

For binary outputs, the JSON structure returned by the REST PI is the following:
```
{
   "NAME_predictions": false,
   "NAME_probabilities_False": 0.76,
   "NAME_probabilities_True": 0.24,
   "NAME_probability": 0.76
}
```

For numerical outputs, the JSON structure returned by the REST PI is the following:

`````
{"NAME_predictions": 0.381}
`````

For categorical outputs, the JSON structure returned by the REST PI is the following:

```
{
   "NAME_predictions": "CLASSNAMEK",
   "NAME_probability": 0.62,
   "NAME_probabilities_CLASSNAME1": 0.099,
   "NAME_probabilities_CLASSNAME2": 0.095,
   ...
   "NAME_probabilities_CLASSNAMEN": 0.077
}
```

For set outputs, the JSON structure returned by the REST PI is the following:
```
{
   "NAME_predictions":[
      "CLASSNAMEI",
      "CLASSNAMEJ",
      "CLASSNAMEK"
   ],
   "NAME_probabilities_CLASSNAME1":0.490,
   "NAME_probabilities_CLASSNAME2":0.245,
   ...
   "NAME_probabilities_CLASSNAMEN":0.341,
   "NAME_probability":[
      0.53,
      0.62,
      0.95
   ]
}
```

For sequence outputs, the JSON structure returned by the REST PI is the following:
```
{
   "NAME_predictions":[
      "TOKEN1",
      "TOKEN2",
      "TOKEN3"
   ],
   "NAME_last_predictions": "TOKEN3",
   "NAME_probabilities":[
      0.106,
      0.122,
      0.118,
      0.133
   ],
   "NAME_probability": -6.4765729904174805
}
```

For text outputs, the JSON structure returned by the REST PI is the same as the sequence one.


## batch_predict

### Input format

You can also make a POST request on the /batch_predict endpoint to run inference on multiple samples at once.

Requests must be submitted as form data, with one of fields being `dataset`: a JSON encoded string representation of the data to be predicted.

The dataset JSON string is expected to be in the Pandas `split` format to reduce payload size.
This format divides the dataset into three parts:

- `columns`: `List[str]`
- `index` (optional): `List[Union[str, int]]`
- `data`: `List[List[object]]`

Additional form fields can be used to provide file resources like images that are referenced within the dataset.

An example of batch prediction:

```
curl http://0.0.0.0:8000/batch_predict -X POST -F 'dataset={"columns": ["a", "b"], "data": [[1, 2], [3, 4]]}'
```

### Output format

The output format is  a JSON that is independent of the number of inputs and their data types, it only depends on the number of outputs the model pipeline was trained to predict and their data types.

At the moment, Ludwig can predict binary, numerical, categorical, set, sequence and text outputs.

For binary outputs, the JSON structure returned by the REST PI is the following:

```
{
   "index": [0, 1],
   "columns": [
      "NAME_predictions",
      "NAME_probabilities_False",
      "NAME_probabilities_True",
      "NAME_probability"
   ],
   "data": [
      [false, 0.768, 0.231, 0.768],
      [true, 0.372, 0.627, 0.627]
   ]
}
```

For numerical outputs, the JSON structure returned by the REST PI is the following:

```
{"index":[0, 1],"columns":["NAME_predictions"],"data":[[0.381],[0.202]]}
```

For categorical outputs, the JSON structure returned by the REST PI is the following:
```
{
   "index": [0, 1],
   "columns": [
      "NAME_predictions",
      "NAME_probabilities_CLASSNAME1",
      "NAME_probabilities_CLASSNAME2",
      ...
      "NAME_probabilities_CLASSNAMEN",
      "NAME_probability"
   ],
   "data": [
      ["CLASSNAMEK", 0.099, 0.095, ... 0.077, 0.623],
      ["CLASSNAMEK", 0.092, 0.061, ... 0.084, 0.541]
   ]
}
```

For set outputs, the JSON structure returned by the REST PI is the following:
```
{
   "index": [0, 1],
   "columns": [
      "NAME_predictions",
      "NAME_probabilities_CLASSNAME1",
      "NAME_probabilities_CLASSNAME2",
      ...
      "NAME_probabilities_CLASSNAMEK",
      "NAME_probability"
   ],
   "data": [
      [
         ["CLASSNAMEI", "CLASSNAMEJ", "CLASSNAMEK"],
         0.490,
         0.453,
         ...
         0.500,
         [0.53, 0.62, 0.95]
      ],
      [
         ["CLASSNAMEM", "CLASSNAMEN", "CLASSNAMEO"],
         0.481,
         0.466,
         ...
         0.485,
         [0.63, 0.72, 0.81]
      ]
   ]
}
```

For sequence outputs, the JSON structure returned by the REST PI is the following:
```
{
   "index": [0, 1],
   "columns": [
      "NAME_predictions",
      "NAME_last_predictions",
      "NAME_probabilities",
      "NAME_probability"
   ],
   "data": [
      [
         ["TOKEN1", "TOKEN1", "TOKEN1"],
         "TOKEN3",
         [0.106, 0.122, … 0.083],
         -6.476
      ],
      [
         ["TOKEN4", "TOKEN5", "TOKEN6"],
         "TOKEN6",
         [0.108, 0.127, … 0.083],
         -6.482
      ]
   ]
}
```

For text outputs, the JSON structure returned by the REST PI is the same as the sequence one.
