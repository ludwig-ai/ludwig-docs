# Exporting Ludwig Models

There are a number of ways to export models in Ludwig. However, you may want to consider whether or not model exporting is necessary. Trained model weights are automatically saved locally at the end of training and these weights can be loaded into a LudwigModel directly:

```ludwig_model = LudwigModel.load("results/experiment_run/model")```

See [LudwigModel.load](./api/LudwigModel.md#load) for more details.
## TorchScript Export

A subset of Ludwig Models can be exported to Torchscript end-to-end. This means
that, in addition to the model itself, the preprocessing and postprocessing steps can be exported to TorchScript as well, ensuring that the model can be used for inference in a production environment out-of-the-box.

To get started, simply run the [`export_torchscript`](/latest/user_guide/command_line_interface/#export_torchscript) command:

```
ludwig export_torchscript -m=results/experiment_run/model
```

As long as `--model_only` is not specified, then three files are output by this command. The most important are `inference_preprocessor.pt`, `inference_predictor_<DEVICE>.pt`, and `inference_postprocessor.pt`.

The `inference_preprocessor.pt` file contains the preprocessor, which is a `torch.nn.Module` that takes in a dictionary of raw data and outputs a dictionary of tensors. The `inference_predictor_<DEVICE>.pt` file contains the predictor, which is a `torch.nn.Module` that takes in a dictionary of tensors and outputs a dictionary of tensors. The `inference_postprocessor.pt` file contains the postprocessor, which is a `torch.nn.Module` that takes in a dictionary of tensors and outputs a dictionary of postprocessed data containing the same keys as the Ludwig `predict` command DataFrame output. These files can each be loaded separately (and on separate devices) to run inference in a staged manner. This can be particularly useful with a tool like [NVIDIA Triton](https://developer.nvidia.com/nvidia-triton-inference-server), which can be used to manage each stage of the pipeline independently.

You can get started using these modules immediately by loading them with `torch.jit.load`:

```
>>> import torch
>>> preprocessor = torch.jit.load("model/inference_preprocessor.pt")
>>> predictor = torch.jit.load("model/inference_predictor_cpu.pt")
>>> postprocessor = torch.jit.load("model/inference_postprocessor.pt")
```

Once you've done this, you can pass in raw data in the form of a dictionary and get back predictions in the form of a dictionary:

```
>>> raw_data = {
    'text_feature': ['This is a test.'],
    'category_feature': ['cat1', 'cat2'],
    'numerical_feature': [1.0, 2.0],
    'vector_feature': [[1.0, 2.0], [3.0, 4.0]],
}
>>> preprocessed_data = preprocessor(raw_data)
>>> predictions = predictor(preprocessed_data)
>>> postprocessed_data = postprocessor(predictions)
>>> postprocessed_data
{
    'probabilities': [[0.8, 0.2]],
    'predictions': ['class1', 'class2'],
}
```

If you have your data stored in a DataFrame, you can use the `to_inference_module_input_from_dataframe` function to convert it to the correct format:

```
>>> import pandas as pd
>>> from ludwig.utils.inference_utils import to_inference_module_input_from_dataframe

>>> df = pd.read_csv('test.csv')
>>> raw_data = to_inference_module_input_from_dataframe(df)
>>> raw_data
{
    'text_feature': ['This is a test.'],
    'category_feature': ['cat1', 'cat2'],
    'numerical_feature': [1.0, 2.0],
    'vector_feature': [[1.0, 2.0], [3.0, 4.0]],
}
```

### Using the InferenceModule class

For convenience, we've provided a wrapper class called `InferenceModule` that can be used to load and use the exported model. The `InferenceModule` class is a subclass of `torch.nn.Module` and can be used in the same way. The `InferenceModule` class can be used to load the preprocessor, predictor, and postprocessor in a single step:

```
>>> from ludwig.models.inference import InferenceModule
>>> inference_module = InferenceModule.from_directory('model/')
>>> raw_data = {...}
>>> preprocessed_data = inference_module.preprocessor_forward(raw_data)
>>> predictions = inference_module.predictor_forward(preprocessed_data)
>>> postprocessed_data = inference_module.postprocessor_forward(predictions)
```

You can also call the preprocessor, predictor, and postprocessor separately:

```
>>> inference_module.preprocessor_forward(raw_data)
{...}
```

You can also use the `InferenceModule.predict` method to input a DataFrame and output a DataFrame, similar to how you would with the `LudwigModel.predict` method:

```
>>> input_df = pd.read_csv('test.csv')
>>> output_df = inference_module.predict(input_df)
```

Finally, you can convert the `InferenceModule` to TorchScript if you need a monolithic artifact for inference. Note that you can still call the `InferenceModule.preprocessor_forward`, `InferenceModule.predictor_forward`, and `InferenceModule.postprocessor_forward` methods, but `InferenceModule.predict` will no longer work after this conversion.

```
>>> scripted_module = torch.jit.script(inference_module)
>>> raw_data = {...}
>>> scripted_module(raw_data)
{...}
>>> input_df = pd.read_csv('test.csv')
>>> scripted_module.predict(input_df)  # Will fail
```

### Current Limitations

TorchScript only implements a subset of Python libraries. This means that there are a few of Ludwig's preprocessing steps that are not supported.

#### Image and Audio Features

If you are using the preprocessor module directly or using the `forward` method of the `InferenceModule` module, loading `image` and `audio` files from filepaths is not supported. Instead, one has to load the data as either a batched tensor or a List of tensors ahead of passing it in to the module.

If you are using the `predict` method of the `InferenceModule` module, then filepath strings will be loaded for you.

#### Date Features

TorchScript does not implement the `datetime` module, which means that `date` features cannot be parsed from string.

If you are using the preprocessor module directly or using the `forward` method of the `InferenceModule` module, a vector of integers representing the date must be passed in. Given a `datetime_obj`, the following code can be used to generate the vector:

```
def create_vector_from_datetime_obj(datetime_obj):
    yearday = datetime_obj.toordinal() - date(datetime_obj.year, 1, 1).toordinal() + 1

    midnight = datetime_obj.replace(hour=0, minute=0, second=0, microsecond=0)
    second_of_day = (datetime_obj - midnight).seconds

    return [
        datetime_obj.year,
        datetime_obj.month,
        datetime_obj.day,
        datetime_obj.weekday(),
        yearday,
        datetime_obj.hour,
        datetime_obj.minute,
        datetime_obj.second,
        second_of_day,
    ]
```

This function is also available in the `ludwig.utils.date_utils` module for convenience.

If you are using the `predict` method of the `InferenceModule` module, then date strings will be featurized for you.

#### GPU Support

Input features fed in as strings can only be preprocessed on CPU. In Ludwig, these would be the following features:

- `binary`
- `category`
- `set`
- `sequence`
- `text`
- `vector`
- `bag`
- `timeseries`

If you are using the preprocessor module directly or using the `forward` method of the `InferenceModule` module, then the user can pass in `binary`, `vector`, and `time series` features as tensors to preprocess them on the GPU.

If you are using the `predict` method of the `InferenceModule` module, then all of the above features will be cast first to string and then preprocessed on CPU.

#### NaN Handling

For the majority of features, NaNs are handled by the preprocessor in the same way that they are handled by the original Ludwig Model. The exceptions are the following features:

- `image`
- `audio`
- `date`
- `vector`

#### HuggingFace Models

HuggingFace models are not yet supported for TorchScript export, though we are working on it!

## Carton Export

[Carton](https://carton.run/) is a library that allows users to efficiently run ML models from several programming languages (including C, C++, Rust, and more).

A subset of Ludwig Models can be exported to Carton. In addition to the model itself, the preprocessing and postprocessing steps are included in the exported model as well, ensuring that the model can be used for inference in a production environment out-of-the-box.

To get started, simply run the [`export_carton`](/latest/user_guide/command_line_interface/#export_carton) command:

```
ludwig export_carton -m=results/experiment_run/model
```
This will produce a file that can be loaded by Carton from any supported programming language.

For example, from Python, you can load and run the model as follows:

```py
import cartonml as carton

async def main():
    model = await carton.load("/path/to/model.carton")
    output = await model.infer({
        "x": np.zeros(5)
    })
```

See the [Carton quickstart guide](https://carton.run/quickstart) for usage from other programming languages.

## Triton Export

Coming soon...

## Neuropod Export

Coming soon...

## MLFlow Export
