{% from './macros/includes.md' import render_fields, render_yaml %}
{% set mv_details = "See [Missing Value Strategy](./input_features.md#missing-value-strategy) for details." %}
{% set norm_details = "See [Normalization](../combiner.md#normalization) for details." %}
{% set details = {"missing_value_strategy": mv_details, "norm": norm_details, "norm_params": norm_details} %}

Date features are like `2023-06-25 15:00:00`, `2023-06-25`, `6-25-2023`, or `6/25/2023`.

# Preprocessing

Ludwig will try to infer the date format automatically, but a specific format can be provided. The date string spec is
the same as the one described in python's [datetime](https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior).

{% set preprocessing = get_feature_preprocessing_schema("date") %}
{{ render_yaml(preprocessing, parent="preprocessing") }}

```yaml
name: date_feature_name
type: date
preprocessing:
  missing_value_strategy: fill_with_const
  fill_value: ''
  datetime_format: "%d %b %Y"
```

Parameters:

{{ render_fields(schema_class_to_fields(preprocessing), details=details) }}

Preprocessing parameters can also be defined once and applied to all date input features using the [Type-Global Preprocessing](../defaults.md#type-global-preprocessing) section.

# Input Features

Input date features are transformed into a int tensors of size `N x 9` (where `N` is the size of the dataset and the 9 dimensions contain year, month, day, weekday, yearday, hour, minute, second, and second of day).

For example, the date `2022-06-25 09:30:59` would be deconstructed into:

```python
[
  2022,   # Year
  6,      # June
  25,     # 25th day of the month
  5,      # Weekday: Saturday
  176,    # 176th day of the year
  9,      # Hour
  30,     # Minute
  59,     # Seconds
  34259,  # 34259th second of the day
]
```

The encoder parameters specified at the feature level are:

- **`tied`** (default `null`): name of another input feature to tie the weights of the encoder with. It needs to be the name of
a feature of the same type and with the same encoder parameters.

Currently there are two encoders supported for dates: `DateEmbed` (default) and `DateWave`. The encoder can be set by specifying `embed` or `wave` in the feature's `encoder` parameter in the input feature's configuration.

Example date feature entry in the input features list:

```yaml
name: date_feature_name
type: date
encoder: 
    type: embed
```

Encoder type and encoder parameters can also be defined once and applied to all date input features using the [Type-Global Encoder](../defaults.md#type-global-encoder) section.

## Encoders

### Embed Encoder

This encoder passes the year through a fully connected layer of one neuron and embeds all other elements for the date, concatenates them and passes the concatenated representation through fully connected layers.

{% set encoder_embed = get_encoder_schema("date", "embed") %}
{{ render_yaml(encoder_embed, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(encoder_embed, exclude=["type"]), details=details) }}

### Wave Encoder

This encoder passes the year through a fully connected layer of one neuron and represents all other elements for the date by taking the cosine of their value with a different period (12 for months, 31 for days, etc.), concatenates them and passes the concatenated representation through fully connected layers.

{% set encoder_wave = get_encoder_schema("date", "wave") %}
{{ render_yaml(encoder_wave, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(encoder_wave, exclude=["type"]), details=details) }}

# Output Features

There is currently no support for date as an output feature. Consider using the [`TEXT` type](../../features/text_features).
