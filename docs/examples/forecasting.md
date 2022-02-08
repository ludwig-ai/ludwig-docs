While direct timeseries prediction is a work in progress Ludwig can ingest timeseries input feature data and make numerical predictions. Below is an example of a model trained to forecast timeseries at five different horizons.

| timeseries_data       | y1    | y2    | y3    | y4    | y5    |
| --------------------- | ----- | ----- | ----- | ----- | ----- |
| 15.07 14.89 14.45 ... | 16.92 | 16.67 | 16.48 | 17.00 | 17.02 |
| 14.89 14.45 14.30 ... | 16.67 | 16.48 | 17.00 | 17.02 | 16.48 |
| 14.45 14.3 14.94 ...  | 16.48 | 17.00 | 17.02 | 16.48 | 15.82 |

```
ludwig experiment \
--dataset timeseries_data.csv \
  --config_file config.yaml
```

With `config.yaml`:

```yaml
input_features:
    -
        name: timeseries_data
        type: timeseries

output_features:
    -
        name: y1
        type: numerical
    -
        name: y2
        type: numerical
    -
        name: y3
        type: numerical
    -
        name: y4
        type: numerical
    -
        name: y5
        type: numerical
```
