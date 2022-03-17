| year | duration | nominations | categories         | rating |
| ---- | -------- | ----------- | ------------------ | ------ |
| 1921 | 3240     | 0           | comedy drama       | 8.4    |
| 1925 | 5700     | 1           | adventure comedy   | 8.3    |
| 1927 | 9180     | 4           | drama comedy scifi | 8.4    |

```
ludwig experiment \
--dataset movie_ratings.csv \
  --config config.yaml
```

With `config.yaml`:

```yaml
input_features:
    -
        name: year
        type: number
    -
        name: duration
        type: number
    -
        name: nominations
        type: number
    -
        name: categories
        type: set

output_features:
    -
        name: rating
        type: number
```
