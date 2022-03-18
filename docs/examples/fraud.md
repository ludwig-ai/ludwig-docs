| transaction_id | card_id | customer_id | customer_zipcode | merchant_id | merchant_name | merchant_category | merchant_zipcode | merchant_country | transaction_amount | authorization_response_code | atm_network_xid | cvv_2_response_xflg | fraud_label |
| -------------- | ------- | ----------- | ---------------- | ----------- | ------------- | ----------------- | ---------------- | ---------------- | ------------------ | --------------------------- | --------------- | ------------------- | ----------- |
| 469483         | 9003    | 1085        | 23039            | 893         | Wright Group  | 7917              | 91323            | GB               | 1962               | C                           | C               | N                   | 0           |
| 926515         | 9009    | 1001        | 32218            | 1011        | Mums Kitchen  | 5813              | 10001            | US               | 1643               | C                           | D               | M                   | 1           |
| 730021         | 9064    | 1174        | 9165             | 916         | Keller        | 7582              | 38332            | DE               | 1184               | D                           | B               | M                   | 0           |

```
ludwig experiment \
--dataset transactions.csv \
  --config config.yaml
```

With `config.yaml`:

```yaml
input_features:
  -
    name: customer_id
    type: category
  -
    name: card_id
    type: category
  -
    name: merchant_id
    type: category
  -
    name: merchant_category
    type: category
  -
    name: merchant_zipcode
    type: category
  -
    name: transaction_amount
    type: number
  -
    name: authorization_response_code
    type: category
  -
    name: atm_network_xid
    type: category
  -
    name: cvv_2_response_xflg
    type: category

combiner:
    type: concat
    num_fc_layers: 1
    output_size: 48

output_features:
  -
    name: fraud_label
    type: binary
```
