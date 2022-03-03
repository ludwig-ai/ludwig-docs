Ludwig can train on any *table-like* dataset, meaning that every feature has its own **column** and every example its own **row**.

In this example, we'll use this [Rotten Tomatoes](https://www.kaggle.com/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset) dataset, a CSV file with variety of feature types and a binary target. 

We've taken the data from the above link and prepared a dataset for you to use in this first example. Click [here](data/rotten_tomatoes.csv)
 to download the prepared dataset.

Let's take a look at the first 10 rows to see how the data is arranged:

=== "CLI"

    ``` sh
    head -n 10 rotten_tomatoes.csv
    ```

=== "Python"

    ``` python
    import pandas as pd

    df = pd.read_csv('rotten_tomatoes.csv')
    df.head(10)
    ```
