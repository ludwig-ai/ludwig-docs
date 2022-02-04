Ludwig can train on any *table-like* dataset, meaning that every feature has its own **column** and every example its own **row**.

In this example, we'll use the well-known [Iris](https://archive.ics.uci.edu/ml/datasets/Iris) dataset, a CSV file with 4 numerical features and a categorical target. 

Ludwig provides Iris and many other popular benchmark datasets through a simple interface:

=== "CLI"

    ``` sh
    ludwig datasets download iris
    ```

=== "Python"

    ``` python
    from ludwig.datasets import iris

    df = iris.load()
    ```

Take a look at the first 10 rows to see how the data is arranged:

=== "CLI"

    ``` sh
    head -n 10 iris.csv
    ```

=== "Python"

    ``` python
    df.head(10)
    ```
