Ludwig can train on any *table-like* dataset, meaning that every feature has its own **column** and every example its own **row**.

In this example, we'll use this [Rotten Tomatoes](https://www.kaggle.com/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset) dataset, a CSV file with variety of feature types and a binary target.

Download the data locally [here](./../data/rotten_tomatoes.csv).

Let's take a look at the first 5 rows to see how the data is arranged:

=== "CLI"

    ``` sh
    head -n 5 rotten_tomatoes.csv
    ```

=== "Python"

    ``` python
    import pandas as pd

    df = pd.read_csv('rotten_tomatoes.csv')
    df.head()
    ```

Your results should look a little something like this:

|     movie_title      | content_rating |                                  genres                                  | runtime | top_critic | review_content                                                                                                                                                                                                   | recommended |
| :------------------: | :------------: | :----------------------------------------------------------------------: | :-----: | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| Deliver Us from Evil |       R        |                        Action & Adventure, Horror                        |  117.0  | TRUE       | Director Scott Derrickson and his co-writer, Paul Harris Boardman, deliver a routine procedural with unremarkable frights.                                                                                       | 0           |
|       Barbara        |     PG-13      |                     Art House & International, Drama                     |  105.0  | FALSE      | Somehow, in this stirring narrative, Barbara manages to keep hold of her principles, and her humanity and courage, and battles to save a dissident teenage girl whose life the Communists are trying to destroy. | 1           |
|   Horrible Bosses    |       R        |                                  Comedy                                  |  98.0   | FALSE      | These bosses cannot justify either murder or lasting comic memories, fatally compromising a farce that could have been great but ends up merely mediocre.                                                        | 0           |
|    Money Monster     |       R        |                                  Drama                                   |  98.0   | FALSE      | A satire about television that feels like it was made by the kind of people who claim they don't even watch TV.                                                                                                  | 0           |
|    Battle Royale     |       NR       | Action & Adventure, Art House & International, Drama, Mystery & Suspense |  114.0  | FALSE      | Battle Royale is The Hunger Games not diluted for young audiences.                                                                                                                                               | 1           |
