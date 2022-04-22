Model pipelines trained with Ludwig can be served by spawning a Rest API using the [FastAPI](https://fastapi.tiangolo.com/) library.

Let's serve the model we just created.

=== "CLI"

    ```sh
    ludwig serve --model_path ./results/experiment_run/model
    ```

=== "Docker CLI"

    ```sh
    docker run -t -i --mount type=bind,source={absolute/path/to/rotten_tomatoes_data},target=/rotten_tomatoes_data ludwigai/ludwig serve --model_path /rotten_tomatoes_data/results/experiment_run/model
    ```

Now that our server is up and running, you can make a POST request on the endpoint to get predictions back:

``` sh
curl http://0.0.0.0:8000/predict -X POST -F "movie_title=Friends With Money" -F "content_rating=R" -F "genres=Art House & International, Comedy, Drama" -F "runtime=88.0" -F "top_critic=TRUE" -F "review_content=The cast is terrific, the movie isn't."
```

Since the output feature is a binary type feature, the output from the POST call will look something like this:

```
{
   "review_content_predictions": false,
   "review_content_probabilities_False": 0.76,
   "review_content_probabilities_True": 0.24,
   "review_content_probability": 0.76
}
```

!!! note

    Users can also send POST requests to the [`/batch_predict`](../..//user_guide/serving/#batch_predict) endpoint to run inference on multiple examples at once. Read more about [ludwig serve](../../user_guide/serving) to learn more about ludwig deployments.
