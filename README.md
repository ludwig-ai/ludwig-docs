# Ludwig documentation

Website: [ludwig.ai](ludwig.ai)

Ludwig's documentation is built using [MkDocs](https://www.mkdocs.org/) and the
beautiful [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
theme, deployed using [Mike](https://github.com/jimporter/mike).

## Edit-refresh development

1. Install requirements.

    ```
    pip install -r requirements.txt
    ```

2. If the content of
   [contributing guide](https://github.com/ludwig-ai/ludwig/blob/master/CONTRIBUTING.md)
   in the Ludwig code repository has changed, then run

    ```
    python code_doc_autogen.py
    ```
    
    in order to download it into `docs/developer_guide/contributing.md` in this
    repository (since all documentation content is served from local files).

    Otherwise, skip this step.

3. In terminal, keep a window running:

    ```
    mkdocs serve
    ```

4. Navigate to <http://localhost:8000> to view your changes.

> :bulb: No need to return to your terminal -- changes will be automatically
> reflected as you save changes to files.

## Versioned docs

The full `ludwig.ai` website is deployed using `mike`, a wrapper around
`mkdocs`, which deploys includes previous snapshots of documentation for older
versions of Ludwig.

To see how the fully rendered `ludwig.ai` website with multiple versions looks:

1. Export the ludwig version to an environment variable:

    ```
    export LUDWIG_VERSION=$(python -c "import ludwig; print('.'.join(ludwig.__version__.split('.')[:2]))")
    ```

2. Run `mike deploy`

    ```
    mike deploy --update-aliases $LUDWIG_VERSION latest --ignore
    ```

3. In a separate tab, run the `mike` web server:

    ```
    mike serve --ignore
    ```

4. Navigate to <http://localhost:8000> to view.

> :warning: `mike serve` is **not** edit-refreshable. In order to see changes reflected, re-run `mike deploy` and `mike serve`.
>
> ```
> mike deploy --update-aliases $LUDWIG_VERSION latest --ignore
> mike serve --ignore
> ```

## Updating docs for older Ludwig versions

The CI system will by default publish new docs for the latest version every day.

Updating docs for an older version of Ludwig needs to be done manually.

Create a new branch:

```
git checkout -b $VERSION
```

Install the relevant version of Ludwig and generate documentation:

```
pip install ludwig==$VERSION
python code_doc_autogen.py
```

Use the `--push` option to publish the changes to the remote repo. Be sure to
only include the major and minor version (e.g., `0.5` instead of `0.5.1`):

```
mike deploy --push $MAJOR_MINOR_VERSION
```

## Regenerating API documentation

Markdown files under `docs/user_guide/api/` are generated automatically. To
regenerate these files, run:

```
python code_doc_autogen.py
```
