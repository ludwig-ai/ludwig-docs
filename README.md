# Ludwig documentation

## Building docs locally

Ludwig's documentation is build using [MkDocs](https://www.mkdocs.org/) and the beautiful [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.
In order to create Ludwig's documentation you have to install them:

```
pip install -r requirements.txt
```

Generate `api.md` from source:

```
python code_doc_autogen.py
```

Run the web server:

```
mkdocs serve
```

Navigate to <http://localhost:8000> to view the docs.

### Test versioning locally

Deploy docs for the current version of Ludwig in your environment:

```
export LUDWIG_VERSION=$(python -c "import ludwig; print('.'.join(ludwig.__version__.split('.')[:2]))")
mike deploy --update-aliases $LUDWIG_VERSION latest
```

Run the web server:

```
mike serve
```

Navigate to <http://localhost:8000> to view the docs.

## Updating docs for older Ludwig versions

The CI system will by default publish new docs for the latest version of Ludwig when commit is made to the master branch.

To update docs for an older version of Ludwig, create a new branch:

```
git checkout -b $VERSION
```

Make sure the correct version of Ludwig is installed locally and generate the API docs:

```
pip install ludwig[full]==$VERSION
python code_doc_autogen.py
```

Use the `--push` option to publish the changes to the remote repo. Be sure to only include the major and minor version (e.g., `0.5` instead of `0.5.1`) for the docs:

```
mike deploy --push $MAJOR_MINOR_VERSION
```
