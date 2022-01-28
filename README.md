Ludwig documentation
====================

Ludwig's documentation is build using [MkDocs](https://www.mkdocs.org/) and the beautiful [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.
In order to create Ludwig's documentation you have to install them:

```
pip install -r requirements.txt
```

Generate `api.md` from source:

```
python code_doc_autogen.py
```

Deploy docs for the current version of Ludwig in your environment:

```
export LUDWIG_VERSION=$(python -c "import ludwig; print('.'.join(ludwig.__version__.split('.')[:2]))")
mike deploy --push --update-aliases $LUDWIG_VERSION latest
```

Run the web server:

```
mike serve
```

Navigate to http://localhost:8000 to view the docs.
