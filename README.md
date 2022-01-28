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

Test it (from the `src` directory):

```
mkdocs serve
```

Finally build the static website (from the `src` directory):

```
mkdocs build
```

It will create the static website in `$LUDWIG_HOME/docs/`.

## Publish a new version

```
mike deploy --push --update-aliases 0.5 latest
```
