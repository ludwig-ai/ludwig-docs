"""MkDocs hook that silences noisy import-time warnings from Ludwig dependencies.

When mkdocstrings/griffe imports Ludwig to inspect its API, libraries like
sentencepiece and bitsandbytes emit deprecation and configuration warnings that
clutter the build output.  This hook installs warning filters early so they are
suppressed before any Ludwig imports occur.

bitsandbytes also prints a raw message to stdout when compiled without GPU
support (not a warning, so ``warnings.filterwarnings`` cannot catch it).  We
suppress that by pre-importing the library with stdout temporarily redirected.
"""

from __future__ import annotations

import contextlib
import io
import warnings


def on_startup(**kwargs) -> None:  # noqa: ARG001
    # sentencepiece SWIG wrappers lack __module__
    warnings.filterwarnings("ignore", message="builtin type Swig.*has no __module__")
    warnings.filterwarnings("ignore", message="builtin type swigvarlink.*has no __module__")

    # bitsandbytes GPU-support warning (irrelevant for doc builds)
    warnings.filterwarnings("ignore", message=".*bitsandbytes was compiled without GPU support.*")

    # bitsandbytes prints to stdout when compiled without GPU support.
    # Pre-import it with stdout suppressed so the message never reaches the
    # console.  The import is guarded so missing bitsandbytes is not an error.
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            import bitsandbytes  # noqa: F401
    except Exception:
        pass
