"""Griffe extension that cleans Ludwig-style docstring section headers.

Ludwig docstrings use custom ``# Section`` headers (``# Inputs``, ``# Return``,
etc.) alongside Sphinx ``:param:``/``:return:`` directives.  mkdocstrings already
creates Parameters/Returns sections from the Sphinx directives, so the custom
headers must be stripped to avoid duplication.

``# Example usage:`` is converted to bold text so it's preserved in the output.
"""

from __future__ import annotations

import re
from typing import Any

import griffe

_STRIP_HEADERS = re.compile(
    r"^\s*#\s*(?:Inputs|Return|Returns|Raises|Args|String)\s*$",
    re.MULTILINE,
)

_EXAMPLE_HEADER = re.compile(
    r"^\s*#\s*(Example\s+usage:?)\s*$",
    re.MULTILINE,
)


class LudwigDocstrings(griffe.Extension):
    """Strip Ludwig custom section headers that duplicate mkdocstrings sections."""

    def on_instance(self, *, obj: griffe.Object, **kwargs: Any) -> None:
        if obj.docstring:
            obj.docstring.value = self._clean(obj.docstring.value)

    @staticmethod
    def _clean(text: str) -> str:
        text = _STRIP_HEADERS.sub("", text)
        text = _EXAMPLE_HEADER.sub(r"    **\1**", text)
        return text
