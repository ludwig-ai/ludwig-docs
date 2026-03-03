"""MkDocs hook that syncs CONTRIBUTING.md from the Ludwig repository.

On ``on_pre_build`` the hook downloads the latest ``CONTRIBUTING.md`` from the
Ludwig main branch and rewrites old ``ludwig-ai.github.io/ludwig-docs/`` URLs
to ``ludwig.ai/``.  If the download fails the existing local copy is kept.
"""

from __future__ import annotations

import logging
import pathlib
import urllib.request

log = logging.getLogger("mkdocs.hooks.contributing")

_SOURCE_URL = (
    "https://raw.githubusercontent.com/ludwig-ai/ludwig/main/CONTRIBUTING.md"
)
_DEST = pathlib.Path("docs/developer_guide/contributing.md")


def on_pre_build(**kwargs) -> None:  # noqa: ARG001
    try:
        with urllib.request.urlopen(_SOURCE_URL, timeout=30) as resp:
            text = resp.read().decode()
    except Exception:
        log.warning(
            "Failed to download CONTRIBUTING.md; using existing local copy."
        )
        return

    text = text.replace("ludwig-ai.github.io/ludwig-docs/", "ludwig.ai/")
    _DEST.parent.mkdir(parents=True, exist_ok=True)
    _DEST.write_text(text)
    log.info("Synced CONTRIBUTING.md from Ludwig repository.")
