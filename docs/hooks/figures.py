"""Publish the figures the examples wrote, without copying them into ``docs/``.

``examples/outputs/`` is the output of a real run -- 13 MB of PNGs and GIFs that
belong to the examples, not to the documentation. Rather than duplicate them
under ``docs/``, this hook registers each one with MkDocs at build time, so pages
can link to ``outputs/<example>/<file>`` and the file lands in the built site
exactly once.

A checkout where the examples have never been run still has to build, and build
``--strict``: a page linking to a figure that does not exist is a broken link,
and a broken link fails the build. So any referenced figure that is missing gets
a 1x1 placeholder registered in its place. The page renders with its caption, the
build stays strict for *real* broken links, and the log says how many figures
were stubbed and how to get the real ones.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from mkdocs.structure.files import File

log = logging.getLogger("mkdocs.hooks.figures")

#: Where the examples write, relative to the repository root.
SOURCE = Path("examples/outputs")

#: The prefix pages link under, and the directory published in the built site.
PREFIX = "outputs"

#: What to publish. Anything else an example emits (``.json``, ``.tex``) is quoted
#: inline in the prose instead, where it can be given context.
SUFFIXES = (".png", ".gif")

#: A figure reference in a Markdown page: ``](../outputs/05_planning/x.png)``.
REFERENCE = re.compile(rf"\]\((?:\.\./)*({PREFIX}/[^)\s]+\.(?:png|gif))")

#: A 1x1 transparent pixel, per format. Small enough to be free, valid enough
#: that a browser renders it rather than showing a broken-image icon.
PLACEHOLDER: dict[str, bytes] = {
    ".png": (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
        b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    ),
    ".gif": (
        b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04"
        b"\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D"
        b"\x01\x00;"
    ),
}


def _available(root: Path) -> dict[str, Path]:
    """Every publishable figure on disk, keyed by the URI pages link to."""
    source = root / SOURCE
    if not source.is_dir():
        return {}
    return {
        f"{PREFIX}/{path.relative_to(source).as_posix()}": path
        for path in sorted(source.rglob("*"))
        if path.suffix.lower() in SUFFIXES and path.is_file()
    }


def _referenced(docs_dir: Path) -> set[str]:
    """Every figure URI the pages actually link to."""
    found: set[str] = set()
    for page in docs_dir.rglob("*.md"):
        found.update(REFERENCE.findall(page.read_text(encoding="utf-8")))
    return found


def _register(config, uri: str, path: Path) -> File:
    """A ``File`` for *path*, served at *uri*, across MkDocs versions."""
    generated = getattr(File, "generated", None)
    if generated is not None:  # MkDocs >= 1.6
        return generated(config, uri, abs_src_path=str(path))
    return File(uri, str(path.parent), config["site_dir"], config["use_directory_urls"])


def on_files(files, config):
    root = Path(config["docs_dir"]).parent
    available = _available(root)
    for uri, path in available.items():
        if files.get_file_from_path(uri) is None:
            files.append(_register(config, uri, path))

    missing = sorted(_referenced(Path(config["docs_dir"])) - available.keys())
    for uri in missing:
        placeholder = PLACEHOLDER.get(Path(uri).suffix.lower())
        if placeholder is not None and files.get_file_from_path(uri) is None:
            files.append(File.generated(config, uri, content=placeholder))

    log.info("figures: published %d from %s", len(available), SOURCE)
    if missing:
        # INFO, not WARNING: a checkout where the examples have never been run is
        # a normal state, and --strict turns a warning here into a failed build.
        log.info(
            "figures: %d referenced figure(s) missing and stubbed with a placeholder "
            "-- run the examples to generate them (first missing: %s)",
            len(missing),
            missing[0],
        )
    return files
