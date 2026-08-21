"""Render the library's docstrings, which are reStructuredText-flavoured.

Every module in ``xwm`` documents itself in the style Python itself uses:
``:mod:`` cross-references, RST simple tables, doctest examples. Sphinx reads
that; MkDocs does not, so a ``:func:`xwm.objectives.sigreg``` would reach the
page verbatim and a table would arrive as a wall of ``=`` signs.

This Griffe extension rewrites those four constructs into Markdown before
mkdocstrings parses the docstring. It is deliberately narrow: the Google-style
sections (``Args:``, ``Returns:``, ``Raises:``) that carry the parameter
documentation are left untouched for the Google parser to handle.

Converted, in this order
------------------------
RST simple tables      a Markdown pipe table, header row optional
``:role:`target```     an inline code span; ``~`` keeps the last component only
``>>>`` blocks         a fenced ``pycon`` block, output included
``text::``             ``text:``, leaving the indented literal block as code

Tables go first because their columns are read off the ``=`` rule by character
offset: rewriting a ``:mod:`xwm.core``` into ``xwm.core`` first would shorten the
cell and shift every column after it.
"""

from __future__ import annotations

import re

from griffe import Extension, Object

#: Sphinx roles the docstrings use. The target becomes a code span rather than a
#: link: an unresolvable cross-reference fails a ``--strict`` build, and these
#: names are already linked from the API reference nav.
ROLE = re.compile(
    r":(?:mod|class|func|meth|data|attr|obj|exc|ref|term|doc|py:\w+):"
    r"`(?P<tilde>~)?(?P<target>[^`<>]+?)(?:\s*<[^`>]+>)?`"
)

#: A run of ``=`` per column, which opens, closes and optionally splits a table.
SEPARATOR = re.compile(r"^(?P<indent>\s*)(?P<rule>=+(?:\s+=+)+)\s*$")

#: A line ending a paragraph that introduces an RST literal block.
LITERAL = re.compile(r"(?<![:\s])::\s*$")


def _roles(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        target = match["target"].strip()
        if match["tilde"]:
            target = target.rsplit(".", 1)[-1]
        return f"`{target}`"

    return ROLE.sub(replace, text)


def _spans(rule: str, offset: int) -> list[tuple[int, int]]:
    """Column ``(start, stop)`` slices, read off a separator line."""
    return [(m.start() + offset, m.end() + offset) for m in re.finditer(r"=+", rule)]


def _cells(line: str, spans: list[tuple[int, int]]) -> list[str]:
    cells = []
    for i, (start, _) in enumerate(spans):
        # The final column runs to the end of the line; the others may also
        # overflow their rule by a character or two, so read to the next start
        # rather than to the end of their own rule.
        end = len(line) if i == len(spans) - 1 else spans[i + 1][0]
        cells.append(line[start:end].strip().replace("|", r"\|"))
    return cells


def _row(cells: list[str], width: int) -> str:
    padded = cells + [""] * (width - len(cells))
    return "| " + " | ".join(padded[:width]) + " |"


def _table(block: list[str], spans: list[tuple[int, int]], indent: str) -> list[str]:
    """Turn one RST simple table into Markdown pipe rows."""
    rules = [i for i, line in enumerate(block) if SEPARATOR.match(line)]
    width = len(spans)
    body_start = rules[1] + 1 if len(rules) >= 3 else rules[0] + 1
    head = (
        _cells(block[rules[0] + 1], spans)
        if len(rules) >= 3
        else [""] * width  # a headerless table still needs a header row
    )
    rows = [
        _cells(line, spans)
        for i, line in enumerate(block[body_start : rules[-1]], start=body_start)
        if line.strip()
    ]
    out = [indent + _row(head, width), indent + _row(["---"] * width, width)]
    out += [indent + _row(cells, width) for cells in rows]
    return out


def _tables(text: str) -> str:
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        match = SEPARATOR.match(lines[i])
        if match is None:
            out.append(lines[i])
            i += 1
            continue
        # Collect to the blank line that ends the table, then convert if it is
        # well formed (opened and closed by a rule); otherwise pass it through.
        end = i
        while end < len(lines) and lines[end].strip():
            end += 1
        block = lines[i:end]
        rules = [n for n, line in enumerate(block) if SEPARATOR.match(line)]
        indent = match["indent"]
        spans = _spans(match["rule"], len(indent))
        if len(rules) >= 2 and len(spans) >= 2:
            out += _table(block, spans, indent)
        else:
            out += block
        i = end
    return "\n".join(out)


def _doctests(text: str) -> str:
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].lstrip()
        if not stripped.startswith(">>>"):
            out.append(lines[i])
            i += 1
            continue
        indent = lines[i][: len(lines[i]) - len(stripped)]
        block: list[str] = []
        while i < len(lines) and lines[i].strip():
            block.append(lines[i][len(indent) :] if lines[i].startswith(indent) else lines[i])
            i += 1
        out += [f"{indent}```pycon", *[indent + line for line in block], f"{indent}```"]
    return "\n".join(out)


def _rst_to_markdown(text: str) -> str:
    # Tables first: their columns are read off the ``=`` rule by character
    # offset, so nothing may change the width of a cell before they are parsed.
    text = _tables(text)
    text = _roles(text)
    text = _doctests(text)
    return LITERAL.sub(":", text)


class RstToMarkdown(Extension):
    """Rewrite RST constructs in every docstring Griffe loads."""

    def on_instance(self, *, obj: Object, **kwargs) -> None:
        if obj.docstring is not None and obj.docstring.value:
            obj.docstring.value = _rst_to_markdown(obj.docstring.value)
