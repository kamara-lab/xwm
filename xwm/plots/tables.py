"""Result tables and metrics, as JSON and LaTeX.

Numbers that took an hour of compute to produce should not have to be
copy-pasted out of a terminal by hand. Every example in xwm routes its results
through :func:`save_table`, which writes both formats from one call:

* **JSON** for anything that reads the numbers back -- a later script, a sweep
  aggregator, a plot. It carries *full precision*, because rounding belongs to
  display and not to storage.
* **LaTeX** for anything a person reads, where the rounding is the point.

:func:`markdown_table` remains for printing a table to a terminal; it is just
not written to disk.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

Row = Sequence[Any]

#: Characters LaTeX treats specially in ordinary text.
_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def escape_latex(text: str) -> str:
    """Escape LaTeX special characters. Metric names like ``feature_std`` need it."""
    return "".join(_LATEX_ESCAPES.get(ch, ch) for ch in text)


def format_cell(value: Any, float_format: str = "{:.4f}") -> str:
    """Render one cell: real numbers via ``float_format``, everything else via ``str``.

    Unwraps 0-d NumPy and JAX scalars first. They are not Python ``float``, so a
    bare ``isinstance`` check silently lets them through unformatted -- and they
    are exactly what a training loop hands to a results table.
    """
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    item = getattr(value, "item", None)
    if item is not None and getattr(value, "ndim", 0) == 0:
        value = item()
    if isinstance(value, bool):  # numpy bool unwraps to a Python bool
        return "yes" if value else "no"
    if isinstance(value, float):
        return float_format.format(value)
    if isinstance(value, int):
        return str(value)
    return str(value)


def _cells(rows: Sequence[Row], float_format: str) -> list[list[str]]:
    return [[format_cell(v, float_format) for v in row] for row in rows]


def markdown_table(
    headers: Sequence[str],
    rows: Sequence[Row],
    *,
    float_format: str = "{:.4f}",
) -> str:
    """A GitHub-flavoured Markdown table, column-aligned for readability.

    For printing to a terminal or pasting into a README; :func:`save_table`
    writes JSON and LaTeX to disk, not this.
    """
    body = _cells(rows, float_format)
    head = [str(h) for h in headers]
    widths = [
        max(len(head[i]), *(len(r[i]) for r in body)) if body else len(head[i])
        for i in range(len(head))
    ]
    line = "| " + " | ".join(h.ljust(w) for h, w in zip(head, widths, strict=True)) + " |"
    rule = "| " + " | ".join("-" * w for w in widths) + " |"
    out = [line, rule]
    out += [
        "| " + " | ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) + " |"
        for row in body
    ]
    return "\n".join(out)


def latex_table(
    headers: Sequence[str],
    rows: Sequence[Row],
    *,
    caption: str | None = None,
    label: str | None = None,
    align: str | None = None,
    float_format: str = "{:.4f}",
    escape: bool = True,
    booktabs: bool = True,
) -> str:
    """A ``tabular`` (optionally wrapped in ``table``) using booktabs rules.

    Args:
        align: column spec such as ``"lrrr"``; defaults to left for the first
            column and right for the rest, which is what numeric tables want.
        escape: escape LaTeX specials in cells. Turn it off to pass math through.
        booktabs: use ``\\toprule``/``\\midrule``/``\\bottomrule`` (needs the
            ``booktabs`` package); otherwise plain ``\\hline``.
    """
    body = _cells(rows, float_format)
    head = [str(h) for h in headers]
    if escape:
        head = [escape_latex(h) for h in head]
        body = [[escape_latex(c) for c in row] for row in body]
    if align is None:
        align = "l" + "r" * (len(head) - 1)
    if len(align) != len(head):
        raise ValueError(f"align {align!r} has {len(align)} columns, headers have {len(head)}")

    top, mid, bottom = (
        (r"\toprule", r"\midrule", r"\bottomrule") if booktabs else (r"\hline",) * 3
    )
    lines = [f"\\begin{{tabular}}{{{align}}}", top, " & ".join(head) + r" \\", mid]
    lines += [" & ".join(row) + r" \\" for row in body]
    lines += [bottom, r"\end{tabular}"]
    tabular = "\n".join(lines)

    if caption is None and label is None:
        return tabular
    wrapped = [r"\begin{table}[t]", r"\centering", tabular]
    if caption is not None:
        wrapped.append(f"\\caption{{{escape_latex(caption) if escape else caption}}}")
    if label is not None:
        wrapped.append(f"\\label{{{label}}}")
    wrapped.append(r"\end{table}")
    return "\n".join(wrapped)


def jsonable(value: Any) -> Any:
    """Convert a value to something :mod:`json` can encode, without rounding.

    NumPy and JAX scalars become Python numbers; non-finite floats become
    ``None``, since JSON has no NaN or Infinity and emitting bare ``NaN``
    produces a file that strict parsers reject.
    """
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (int,)):
        return value
    item = getattr(value, "item", None)
    if item is not None and getattr(value, "ndim", 0) == 0:
        value = item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if isinstance(value, (int, float)):
        return value
    return str(value)


def table_to_dict(
    headers: Sequence[str],
    rows: Sequence[Row],
    *,
    caption: str | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """The JSON payload for a table: full-precision values keyed by column."""
    columns = [str(h) for h in headers]
    payload: dict[str, Any] = {
        "columns": columns,
        "rows": [
            {column: jsonable(value) for column, value in zip(columns, row, strict=True)}
            for row in rows
        ],
    }
    if caption is not None:
        payload["caption"] = caption
    if label is not None:
        payload["label"] = label
    return payload


def save_json(path: str | Path, payload: Any) -> Path:
    """Write ``payload`` as pretty-printed JSON, creating parent directories."""
    path = Path(path).with_suffix(".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=False) + "\n")
    return path


def save_metrics(path_stem: str | Path, metrics: Mapping[str, Any]) -> Path:
    """Write a flat mapping of scalar metrics to ``<stem>.json``."""
    return save_json(Path(path_stem), dict(metrics))


def save_table(
    path_stem: str | Path,
    headers: Sequence[str],
    rows: Sequence[Row],
    *,
    caption: str | None = None,
    label: str | None = None,
    align: str | None = None,
    float_format: str = "{:.4f}",
) -> dict[str, Path]:
    """Write ``<stem>.json`` (full precision) and ``<stem>.tex`` (formatted).

    Returns a mapping from extension to the path written.
    """
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)

    json_path = save_json(stem, table_to_dict(headers, rows, caption=caption, label=label))
    tex = stem.with_suffix(".tex")
    tex.write_text(
        latex_table(
            headers, rows, caption=caption, label=label, align=align, float_format=float_format
        )
        + "\n"
    )
    return {"json": json_path, "tex": tex}
