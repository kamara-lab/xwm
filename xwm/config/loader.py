"""Reading a config: TOML in, dataclass tree out, overrides applied loudly.

Three rules, each from a way configs usually fail:

**An unknown key is an error, naming the valid ones.** A silently ignored
``train.lr=1e-3`` (the field is ``learning_rate``) produces a run at the default
rate that looks exactly like the run you asked for. ``deploy/_shared.py`` already
records this lesson about environment-variable presets; this is the same failure
with a different spelling.

**Overrides are coerced by the field's declared type**, not guessed from the
string, so ``train.steps=4`` is an ``int`` and ``eval.plan.horizon=5`` is too,
while ``model.kwargs.size=small`` -- inside an untyped dict -- stays a string.

**The resolved config is written back as JSON.** TOML has no writer in the
standard library, and rather than add a dependency the run directory keeps the
source file verbatim *and* the fully resolved tree that ``xwm eval`` reads back.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import tomllib
import types
import typing
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .schema import ExperimentConfig

__all__ = [
    "apply_overrides",
    "config_hash",
    "from_dict",
    "load",
    "save",
    "to_dict",
]


def _is_dataclass_type(annotation: Any) -> bool:
    return dataclasses.is_dataclass(annotation) and isinstance(annotation, type)


def _resolve_hints(cls: type) -> dict[str, Any]:
    """Field annotations, with ``from __future__ import annotations`` undone."""
    import xwm.bench.protocol as protocol
    import xwm.config.schema as schema

    return typing.get_type_hints(cls, {**vars(schema), **vars(protocol)})


def _unwrap(annotation: Any) -> Any:
    """The interesting half of ``X | None``."""
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def coerce(annotation: Any, raw: Any) -> Any:
    """Turn a TOML or command-line value into what the field is declared to hold."""
    annotation = _unwrap(annotation)
    origin = typing.get_origin(annotation)
    if annotation is Any or annotation is None:
        return raw
    if raw is None:
        return None
    if origin in (tuple, list) or annotation in (tuple, list):
        args = typing.get_args(annotation)
        items = raw if isinstance(raw, (list, tuple)) else [raw]
        if args and args[-1] is not Ellipsis and len(args) == len(items):
            items = [coerce(a, v) for a, v in zip(args, items, strict=True)]
        elif args:
            items = [coerce(args[0], v) for v in items]
        return tuple(items) if (origin is tuple or annotation is tuple) else list(items)
    if origin is dict or annotation is dict:
        return dict(raw)
    if annotation is bool:
        if isinstance(raw, str):
            if raw.lower() not in ("true", "false", "1", "0", "yes", "no"):
                raise ValueError(f"expected a boolean, got {raw!r}")
            return raw.lower() in ("true", "1", "yes")
        return bool(raw)
    if annotation in (int, float, str):
        return annotation(raw)
    return raw


def from_dict(data: Mapping[str, Any], cls: type = ExperimentConfig) -> Any:
    """Build a config tree, rejecting anything the schema does not declare."""
    hints = _resolve_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(
            f"{cls.__name__} has no field(s) {', '.join(repr(u) for u in unknown)}; "
            f"valid keys are {sorted(known)}"
        )
    built: dict[str, Any] = {}
    for name, value in data.items():
        annotation = _unwrap(hints[name])
        if _is_dataclass_type(annotation) and isinstance(value, Mapping):
            built[name] = from_dict(value, annotation)
        else:
            built[name] = coerce(hints[name], value)
    return cls(**built)


def to_dict(config: Any) -> dict[str, Any]:
    """A JSON-safe mapping, round-trippable through :func:`from_dict`."""
    return dataclasses.asdict(config)


def _split_path(item: str) -> tuple[list[str], str]:
    if "=" not in item:
        raise ValueError(f"override {item!r} is not `key=value`; for example train.steps=2000")
    path, _, value = item.partition("=")
    return path.strip().split("."), value.strip()


def _parse_scalar(text: str) -> Any:
    """Best-effort literal, for values landing in an untyped dict."""
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        return [_parse_scalar(p.strip()) for p in inner.split(",") if p.strip()] if inner else []
    for parse in (json.loads,):
        try:
            return parse(text)
        except (ValueError, TypeError):
            pass
    return text


def apply_overrides(data: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply ``a.b.c=value`` strings onto a nested mapping, in place of nothing.

    A path that does not exist in the schema raises, listing what does -- except
    below an untyped ``dict`` field such as ``model.kwargs``, where any key is
    legitimate and the value is parsed as a literal.
    """
    data = json.loads(json.dumps(data))  # a deep copy that is already JSON-safe
    for item in overrides:
        path, raw = _split_path(item)
        node, cls = data, ExperimentConfig
        for depth, part in enumerate(path[:-1]):
            hints = _resolve_hints(cls) if cls is not None else {}
            if cls is not None and part not in hints:
                raise ValueError(
                    f"override {item!r}: {cls.__name__} has no field {part!r}; "
                    f"valid keys are {sorted(hints)}"
                )
            annotation = _unwrap(hints.get(part)) if cls is not None else None
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(
                    f"override {item!r}: {'.'.join(path[: depth + 1])} is a value, not a section"
                )
            cls = annotation if _is_dataclass_type(annotation) else None
        leaf = path[-1]
        if cls is not None:
            hints = _resolve_hints(cls)
            if leaf not in hints:
                raise ValueError(
                    f"override {item!r}: {cls.__name__} has no field {leaf!r}; "
                    f"valid keys are {sorted(hints)}"
                )
            node[leaf] = coerce(hints[leaf], _parse_scalar(raw))
        else:
            node[leaf] = _parse_scalar(raw)
    return data


def load(path: str | Path, *, overrides: Sequence[str] = ()) -> ExperimentConfig:
    """Read a TOML config, follow one level of ``extends``, and apply overrides."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no config at {path}")
    data = tomllib.loads(path.read_text())
    parent = data.pop("extends", None)
    if parent is not None:
        base = tomllib.loads((path.parent / parent).resolve().read_text())
        base.pop("extends", None)
        data = _merge(base, data)
    return from_dict(apply_overrides(data, overrides))


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Depth-first merge: a section in the child updates, it does not replace."""
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def config_hash(config: Any) -> str:
    """A short digest of everything that defines the run.

    Deliberately excludes ``output_dir``, ``name`` and ``log``: two runs that
    differ only in where they are written, or in whether a viewer was watching,
    are the same experiment, and should not look like different ones in a
    results table -- nor land in differently-named directories.
    """
    payload = to_dict(config)
    for cosmetic in ("output_dir", "name", "log"):
        payload.pop(cosmetic, None)
    digest = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(digest).hexdigest()[:12]


def save(config: ExperimentConfig, directory: str | Path, *, source: Path | None = None) -> Path:
    """Write the resolved config as ``config.json``, keeping the source beside it."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "config.json"
    target.write_text(json.dumps(to_dict(config), indent=2, default=str) + "\n")
    if source is not None and Path(source).exists():
        (directory / "config.toml").write_text(Path(source).read_text())
    return target
