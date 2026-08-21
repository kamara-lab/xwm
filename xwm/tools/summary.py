"""Model introspection."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax

from ..core.types import PyTree


def count_params(tree: PyTree) -> int:
    """Total number of inexact-array scalars in ``tree``."""
    leaves = jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_inexact_array))
    return sum(int(x.size) for x in leaves)


def param_bytes(tree: PyTree) -> int:
    """Bytes occupied by the parameters, at their current dtypes."""
    leaves = jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_inexact_array))
    return sum(int(x.size) * x.dtype.itemsize for x in leaves)


def _children(module: Any) -> list[tuple[str, Any]]:
    if not isinstance(module, eqx.Module):
        return []
    out = []
    for field in module.__dataclass_fields__:
        value = getattr(module, field, None)
        if isinstance(value, eqx.Module):
            out.append((field, value))
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                if isinstance(item, eqx.Module):
                    out.append((f"{field}[{i}]", item))
    return out


def summary(model: PyTree, *, max_depth: int = 2, collapse_lists: bool = True) -> str:
    """A parameter-count tree, in the spirit of ``torchinfo``.

    Args:
        max_depth: how far to descend before summarising a subtree as a total.
        collapse_lists: print ``blocks[0]`` and note the repeat count instead of
            listing every identical transformer block.
    """
    total = count_params(model)
    lines = [
        f"{type(model).__name__}: {total:,} params ({param_bytes(model) / 1e6:.1f} MB)",
    ]

    def walk(module: Any, depth: int, prefix: str) -> None:
        if depth > max_depth:
            return
        children = _children(module)
        seen_prefixes: set[str] = set()
        for name, child in children:
            base = name.split("[")[0]
            if collapse_lists and "[" in name:
                if base in seen_prefixes:
                    continue
                seen_prefixes.add(base)
                repeats = sum(1 for n, _ in children if n.split("[")[0] == base)
                label = f"{base}[x{repeats}]"
                count = count_params(child) * repeats
            else:
                label, count = name, count_params(child)
            share = 100.0 * count / total if total else 0.0
            lines.append(
                f"{'  ' * depth}{prefix}{label}: {type(child).__name__} "
                f"-- {count:,} ({share:.1f}%)"
            )
            walk(child, depth + 1, prefix)

    walk(model, 1, "")
    return "\n".join(lines)
