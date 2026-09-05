"""The ``xwm`` console script: train, evaluate, and ask what is available.

Deliberately not imported by ``xwm/__init__.py``. Importing the library should
not cost an argparse tree, and a subcommand's own imports -- a simulator, a
dataset reader -- happen when that subcommand runs and not before.

    xwm tasks list
    xwm train configs/pusht/smoke.toml train.steps=2000
    xwm eval runs/pusht-synthetic-jepa-action-s0-9f2c1a0b4e7d
    xwm doctor

Overrides are ``key.path=value`` and are checked against the config schema, so
a misspelled one stops the run instead of silently doing nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

__all__ = ["main"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xwm",
        description="Train and evaluate action-conditioned world models.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train a model from a config file")
    train.add_argument("config", help="path to a TOML config")
    train.add_argument("overrides", nargs="*", help="key.path=value")
    train.add_argument("--output-dir", default=None, help="where run directories go")
    train.add_argument("--name", default=None, help="name this run")
    train.add_argument("--quiet", action="store_true")
    train.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve the config, print it and the run directory, and stop",
    )
    train.add_argument("--eval", action="store_true", help="evaluate immediately after training")

    evaluate = sub.add_parser("eval", help="evaluate a trained run")
    evaluate.add_argument("run", help="a run directory written by `xwm train`")
    evaluate.add_argument("overrides", nargs="*", help="key.path=value")
    evaluate.add_argument("--checkpoint", default=None, help="weights, if not in the run")
    evaluate.add_argument(
        "--policy", action="append", default=None, help="repeatable; overrides eval.policy"
    )
    evaluate.add_argument("--quiet", action="store_true")

    for name, help_text in (
        ("tasks", "benchmark tasks"),
        ("datasets", "recorded datasets"),
        ("models", "registered model families"),
    ):
        listing = sub.add_parser(name, help=f"list or describe {help_text}")
        listing.add_argument("action", nargs="?", default="list", choices=["list", "describe"])
        listing.add_argument("target", nargs="?", default=None)

    sub.add_parser("doctor", help="report which optional pieces this environment has")
    return parser


def _cmd_train(args) -> int:
    from ..config.loader import load

    overrides = list(args.overrides)
    if args.output_dir:
        overrides.append(f"output_dir={args.output_dir}")
    if args.name:
        overrides.append(f"name={args.name}")
    config = load(args.config, overrides=overrides)

    from .train import run, run_directory

    if args.dry_run:
        from ..config.loader import to_dict

        print(json.dumps(to_dict(config), indent=2, default=str))
        print(f"\nwould write to {run_directory(config)}")
        return 0

    directory = run(config, source=Path(args.config), progress=not args.quiet)
    print(f"run written to {directory}")
    if args.eval:
        from .evaluate import run as run_eval

        print(run_eval(config, directory, progress=not args.quiet))
    return 0


def _cmd_eval(args) -> int:
    from ..config.loader import apply_overrides, from_dict, to_dict
    from .evaluate import load_run, run

    config, directory = load_run(args.run)
    overrides = list(args.overrides)
    if args.policy:
        overrides.append(f"eval.policy=[{','.join(args.policy)}]")
    if overrides:
        config = from_dict(apply_overrides(to_dict(config), overrides))
    path = run(config, directory, checkpoint=args.checkpoint, progress=not args.quiet)
    print(f"results written to {path}")
    return 0


def _cmd_listing(args) -> int:
    if args.command == "tasks":
        from .. import tasks as module

        names, describe = module.available(), module.describe
    elif args.command == "datasets":
        from .. import datasets as module

        names, describe = module.available(), module.describe
    else:
        from ..families import registry

        names, describe = registry.available(), None

    if args.action == "describe":
        if args.target is None:
            print(f"describe what? one of: {', '.join(names)}", file=sys.stderr)
            return 2
        if describe is None:
            print(f"{args.target}: a registered model family", file=sys.stderr)
            return 0
        import dataclasses

        entry = describe(args.target)
        for name, value in dataclasses.asdict(entry).items():
            print(f"{name:>18}: {value}")
        return 0

    for name in names:
        summary = ""
        if describe is not None:
            text = getattr(describe(name), "summary", "")
            # Split on sentence ends, not on every dot: these summaries name
            # modules like `xwm.data.PushWorld`, and splitting on "." truncates
            # them mid-identifier.
            summary = text.split(". ")[0].rstrip(".") if text else ""
        print(f"{name:<34} {summary}")
    return 0


def _cmd_doctor(args) -> int:
    from ..datasets.cache import which_readers
    from ..envs import which_backends
    from ..tools.cache import cache_dir

    def report(title, states):
        print(title)
        for name, ok in sorted(states.items()):
            print(f"  {'yes' if ok else ' no'}  {name}")

    import xwm

    print(f"xwm {xwm.__version__}")
    import jax

    print(f"jax {jax.__version__} on {jax.default_backend()} ({len(jax.devices())} device(s))")
    print(f"cache {cache_dir()}")
    report("dataset readers:", which_readers())
    report("render backends:", which_backends())
    report("simulators:", which_simulators())
    return 0


def which_simulators() -> dict[str, bool]:
    """Which simulators this environment can run.

    The counterpart of :func:`xwm.datasets.which_readers` and
    :func:`xwm.envs.which_backends`: ask before planning a run rather than
    finding out from a traceback halfway through one.
    """
    from importlib.util import find_spec

    states = {}
    for name, modules in {
        "synthetic (xwm.data)": (),
        "pusht (gym_pusht)": ("gymnasium", "gym_pusht"),
        "ogbench": ("ogbench", "mujoco"),
        "franka (newton)": ("newton", "warp"),
    }.items():
        try:
            states[name] = all(find_spec(m) is not None for m in modules)
        except (ImportError, ValueError):  # pragma: no cover - broken installs
            states[name] = False
    return states


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code rather than raising."""
    args = _parser().parse_args(argv)
    handlers = {
        "train": _cmd_train,
        "eval": _cmd_eval,
        "tasks": _cmd_listing,
        "datasets": _cmd_listing,
        "models": _cmd_listing,
        "doctor": _cmd_doctor,
    }
    try:
        return handlers[args.command](args)
    except (ValueError, KeyError, FileNotFoundError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
