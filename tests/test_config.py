"""Configs: does an override reach the run, and does a wrong one stop it?

The failure this file exists to prevent is silent. A misspelled override that is
ignored produces a run at the default setting which looks exactly like the run
that was asked for, and the mistake surfaces days later as an unreproducible
number.
"""

import json

import pytest

import xwm
from xwm.config.loader import config_hash, from_dict, load, save, to_dict

BASE = """
seed = 3
output_dir = "somewhere"

[task]
name = "pusht/synthetic"

[model]
name = "jepa/action"

[model.kwargs]
size = "tiny"

[train]
steps = 123

[eval.plan]
horizon = 4
receding_horizon = 2
"""


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "base.toml"
    path.write_text(BASE)
    return path


# -- reading ----------------------------------------------------------------
def test_load_builds_the_whole_tree(config_file):
    config = load(config_file)
    assert config.seed == 3
    assert config.model.kwargs == {"size": "tiny"}
    assert config.train.steps == 123
    assert config.eval.plan.horizon == 4
    assert config.train.batch_size == 64  # untouched default


def test_extends_merges_rather_than_replaces(tmp_path, config_file):
    child = tmp_path / "child.toml"
    child.write_text('extends = "base.toml"\n[train]\nsteps = 9\n')
    config = load(child)
    assert config.train.steps == 9
    assert config.seed == 3  # inherited
    assert config.eval.plan.horizon == 4  # a section the child never mentions


def test_an_unknown_key_is_refused_by_name(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("[train]\nlr = 0.1\n")
    with pytest.raises(ValueError, match="TrainConfig has no field.*'lr'"):
        load(path)


def test_a_missing_file_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "nope.toml")


# -- overrides --------------------------------------------------------------
@pytest.mark.parametrize(
    ("override", "path", "expected"),
    [
        ("train.steps=7", ("train", "steps"), 7),
        ("train.learning_rate=1e-3", ("train", "learning_rate"), 1e-3),
        ("eval.plan.horizon=9", ("eval", "plan", "horizon"), 9),
        ("eval.plan.warm_start=false", ("eval", "plan", "warm_start"), False),
        ("seed=11", ("seed",), 11),
        ("eval.policy=[noop,replay]", ("eval", "policy"), ("noop", "replay")),
    ],
)
def test_overrides_are_coerced_to_the_declared_type(config_file, override, path, expected):
    config = load(config_file, overrides=[override])
    value = config
    for part in path:
        value = getattr(value, part)
    assert value == expected
    assert type(value) is type(expected)


def test_overrides_into_an_untyped_dict_are_literals(config_file):
    config = load(config_file, overrides=["model.kwargs.patch_size=8", "model.kwargs.size=small"])
    assert config.model.kwargs == {"size": "small", "patch_size": 8}


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        ("train.lr=1e-3", "TrainConfig has no field 'lr'"),
        ("nonsense.x=1", "ExperimentConfig has no field 'nonsense'"),
        ("eval.plan.horizen=3", "PlanConfig has no field 'horizen'"),
        ("noequals", "is not `key=value`"),
    ],
)
def test_a_bad_override_fails_loudly(config_file, bad, message):
    with pytest.raises(ValueError, match=message):
        load(config_file, overrides=[bad])


def test_the_error_lists_the_valid_keys(config_file):
    with pytest.raises(ValueError, match="learning_rate"):
        load(config_file, overrides=["train.lr=1e-3"])


# -- round trips and hashing -------------------------------------------------
def test_to_dict_round_trips(config_file):
    config = load(config_file)
    assert to_dict(from_dict(to_dict(config))) == to_dict(config)


def test_config_hash_is_stable_under_key_order(config_file):
    config = load(config_file)
    shuffled = dict(reversed(list(to_dict(config).items())))
    assert config_hash(from_dict(shuffled)) == config_hash(config)


def test_config_hash_changes_with_any_value(config_file):
    base = config_hash(load(config_file))
    assert config_hash(load(config_file, overrides=["train.steps=124"])) != base
    assert config_hash(load(config_file, overrides=["eval.plan.horizon=5"])) != base


def test_config_hash_ignores_where_it_is_written(config_file):
    """Two runs differing only in output path are the same experiment."""
    base = config_hash(load(config_file))
    assert config_hash(load(config_file, overrides=["output_dir=/tmp/elsewhere"])) == base
    assert config_hash(load(config_file, overrides=["name=other"])) == base


def test_save_writes_the_resolved_config_and_keeps_the_source(tmp_path, config_file):
    config = load(config_file, overrides=["train.steps=5"])
    save(config, tmp_path / "run", source=config_file)
    resolved = json.loads((tmp_path / "run" / "config.json").read_text())
    assert resolved["train"]["steps"] == 5
    # The source is kept verbatim, so what was asked for survives beside what it meant.
    assert (tmp_path / "run" / "config.toml").read_text() == BASE


# -- the shipped configs -----------------------------------------------------
@pytest.mark.parametrize("path", sorted(__import__("pathlib").Path("configs").rglob("*.toml")))
def test_every_shipped_config_loads(path):
    config = load(path)
    assert config.task.name in xwm.tasks.available()
    assert config.model.name in xwm.families.available()
