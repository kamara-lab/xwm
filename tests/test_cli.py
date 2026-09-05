"""The console script, driven through ``main([...])`` rather than a subprocess.

Direct calls keep the tests fast and give real tracebacks when something breaks.
What is checked is the contract a person relies on: a command either does the
thing and returns 0, or explains itself and returns non-zero.
"""

import json

import pytest

from xwm.bench.protocol import SCHEMA
from xwm.cli import main, which_simulators


@pytest.fixture
def runs(tmp_path, monkeypatch):
    """An isolated cache and output directory, so tests never touch the real ones."""
    monkeypatch.setenv("XWM_DATA_HOME", str(tmp_path / "cache"))
    return tmp_path / "runs"


@pytest.fixture
def smoke_config(tmp_path):
    """The shipped smoke config, made smaller still."""
    path = tmp_path / "smoke.toml"
    path.write_text(
        "[task]\nname = 'pusht/synthetic'\n"
        "[task.overrides]\nnative_size = [16, 16]\n"
        "[model]\nname = 'jepa/action'\n"
        "[model.kwargs]\nsize = 'tiny'\npatch_size = 8\nfreeze_encoder = false\n"
        "[train]\nsteps = 2\nbatch_size = 2\n"
        "[eval]\nepisodes = 2\npolicy = ['noop']\n"
        "[eval.plan]\nhorizon = 2\nreceding_horizon = 1\n"
    )
    return path


# -- listings ---------------------------------------------------------------
@pytest.mark.parametrize(
    "argv",
    [
        ["tasks", "list"],
        ["datasets", "list"],
        ["models", "list"],
        ["doctor"],
        ["tasks", "describe", "pusht/synthetic"],
        ["datasets", "describe", "lerobot/pusht"],
    ],
)
def test_informational_commands_succeed(argv, capsys):
    assert main(argv) == 0
    assert capsys.readouterr().out.strip()


def test_describe_without_a_target_explains_itself(capsys):
    assert main(["tasks", "describe"]) == 2
    assert "describe what?" in capsys.readouterr().err


def test_an_unknown_task_is_an_error_not_a_traceback(capsys):
    assert main(["tasks", "describe", "no/such"]) == 1
    assert "unknown task" in capsys.readouterr().err


def test_help_exits_cleanly():
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0


def test_which_simulators_reports_the_dependency_free_one():
    states = which_simulators()
    assert states["synthetic (xwm.data)"] is True
    assert set(states) >= {"pusht (gym_pusht)", "ogbench", "franka (newton)"}


# -- train ------------------------------------------------------------------
def test_dry_run_resolves_without_training(smoke_config, runs, capsys):
    argv = ["train", str(smoke_config), "train.steps=99", f"--output-dir={runs}", "--dry-run"]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert '"steps": 99' in out
    assert "would write to" in out
    assert not runs.exists()  # nothing was written


def test_train_writes_a_complete_run_directory(smoke_config, runs):
    assert main(["train", str(smoke_config), f"--output-dir={runs}", "--quiet"]) == 0
    (directory,) = list(runs.iterdir())
    assert (directory / "config.json").exists()
    assert (directory / "config.toml").exists()  # the source, verbatim
    assert (directory / "checkpoint" / "model.eqx").exists()
    assert (directory / "checkpoint" / "model.eqx.json").exists()  # rebuilds the weights
    history = json.loads((directory / "history.json").read_text())
    assert history and "loss" in history[0]


def test_a_bad_override_stops_the_run(smoke_config, runs, capsys):
    assert main(["train", str(smoke_config), "train.lr=1", f"--output-dir={runs}"]) == 1
    assert "has no field 'lr'" in capsys.readouterr().err
    assert not runs.exists()


def test_the_task_owns_the_model_shape(smoke_config, runs, capsys):
    """A config that contradicts its task must fail, not quietly win."""
    code = main(["train", str(smoke_config), "model.kwargs.action_dim=7", f"--output-dir={runs}"])
    assert code == 1
    assert "which task" in capsys.readouterr().err


# -- eval -------------------------------------------------------------------
def test_eval_writes_results_matching_the_schema(smoke_config, runs, capsys):
    main(["train", str(smoke_config), f"--output-dir={runs}", "--quiet"])
    (directory,) = list(runs.iterdir())
    assert main(["eval", str(directory), "--quiet"]) == 0

    payload = json.loads((directory / "eval.json").read_text())
    assert payload["schema"] == SCHEMA
    assert payload["task"] == "pusht/synthetic"
    assert payload["model"]["name"] == "jepa/action"
    assert payload["model"]["config_hash"]
    assert len(payload["instances"]) == payload["episodes"] == 2
    numbers = payload["results"]["noop"]
    assert 0.0 <= numbers["success_rate"] <= 1.0
    assert len(numbers["per_episode"]) == 2
    # A table for a paper, beside the numbers for a program.
    assert (directory / "eval_table.tex").exists()


def test_eval_can_select_policies(smoke_config, runs):
    main(["train", str(smoke_config), f"--output-dir={runs}", "--quiet"])
    (directory,) = list(runs.iterdir())
    assert main(["eval", str(directory), "--policy", "replay", "--policy", "noop", "--quiet"]) == 0
    payload = json.loads((directory / "eval.json").read_text())
    assert set(payload["results"]) == {"replay", "noop"}


def test_eval_needs_a_run_directory(tmp_path, capsys):
    assert main(["eval", str(tmp_path)]) == 1
    assert "no config.json" in capsys.readouterr().err


def test_instances_are_recorded_so_a_run_can_be_repeated(smoke_config, runs):
    main(["train", str(smoke_config), f"--output-dir={runs}", "--quiet"])
    (directory,) = list(runs.iterdir())
    main(["eval", str(directory), "--quiet"])
    first = json.loads((directory / "eval.json").read_text())
    main(["eval", str(directory), "--quiet"])
    second = json.loads((directory / "eval.json").read_text())
    assert first["instances"] == second["instances"]
    assert first["results"]["noop"]["success_rate"] == second["results"]["noop"]["success_rate"]
