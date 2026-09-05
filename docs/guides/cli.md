# The `xwm` command

Installing the package puts an `xwm` command on the path. It exists so that an
experiment is a file rather than a script: the same config trains, evaluates and
reproduces, and the run directory holds everything needed to do it again.

```bash
xwm doctor                                  # what this environment can actually run
xwm tasks list                              # registered benchmark tasks
xwm datasets describe lerobot/pusht         # what a download would cost, before it starts
xwm models list                             # registered model families

xwm train configs/pusht/smoke.toml          # train, writing runs/<name>-<hash>/
xwm train configs/pusht/smoke.toml --eval   # ... and evaluate immediately
xwm eval runs/smoke-69bc402687cb            # evaluate a finished run
```

## Overrides

Any config field can be set from the command line with a dotted path:

```bash
xwm train configs/pusht/smoke.toml train.steps=2000 eval.plan.horizon=8
xwm eval runs/my-run eval.episodes=50 eval.policy=[replay,noop]
```

Values are coerced by the field's declared type, so `train.steps=4` is an `int`
and `eval.plan.warm_start=false` is a `bool`. Keys under an untyped dict such as
`model.kwargs` are parsed as literals.

**A path the schema does not have is an error**, listing the ones it does:

```
$ xwm train configs/pusht/smoke.toml train.lr=1e-3
error: override 'train.lr=1e-3': TrainConfig has no field 'lr'; valid keys are
['batch_size', 'checkpoint_every', 'clip_length', 'grad_clip', 'learning_rate', ...]
```

This is deliberate. A silently ignored override produces a run at the default
setting that looks exactly like the run you asked for, and the mistake surfaces
days later as a number nobody can reproduce.

`--dry-run` resolves the config, prints it and the directory it would write, and
stops -- the cheap answer to "did my override apply?".

## What a config may not say

`action_dim`, `img_size` and `in_channels` come from the task, not the config.
A model cannot be built that disagrees with the data it is about to be trained
on, and trying is an error rather than an override:

```
$ xwm train configs/pusht/smoke.toml model.kwargs.action_dim=7
error: model.kwargs sets ['action_dim'], which task 'pusht/synthetic' determines
(action_dim=2). Change the task, or its overrides, rather than the model.
```

## Run directories

```
runs/<name>-<config hash>/
  config.toml       the source file, verbatim
  config.json       the resolved config, which `xwm eval` reads back
  checkpoint/model.eqx        weights
  checkpoint/model.eqx.json   what built them
  history.json      training metrics
  training_curve.png
  eval.json         results, schema xwm.bench.goal_reaching/1
  eval_table.{json,tex}
```

The hash covers everything that defines the run and deliberately excludes
`output_dir` and `name`: two runs differing only in where they are written are
the same experiment.

## Relation to the examples and to Modal

The eight numbered examples and the `deploy/` Modal apps are unchanged and
remain the way the published figures are produced. They configure themselves
through `XWM_*` environment variables and `deploy/_shared.py`'s presets; a
preset entry maps one-to-one onto a CLI override (`BATCH=64` becomes
`train.batch_size=64`), so a future benchmark app is a thin wrapper around
`xwm train` rather than a second copy of a pipeline.
