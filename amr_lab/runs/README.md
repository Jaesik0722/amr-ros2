# amr_lab/runs: how the runs were actually driven

`train.py`, `eval_pairs.py` and `capture.py` take a lot of flags, and every number this
project reports came from a particular combination of them repeated a particular number
of times. These are those combinations, kept as they were run rather than tidied into
something that never executed.

They are a record first and a tool second. A script here names checkpoints by the
directory the run that produced them wrote, so one will only reproduce its own results on
a machine that has those runs; what it always shows is the exact arguments, the seeds and
the order.

## Conventions

Every script resolves the repository from its own path, so it can be called from
anywhere:

```bash
./amr_lab/runs/pairs13.sh
```

Two environment variables matter:

| Variable | Default | What it is |
|---|---|---|
| `ISAAC_PYTHON` | `$HOME/anaconda3/envs/isaaclab/bin/python` | the interpreter Isaac Lab is installed in |
| `AMR_RESULTS` | `$HOME/amr_results` | where CSVs, logs and recordings are written |

Results go outside the repository on purpose: they are data, not source. Checkpoints land
in `logs/` at the repository root, which is ignored.

## What is here

| Prefix | What it drives |
|---|---|
| `campaign*.sh` | training campaigns: the four hitch conditions, both cart constructions, three seeds |
| `pairs*.sh` | paired evaluation with `eval_pairs.py`: the tables, the ablations, the robustness rows |
| `authority2.sh` | the open-loop authority grid, measured without a policy |
| `highlight*.sh`, `jack200*.sh`, `caster_rear.sh` | recording the clips the videos are made of |
| `fullpose_zeroshot.sh`, `rb_*.sh`, `sens*.sh` | the out-of-distribution replays |
| `status.sh`, `watch*.sh` | watching a long campaign without blocking on it |

The numbering is chronological, not a ranking: `pairs13.sh` is the thirteenth evaluation
session, and an earlier one is not superseded unless its own comment says so.

## A warning about reproducing them

Per-scenario outcomes are **not** reproducible run to run. Two runs of the same
configuration on the same 256 scenarios flip about one scenario in eight, and the
aggregate over 1,024 scenarios moves by one to two points. The ranking of the conditions
is stable; a single episode is not. A script that searches for a scenario to record
therefore searches and records under the same settings, because a scenario found one way
can come back the other way when it is recorded under another.

The absolute level also depends on how many environments share the PhysX scene: the same
checkpoint on the same 1,024 scenarios reaches 75.2% at 64 environments and 60.4% at 256.
Everything reported was measured at 256, which is what these scripts pass.
