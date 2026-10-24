#!/usr/bin/env bash
# Size the hitch yaw servo without also resizing the hook pitch servo.
#
# `--hitch-torque A B` raised cfg.hitch_torque to B, and the pitch actuator took its
# effort limit from the same field, so the existing 20 and 30 N.m rows changed both
# joints at once. At 30 N.m the pitch servo presses the robot's tail down with about
# 559 N and moves normal force off the drive wheels onto frictionless support casters,
# which is a mechanism with nothing to do with hitch authority. `--pitch-torque 10.6`
# pins the hook servo at the fitted value so the sweep is the yaw limit alone.
#
# Seed 1 only, to pair with the existing 10.6 N.m seed-1 baseline (49.1%).
set -u
PY="env PYTHONUNBUFFERED=1 $HOME/anaconda3/envs/isaaclab/bin/python"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1   # the repository root
LOG=$HOME/amr_results/amr_lab/yawonly.log
OUT=$HOME/amr_results/amr_lab/pairs
TRAIN="--headless --num-envs 2048 --episode-length 20 --floor-friction 0.6 1.0 \
  --max-iterations 3000 --cart-wheels caster4"
EVAL="--headless --num-envs 256 --episodes 4 --episode-length 20 \
  --floor-friction 0.6 1.0 --cart-wheels caster4 --scenario-seed 7 --curriculum 0.024"
say() { echo "$*" | tee -a "$LOG"; }

train() {  # run-name extra...
  local name=$1; shift
  if ls logs/amr_lab/*/*_"$name"/model_2999.pt >/dev/null 2>&1; then
    say "# $name already trained"; return; fi
  say ""; say "---- train $name  [$(date '+%F %T')]"
  $PY amr_lab/train.py $TRAIN --run-name "$name" "$@" > /tmp/yo_$name.log 2>&1
  if ls logs/amr_lab/*/*_"$name"/model_2999.pt >/dev/null 2>&1; then
    say "# $name done [$(date '+%F %T')]"
  else say "# FAILED: $name"; tail -5 /tmp/yo_$name.log | tee -a "$LOG"; fi
  sleep 30
}

evaluate() {  # run-name label extra...
  local run=$1 label=$2; shift 2
  local hits c
  hits=$(ls -d logs/amr_lab/*/*_"$run" 2>/dev/null | wc -l)
  if [ "$hits" -ne 1 ]; then say "# $run resolves to $hits directories, skipped"; return; fi
  c=$(ls -d logs/amr_lab/*/*_"$run")/model_2999.pt
  if [ -s "$OUT/$label.csv" ]; then say "# $label already done"; return; fi
  say ""; say "---- eval $label  [$(date '+%F %T')]"; say "# $c"
  timeout 2400 $PY amr_lab/eval_pairs.py $EVAL --checkpoint "$c" --label "$label" \
      --out "$OUT/$label.csv" "$@" > /tmp/yo_eval_$label.log 2>&1
  if [ -s "$OUT/$label.csv" ]; then
    sed -n '/^=== /,$p' /tmp/yo_eval_$label.log | tee -a "$LOG"
  else say "# FAILED: $label"; tail -15 /tmp/yo_eval_$label.log | tee -a "$LOG"; fi
  sleep 15
}

say "======== yaw-only hitch sizing  [$(date '+%F %T')] ========"
train yo_yaw20_s1 --seed 1 --hitch-torque 20 20 --pitch-torque 10.6
train yo_yaw30_s1 --seed 1 --hitch-torque 30 30 --pitch-torque 10.6
evaluate yo_yaw20_s1 yo_yaw20_s1 --hitch-torque 20 20 --pitch-torque 10.6
evaluate yo_yaw30_s1 yo_yaw30_s1 --hitch-torque 30 30 --pitch-torque 10.6
say ""; say "YAWONLY DONE $(date '+%F %T')"
