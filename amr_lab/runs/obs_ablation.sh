#!/usr/bin/env bash
# Retrain the observation ablations. The existing s2r_legacyobs checkpoints are dated
# 2026-10-19/20 and therefore predate the cart-friction correction of trap 32, under
# which every amr_lab result changed; they cannot appear in the manuscript beside
# post-fix numbers. Four runs: the legacy observation set (lateral body velocity and the
# tow-bar angle restored, neither of which the hardware can measure) at three seeds, and
# the privileged actor observation at one seed.
set -u
PY="env PYTHONUNBUFFERED=1 $HOME/anaconda3/envs/isaaclab/bin/python"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1   # the repository root
LOG=$HOME/amr_results/amr_lab/obs_ablation.log
TRAIN="--headless --num-envs 2048 --episode-length 20 --floor-friction 0.6 1.0 \
  --max-iterations 3000 --cart-wheels caster4"
say() { echo "$*" | tee -a "$LOG"; }

one() {  # run-name extra...
  local name=$1; shift
  if ls logs/amr_lab/*/*_"$name"/model_2999.pt >/dev/null 2>&1; then
    say "# $name already trained"; return
  fi
  say ""; say "---- train $name  [$(date '+%F %T')]"
  $PY amr_lab/train.py $TRAIN --run-name "$name" "$@" \
      > /tmp/obs_$name.log 2>&1
  if ls logs/amr_lab/*/*_"$name"/model_2999.pt >/dev/null 2>&1; then
    say "# $name done [$(date '+%F %T')]"
  else
    say "# FAILED: $name"; tail -5 /tmp/obs_$name.log | tee -a "$LOG"
  fi
  sleep 30
}

say "======== observation ablations, retrained post friction fix  [$(date '+%F %T')] ========"
one ab_legacyobs_s1 --legacy-obs --seed 1
one ab_legacyobs_s2 --legacy-obs --seed 2
one ab_legacyobs_s3 --legacy-obs --seed 3
one ab_trueobs_s1   --true-cart-obs --seed 1
say ""; say "OBS ABLATION DONE $(date '+%F %T')"
