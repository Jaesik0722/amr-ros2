#!/usr/bin/env bash
# Rollout repeatability. No training, no new scenarios: the same checkpoint over the same
# 1,024 scenarios, several times.
#
# Why this is needed. Two rollouts of the mechanically locked bay policy over an
# identical scenario set (verified: max |diff| 0.000e+00 over start pose, goal, payload,
# friction, hitch limit, rolling, scrub, and the same environment index for all 1,024)
# reported 592 and 555 successes. 18.5% of the scenarios changed outcome. The scenario is
# a lookup, so that spread is not scenario sampling; it is the per-step sensor noise
# realisation, which is drawn from the generator the policy is stepping and is not part
# of the lookup, plus PhysX's own non-determinism on the GPU.
#
# This is the third variance component, and the paper needs all three kept apart:
#   scenario sampling   removed by the lookup (--scenario-seed)
#   rollout noise       measured here, by repeating one rollout
#   training seed       measured by the campaign's three seeds, and not substitutable
#                       by any number of evaluation episodes
#
# Four repeats each of the two conditions whose difference is being claimed, at both
# approach distributions. If the +14.4-point active-minus-rigid gap at the trained
# approach survives this spread and the 1.5-point gap at a square approach does not,
# then "difference withheld" for the square case is a measurement statement and not a
# hedge.
set -u
PY="env PYTHONUNBUFFERED=1 $HOME/anaconda3/envs/isaaclab/bin/python"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1   # the repository root
OUT=$HOME/amr_results/amr_lab/pairs
LOG=$OUT/pairs4.log
COMMON="--headless --num-envs 256 --episodes 4 --episode-length 20 \
  --floor-friction 0.6 1.0 --cart-wheels caster4 --scenario-seed 7"

say() { echo "$*" | tee -a "$LOG"; }

ck() { ls -1 logs/amr_lab/$1/*_$2/model_*.pt 2>/dev/null \
       | sed 's#\(.*model_\)\([0-9]*\)\(\.pt\)#\2 &#' | sort -n | tail -1 | cut -d' ' -f2; }

one() {  # tag run label curriculum extra...
  local tag=$1 run=$2 label=$3 cur=$4; shift 4
  local c; c=$(ck "$tag" "$run")
  if [ -z "$c" ]; then say "# no checkpoint for $run"; return; fi
  if [ -s "$OUT/$label.csv" ]; then say "# $label already done"; return; fi
  say ""
  say "---- $label  curriculum=$cur  [$(date '+%F %T')]"
  timeout 2400 $PY amr_lab/eval_pairs.py $COMMON --curriculum "$cur" \
      --checkpoint "$c" --label "$label" --out "$OUT/$label.csv" "$@" \
      > /tmp/pairs4_$label.log 2>&1
  if [ -s "$OUT/$label.csv" ]; then
    sed -n '/^=== /,$p' /tmp/pairs4_$label.log | tee -a "$LOG"
  else
    say "# FAILED: $label"; tail -5 /tmp/pairs4_$label.log | tee -a "$LOG"
  fi
  sleep 20
}

say "======== rollout repeatability  [$(date '+%F %T')] ========"
for k in 1 2 3; do
  one caster4         by_active_s1 rep${k}_bay_active_full 1.0    --bay
  one caster4_passive by_rigid_s1  rep${k}_bay_rigid_full  1.0    --bay --rigid-lock
  one caster4         by_active_s1 rep${k}_bay_active_sq   0.0125 --bay
  one caster4_passive by_rigid_s1  rep${k}_bay_rigid_sq    0.0125 --bay --rigid-lock
done
say ""
say "PAIRS4 DONE $(date '+%F %T')"
