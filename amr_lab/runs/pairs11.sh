#!/usr/bin/env bash
# Evaluate the retrained observation ablations on the corrected protocol: the same
# 1024 fixed scenarios, the same curriculum point and the same free-pose settings the
# headline table uses, so the rows are paired with `free_active` scenario by scenario.
#
# The observation flag must match the one the policy was TRAINED with: the actor's
# input dimension differs between the sets, so a mismatch either fails to load or,
# worse, loads and feeds the network a differently ordered vector.
#
# Checkpoints are globbed by run name here, unlike pairs7, because these four run names
# exist exactly once each -- they were created by obs_ablation.sh this morning and no
# budget run reused them. The glob is asserted to resolve to one path before use.
set -u
PY="env PYTHONUNBUFFERED=1 $HOME/anaconda3/envs/isaaclab/bin/python"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1   # the repository root
OUT=$HOME/amr_results/amr_lab/pairs
LOG=$OUT/pairs11.log
COMMON="--headless --num-envs 256 --episodes 4 --episode-length 20 \
  --floor-friction 0.6 1.0 --scenario-seed 7 --curriculum 0.024 --cart-wheels caster4"
say() { echo "$*" | tee -a "$LOG"; }

one() {  # run-name label extra...
  local run=$1 label=$2; shift 2
  local hits c
  hits=$(ls -d logs/amr_lab/*/*_"$run" 2>/dev/null | wc -l)
  if [ "$hits" -ne 1 ]; then say "# $run resolves to $hits directories, skipped"; return; fi
  c=$(ls -d logs/amr_lab/*/*_"$run")/model_2999.pt
  if [ ! -f "$c" ]; then say "# missing checkpoint: $c"; return; fi
  if [ -s "$OUT/$label.csv" ]; then say "# $label already done"; return; fi
  say ""; say "---- $label  [$(date '+%F %T')]"; say "# $c"
  timeout 2400 $PY amr_lab/eval_pairs.py $COMMON \
      --checkpoint "$c" --label "$label" --out "$OUT/$label.csv" "$@" \
      > /tmp/pairs11_$label.log 2>&1
  if [ -s "$OUT/$label.csv" ]; then
    sed -n '/^=== /,$p' /tmp/pairs11_$label.log | tee -a "$LOG"
  else
    say "# FAILED: $label"; tail -15 /tmp/pairs11_$label.log | tee -a "$LOG"
  fi
  sleep 20
}

say "======== observation ablations, corrected protocol  [$(date '+%F %T')] ========"
one ab_legacyobs_s1 ab_legacy_s1 --legacy-obs
one ab_legacyobs_s2 ab_legacy_s2 --legacy-obs
one ab_legacyobs_s3 ab_legacy_s3 --legacy-obs
one ab_trueobs_s1   ab_trueobs_s1 --true-cart-obs
say ""; say "PAIRS11 DONE $(date '+%F %T')"
