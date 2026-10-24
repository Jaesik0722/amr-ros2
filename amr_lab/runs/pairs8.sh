#!/usr/bin/env bash
# The transfer-budget table, re-read on the corrected protocol over three seeds.
#
# Table "Transfer budget" in the manuscript still carries the pre-friction-fix numbers
# (82.3% as the trained-condition reference). Trap 41 in CLAUDE.md records that those
# predate the cart-friction correction, and they also come from `play.py`'s arrival-order
# estimator. Each variant is therefore re-evaluated here: the active checkpoints of all
# three seeds, unchanged, on the same 1,024 scenarios, with one condition of the
# simulation moved outside the box it was trained in.
set -u
PY="env PYTHONUNBUFFERED=1 $HOME/anaconda3/envs/isaaclab/bin/python"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1   # the repository root
OUT=$HOME/amr_results/amr_lab/pairs
LOG=$OUT/pairs8.log
BASE="--headless --num-envs 256 --episodes 4 --episode-length 20 --cart-wheels caster4 \
  --scenario-seed 7 --curriculum 0.024"
say() { echo "$*" | tee -a "$LOG"; }

A1=logs/amr_lab/caster4/2026-10-20_21-17-35_fr_active_s1/model_2999.pt
A2=logs/amr_lab/caster4/2026-10-21_19-59-25_fr_active_s2/model_2999.pt
A3=logs/amr_lab/caster4/2026-10-22_01-39-47_fr_active_s3/model_2999.pt

one() {  # checkpoint label extra...
  local c=$1 label=$2; shift 2
  if [ ! -f "$c" ]; then say "# missing checkpoint: $c"; return; fi
  if [ -s "$OUT/$label.csv" ]; then say "# $label already done"; return; fi
  say ""; say "---- $label  [$(date '+%F %T')]"
  timeout 2400 $PY amr_lab/eval_pairs.py $BASE \
      --checkpoint "$c" --label "$label" --out "$OUT/$label.csv" "$@" \
      > /tmp/pairs8_$label.log 2>&1
  if [ -s "$OUT/$label.csv" ]; then
    sed -n '/^=== /,$p' /tmp/pairs8_$label.log | tee -a "$LOG"
  else
    say "# FAILED: $label"; tail -5 /tmp/pairs8_$label.log | tee -a "$LOG"
  fi
  sleep 20
}

say "======== transfer budget, three seeds  [$(date '+%F %T')] ========"
k=0
for c in "$A1" "$A2" "$A3"; do
  k=$((k+1))
  one "$c" "tb_perfect_s$k"  --floor-friction 0.6 1.0 --perfect-sensors
  one "$c" "tb_mu0406_s$k"   --floor-friction 0.4 0.6
  one "$c" "tb_caster_s$k"   --floor-friction 0.6 1.0 --caster-init 3.1416
  one "$c" "tb_heading_s$k"  --floor-friction 0.6 1.0 --heading-spread 3.1416
  one "$c" "tb_delay200_s$k" --floor-friction 0.6 1.0 --latency 4 4
done
say ""; say "PAIRS8 DONE $(date '+%F %T')"
