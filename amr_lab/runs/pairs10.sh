#!/usr/bin/env bash
# The command-delay sweep, three seeds, on the corrected protocol. pairs8 measured the
# 200 ms point; the manuscript's table is a sweep from zero to 400 ms, because the claim
# is about the shape of the curve and not about one point on it. Latency is in control
# steps of 50 ms: 0, 2, 4, 6, 8 steps = 0, 100, 200, 300, 400 ms.
set -u
PY="env PYTHONUNBUFFERED=1 $HOME/anaconda3/envs/isaaclab/bin/python"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1   # the repository root
OUT=$HOME/amr_results/amr_lab/pairs
LOG=$OUT/pairs10.log
BASE="--headless --num-envs 256 --episodes 4 --episode-length 20 --cart-wheels caster4 \
  --floor-friction 0.6 1.0 --scenario-seed 7 --curriculum 0.024"
say() { echo "$*" | tee -a "$LOG"; }
A1=logs/amr_lab/caster4/2026-10-20_21-17-35_fr_active_s1/model_2999.pt
A2=logs/amr_lab/caster4/2026-10-21_19-59-25_fr_active_s2/model_2999.pt
A3=logs/amr_lab/caster4/2026-10-22_01-39-47_fr_active_s3/model_2999.pt
one() {
  local c=$1 label=$2; shift 2
  if [ ! -f "$c" ]; then say "# missing checkpoint: $c"; return; fi
  if [ -s "$OUT/$label.csv" ]; then say "# $label already done"; return; fi
  say ""; say "---- $label  [$(date '+%F %T')]"
  timeout 2400 $PY amr_lab/eval_pairs.py $BASE \
      --checkpoint "$c" --label "$label" --out "$OUT/$label.csv" "$@" \
      > /tmp/pairs10_$label.log 2>&1
  if [ -s "$OUT/$label.csv" ]; then
    sed -n '/^=== /,$p' /tmp/pairs10_$label.log | tee -a "$LOG"
  else
    say "# FAILED: $label"; tail -5 /tmp/pairs10_$label.log | tee -a "$LOG"
  fi
  sleep 20
}
say "======== command-delay sweep  [$(date '+%F %T')] ========"
k=0
for c in "$A1" "$A2" "$A3"; do
  k=$((k+1))
  one "$c" "dl0_s$k"   --latency 0 0
  one "$c" "dl100_s$k" --latency 2 2
  one "$c" "dl300_s$k" --latency 6 6
  one "$c" "dl400_s$k" --latency 8 8
done
say ""; say "PAIRS10 DONE $(date '+%F %T')"
