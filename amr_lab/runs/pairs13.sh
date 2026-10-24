#!/usr/bin/env bash
# Re-read the DC torque-speed block with the reported torque equal to the applied one.
#
# The earlier runs clipped the REPORTED torque to a symmetric ceiling on |beta_dot|
# while the actuator applied the four-quadrant bound. The two differ on the braking
# side, which is the side this joint mostly works on: at half the no-load speed the
# actuator permits -10.6 N.m while the reported value was capped at 5.3. The success
# and jackknife figures came from the physics and are unaffected; the torque column and
# the near-limit fraction were properties of the reporting, so they are re-measured.
#
# `near_limit` now counts against the ceiling in force at that speed rather than the
# flat limit, which is the only comparison that means anything under a DC curve.
set -u
PY="env PYTHONUNBUFFERED=1 $HOME/anaconda3/envs/isaaclab/bin/python"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1   # the repository root
OUT=$HOME/amr_results/amr_lab/pairs
LOG=$OUT/pairs13.log
COMMON="--headless --num-envs 256 --episodes 4 --episode-length 20 \
  --floor-friction 0.6 1.0 --cart-wheels caster4 --scenario-seed 7 --curriculum 0.024"
say() { echo "$*" | tee -a "$LOG"; }
A1=logs/amr_lab/caster4/2026-10-20_21-17-35_fr_active_s1/model_2999.pt
A2=logs/amr_lab/caster4/2026-10-21_19-59-25_fr_active_s2/model_2999.pt
A3=logs/amr_lab/caster4/2026-10-22_01-39-47_fr_active_s3/model_2999.pt

one() {  # checkpoint label extra...
  local c=$1 label=$2; shift 2
  if [ ! -f "$c" ]; then say "# missing checkpoint: $c"; return; fi
  if [ -s "$OUT/$label.csv" ]; then say "# $label already done"; return; fi
  say ""; say "---- $label  [$(date '+%F %T')]"
  timeout 2400 $PY amr_lab/eval_pairs.py $COMMON \
      --checkpoint "$c" --label "$label" --out "$OUT/$label.csv" "$@" \
      > /tmp/pairs13_$label.log 2>&1
  if [ -s "$OUT/$label.csv" ]; then
    sed -n '/^=== /,$p' /tmp/pairs13_$label.log | tee -a "$LOG"
  else say "# FAILED: $label"; tail -15 /tmp/pairs13_$label.log | tee -a "$LOG"; fi
  sleep 15
}

say "======== DC torque-speed, reported torque = applied  [$(date '+%F %T')] ========"
k=0
for c in "$A1" "$A2" "$A3"; do
  k=$((k+1))
  one "$c" "dc2_hitch_s$k"  --hitch-dc
  one "$c" "fl2_hitch_s$k"
done
say ""; say "PAIRS13 DONE $(date '+%F %T')"
