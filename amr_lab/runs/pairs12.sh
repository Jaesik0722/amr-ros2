#!/usr/bin/env bash
# Re-read every bay block under the corrected wall-gap metric.
#
# `_bay_gap` measured the distance from the four deck corners to the three wall boxes
# and took the minimum, on the argument that the nearest point of a convex polygon to a
# plane is a vertex. The walls are boxes, not planes, and a box has corners: the deck
# can lie across the front corner of a rack with all four of its own corners well
# outside the wall. Over 2.85 million sampled deck poses that the colliders allow, that
# form reported at least 10 mm of clearance while the deck was touching a wall in 0.84%
# of poses -- 2.33% of the poses within 200 mm of a wall -- and one searched grazing
# pose reads 366 mm of clearance at a true gap of 0.3 mm. Contact terminates the
# episode and feeds the shaping term, so the figures it produced are re-read here.
#
# The policies are NOT retrained: they were trained under the old form, and that is a
# fact about them to report rather than to erase. What changes is the measurement.
#
# Checkpoints are explicit. The bay runs were resumed, so several directories share a
# run name and a glob that sorts on the iteration number can break the tie either way.
# These paths are the ones the published tables were produced from.
set -u
PY="env PYTHONUNBUFFERED=1 $HOME/anaconda3/envs/isaaclab/bin/python"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1   # the repository root
OUT=$HOME/amr_results/amr_lab/pairs
LOG=$OUT/pairs12.log
COMMON="--headless --num-envs 256 --episodes 4 --episode-length 20 \
  --floor-friction 0.6 1.0 --cart-wheels caster4 --scenario-seed 7 --bay"
say() { echo "$*" | tee -a "$LOG"; }

A1=logs/amr_lab/caster4/2026-10-21_14-26-34_by_active_s1/model_2999.pt
P1=logs/amr_lab/caster4_passive/2026-10-21_16-41-12_by_passive_s1/model_2999.pt
S1=logs/amr_lab/caster4_passive/2026-10-21_17-53-50_by_servo_s1/model_2999.pt
R1=logs/amr_lab/caster4_passive/2026-10-21_19-50-48_by_rigid_s1/model_2999.pt
A2=logs/amr_lab/caster4/2026-10-22_09-11-42_by_active_s2/model_2999.pt
P2=logs/amr_lab/caster4_passive/2026-10-22_10-12-18_by_passive_s2/model_2999.pt
S2=logs/amr_lab/caster4_passive/2026-10-22_11-53-58_by_servo_s2/model_2999.pt
R2=logs/amr_lab/caster4_passive/2026-10-22_14-00-56_by_rigid_s2/model_2999.pt
A3=logs/amr_lab/caster4/2026-10-22_15-04-37_by_active_s3/model_2999.pt
P3=logs/amr_lab/caster4_passive/2026-10-22_16-37-13_by_passive_s3/model_2999.pt
S3=logs/amr_lab/caster4_passive/2026-10-22_18-46-55_by_servo_s3/model_2999.pt
R3=logs/amr_lab/caster4_passive/2026-10-22_19-40-50_by_rigid_s3/model_2999.pt

one() {  # checkpoint label curriculum extra...
  local c=$1 label=$2 cur=$3; shift 3
  if [ ! -f "$c" ]; then say "# missing checkpoint: $c"; return; fi
  if [ -s "$OUT/$label.csv" ]; then say "# $label already done"; return; fi
  say ""; say "---- $label  curriculum=$cur  [$(date '+%F %T')]"; say "# $c"
  timeout 2400 $PY amr_lab/eval_pairs.py $COMMON --curriculum "$cur" \
      --checkpoint "$c" --label "$label" --out "$OUT/$label.csv" "$@" \
      > /tmp/pairs12_$label.log 2>&1
  if [ -s "$OUT/$label.csv" ]; then
    sed -n '/^=== /,$p' /tmp/pairs12_$label.log | tee -a "$LOG"
  else
    say "# FAILED: $label"; tail -15 /tmp/pairs12_$label.log | tee -a "$LOG"
  fi
  sleep 15
}

say "======== bay blocks, corrected wall gap  [$(date '+%F %T')] ========"
for s in 1 2 3; do
  eval "A=\$A$s P=\$P$s S=\$S$s R=\$R$s"
  say ""; say "######## seed $s, trained approach (curriculum 1.0)"
  one "$A" cg${s}_bay_active_full  1.0
  one "$P" cg${s}_bay_passive_full 1.0 --passive-hitch
  one "$S" cg${s}_bay_servo_full   1.0 --passive-hitch --lock-hitch
  one "$R" cg${s}_bay_rigid_full   1.0 --rigid-lock
  say ""; say "######## seed $s, near-square approach (curriculum 0.0125)"
  one "$A" cg${s}_bay_active_sq  0.0125
  one "$P" cg${s}_bay_passive_sq 0.0125 --passive-hitch
  one "$S" cg${s}_bay_servo_sq   0.0125 --passive-hitch --lock-hitch
  one "$R" cg${s}_bay_rigid_sq   0.0125 --rigid-lock
done
say ""; say "PAIRS12 DONE $(date '+%F %T')"
