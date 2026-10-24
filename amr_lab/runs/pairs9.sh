#!/usr/bin/env bash
# The last two tables: numerical settings and the DC torque-speed hitch, three seeds.
# pairs7 covers seed 1 of the numerical variants across all four conditions; the
# manuscript's tables are the active policy over three seeds, so seeds 2 and 3 are added
# here, together with the torque-speed actuator the flat-ceiling result has to survive.
set -u
PY="env PYTHONUNBUFFERED=1 $HOME/anaconda3/envs/isaaclab/bin/python"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1   # the repository root
OUT=$HOME/amr_results/amr_lab/pairs
LOG=$OUT/pairs9.log
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
      > /tmp/pairs9_$label.log 2>&1
  if [ -s "$OUT/$label.csv" ]; then
    sed -n '/^=== /,$p' /tmp/pairs9_$label.log | tee -a "$LOG"
  else
    say "# FAILED: $label"; tail -5 /tmp/pairs9_$label.log | tee -a "$LOG"
  fi
  sleep 20
}
say "======== numerics and the DC servo, three seeds  [$(date '+%F %T')] ========"
one "$A2" nm_phys400_s2  --physics-hz 400
one "$A3" nm_phys400_s3  --physics-hz 400
one "$A2" nm_solver16_s2 --solver-iters 16 4
one "$A3" nm_solver16_s3 --solver-iters 16 4
one "$A1" dc_hitch_s1 --hitch-dc
one "$A2" dc_hitch_s2 --hitch-dc
one "$A3" dc_hitch_s3 --hitch-dc
say ""; say "PAIRS9 DONE $(date '+%F %T')"
