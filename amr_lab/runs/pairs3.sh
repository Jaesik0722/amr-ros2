#!/usr/bin/env bash
# Entry-time and servo diagnostics. No training: this re-reads the seed-1 checkpoints a
# third time, now that eval_pairs.py also records
#
#   entry_{t,py,yaw,beta,clear}  the state of the deck the first time it crossed the
#                                mouth plane between the walls, so a bay failure can be
#                                split into one that happened before the entry and one
#                                that happened after it, and the entry itself can be
#                                read for the lateral offset, the heading and the hitch
#                                angle it was taken at;
#   track_err, track_max, sat_t  how far the joint sits from the angle it is commanded
#                                and when the servo first spends more than 90% of its
#                                torque, which is what the 31.8%-of-the-episode figure
#                                is made of and what it does not say: *when*.
#
# The eight bay runs are the two protocols of pairs.sh so that the new columns can be
# read against numbers that are already analysed; the four free-space runs are for the
# tracking columns only (there is no bay to enter there).
set -u
PY="env PYTHONUNBUFFERED=1 $HOME/anaconda3/envs/isaaclab/bin/python"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1   # the repository root
OUT=$HOME/amr_results/amr_lab/pairs
LOG=$OUT/pairs3.log
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
  say "# checkpoint: $c"
  timeout 2400 $PY amr_lab/eval_pairs.py $COMMON --curriculum "$cur" \
      --checkpoint "$c" --label "$label" --out "$OUT/$label.csv" "$@" \
      > /tmp/pairs3_$label.log 2>&1
  if [ -s "$OUT/$label.csv" ]; then
    sed -n '/^=== /,$p' /tmp/pairs3_$label.log | tee -a "$LOG"
  else
    say "# FAILED: $label"; tail -5 /tmp/pairs3_$label.log | tee -a "$LOG"
  fi
  sleep 20
}

say "======== entry and servo diagnostics  [$(date '+%F %T')] ========"

one caster4         by_active_s1  dx_bay_active_full  1.0 --bay
one caster4_passive by_passive_s1 dx_bay_passive_full 1.0 --bay --passive-hitch
one caster4_passive by_servo_s1   dx_bay_servo_full   1.0 --bay --passive-hitch --lock-hitch
one caster4_passive by_rigid_s1   dx_bay_rigid_full   1.0 --bay --rigid-lock

one caster4         by_active_s1  dx_bay_active_sq  0.0125 --bay
one caster4_passive by_passive_s1 dx_bay_passive_sq 0.0125 --bay --passive-hitch
one caster4_passive by_servo_s1   dx_bay_servo_sq   0.0125 --bay --passive-hitch --lock-hitch
one caster4_passive by_rigid_s1   dx_bay_rigid_sq   0.0125 --bay --rigid-lock

one caster4         fr_active_s1  dx_free_active 0.024
one caster4_passive fr_passive_s1 dx_free_passive 0.024 --passive-hitch
one caster4_passive fr_servo_s1   dx_free_servo   0.024 --passive-hitch --lock-hitch
one caster4_passive fr_rigid_s1   dx_free_rigid   0.024 --rigid-lock

say ""
say "PAIRS3 DONE $(date '+%F %T')"
