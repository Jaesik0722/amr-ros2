#!/usr/bin/env bash
# Seeds 2 and 3 on the corrected protocol, then rollout repeatability.
#
# The campaign's own evaluations run through `play.py`, which collects episodes in the
# order they finish. That estimator is biased, by an amount that depends on the
# condition: on one bay rollout it read 71.7% for the mechanically locked policy where
# one row per scenario reads 56.6%, and 66.0% against 57.2% for the active one. The
# ranking in `review4.txt` is a product of that difference in bias. So the three seeds
# have to be re-read here, on fixed scenarios and at the approach distribution training
# ended with, before anything is said about seed variation.
#
# Block A: the eight bay runs of seeds 2 and 3 (seed 1 is already in pairs/*_full.csv).
# Block B: the eight free-space runs of seeds 2 and 3.
# Block C: three more rollouts each of the two bay conditions whose difference is being
#          claimed, at both approach distributions, to separate rollout noise from the
#          seed-to-seed spread that blocks A and B measure.
set -u
PY="env PYTHONUNBUFFERED=1 $HOME/anaconda3/envs/isaaclab/bin/python"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1   # the repository root
OUT=$HOME/amr_results/amr_lab/pairs
LOG=$OUT/pairs5.log
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
      > /tmp/pairs5_$label.log 2>&1
  if [ -s "$OUT/$label.csv" ]; then
    sed -n '/^=== /,$p' /tmp/pairs5_$label.log | tee -a "$LOG"
  else
    say "# FAILED: $label"; tail -5 /tmp/pairs5_$label.log | tee -a "$LOG"
  fi
  sleep 20
}

say "======== seeds 2-3 and repeatability  [$(date '+%F %T')] ========"

for s in 2 3; do
  say ""; say "######## bay, seed $s, trained approach"
  one caster4         by_active_s$s  s${s}_bay_active  1.0 --bay
  one caster4_passive by_passive_s$s s${s}_bay_passive 1.0 --bay --passive-hitch
  one caster4_passive by_servo_s$s   s${s}_bay_servo   1.0 --bay --passive-hitch --lock-hitch
  one caster4_passive by_rigid_s$s   s${s}_bay_rigid   1.0 --bay --rigid-lock
done

for s in 2 3; do
  say ""; say "######## free space, seed $s"
  one caster4         fr_active_s$s  s${s}_free_active  0.024
  one caster4_passive fr_passive_s$s s${s}_free_passive 0.024 --passive-hitch
  one caster4_passive fr_servo_s$s   s${s}_free_servo   0.024 --passive-hitch --lock-hitch
  one caster4_passive fr_rigid_s$s   s${s}_free_rigid   0.024 --rigid-lock
done

say ""; say "######## rollout repeatability, seed 1"
for k in 1 2 3; do
  one caster4         by_active_s1 rep${k}_bay_active_full 1.0    --bay
  one caster4_passive by_rigid_s1  rep${k}_bay_rigid_full  1.0    --bay --rigid-lock
  one caster4         by_active_s1 rep${k}_bay_active_sq   0.0125 --bay
  one caster4_passive by_rigid_s1  rep${k}_bay_rigid_sq    0.0125 --bay --rigid-lock
done

say ""
say "PAIRS5 DONE $(date '+%F %T')"
