#!/usr/bin/env bash
# Re-evaluate the seed-1 checkpoints the campaign already produced, on a fixed set of
# scenarios and at a stated point on the curriculum. No training: this reads checkpoints.
#
# Two things are being corrected here.
#
# 1. `common_step_counter` is zero in a fresh process and the curricula are read off it,
#    so every evaluation so far sampled the *start* of the schedule. For the bay that is
#    +-0.26 rad of approach against the +-1.2 rad training ended on, so the recorded bay
#    numbers describe a near-square reverse and not the task the policies trained for.
#    The free-space schedule is 4,000,000 steps long and training only reaches 2.4% of
#    it, so there the error is 30.0 deg against 33.6 deg -- small, but stated rather than
#    assumed.
# 2. Episodes were drawn from the generator the policy was stepping, so no two conditions
#    saw the same ones. `--scenario-seed` makes each episode a lookup keyed by its index,
#    which gives the paired comparison: both reached, one reached, neither did.
set -u
PY="env PYTHONUNBUFFERED=1 $HOME/anaconda3/envs/isaaclab/bin/python"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1   # the repository root
OUT=$HOME/amr_results/amr_lab/pairs
LOG=$OUT/pairs.log
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
      > /tmp/pairs_$label.log 2>&1
  if [ -s "$OUT/$label.csv" ]; then
    sed -n '/^=== /,$p' /tmp/pairs_$label.log | tee -a "$LOG"
  else
    say "# FAILED: $label"; tail -5 /tmp/pairs_$label.log | tee -a "$LOG"
  fi
  sleep 30
}

say "======== paired re-evaluation  [$(date '+%F %T')] ========"

# The bay, over the approach distribution training ended on.
one caster4         by_active_s1  bay_active_full  1.0 --bay
one caster4_passive by_passive_s1 bay_passive_full 1.0 --bay --passive-hitch
one caster4_passive by_servo_s1   bay_servo_full   1.0 --bay --passive-hitch --lock-hitch
one caster4_passive by_rigid_s1   bay_rigid_full   1.0 --bay --rigid-lock

# The bay again, at the point on the curriculum the campaign's evaluations actually read
# (600-ish control steps of a 48,000-step schedule), so that the difference between this
# block and the one above is the protocol and nothing else.
one caster4         by_active_s1  bay_active_square  0.0125 --bay
one caster4_passive by_passive_s1 bay_passive_square 0.0125 --bay --passive-hitch
one caster4_passive by_servo_s1   bay_servo_square   0.0125 --bay --passive-hitch --lock-hitch
one caster4_passive by_rigid_s1   bay_rigid_square   0.0125 --bay --rigid-lock

# Free space, at the 2.4% of its schedule that 3000 iterations reach.
one caster4         fr_active_s1  free_active  0.024
one caster4_passive fr_passive_s1 free_passive 0.024 --passive-hitch
one caster4_passive fr_servo_s1   free_servo   0.024 --passive-hitch --lock-hitch
one caster4_passive fr_rigid_s1   free_rigid   0.024 --rigid-lock

say ""
say "======== paired re-evaluation finished  [$(date '+%F %T')] ========"
