#!/usr/bin/env bash
# Two follow-ups to the paired re-evaluation, both on checkpoints that already exist.
#
# 1. How much does the arrival-order truncation move the bay? `play.py` keeps the first
#    1024 episodes to finish, and the 20 s timeouts finish last, so the slowest failures
#    are under-sampled. In free space the bias measured 0.1-1.9 points; the bay has
#    contact terminations as well, which end episodes early, so it has to be measured
#    there rather than carried over.
# 2. Is the episode clock what binds the locked configurations? In free space every
#    rigid-lock failure is a timeout, at a median 632 mm of position error and 125 deg of
#    cart-heading error: the position is nearly right and the heading is not. Equation
#    (18) prices a heading correction in workspace, so the prediction is that more time
#    buys the locked hitch some of those episodes. 60 s is three times what the policy
#    trained on and therefore out of distribution - it tests the explanation, it is not a
#    headline number.
set -u
PY="env PYTHONUNBUFFERED=1 $HOME/anaconda3/envs/isaaclab/bin/python"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1   # the repository root
OUT=$HOME/amr_results/amr_lab/pairs

ck() { ls -1 logs/amr_lab/$1/*_$2/model_*.pt 2>/dev/null \
       | sed 's#\(.*model_\)\([0-9]*\)\(\.pt\)#\2 &#' | sort -n | tail -1 | cut -d' ' -f2; }

one() {  # tag run label episode-length curriculum extra...
  local tag=$1 run=$2 label=$3 len=$4 cur=$5; shift 5
  local c; c=$(ck "$tag" "$run")
  [ -n "$c" ] || { echo "# no checkpoint for $run"; return; }
  echo "---- $label  len=$len curriculum=$cur  [$(date '+%F %T')]"
  timeout 3000 $PY amr_lab/eval_pairs.py --headless --num-envs 256 --episodes 4 \
      --episode-length "$len" --floor-friction 0.6 1.0 --cart-wheels caster4 \
      --scenario-seed 7 --curriculum "$cur" --checkpoint "$c" --label "$label" \
      --out "$OUT/$label.csv" "$@" > /tmp/pairs_$label.log 2>&1
  sed -n '/^=== /,$p' /tmp/pairs_$label.log
  sleep 30
}

one caster4         by_active_s1 bay_active_square_ord 20 0.0125 --bay
one caster4_passive by_rigid_s1  bay_rigid_square_ord  20 0.0125 --bay --rigid-lock
one caster4         fr_active_s1 free_active_60s 60 0.024
one caster4_passive fr_rigid_s1  free_rigid_60s  60 0.024 --rigid-lock
one caster4_passive fr_servo_s1  free_servo_60s  60 0.024 --passive-hitch --lock-hitch
one caster4_passive fr_passive_s1 free_passive_60s 60 0.024 --passive-hitch
echo "PAIRS2 DONE $(date '+%F %T')"
