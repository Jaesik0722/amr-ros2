#!/usr/bin/env bash
# The near-square approach for seeds 2 and 3. The trained-approach distribution is now
# measured on all three seeds, but "the mechanically locked hitch is competitive on a
# square approach" rests on seed 1 alone, and the seed spread on this task is larger than
# anything else being measured (bay active ranges 29.5-50.0 over three seeds against a
# rollout spread of 1.1). One seed cannot carry that claim either way.
set -u
PY="env PYTHONUNBUFFERED=1 $HOME/anaconda3/envs/isaaclab/bin/python"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1   # the repository root
OUT=$HOME/amr_results/amr_lab/pairs
LOG=$OUT/pairs6.log
COMMON="--headless --num-envs 256 --episodes 4 --episode-length 20 \
  --floor-friction 0.6 1.0 --cart-wheels caster4 --scenario-seed 7"
say() { echo "$*" | tee -a "$LOG"; }
ck() { ls -1 logs/amr_lab/$1/*_$2/model_*.pt 2>/dev/null \
       | sed 's#\(.*model_\)\([0-9]*\)\(\.pt\)#\2 &#' | sort -n | tail -1 | cut -d' ' -f2; }
one() {
  local tag=$1 run=$2 label=$3 cur=$4; shift 4
  local c; c=$(ck "$tag" "$run")
  if [ -z "$c" ]; then say "# no checkpoint for $run"; return; fi
  if [ -s "$OUT/$label.csv" ]; then say "# $label already done"; return; fi
  say ""; say "---- $label  curriculum=$cur  [$(date '+%F %T')]"
  timeout 2400 $PY amr_lab/eval_pairs.py $COMMON --curriculum "$cur" \
      --checkpoint "$c" --label "$label" --out "$OUT/$label.csv" "$@" \
      > /tmp/pairs6_$label.log 2>&1
  if [ -s "$OUT/$label.csv" ]; then
    sed -n '/^=== /,$p' /tmp/pairs6_$label.log | tee -a "$LOG"
  else
    say "# FAILED: $label"; tail -5 /tmp/pairs6_$label.log | tee -a "$LOG"
  fi
  sleep 20
}
say "======== near-square approach, seeds 2 and 3  [$(date '+%F %T')] ========"
for s in 2 3; do
  one caster4         by_active_s$s  sq${s}_bay_active  0.0125 --bay
  one caster4_passive by_passive_s$s sq${s}_bay_passive 0.0125 --bay --passive-hitch
  one caster4_passive by_servo_s$s   sq${s}_bay_servo   0.0125 --bay --passive-hitch --lock-hitch
  one caster4_passive by_rigid_s$s   sq${s}_bay_rigid   0.0125 --bay --rigid-lock
done
say ""; say "PAIRS6 DONE $(date '+%F %T')"
