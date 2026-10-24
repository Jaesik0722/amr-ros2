#!/bin/bash
# Review item A: remove the learner.
#
# The review asks whether the passive condition fails because of the hitch or because
# PPO failed to learn it. Every experiment that uses a policy is open to the second
# reading, so this uses none: the cross-entropy method searches open-loop command
# sequences with the simulator itself as its model, scored in the dynamics it plans in.
# Model error is zero and the budget per episode is about 41,000 full rollouts, which is
# thousands of times what a 20 Hz controller could spend, so the result is an upper
# bound on what any controller of this command set could reach rather than an estimate
# of what one would.
#
# The active condition is the control: a search that cannot solve it proves nothing
# about the passive one. A pilot solved 1 of 3 active scenarios at this budget and 0 of
# 4 at a tenth of it, so the budget below is the one that works and 20 scenarios per
# condition is what makes the comparison testable.
cd "$HOME/Repositories/amr-ros2" || exit 1
RES=$HOME/amr_results/amr_lab
OUT=$RES/search_results.txt
PY="env PYTHONUNBUFFERED=1 $HOME/anaconda3/envs/isaaclab/bin/python"
COMMON="--headless --num-envs 2048 --episode-length 20 --cart-wheels caster4 --seed 1
        --floor-friction 0.6 1.0 --scenarios 20 --cem-iters 20 --segment 10
        --elite-frac 0.03"

{
  echo ""
  echo "=========== simulator-as-model search  $(date '+%F %T') ==========="
} | tee -a "$OUT"

for cond in "active:" "passive:--passive-hitch"; do
  name=${cond%%:*}; extra=${cond##*:}
  [ -f "$RES/state/search_$name.done" ] && { echo "# done: $name"; continue; }
  { echo ""; echo "---- search :: $name"; } | tee -a "$OUT"
  timeout 36000 $PY amr_lab/search.py $COMMON $extra > /tmp/search_$name.log 2>&1
  grep -E '^\[search\] scenario|^===|^  scenarios|^  budget' /tmp/search_$name.log \
      | tee -a "$OUT"
  grep -q 'scenarios solved' /tmp/search_$name.log && touch "$RES/state/search_$name.done"
  sync
done

{ echo ""; echo "=========== search finished $(date '+%F %T') ==========="; } | tee -a "$OUT"
touch "$RES/state/search.done"
