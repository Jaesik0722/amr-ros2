#!/bin/bash
# Re-measure the open-loop authority grid AFTER the joint-friction correction of
# 2026-10-20 (CLAUDE.md trap 32). The grid behind Table VIII of the paper was run on
# 2026-10-17, i.e. on a cart whose casters had essentially no friction, so both of its
# stated conclusions ("the fitted servo is sufficient", "1.1-3.3 N.m of its 10.6") are
# measurements of the bug. No policy is involved, so this does not have to wait for the
# training campaign: the physics is stepped at a fixed timestep and GPU contention
# changes the wall clock, not the result.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1   # the repository root
PY=${ISAAC_PYTHON:-$HOME/anaconda3/envs/isaaclab/bin/python}
RES=${AMR_RESULTS:-$HOME/amr_results}/amr_lab
GRID="--payloads 0 20 60 100 --hitch-torques 10.6 20 30 --frictions 0.4 0.7 1.0 --repeats 8"
for w in caster4 trailer; do
  echo "=== $w authority, post friction fix, $(date '+%F %T')" | tee "$RES/authority2_$w.txt"
  $PY amr_lab/authority.py --headless --cart-wheels "$w" $GRID \
      >>"$RES/authority2_$w.txt" 2>"/tmp/auth2_$w.log"
  echo "# exit=$? at $(date '+%F %T')" >>"$RES/authority2_$w.txt"
done
echo "AUTHORITY RE-MEASURE DONE $(date '+%F %T')"
