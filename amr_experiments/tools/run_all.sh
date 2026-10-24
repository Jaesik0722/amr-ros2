#!/usr/bin/env bash
# The whole simulation campaign, then the tables and figures.
#
#   ./amr_experiments/tools/run_all.sh                 # everything
#   ./amr_experiments/tools/run_all.sh session         # only the shared-world block
#   ./amr_experiments/tools/run_all.sh e3 e4           # only these
#
# Blocks:
#   session  E1, E2, E5, E6 in one simulator (same world)
#   e3       hitch engagement, no cameras
#   e4       narrow passage, one simulator per corridor width
#   e7       the whole mission once, end to end, in its own world
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TOOLS="$REPO/amr_experiments/tools"
RESULTS="${AMR_RESULTS:-$HOME/amr_results}"
export AMR_RESULTS="$RESULTS"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-100}"
mkdir -p "$RESULTS"
WANTED=("$@")

want() {
  [ ${#WANTED[@]} -eq 0 ] && return 0
  for w in "${WANTED[@]}"; do [ "$w" = "$1" ] && return 0; done
  return 1
}

if want session; then
  "$TOOLS/run_session.sh" "e1 --trials 30" "e2 --repeats 6" "e5 --repeats 3" \
    "e6 --cycles 30"
fi
want e3 && "$TOOLS/run_experiment.sh" e3 --repeats 5
if want e4; then
  for width in 1.20 1.00 0.90; do
    "$TOOLS/run_experiment.sh" e4 --width "$width" --repeats 3
  done
fi
# Last, because it is the integration check rather than a measurement: if the campaign
# produced numbers but the stages no longer chain, this is where that shows.
want e7 && "$TOOLS/run_experiment.sh" e7

echo "===== $(date -u +%H:%M:%S)  analysing"
set +u          # the ROS setup scripts read unset variables
source /opt/ros/jazzy/setup.bash
source "$REPO/install/setup.bash"
set -u
ros2 run amr_experiments analyze --results "$RESULTS" --out "$RESULTS/report"
echo "===== $(date -u +%H:%M:%S)  campaign finished"
