#!/usr/bin/env bash
# Run several experiments in ONE simulator session.
#
#   ./amr_experiments/tools/run_session.sh "e1 --trials 30" "e2 --repeats 6"
#
# E1, E2, E5 and E6 all use the same world (cameras, Nav2, SLAM), so they share a
# simulator: every restart is a chance to leave a second simulator publishing on the
# same ROS domain, and that is what invalidated the first campaign. E3 (no cameras) and
# E4 (corridor geometry) need their own world; use run_experiment.sh for those.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/amr_experiments/tools/lib.sh"
RESULTS="${AMR_RESULTS:-$HOME/amr_results}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-100}"
mkdir -p "$RESULTS"
[ $# -gt 0 ] || { echo 'usage: run_session.sh "e1 --trials 30" ["e2 --repeats 6" ...]'; exit 2; }

set +u
source /opt/ros/jazzy/setup.bash
source "$REPO/install/setup.bash"
set -u

require_clean || exit 1
trap stop_all EXIT

mapfile -t CART_ARGS < <(cart_sim_args) || exit 2
record_campaign "$RESULTS"
echo "[run] session  cart wheels: $CART_WHEELS  results: $RESULTS"
start_sim "$RESULTS/session_sim.log" --headless "${CART_ARGS[@]}" || exit 1
start_nav "$RESULTS/session_bringup.log" || exit 1

declare -A RUNNERS=([e1]=e1_charger_docking [e2]=e2_cart_docking [e3]=e3_engagement
                    [e4]=e4_corridor [e5]=e5_towing [e6]=e6_endurance)

for entry in "$@"; do
  read -r name rest <<< "$entry"
  runner="${RUNNERS[$name]:-}"
  [ -n "$runner" ] || { echo "[session] unknown experiment $name"; continue; }
  echo "===== $(date -u +%H:%M:%S)  $name  $rest"
  # shellcheck disable=SC2086
  ros2 run amr_experiments "$runner" --out "$RESULTS/$name" $rest 2>&1 \
    | tee "$RESULTS/${name}_run.log"
  echo "===== $(date -u +%H:%M:%S)  $name finished with status ${PIPESTATUS[0]}"
  if ! running "$SIM_PATTERN"; then
    echo "[session] the simulator is gone; stopping here" >&2
    exit 1
  fi
done
echo "===== $(date -u +%H:%M:%S)  session finished"
