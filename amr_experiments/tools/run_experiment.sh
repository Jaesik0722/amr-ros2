#!/usr/bin/env bash
# Run one experiment that needs a world of its own, end to end.
#
#   ./amr_experiments/tools/run_experiment.sh e3 --repeats 5
#   ./amr_experiments/tools/run_experiment.sh e4 --width 1.0 --repeats 3
#   ./amr_experiments/tools/run_experiment.sh e7
#
# E1, E2, E5 and E6 share a world; run those with run_session.sh instead.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/amr_experiments/tools/lib.sh"
EXP="${1:?usage: run_experiment.sh e1|e2|e3|e4|e5|e6|e7 [runner options]}"
shift
RESULTS="${AMR_RESULTS:-$HOME/amr_results}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-100}"
mkdir -p "$RESULTS"

mapfile -t CART_ARGS < <(cart_sim_args) || exit 2
SIM_ARGS=(--headless "${CART_ARGS[@]}")
NEEDS_NAV=1
case "$EXP" in
  e3) SIM_ARGS+=(--no-cameras); NEEDS_NAV=0 ;;       # hitch only: no Nav2, no ArUco
  e4)
    WIDTH=""
    for ((i = 1; i <= $#; i++)); do
      [ "${!i}" = "--width" ] && { j=$((i + 1)); WIDTH="${!j}"; }
    done
    [ -n "$WIDTH" ] || { echo "e4 needs --width"; exit 2; }
    # The cart's default spawn pose (3.0, -7.5) is inside the corridor's wing wall,
    # which stands at x = 3.0. PhysX resolving that penetration left the four-caster
    # cart's articulation with a NaN pose and every trial of the block failed on a NaN
    # goal (see e4_1p00_nan_cartpose). Spawn it where the trials park it anyway.
    SIM_ARGS+=(--corridor "$WIDTH" --cart-pose 0.6 -4.5 0)
    ;;
  e7)
    # The scenario drives the passage twice, so the world needs it, and the cart starts
    # parked on the near side of it (E4's parking pose) instead of in the open floor.
    WIDTH="1.00"
    for ((i = 1; i <= $#; i++)); do
      [ "${!i}" = "--width" ] && { j=$((i + 1)); WIDTH="${!j}"; }
    done
    SIM_ARGS+=(--corridor "$WIDTH" --cart-pose 0.6 -4.5 0)
    ;;
esac
# The cart mass is a simulator property, so pass it to both sides.
for ((i = 1; i <= $#; i++)); do
  case "${!i}" in
    --cart-mass) j=$((i + 1)); SIM_ARGS+=(--cart-mass "${!j}") ;;
    --cart-mass=*) SIM_ARGS+=(--cart-mass "${!i#*=}") ;;
  esac
done

case "$EXP" in
  e1) RUNNER=e1_charger_docking ;;
  e2) RUNNER=e2_cart_docking ;;
  e3) RUNNER=e3_engagement ;;
  e4) RUNNER=e4_corridor ;;
  e5) RUNNER=e5_towing ;;
  e6) RUNNER=e6_endurance ;;
  e7) RUNNER=e7_scenario ;;
  *) echo "unknown experiment $EXP"; exit 2 ;;
esac

echo "[run] $EXP  sim: ${SIM_ARGS[*]}  results: $RESULTS"
record_campaign "$RESULTS"
set +u          # the ROS setup scripts read unset variables
source /opt/ros/jazzy/setup.bash
source "$REPO/install/setup.bash"
set -u

require_clean || exit 1
trap stop_all EXIT

start_sim "$RESULTS/${EXP}_sim.log" "${SIM_ARGS[@]}" || exit 1
# A bringup can come up around a map that does not cover the robot, and then every
# goal of every trial fails identically. The stage is fine, so start the stack again
# rather than throwing the run away.
if [ "$NEEDS_NAV" = 1 ]; then
  for attempt in 1 2 3; do
    start_nav "$RESULTS/${EXP}_bringup.log" && break
    [ "$attempt" = 3 ] && { echo "[run] the ROS stack never came up" >&2; exit 1; }
    echo "[run] bringup attempt $attempt failed; starting the ROS stack again"
    mv "$RESULTS/${EXP}_bringup.log" "$RESULTS/${EXP}_bringup_failed$attempt.log" 2>/dev/null
    stop_ros
  done
fi

echo "[run] $RUNNER $*"
ros2 run amr_experiments "$RUNNER" --out "$RESULTS/$EXP" "$@" 2>&1 \
  | tee "$RESULTS/${EXP}_run.log"
STATUS=${PIPESTATUS[0]}
echo "[run] $EXP finished with status $STATUS"
exit "$STATUS"
