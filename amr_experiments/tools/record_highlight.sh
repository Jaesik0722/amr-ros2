#!/usr/bin/env bash
# Record a highlight video of one experiment: bring up a world with the cameras on,
# run a few trials, and compose the panels into one captioned clip.
#
#   ./amr_experiments/tools/record_highlight.sh e1 --trials 3
#   ./amr_experiments/tools/record_highlight.sh e4 --width 1.00 --repeats 1
#
# The videos are for showing what the numbers mean, so they run a handful of trials,
# not a campaign; the measured results come from run_experiment.sh. Output goes to
# $AMR_VIDEOS (default ~/amr_results/videos)/<exp>/<exp>_highlight.mp4.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/amr_experiments/tools/lib.sh"

EXP="${1:?usage: record_highlight.sh e1|e2|e3|e4|e5|e6|e7 [runner options]}"
shift
VIDEOS="${AMR_VIDEOS:-$HOME/amr_results/videos}"
OUT="$VIDEOS/$EXP"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-100}"
mkdir -p "$OUT"

# Per experiment: the runner, the overview camera (position then target), how long the
# capture may run, which docking camera is the interesting one, and the title card.
mapfile -t CART_ARGS < <(cart_sim_args) || exit 2
SIM_ARGS=(--headless --chase-camera "${CART_ARGS[@]}")
NEEDS_NAV=1
RVIZ=1
SPEED=3
DOCK=rear
case "$EXP" in
  e1)
    RUNNER=e1_charger_docking
    OVERVIEW=(6.0 3.0 4.5 7.0 -1.5 0.0)
    SECS=${HL_SECONDS:-260}
    DOCK=front
    TITLE='E1  Charger docking  -  Nav2 approach, ArUco servo, wireless charging'
    ;;
  e2)
    RUNNER=e2_cart_docking
    OVERVIEW=(5.5 -1.5 4.5 3.5 -6.5 0.0)
    SECS=${HL_SECONDS:-280}
    TITLE='E2  Cart docking  -  indoor GPS pose, reverse docking, hook engagement'
    ;;
  e3)
    RUNNER=e3_engagement
    OVERVIEW=(3.0 -5.2 2.0 3.0 -7.4 0.3)
    SECS=${HL_SECONDS:-220}
    NEEDS_NAV=0
    RVIZ=0
    SPEED=2
    SIM_ARGS+=(--no-cameras)
    DOCK=none
    TITLE='E3  Engagement detection  -  hitch effort tells a hooked cart from a missed one'
    ;;
  e4)
    RUNNER=e4_corridor
    OVERVIEW=(4.2 1.5 7.5 4.2 -4.5 0.0)
    SECS=${HL_SECONDS:-400}
    TITLE='E4  Narrow passage  -  towing footprint vs fixed hitch'
    ;;
  e5)
    RUNNER=e5_towing
    OVERVIEW=(5.0 -1.0 6.5 5.0 -5.0 0.0)
    SECS=${HL_SECONDS:-360}
    TITLE='E5  Towing  -  off-tracking and hitch angle along the route'
    ;;
  e6)
    RUNNER=e6_endurance
    OVERVIEW=(4.5 2.0 7.5 5.0 -4.0 0.0)
    SECS=${HL_SECONDS:-400}
    TITLE='E6  Endurance  -  one unattended charge-fetch-tow-release cycle'
    ;;
  e7)
    # The whole mission in one run, so the overview has to hold the entire floor: the
    # cart park in the west, the passage in the middle, the charger in the east.
    RUNNER=e7_scenario
    # 10.5 m up put the camera above the warehouse roof and the panel came out black.
    # The E4 vantage point is inside the building and already frames the passage and the
    # floor on both sides of it, which is most of this mission.
    OVERVIEW=(4.2 1.5 7.5 4.2 -4.5 0.0)
    SECS=${HL_SECONDS:-760}
    # One pass of each stage runs in about three and a half minutes, so this clip is
    # played at half the speed of the campaign clips: there is one chance to see each
    # stage rather than six, and x3 goes past too fast to read.
    SPEED=2
    DOCK=both
    TITLE='Full mission  -  fetch, couple, tow through the passage, deliver, charge, return'
    ;;
  *) echo "unknown experiment $EXP"; exit 2 ;;
esac
# A clip made for a side-by-side comparison often wants a different vantage point than
# the one that suits a single run: for E5, looking straight down shows the cart cutting
# the corner, which is the whole subject, where the default oblique view does not.
# HL_OVERVIEW overrides it as six numbers: position then target.
if [ -n "${HL_OVERVIEW:-}" ]; then
  read -r -a OVERVIEW <<< "$HL_OVERVIEW"
fi
SIM_ARGS+=(--overview-camera "${OVERVIEW[@]}")
CAMERAS=overview,chase
case "$DOCK" in
  none) ;;
  both) CAMERAS="overview,chase,front,rear" ;;
  *) CAMERAS="overview,chase,$DOCK" ;;
esac

# E3 drives the hitch only: no Nav2, no map to show and no ArUco camera, so its clip is
# the overview and the chase view side by side.

case "$EXP" in
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
    WIDTH=1.00
    for ((i = 1; i <= $#; i++)); do
      [ "${!i}" = "--width" ] && { j=$((i + 1)); WIDTH="${!j}"; }
    done
    SIM_ARGS+=(--corridor "$WIDTH" --cart-pose 0.6 -4.5 0)
    ;;
esac

echo "[video] $EXP  sim: ${SIM_ARGS[*]}  out: $OUT"
set +u
source /opt/ros/jazzy/setup.bash
source "$REPO/install/setup.bash"
set -u

require_clean || exit 1
trap stop_all EXIT

start_sim "$OUT/sim.log" "${SIM_ARGS[@]}" || exit 1
[ "$NEEDS_NAV" = 1 ] && { start_nav "$OUT/bringup.log" || exit 1; }

HL_TITLE="$TITLE" HL_RVIZ="$RVIZ" HL_CAMERAS="$CAMERAS" \
  xvfb-run -a -s "-screen 0 960x540x24" \
  "$REPO/amr_experiments/tools/capture_highlight.sh" "$OUT" "$SECS" "$RUNNER" \
  --out "$OUT/trials" "$@"
STATUS=$?
echo "[video] capture finished with status $STATUS"

stop_all
trap - EXIT

COMPOSE=("$REPO/amr_experiments/tools/compose_highlight.py" "$OUT"
         --name "${EXP}_highlight.mp4" --speed "$SPEED" --dock "$DOCK")
[ "$RVIZ" = 1 ] || COMPOSE+=(--no-rviz)
python3 "${COMPOSE[@]}"
