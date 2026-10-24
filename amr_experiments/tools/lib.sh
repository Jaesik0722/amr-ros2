# Shared helpers for the experiment runners: start the simulator and the ROS stack,
# wait until they really publish, and stop them so nothing survives into the next run.
#
# Two simulators on one ROS domain publish /clock, /odom and /tf at the same time and
# every trial after that is meaningless, so the stop has to be verified, not assumed.
# The same goes for the ROS stack: a stop that matched only /opt/ros/jazzy/lib left this
# repo's own Python nodes (cart_towing, the two ArUco detectors) and the `ros2 launch`
# parent alive. Forty-six of them accumulated over one day of runs, all on domain 100,
# and a stale cart_towing still in towing mode kept republishing the cart-hull footprint
# over Nav2's: the docking server then reported a collision at the charger and the robot
# could not undock for the rest of the campaign.

# Anchored patterns. Two rules:
#   - anchor on the absolute path of the executable, because a bare pattern like
#     "amr_sim.py" also matches the shell running this script (CLAUDE.md);
#   - match amr_sim.py as well, so an Isaac Sim the user opened for their own work is
#     never touched - only the simulators these experiments start.
SIM_PATTERN='^/home/[^ ]*/bin/python[0-9.]* [^ ]*amr_isaac/amr_sim\.py'
# Everything the bringup launch starts, in the order it has to be stopped: the launch
# parent first, so it does not report its children dying, then the Nav2 C++ nodes, then
# this repo's Python nodes and command-line tools (lib/amr_*/, never install/amr_*,
# which also appears in the simulator's own arguments - CLAUDE.md).
LAUNCH_PATTERN='^/usr/bin/python3 /opt/ros/jazzy/bin/ros2 launch amr_bringup'
NAV_PATTERN='^/opt/ros/jazzy/lib'
NODE_PATTERN='^/usr/bin/python3 /home/[^ ]*/lib/amr_[a-z_]*/'
ROS_PATTERNS=("$LAUNCH_PATTERN" "$NAV_PATTERN" "$NODE_PATTERN")
ROS_PATTERN="$NAV_PATTERN"   # kept for scripts that only check the Nav2 nodes

# Factory carts come as single-axle trailers and as four-caster trolleys, and the two
# behave like different vehicles, so which one a campaign measured has to travel with
# its numbers. $AMR_CART_WHEELS picks it; the choice is echoed into the console log and
# written beside the CSVs, because a results directory that does not say which cart it
# used is not evidence of anything.
CART_WHEELS="${AMR_CART_WHEELS:-trailer}"

cart_sim_args() {
  case "$CART_WHEELS" in
    trailer|caster4) printf '%s\n' --cart-wheels "$CART_WHEELS" ;;
    *) echo "[lib] unknown AMR_CART_WHEELS $CART_WHEELS: trailer or caster4" >&2; return 1 ;;
  esac
}

record_campaign() {   # record_campaign <results dir>
  mkdir -p "$1"
  printf 'cart_wheels: %s\n' "$CART_WHEELS" > "$1/campaign.txt"
}

running() { pgrep -f "$1" > /dev/null 2>&1; }

ros_running() {
  local pattern
  for pattern in "${ROS_PATTERNS[@]}"; do running "$pattern" && return 0; done
  return 1
}

stop_pattern() {   # stop_pattern <pattern> [seconds]
  local pattern="$1" deadline=$((SECONDS + ${2:-20})) pid
  for pid in $(pgrep -f "$pattern" 2>/dev/null); do kill -TERM "$pid" 2>/dev/null; done
  while running "$pattern" && [ "$SECONDS" -lt "$deadline" ]; do sleep 1; done
  for pid in $(pgrep -f "$pattern" 2>/dev/null); do kill -KILL "$pid" 2>/dev/null; done
  sleep 1
  ! running "$pattern"
}

# Stop the ROS stack but leave the simulator running, so a bringup that came up with
# an unusable map can be started again without rebuilding the stage.
stop_ros() {
  local pattern
  for pattern in "${ROS_PATTERNS[@]}"; do
    stop_pattern "$pattern" 15 || echo "[lib] warning: ROS processes survived: $pattern"
  done
}

stop_all() {
  local pattern
  for pattern in "${ROS_PATTERNS[@]}"; do
    stop_pattern "$pattern" 15 || echo "[lib] warning: ROS processes survived: $pattern"
  done
  stop_pattern "$SIM_PATTERN" 25 || echo "[lib] warning: the simulator survived"
}

require_clean() {
  if running "$SIM_PATTERN" || ros_running; then
    echo "[lib] a simulator or ROS stack is already running; stopping it first"
    stop_all
  fi
  if running "$SIM_PATTERN" || ros_running; then
    echo "[lib] could not get a clean machine; refusing to start" >&2
    pgrep -af "$SIM_PATTERN" >&2
    pgrep -af "$NODE_PATTERN" >&2
    return 1
  fi
}

# A topic appearing in `ros2 topic list` only means some node advertised it. The
# simulator advertises /clock while it is still building the stage, so a runner that
# starts then waits for sim time and gives up. Wait for an actual message.
# Subscribe best effort: /scan and /livox/lidar are published best effort, and a
# reliable subscriber never receives from them, so a reliable wait on /scan times out
# on a perfectly healthy simulator. Best effort is compatible with both kinds of
# publisher, so it is the right default for a liveness check.
wait_topic() {   # wait_topic <topic> <timeout seconds>
  local topic="$1" deadline=$((SECONDS + $2))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if timeout 5 ros2 topic echo --once --qos-reliability best_effort "$topic" \
        > /dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

# The published map's width and height in cells, or nothing if no map arrived.
map_cells() {
  timeout 10 ros2 topic echo --once --qos-reliability best_effort --field info /map \
      2>/dev/null \
    | awk '/^width:/ {w=$2} /^height:/ {h=$2} END {if (w && h) print w, h}'
}

start_sim() {   # start_sim <log> <sim args...>
  local log="$1"; shift
  setsid env AMR_WS="$REPO" "$REPO/amr_isaac/run_sim.sh" "$@" > "$log" 2>&1 &
  SIM_PID=$!
  echo "[lib] simulator pid $SIM_PID: $*"
  while ! wait_topic /clock 10; do
    kill -0 "$SIM_PID" 2>/dev/null || { echo "[lib] the simulator died, see $log" >&2; return 1; }
  done
  wait_topic /odom 60 || { echo "[lib] no /odom from the simulator" >&2; return 1; }
  # Falling back to a bare ground plane is fine when someone is driving the robot
  # around by hand, but an experiment measured in a world with no warehouse in it is
  # not a result, it is a silently wrong number. Refuse it here.
  if grep -q 'asset server unreachable' "$log" 2>/dev/null; then
    echo "[lib] the simulator fell back to an empty ground plane: no warehouse, so the" >&2
    echo "[lib] clearances and the SLAM map would be measured in the wrong world. See $log" >&2
    return 1
  fi
  echo "[lib] simulator ready"
}

start_nav() {   # start_nav <log>
  local log="$1"
  # SLAM has no input without the lidar, and the lidar is rendered on the GPU: when the
  # NVIDIA driver is upgraded under a running system the simulator still publishes
  # /clock and /odom but never a point cloud. Catch that here, where the message can say
  # so, instead of in a map wait that never ends. The topic to wait for is the
  # simulator's own /livox/lidar - /scan does not exist yet, because it is made by the
  # pointcloud_to_laserscan node inside the bringup this function is about to start.
  if ! wait_topic /livox/lidar 90; then
    echo "[lib] the simulator publishes no /livox/lidar; the RTX lidar is not rendering." >&2
    echo "[lib] check nvidia-smi: a driver upgrade under a running kernel module gives" >&2
    echo "[lib] 'Driver/library version mismatch' and needs a reboot." >&2
    return 1
  fi
  # The first frames of the RTX lidar are partial, and slam_toolbox keeps whichever
  # scan it sees first as the only node of its pose graph until the robot moves.
  sleep 5
  setsid ros2 launch amr_bringup sim.launch.py rviz:=false slam:=true nav:=true \
    > "$log" 2>&1 &
  NAV_PID=$!
  echo "[lib] ROS stack pid $NAV_PID"
  # Bounded: Nav2's lifecycle manager gives up on its own and then the launch parent
  # stays alive forever with half a stack under it, which an unbounded wait never sees.
  local deadline=$((SECONDS + 180))
  while ! wait_topic /map 10; do
    kill -0 "$NAV_PID" 2>/dev/null || { echo "[lib] the ROS stack died, see $log" >&2; return 1; }
    if grep -q 'Failed to bring up all requested nodes' "$log" 2>/dev/null; then
      echo "[lib] Nav2 bringup aborted, see $log" >&2
      return 1
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "[lib] no /map after 180 s, see $log" >&2
      return 1
    fi
  done
  sleep 5          # the first costmap update
  # A /map message only means slam_toolbox published something. Once it built its pose
  # graph from a partial first scan it published a 74 x 194 cell map (3.7 x 9.7 m) that
  # did not contain the robot: the planner answered every goal with "Start Coordinates
  # was outside bounds", so the robot never moved, so no new scan ever entered the graph
  # and the map never grew. Twenty trials of E4 at 1.00 m were lost that way. A healthy
  # first map of this warehouse is about 396 x 322 cells, so refuse anything far smaller
  # and let the caller start the stack again.
  local cells w h
  deadline=$((SECONDS + 90))
  while :; do
    cells=$(map_cells)
    w="${cells% *}"; h="${cells#* }"
    [ -n "$cells" ] && [ "$w" -ge 250 ] && [ "$h" -ge 250 ] && break
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "[lib] the SLAM map is ${w:-no} x ${h:-map} cells, too small to plan in;" >&2
      echo "[lib] slam_toolbox started from a partial scan, see $log" >&2
      return 1
    fi
    sleep 5
  done
  echo "[lib] map is $w x $h cells"
  # One cart_towing, or two of them fight over the Nav2 footprint.
  local towing
  towing=$(pgrep -cf 'lib/amr_docking/cart_towing')
  if [ "$towing" != 1 ]; then
    echo "[lib] $towing cart_towing nodes are running; expected exactly 1" >&2
    return 1
  fi
  echo "[lib] Nav2 and SLAM ready"
}
