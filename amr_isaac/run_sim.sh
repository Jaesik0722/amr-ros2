#!/usr/bin/env bash
# Start the Isaac Sim 5.1 simulation of the AMR.
#   ./run_sim.sh [--headless] [--env warehouse|empty] [--no-cameras] [--no-lidar] [--reimport]
#
# Needs: a built workspace with amr_description (sourced or via AMR_WS), and Isaac Sim 5.1
# installed with pip in a conda env (ISAAC_PYTHON, default: the 'isaaclab' env).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-$HOME/anaconda3/envs/isaaclab/bin/python}"
CACHE="${AMR_SIM_CACHE:-$HOME/.cache/amr_isaac}"
mkdir -p "$CACHE"

# 1) Expand the xacro with the system ROS 2 (Python 3.12) and make mesh paths absolute.
set +u  # ROS setup scripts use unset variables
if ! command -v xacro >/dev/null; then source /opt/ros/jazzy/setup.bash; fi
if [ -n "${AMR_WS:-}" ]; then source "$AMR_WS/install/setup.bash"; fi
set -u
SHARE="$(ros2 pkg prefix amr_description)/share/amr_description"
# Cart and charging station (docking targets and markers come from amr_docking).
DOCKING_ARGS=()
if DOCKING="$(ros2 pkg prefix amr_docking 2>/dev/null)/share/amr_docking" && [ -d "$DOCKING" ]; then
  DOCKING_ARGS=(--targets "$DOCKING/config/docking_targets.yaml" --markers "$DOCKING/markers")
fi
xacro "$SHARE/urdf/amr.urdf.xacro" | sed "s#package://amr_description/#$SHARE/#g" > "$CACHE/amr.urdf.new"
if ! cmp -s "$CACHE/amr.urdf.new" "$CACHE/amr.urdf"; then mv "$CACHE/amr.urdf.new" "$CACHE/amr.urdf"; else rm "$CACHE/amr.urdf.new"; fi

# 2) Run Isaac Sim with its bundled ROS 2 Jazzy libraries (Python 3.11), not the system ones.
ISAAC_PKG="$("$ISAAC_PYTHON" -c 'import isaacsim, os; print(os.path.dirname(isaacsim.__file__))' 2>/dev/null | tail -1)"
exec env -i HOME="$HOME" USER="${USER:-}" PATH=/usr/bin:/bin TERM="${TERM:-xterm}" \
  DISPLAY="${DISPLAY:-}" XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-}" XAUTHORITY="${XAUTHORITY:-}" \
  OMNI_KIT_ACCEPT_EULA=YES \
  ROS_DISTRO=jazzy RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" \
  ${ROS_AUTOMATIC_DISCOVERY_RANGE:+ROS_AUTOMATIC_DISCOVERY_RANGE=$ROS_AUTOMATIC_DISCOVERY_RANGE} \
  LD_LIBRARY_PATH="$ISAAC_PKG/exts/isaacsim.ros2.bridge/jazzy/lib" \
  "$ISAAC_PYTHON" "$HERE/amr_sim.py" --urdf "$CACHE/amr.urdf" --usd-dir "$CACHE/usd" "${DOCKING_ARGS[@]}" "$@"
