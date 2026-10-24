#!/usr/bin/env bash
# Record the panels of one experiment run. Runs inside xvfb-run, started by
# record_highlight.sh, which has already brought up the simulator and the ROS stack.
#
#   capture_highlight.sh OUT_DIR SECONDS RUNNER [runner options...]
#
# Writes OUT_DIR/frames/{overview,chase,front,rear}, OUT_DIR/rviz.mp4 (unless
# HL_RVIZ=0) and OUT_DIR/events.txt, whose lines are "<seconds> <caption>" taken from
# the runner's own per-trial log lines. The caption times are what makes the composed
# video readable, so they are stamped here rather than guessed afterwards.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
GRAB="$REPO/amr_isaac/tools/demo_video/grab_multi.py"
RVIZ_CONFIG="${HL_RVIZ_CONFIG:-$REPO/amr_isaac/tools/demo_video/video.rviz}"

OUT="$(realpath -m "${1:?usage: capture_highlight.sh OUT_DIR SECONDS RUNNER [options]}")"
SECS="${2:?missing SECONDS}"
RUNNER="${3:?missing RUNNER}"
shift 3

rm -rf "$OUT/frames"
mkdir -p "$OUT/frames"
: > "$OUT/events.txt"

RV=
if [ "${HL_RVIZ:-1}" = 1 ]; then
  rviz2 -d "$RVIZ_CONFIG" --ros-args -p use_sim_time:=true > "$OUT/rviz.log" 2>&1 &
  RV=$!
  sleep 10
fi

python3 "$GRAB" "$OUT/frames" "$SECS" ${HL_CAMERAS:+"$HL_CAMERAS"} > "$OUT/grab.log" 2>&1 &
GR=$!
# The grabber writes this once every camera has delivered a frame; starting the clock
# before that would put the captions out of step with the pictures.
until [ -f "$OUT/frames/started" ]; do
  kill -0 "$GR" 2>/dev/null || { echo "the frame grabber died, see $OUT/grab.log" >&2; exit 1; }
  sleep 0.05
done
T0=$(date +%s.%N)

FF=
if [ -n "$RV" ]; then
  ffmpeg -loglevel error -y -f x11grab -framerate 10 -video_size 960x540 -i "$DISPLAY" \
    -t "$SECS" -c:v libx264 -pix_fmt yuv420p "$OUT/rviz.mp4" < /dev/null &
  FF=$!
fi

ev() { echo "$(echo "$(date +%s.%N) - $T0" | bc) $*" >> "$OUT/events.txt"; }

ev "$HL_TITLE"
timeout "$SECS" stdbuf -oL ros2 run amr_experiments "$RUNNER" "$@" 2>&1 \
  | while IFS= read -r line; do
      echo "$line" | tee -a "$OUT/run.log" > /dev/null
      # Only the runner's own per-trial lines become captions; everything else in the
      # log is Nav2 chatter that would flicker past unread.
      case "$line" in
        *"]: trial "*|*"]: cycle "*|*"]: stage "*) ev "${line#*]: }" ;;
      esac
    done
ev end

[ -n "$FF" ] && { kill -INT "$FF" 2>/dev/null; sleep 2; }
kill "$GR" 2>/dev/null
[ -n "$RV" ] && kill "$RV" 2>/dev/null
sleep 1
cat "$OUT/events.txt"
