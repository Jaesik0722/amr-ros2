#!/usr/bin/env bash
# Record the docking/towing demo. The simulator and Nav2 must already be running:
#   ./amr_isaac/run_sim.sh --headless --chase-camera --overview-camera 4.5 5.0 6.5 4.5 -3.8 0.0
#   ros2 launch amr_bringup sim.launch.py rviz:=false slam:=true nav:=true
# then:  xvfb-run -a -s "-screen 0 960x540x24" amr_isaac/tools/demo_video/record.sh OUT_DIR
# and:   python3 amr_isaac/tools/demo_video/compose.py OUT_DIR
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
V="$(realpath -m "${1:?usage: record.sh OUT_DIR}")"
rm -rf "$V/frames"; mkdir -p "$V/frames"; : > "$V/events.txt"
R="--ros-args -p use_sim_time:=true"
rviz2 -d "$HERE/video.rviz" --ros-args -p use_sim_time:=true > "$V/rviz.log" 2>&1 &
RV=$!
sleep 10
python3 "$HERE/grab_multi.py" "$V/frames" 400 > "$V/grab.log" 2>&1 &
GR=$!
until [ -f "$V/frames/started" ]; do sleep 0.05; done
T0=$(date +%s.%N)
ffmpeg -loglevel error -y -f x11grab -framerate 10 -video_size 960x540 -i "$DISPLAY" -t 400 \
  -c:v libx264 -pix_fmt yuv420p "$V/rviz.mp4" < /dev/null &
FF=$!
ev() { echo "$(echo "$(date +%s.%N) - $T0" | bc) $*" >> "$V/events.txt"; }
GV=
goalviz() {   # goal arrow for RViz (/video_goal)
  [ -n "$GV" ] && kill "$GV" 2>/dev/null
  ros2 topic pub -r 2 /video_goal geometry_msgs/msg/PoseStamped "{header: {frame_id: map}, pose: {position: {x: $1, y: $2}, orientation: {z: $3, w: $4}}}" > /dev/null 2>&1 &
  GV=$!
}
sleep 1
ev charger_start; goalviz 8.096 -1.0 0.0 1.0
timeout 300 ros2 run amr_docking dock_charger $R > "$V/dock_charger.log" 2>&1; ev charger_done
sleep 4; ev undock_start
timeout 120 ros2 run amr_docking undock_charger $R > "$V/undock.log" 2>&1; ev undock_done
goalviz 3.0 -6.651 0.7071 0.7071; ev cart_start
timeout 400 stdbuf -oL ros2 run amr_docking dock_cart $R 2>&1 | while IFS= read -r line; do
  echo "$line" >> "$V/dock_cart.log"
  case "$line" in
    *"staging at"*) ev cart_gps;;
    *"At the tow bar"*) ev cart_at_bar;;
    *"Hitch pitch effort"*) ev "effort $(echo "$line" | sed -E 's/.*, ([0-9.]+) after.*/\1/')";;
    *"Cart hooked"*) ev cart_hooked;;
  esac
done
ev tow_start; goalviz 7.0 -2.5 0.0 1.0
timeout 240 ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose "{pose: {header: {frame_id: map}, pose: {position: {x: 7.0, y: -2.5}, orientation: {w: 1.0}}}}" > "$V/tow.log" 2>&1; ev tow_done
kill "$GV" 2>/dev/null
timeout 120 ros2 run amr_docking release_cart $R > "$V/release.log" 2>&1; ev release_done
sleep 3; ev end
kill -INT $FF; sleep 2; kill $GR $RV 2>/dev/null; sleep 1
cat "$V/events.txt"
