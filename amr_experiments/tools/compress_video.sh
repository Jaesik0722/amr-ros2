#!/usr/bin/env bash
# Make a 720p copy of a composed clip, small enough to send.
#
#   ./amr_experiments/tools/compress_video.sh ~/amr_results/videos/e7/e7_highlight.mp4 ...
#
# The copies go to <videos>/compressed/<name>. The full-resolution file stays where it
# is: the panels are 960x540 each, so a 1920x1080 clip halved to 1280x720 still reads,
# but the text in the panels is near its limit and compressing twice makes it mush.
set -uo pipefail
OUT="${AMR_VIDEOS:-$HOME/amr_results/videos}/compressed"
mkdir -p "$OUT"
status=0
for src in "$@"; do
  [ -f "$src" ] || { echo "no such file: $src" >&2; status=1; continue; }
  dst="$OUT/$(basename "$src")"
  ffmpeg -loglevel error -y -i "$src" -vf scale=1280:-2 -c:v libx264 -preset slow \
    -crf 30 -pix_fmt yuv420p -movflags +faststart "$dst" || { status=1; continue; }
  printf '%s  %s\n' "$(du -h "$dst" | cut -f1)" "$dst"
done
exit "$status"
