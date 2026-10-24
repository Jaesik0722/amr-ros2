#!/usr/bin/env bash
# A folding episode under the fixed 200 ms delay, framed so the fold is in the middle of
# the picture.
#
# The auto camera frames the whole start-to-goal line, which is right for the comparison
# grids and wrong for a single clip: the fold happens near the goal and ends up at the
# edge of an otherwise empty floor. This uses a chase camera instead, and searches for
# the seed under that same configuration -- recording a clip under different settings
# from the search is not safe here, because per-scenario outcomes are not reproducible
# across runs (about one scenario in eight flips), so a seed found one way can come back
# the other way when it is recorded.
set -u
PY="env PYTHONUNBUFFERED=1 $HOME/anaconda3/envs/isaaclab/bin/python"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1   # the repository root
OUT=$HOME/amr_results/amr_lab/highlight
LOG=$OUT/record8.log
mkdir -p "$OUT/try8"
say() { echo "$*" >&2; echo "$*" >> "$LOG"; }
FA=$(ls -1 logs/amr_lab/caster4/*_fr_active_s3/model_*.pt | tail -1)

CAM="--follow --eye 2.6 -3.0 2.1 --lookat -0.5 0.0 0.2"
COMMON="--headless --enable_cameras --cart-wheels caster4 --episode-length 20 \
        --floor-friction 0.6 1.0 --num-envs 8 --env-spacing 60 --latency 4 4 $CAM"

say ""; say "======== jackknife at 200 ms, chase framing  [$(date '+%F %T')] ========"
JS=""
for sd in 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
  d="$OUT/try8/s$sd"; rm -rf "$d"; mkdir -p "$d"
  $PY amr_lab/capture.py $COMMON --every 10000 --resolution 320 180 --seed "$sd" \
      --checkpoint "$FA" --save-scenario "$OUT/try8/sc_$sd.pt" --out "$d" \
      > "/tmp/j8_$sd.log" 2>&1
  say "   seed $sd: $(grep -oE 'jackknife=[A-Za-z]+' "/tmp/j8_$sd.log")"
  if grep -q 'jackknife=True' "/tmp/j8_$sd.log"; then JS=$sd; break; fi
done
[ -n "$JS" ] || { say "# none folded in the range"; exit 1; }

d="$OUT/jack200/active"; rm -rf "$d"; mkdir -p "$d"
cp "$OUT/try8/sc_$JS.pt" "$OUT/jack200/scenario.pt"; echo "$JS" > "$OUT/jack200/seed.txt"
say ""; say "---- recording seed $JS  [$(date '+%F %T')]"
$PY amr_lab/capture.py $COMMON --every 1 --resolution 960 540 --visual-meshes \
    --backdrop --seed "$JS" --checkpoint "$FA" \
    --load-scenario "$OUT/jack200/scenario.pt" --out "$d" > /tmp/j8_rec.log 2>&1
say "   $(ls "$d"/*.png | wc -l) frames; $(grep -oE 'episode ended.*' /tmp/j8_rec.log)"
step=$(grep -oE 'ended at step [0-9]+' /tmp/j8_rec.log | grep -oE '[0-9]+')
if grep -q 'jackknife=True' /tmp/j8_rec.log; then
  echo "Jackknife  $(python3 -c "print(f'{int($step*10/20)/10:.1f}')") s" > "$d/outcome.txt"
else
  say "# the recorded run did not fold; outcome written from what it did"
  echo "Did not fold on the recorded run" > "$d/outcome.txt"
fi
say "   outcome: $(cat "$d/outcome.txt")"
say "JACK200 DONE $(date '+%F %T')"
