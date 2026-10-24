#!/usr/bin/env bash
# The first folding seed under 200 ms folds 1.1 s in, which is a true failure and a poor
# clip: it is over before a viewer has seen the task. Walk the whole seed range, keep the
# step each fold happened at, and record the one nearest eight seconds -- long enough to
# show the robot working and then losing the cart.
set -u
PY="env PYTHONUNBUFFERED=1 $HOME/anaconda3/envs/isaaclab/bin/python"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1   # the repository root
OUT=$HOME/amr_results/amr_lab/highlight
LOG=$OUT/record9.log
mkdir -p "$OUT/try9"
say() { echo "$*" >&2; echo "$*" >> "$LOG"; }
FA=$(ls -1 logs/amr_lab/caster4/*_fr_active_s3/model_*.pt | tail -1)

CAM="--follow --eye 2.6 -3.0 2.1 --lookat -0.5 0.0 0.2"
COMMON="--headless --enable_cameras --cart-wheels caster4 --episode-length 20 \
        --floor-friction 0.6 1.0 --num-envs 8 --env-spacing 60 --latency 4 4 $CAM"

say ""; say "======== 200 ms folds, by when they fold  [$(date '+%F %T')] ========"
best=""; bestd=999
for sd in 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24; do
  d="$OUT/try9/s$sd"; rm -rf "$d"; mkdir -p "$d"
  $PY amr_lab/capture.py $COMMON --every 10000 --resolution 320 180 --seed "$sd" \
      --checkpoint "$FA" --save-scenario "$OUT/try9/sc_$sd.pt" --out "$d" \
      > "/tmp/j9_$sd.log" 2>&1
  if grep -q 'jackknife=True' "/tmp/j9_$sd.log"; then
    step=$(grep -oE 'ended at step [0-9]+' "/tmp/j9_$sd.log" | grep -oE '[0-9]+')
    t=$(python3 -c "print(f'{$step/20:.1f}')")
    dist=$(python3 -c "print(abs($step/20 - 8.0))")
    say "   seed $sd: folds at $t s"
    if python3 -c "import sys; sys.exit(0 if $dist < $bestd else 1)"; then
      best=$sd; bestd=$dist
    fi
  else
    say "   seed $sd: no fold"
  fi
done
[ -n "$best" ] || { say "# none folded"; exit 1; }
say "## chosen: seed $best"

d="$OUT/jack200/active"; rm -rf "$d"; mkdir -p "$d"
cp "$OUT/try9/sc_$best.pt" "$OUT/jack200/scenario.pt"; echo "$best" > "$OUT/jack200/seed.txt"
$PY amr_lab/capture.py $COMMON --every 1 --resolution 960 540 --visual-meshes \
    --backdrop --seed "$best" --checkpoint "$FA" \
    --load-scenario "$OUT/jack200/scenario.pt" --out "$d" > /tmp/j9_rec.log 2>&1
say "   $(ls "$d"/*.png | wc -l) frames; $(grep -oE 'episode ended.*' /tmp/j9_rec.log)"
step=$(grep -oE 'ended at step [0-9]+' /tmp/j9_rec.log | grep -oE '[0-9]+')
if grep -q 'jackknife=True' /tmp/j9_rec.log; then
  echo "Jackknife  $(python3 -c "print(f'{int($step*10/20)/10:.1f}')") s" > "$d/outcome.txt"
else
  echo "Did not fold on the recorded run" > "$d/outcome.txt"
  say "# the recorded run did not fold"
fi
say "   outcome: $(cat "$d/outcome.txt")"
say "JACK200B DONE $(date '+%F %T')"
