#!/usr/bin/env bash
# Re-record every clip with the goal drawn on the floor and a camera that frames it.
#
# The review's first point: the clips show the cart moving but not where it was asked to
# go, so a viewer has to wait for the overhead card to find out why one run is a success
# and another a timeout. Two changes fix that together --- a visual-only marker at the
# commanded cart pose, and a fixed camera computed per scenario to hold the cart start and
# the goal in one shot. A chase camera cannot: the goal can be 3.5 m away in any
# direction.
#
# Also records the folding episode under a FIXED 200 ms delay rather than at the trained
# delay, so the clip that follows the latency card is an episode of the case that card
# describes instead of one that merely resembles it.
set -u
PY="env PYTHONUNBUFFERED=1 $HOME/anaconda3/envs/isaaclab/bin/python"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1   # the repository root
OUT=$HOME/amr_results/amr_lab/highlight
LOG=$OUT/record7.log
mkdir -p "$OUT/try7"
say() { echo "$*" >&2; echo "$*" >> "$LOG"; }

ck() { ls -1 logs/amr_lab/$1/*_$2/model_*.pt 2>/dev/null \
       | sed 's#\(.*model_\)\([0-9]*\)\(\.pt\)#\2 &#' | sort -n | tail -1 | cut -d' ' -f2; }
FA=$(ck caster4 fr_active_s3);          FP=$(ck caster4_passive fr_passive_s3)
FS_=$(ck caster4_passive fr_servo_s3);  FR=$(ck caster4_passive fr_rigid_s3)
BA=$(ck caster4 by_active_s3);          BP=$(ck caster4_passive by_passive_s3)
BS_=$(ck caster4_passive by_servo_s3);  BR=$(ck caster4_passive by_rigid_s3)

COMMON="--headless --enable_cameras --cart-wheels caster4 --episode-length 20 \
        --floor-friction 0.6 1.0 --num-envs 8 --env-spacing 60"
RECORD="$COMMON --visual-meshes --backdrop --every 1 --resolution 960 540 --auto-camera"
SEARCH="$COMMON --every 10000 --resolution 320 180 --auto-camera"

note() {  # logfile outdir
  local l=$1 d=$2 txt step
  if grep -q 'reached=True' "$l"; then txt='Success'
  elif grep -q 'jackknife=True' "$l"; then txt='Jackknife'
  elif grep -q 'contact=True' "$l"; then txt='Wall contact'
  elif grep -q 'tipped=True' "$l"; then txt='Cart tipped'
  else txt='Timeout'; fi
  step=$(grep -oE 'ended at step [0-9]+' "$l" | grep -oE '[0-9]+')
  [ -n "$step" ] && txt="$txt  $(python3 -c "print(f'{int($step*10/20)/10:.1f}')") s"
  echo "$txt" > "$d/outcome.txt"
}

record() {  # scene extra-args-string spec...
  local scene=$1 extra=$2; shift 2
  local sc="$OUT/$scene/scenario.pt" sd
  sd=$(cat "$OUT/$scene/seed.txt" 2>/dev/null || echo 17)
  local spec
  for spec in "$@"; do
    local cond=${spec%%:*} rest=${spec#*:} flags ckp
    flags=${rest%%:*}; ckp=${rest#*:}
    local d="$OUT/$scene/$cond"
    rm -f "$d"/*.png
    say ""; say "---- $scene/$cond  seed $sd  [$(date '+%F %T')]"
    $PY amr_lab/capture.py $RECORD --seed "$sd" --checkpoint "$ckp" \
        --load-scenario "$sc" --out "$d" $flags $extra \
        > "/tmp/hl7_${scene}_$cond.log" 2>&1
    say "   $(ls "$d"/*.png 2>/dev/null | wc -l) frames; $(grep -oE 'episode ended.*' "/tmp/hl7_${scene}_$cond.log")"
    note "/tmp/hl7_${scene}_$cond.log" "$d"
  done
}

say ""; say "======== re-record with the goal in shot  [$(date '+%F %T')] ========"
record free "" "active::$FA" "passive:--passive-hitch:$FP" \
    "servo:--passive-hitch --lock-hitch:$FS_" "rigid:--rigid-lock:$FR"
record bay "--no-backward --bay --curriculum 1.0" "active::$BA" \
    "passive:--passive-hitch:$BP" "servo:--passive-hitch --lock-hitch:$BS_" \
    "rigid:--rigid-lock:$BR"
record bay_sq "--no-backward --bay --curriculum 0.0125" "active::$BA" \
    "passive:--passive-hitch:$BP" "servo:--passive-hitch --lock-hitch:$BS_" \
    "rigid:--rigid-lock:$BR"

# The folding episode, under the delay the card is about: 4 control steps at 20 Hz.
say ""; say "======== jackknife under a fixed 200 ms delay  [$(date '+%F %T')] ========"
D200="--latency 4 4"
JS=""
for sd in 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18; do
  d="$OUT/try7/j200_s$sd"; rm -rf "$d"; mkdir -p "$d"
  $PY amr_lab/capture.py $SEARCH --seed "$sd" --checkpoint "$FA" $D200 \
      --save-scenario "$OUT/try7/j200_sc_$sd.pt" --out "$d" \
      > "/tmp/hl7_j200_$sd.log" 2>&1
  say "   seed $sd: $(grep -oE 'jackknife=[A-Za-z]+' "/tmp/hl7_j200_$sd.log")"
  if grep -q 'jackknife=True' "/tmp/hl7_j200_$sd.log"; then JS=$sd; break; fi
done
if [ -n "$JS" ]; then
  rm -rf "$OUT/jack200"; mkdir -p "$OUT/jack200"
  cp "$OUT/try7/j200_sc_$JS.pt" "$OUT/jack200/scenario.pt"
  echo "$JS" > "$OUT/jack200/seed.txt"
  record jack200 "$D200" "active::$FA"
else
  say "# no seed folded under 200 ms in the range"
fi

say ""; say "HIGHLIGHT7 DONE $(date '+%F %T')"
for s in free bay bay_sq jack200; do for c in active passive servo rigid; do
  f="$OUT/$s/$c/outcome.txt"; [ -f "$f" ] && say "   $s/$c: $(cat "$f")"
done; done
