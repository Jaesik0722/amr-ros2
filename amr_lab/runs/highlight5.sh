#!/usr/bin/env bash
# Record the highlight clips. Third attempt; the two before it each failed for their own
# reason, and both reasons are worth keeping written down.
#
#  * highlight.sh and highlight2.sh recorded at --num-envs 1, which capture.py used to
#    force. Every clip failed -- 12 of 12 seeds -- although the checkpoint places the
#    cart on 52.5% of behind-cart goals. At one environment the robot pivots on the spot
#    and the cart does not move at all (0.02 m in 20 s); at eight it tows normally
#    (2.95 m robot, 1.37 m cart, same seed, same scenario). So this records eight
#    environments and films environment 0.
#  * highlight3.sh then searched the BAY scene with the FREE-SPACE checkpoint and found
#    nothing in 18 seeds, which is the expected result of asking a policy trained in open
#    space to back into a slot it has never seen. The bay scene has its own four
#    checkpoints, resolved the same way the pairs scripts resolve them.
#
# The search runs undressed; a four-way bisect showed --visual-meshes and --backdrop give
# byte-identical episodes, so only the chosen scenario is recorded with them.
set -u
PY="env PYTHONUNBUFFERED=1 $HOME/anaconda3/envs/isaaclab/bin/python"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1   # the repository root
OUT=$HOME/amr_results/amr_lab/highlight
LOG=$OUT/record5.log
mkdir -p "$OUT/try5"
# stdout is the function return channel here, so the running commentary goes to stderr.
say() { echo "$*" >&2; echo "$*" >> "$LOG"; }

ck() { ls -1 logs/amr_lab/$1/*_$2/model_*.pt 2>/dev/null \
       | sed 's#\(.*model_\)\([0-9]*\)\(\.pt\)#\2 &#' | sort -n | tail -1 | cut -d' ' -f2; }

# free space
FA=$(ck caster4 fr_active_s3);          FP=$(ck caster4_passive fr_passive_s3)
FS_=$(ck caster4_passive fr_servo_s3);  FR=$(ck caster4_passive fr_rigid_s3)
# loading bay -- its own policies
BA=$(ck caster4 by_active_s3);          BP=$(ck caster4_passive by_passive_s3)
BS_=$(ck caster4_passive by_servo_s3);  BR=$(ck caster4_passive by_rigid_s3)

FLEET="--num-envs 8 --env-spacing 60"
CAM="--resolution 960 540 --follow --eye 1.5 -1.7 1.0 --lookat -0.55 0.0 0.28"
COMMON="--headless --enable_cameras --cart-wheels caster4 --episode-length 20 \
        --floor-friction 0.6 1.0 $FLEET"
SEARCH="$COMMON --every 10000 --resolution 320 180"
RECORD="$COMMON --visual-meshes --backdrop --every 1 $CAM"

note() {  # logfile outdir
  local l=$1 d=$2 txt step
  if grep -q 'reached=True' "$l"; then txt='reached the commanded pose'
  elif grep -q 'jackknife=True' "$l"; then txt='jackknifed'
  elif grep -q 'contact=True' "$l"; then txt='touched a bay wall'
  elif grep -q 'tipped=True' "$l"; then txt='tipped the cart'
  else txt='ran out of time'; fi
  step=$(grep -oE 'ended at step [0-9]+' "$l" | grep -oE '[0-9]+')
  [ -n "$step" ] && txt="$txt at $(python3 -c "print(f'{$step/20:.1f}')") s"
  echo "$txt" > "$d/outcome.txt"
}

search() {  # scene active-checkpoint extra-args...
  local scene=$1 ck=$2; shift 2
  say ""; say "======== $scene: scenario search at 8 envs  [$(date '+%F %T')] ========"
  local sd r
  for sd in 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24; do
    local f="$OUT/try5/${scene}_sc_$sd.pt"
    if [ ! -s "$f" ] || ! grep -q 'episode ended' "/tmp/hl5_try_${scene}_$sd.log" 2>/dev/null; then
      local d="$OUT/try5/${scene}_s$sd"; rm -rf "$d"; mkdir -p "$d"
      $PY amr_lab/capture.py $SEARCH --seed "$sd" --checkpoint "$ck" \
          --save-scenario "$f" --out "$d" "$@" \
          > "/tmp/hl5_try_${scene}_$sd.log" 2>&1
    fi
    r=$(grep -oE 'reached=[A-Za-z]+' "/tmp/hl5_try_${scene}_$sd.log")
    say "   seed $sd: $r $(grep -oE 'jackknife=[A-Za-z]+' "/tmp/hl5_try_${scene}_$sd.log")"
    if [ "$r" = "reached=True" ]; then echo "$sd"; return 0; fi
  done
  return 1
}

record() {  # scene seed spec... ; spec is cond:flags:checkpoint
  local scene=$1 sd=$2; shift 2
  local sc="$OUT/$scene/scenario.pt"
  rm -rf "$OUT/$scene"; mkdir -p "$OUT/$scene"
  cp "$OUT/try5/${scene}_sc_$sd.pt" "$sc"
  echo "$sd" > "$OUT/$scene/seed.txt"
  local extra=$EXTRA spec
  for spec in "$@"; do
    local cond=${spec%%:*} rest=${spec#*:} flags ckp
    flags=${rest%%:*}; ckp=${rest#*:}
    local d="$OUT/$scene/$cond"; rm -rf "$d"; mkdir -p "$d"
    say ""; say "---- $scene/$cond  seed $sd  [$(date '+%F %T')]"
    $PY amr_lab/capture.py $RECORD --seed "$sd" --checkpoint "$ckp" \
        --load-scenario "$sc" --out "$d" $flags $extra \
        > "/tmp/hl5_${scene}_$cond.log" 2>&1
    say "   $(ls "$d"/*.png 2>/dev/null | wc -l) frames; $(grep -oE 'episode ended.*' "/tmp/hl5_${scene}_$cond.log")"
    note "/tmp/hl5_${scene}_$cond.log" "$d"
  done
}

# The free-space scene was already found at seed 17 by the previous attempt; reuse its
# scenario file rather than walking the seeds again.
mkdir -p "$OUT/try5"
cp -n "$OUT/try4/free_sc_17.pt" "$OUT/try5/free_sc_17.pt" 2>/dev/null
cp -n "$OUT/try4/bay_sc_9.pt" "$OUT/try5/bay_sc_9.pt" 2>/dev/null
cp -n /tmp/hl4_try_free_17.log /tmp/hl5_try_free_17.log 2>/dev/null
cp -n /tmp/hl4_try_bay_9.log /tmp/hl5_try_bay_9.log 2>/dev/null
EXTRA=""
# Both scenarios were already found by the previous attempt, at seed 17 in the
# free-space scene and seed 9 in the bay. This pass only re-records them, so that
# the caption of every clip comes from a run that reports its termination cause.
F=17
say "## free scene: seed $F"
record free "$F" "active::$FA" "passive:--passive-hitch:$FP" \
    "servo:--passive-hitch --lock-hitch:$FS_" "rigid:--rigid-lock:$FR"

EXTRA="--no-backward --bay --curriculum 1.0"
B=9
say "## bay scene: seed $B"
record bay "$B" "active::$BA" "passive:--passive-hitch:$BP" \
    "servo:--passive-hitch --lock-hitch:$BS_" "rigid:--rigid-lock:$BR"

say ""; say "HIGHLIGHT5 DONE $(date '+%F %T')"
say "   free scene: seed $F;  bay scene: seed $B"
for s in free bay; do for c in active passive servo rigid; do
  f="$OUT/$s/$c/outcome.txt"; [ -f "$f" ] && say "   $s/$c: $(cat "$f")"
done; done
