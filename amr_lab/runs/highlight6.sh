#!/usr/bin/env bash
# The clips the revised cut needs on top of what highlight5.sh recorded:
#   bay_sq  the SAME bay, approached from the near-square distribution (curriculum
#           0.0125, the +-15 deg column of Table III) instead of the trained one. The
#           result card contrasts the two approach windows, so the video should show
#           both rather than asking the viewer to take the contrast on trust.
#   jack    one episode of the ACTIVE policy folding the cart. The paper's latency
#           finding is that the success rate hides a failure mode, and a cut that only
#           ever shows the powered joint succeeding illustrates the opposite.
#
# The scenario rule is the same one used for the other two scenes -- walk the seeds until
# the active run reaches -- so no scene is picked by a different standard. The jackknife
# clip is the exception by construction: it replays free-space seed 8, which the earlier
# search already recorded as folding at 3.1 s.
set -u
PY="env PYTHONUNBUFFERED=1 $HOME/anaconda3/envs/isaaclab/bin/python"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1   # the repository root
OUT=$HOME/amr_results/amr_lab/highlight
LOG=$OUT/record6.log
mkdir -p "$OUT/try6"
say() { echo "$*" >&2; echo "$*" >> "$LOG"; }

ck() { ls -1 logs/amr_lab/$1/*_$2/model_*.pt 2>/dev/null \
       | sed 's#\(.*model_\)\([0-9]*\)\(\.pt\)#\2 &#' | sort -n | tail -1 | cut -d' ' -f2; }
FA=$(ck caster4 fr_active_s3)
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
  if grep -q 'reached=True' "$l"; then txt='Success'
  elif grep -q 'jackknife=True' "$l"; then txt='Jackknife'
  elif grep -q 'contact=True' "$l"; then txt='Wall contact'
  elif grep -q 'tipped=True' "$l"; then txt='Cart tipped'
  else txt='Timeout'; fi
  step=$(grep -oE 'ended at step [0-9]+' "$l" | grep -oE '[0-9]+')
  [ -n "$step" ] && txt="$txt  $(python3 -c "print(f'{$step/20:.1f}')") s"
  echo "$txt" > "$d/outcome.txt"
}

search() {  # scene active-checkpoint extra-args...
  local scene=$1 ck=$2; shift 2
  say ""; say "======== $scene: scenario search  [$(date '+%F %T')] ========"
  local sd r
  for sd in 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24; do
    local f="$OUT/try6/${scene}_sc_$sd.pt"
    local d="$OUT/try6/${scene}_s$sd"; rm -rf "$d"; mkdir -p "$d"
    $PY amr_lab/capture.py $SEARCH --seed "$sd" --checkpoint "$ck" \
        --save-scenario "$f" --out "$d" "$@" > "/tmp/hl6_try_${scene}_$sd.log" 2>&1
    r=$(grep -oE 'reached=[A-Za-z]+' "/tmp/hl6_try_${scene}_$sd.log")
    say "   seed $sd: $r"
    if [ "$r" = "reached=True" ]; then echo "$sd"; return 0; fi
  done
  return 1
}

record() {  # scene seed extra-args-string spec...
  local scene=$1 sd=$2 extra=$3; shift 3
  local sc="$OUT/$scene/scenario.pt"
  rm -rf "$OUT/$scene"; mkdir -p "$OUT/$scene"
  cp "$OUT/try6/${scene}_sc_$sd.pt" "$sc"
  echo "$sd" > "$OUT/$scene/seed.txt"
  local spec
  for spec in "$@"; do
    local cond=${spec%%:*} rest=${spec#*:} flags ckp
    flags=${rest%%:*}; ckp=${rest#*:}
    local d="$OUT/$scene/$cond"; rm -rf "$d"; mkdir -p "$d"
    say ""; say "---- $scene/$cond  seed $sd  [$(date '+%F %T')]"
    $PY amr_lab/capture.py $RECORD --seed "$sd" --checkpoint "$ckp" \
        --load-scenario "$sc" --out "$d" $flags $extra \
        > "/tmp/hl6_${scene}_$cond.log" 2>&1
    say "   $(ls "$d"/*.png 2>/dev/null | wc -l) frames; $(grep -oE 'episode ended.*' "/tmp/hl6_${scene}_$cond.log")"
    note "/tmp/hl6_${scene}_$cond.log" "$d"
  done
}

SQ="--no-backward --bay --curriculum 0.0125"
B=$(search bay_sq "$BA" $SQ) || { say "# no near-square bay seed reached"; exit 1; }
say "## bay_sq scene: seed $B"
record bay_sq "$B" "$SQ" "active::$BA" "passive:--passive-hitch:$BP" \
    "servo:--passive-hitch --lock-hitch:$BS_" "rigid:--rigid-lock:$BR"

# The folding episode: free-space seed 8, the active policy, replayed from the scenario
# the earlier search saved.
cp -f "$OUT/try4/free_sc_8.pt" "$OUT/try6/jack_sc_8.pt"
say ""; say "## jackknife clip: free-space seed 8, active"
record jack 8 "" "active::$FA"

say ""; say "HIGHLIGHT6 DONE $(date '+%F %T')"
for c in active passive servo rigid; do
  f="$OUT/bay_sq/$c/outcome.txt"; [ -f "$f" ] && say "   bay_sq/$c: $(cat "$f")"
done
f="$OUT/jack/active/outcome.txt"; [ -f "$f" ] && say "   jack/active: $(cat "$f")"
