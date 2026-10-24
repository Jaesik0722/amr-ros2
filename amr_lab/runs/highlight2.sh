#!/usr/bin/env bash
# Pick the scenario each highlight grid is drawn on, then record all four conditions
# on it and write each clip's outcome beside its frames.
#
# The first draw of both scenes folds the active policy. That is an ordinary outcome --
# it places the cart on 36.9% of behind-cart goals and 35.4% of trained bay approaches,
# so most draws end some other way -- but a grid whose point is that the powered joint
# places the cart should not be drawn on one of the failures. This walks seeds until
# the ACTIVE run reaches, records the other three on that same scenario, and records
# which seed was used so the caption can state it.
set -u
PY="env PYTHONUNBUFFERED=1 $HOME/anaconda3/envs/isaaclab/bin/python"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1   # the repository root
OUT=$HOME/amr_results/amr_lab/highlight
LOG=$OUT/record2.log
mkdir -p "$OUT/try"
say() { echo "$*" | tee -a "$LOG"; }

A=logs/amr_lab/caster4/2026-10-22_01-39-47_fr_active_s3/model_2999.pt
P=logs/amr_lab/caster4_passive/2026-10-22_03-05-40_fr_passive_s3/model_2999.pt
S=logs/amr_lab/caster4_passive/2026-10-22_04-39-52_fr_servo_s3/model_2999.pt
R=logs/amr_lab/caster4_passive/2026-10-22_06-11-30_fr_rigid_s3/model_2999.pt
BA=logs/amr_lab/caster4/2026-10-22_01-39-47_fr_active_s3/model_2999.pt

CAM="--resolution 960 540 --follow --eye 1.5 -1.7 1.0 --lookat -0.55 0.0 0.28"
BASE="--headless --enable_cameras --cart-wheels caster4 --visual-meshes --backdrop
      --episode-length 20 --floor-friction 0.6 1.0 --every 1 $CAM"

note() {  # logfile outdir
  local l=$1 d=$2 txt step
  if grep -q 'reached=True' "$l"; then txt='reached the commanded pose'
  elif grep -q 'jackknife=True' "$l"; then txt='jackknifed'
  else txt='ran out of time'; fi
  step=$(grep -oE 'ended at step [0-9]+' "$l" | grep -oE '[0-9]+')
  [ -n "$step" ] && txt="$txt at $(python3 -c "print(f'{$step/20:.1f}')") s"
  echo "$txt" > "$d/outcome.txt"
}

search() {  # scene extra-args...
  local scene=$1; shift
  say ""; say "======== $scene: scenario search  [$(date '+%F %T')] ========"
  local sd r
  for sd in 3 4 5 6 7 8 9 10 11 12 13 14; do
    local d="$OUT/try/${scene}_s$sd"; rm -rf "$d"; mkdir -p "$d"
    $PY amr_lab/capture.py $BASE --seed "$sd" --checkpoint "$A" \
        --save-scenario "$OUT/try/${scene}_sc_$sd.pt" --out "$d" "$@" \
        > "/tmp/hl_try_${scene}_$sd.log" 2>&1
    r=$(grep -oE 'reached=[A-Za-z]+' "/tmp/hl_try_${scene}_$sd.log")
    say "   seed $sd: $r $(grep -oE 'jackknife=[A-Za-z]+' "/tmp/hl_try_${scene}_$sd.log")"
    if [ "$r" = "reached=True" ]; then echo "$sd"; return 0; fi
    sleep 8
  done
  return 1
}

record() {  # scene seed extra-args...
  local scene=$1 sd=$2; shift 2
  local sc="$OUT/$scene/scenario.pt"
  rm -rf "$OUT/$scene"; mkdir -p "$OUT/$scene"
  cp "$OUT/try/${scene}_sc_$sd.pt" "$sc"
  echo "$sd" > "$OUT/$scene/seed.txt"
  local spec
  for spec in "active::$A" "passive:--passive-hitch:$P" \
              "servo:--passive-hitch --lock-hitch:$S" "rigid:--rigid-lock:$R"; do
    local cond=${spec%%:*} rest=${spec#*:} flags ck
    flags=${rest%%:*}; ck=${rest#*:}
    local d="$OUT/$scene/$cond"; rm -rf "$d"; mkdir -p "$d"
    say ""; say "---- $scene/$cond  seed $sd  [$(date '+%F %T')]"
    $PY amr_lab/capture.py $BASE --seed "$sd" --checkpoint "$ck" \
        --load-scenario "$sc" --out "$d" $flags "$@" \
        > "/tmp/hl2_${scene}_$cond.log" 2>&1
    say "   $(ls "$d"/*.png 2>/dev/null | wc -l) frames; $(grep -oE 'episode ended.*' "/tmp/hl2_${scene}_$cond.log")"
    note "/tmp/hl2_${scene}_$cond.log" "$d"
    sleep 8
  done
}

FS=$(search free) || { say "# no free-space seed reached in the range"; exit 1; }
record free "$FS"

BS=$(search bay --no-backward --bay --curriculum 1.0) \
  || { say "# no bay seed reached in the range"; exit 1; }
record bay "$BS" --no-backward --bay --curriculum 1.0

say ""; say "HIGHLIGHT2 DONE $(date '+%F %T')"
say "   free scene: seed $FS;  bay scene: seed $BS"
for s in free bay; do for c in active passive servo rigid; do
  f="$OUT/$s/$c/outcome.txt"; [ -f "$f" ] && say "   $s/$c: $(cat "$f")"
done; done
