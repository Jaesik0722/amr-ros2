#!/bin/bash
# The one-tonne study, started automatically once the video is recorded.
#
# Everything in paper 5 is trained and measured at 0-100 kg on the deck (cart 22.6 kg).
# One tonne is a ten-fold extrapolation outside that distribution, so this is NOT a
# paper-5 result and is written to its own file. Two questions are separated:
#
#   phase 1  open loop, no policy: what can the MACHINE do at these masses?  The payload
#            is centred and the only things that move are mass, floor friction and the
#            wheel torque limit, so a failure here is mechanical.
#   phase 2  the trained policies, replayed at 100/250/500/750/1000 kg.  The gap between
#            phase 1 and phase 2 is what the controller loses by being out of distribution.
#   phase 3  at one tonne only: raise floor friction, then hitch torque, one at a time,
#            to name which actuator or contact is the binding one.
set -u
RES=$HOME/amr_results/amr_lab
STATE=$RES/state
OUT=$RES/tonne_results.txt
[ -f "$STATE/tonne.done" ] && exit 0

while [ ! -f "$STATE/video.done" ]; do sleep 120; done
busy() { pgrep -f "[s]earch.py" >/dev/null || pgrep -f "[t]rain.py" >/dev/null \
      || pgrep -f "[p]lay.py" >/dev/null || pgrep -f "[c]apture.py" >/dev/null \
      || pgrep -f "[a]uthority.py" >/dev/null; }
while busy; do sleep 60; done

cd "$HOME/Repositories/amr-ros2" || exit 1
PY="env PYTHONUNBUFFERED=1 $HOME/anaconda3/envs/isaaclab/bin/python"
EVAL="--headless --num-envs 256 --episodes 4 --episode-length 20 --cart-wheels caster4"
COOLDOWN=60
say() { echo "$@" | tee -a "$OUT"; }

run_dirs() { ls -d logs/amr_lab/"$1"/*_"$2" 2>/dev/null; }
newest_ckpt() {
  local best="" bestn=-1 d f n
  for d in $(run_dirs "$1" "$2"); do
    for f in "$d"/model_*.pt; do
      [ -s "$f" ] || continue
      n=$(basename "$f"); n=${n#model_}; n=${n%.pt}
      [[ "$n" =~ ^[0-9]+$ ]] || continue
      [ "$n" -gt "$bestn" ] && { bestn=$n; best=$f; }
    done
  done
  echo "$best"
}

# seed 1 predates the _sN naming, so its run directory is s2r_active
runname() { [ "$1" = 1 ] && echo s2r_active || echo "s2r_active_s$1"; }

# one play.py run, recorded once (marker keeps it once across re-entries)
one() {
  local label=$1 name=$2; shift 2
  [ -f "$STATE/tonne_$label.done" ] && { say "# $label: already recorded"; return; }
  local ck; ck=$(newest_ckpt caster4 "$name")
  [ -n "$ck" ] || { say "# $label: NO CHECKPOINT for $name"; return; }
  say ""
  say "---- $label   [$(date '+%F %T')]"
  say "# checkpoint: $ck"
  timeout 2400 $PY amr_lab/play.py $EVAL "$@" --checkpoint "$ck" \
      > "/tmp/tonne_$label.log" 2>&1
  [ $? -eq 124 ] && say "# TIMED OUT: $label"
  if grep -q '^=== ' "/tmp/tonne_$label.log"; then
    sed -n '/^=== /,$p' "/tmp/tonne_$label.log" | tee -a "$OUT"
    touch "$STATE/tonne_$label.done"
  else
    say "# NO TABLE: $label (left unmarked, will be retried)"
    tail -5 "/tmp/tonne_$label.log" | sed 's/^/#   /' | tee -a "$OUT"
  fi
  sleep $COOLDOWN
}

say ""
say "================================================================"
say "one-tonne study, entered $(date '+%F %T')"
say "NOT a paper-5 result: the policies are trained at 0-100 kg and this is"
say "a ten-fold extrapolation. Open-loop first, then the policies, then the"
say "binding-constraint test at one tonne."
say "================================================================"

# ---- phase 1: open loop, the machine alone --------------------------------------
if [ ! -f "$STATE/tonne_authority.done" ]; then
  say ""
  say "=========== phase 1: open-loop authority to one tonne $(date '+%F %T') ==========="
  timeout 7200 $PY amr_lab/authority.py --headless --cart-wheels caster4 \
      --payloads 100 250 500 750 1000 \
      --frictions 0.4 0.7 1.0 \
      --wheel-torques 11 20 30 \
      --hitch-torques 10.6 \
      --repeats 4 > /tmp/tonne_authority.log 2>&1
  if grep -q 'payload  cart' /tmp/tonne_authority.log; then
    sed -n '/^=== /,$p' /tmp/tonne_authority.log | tee -a "$OUT"
    touch "$STATE/tonne_authority.done"
  else
    say "# AUTHORITY PRODUCED NO TABLE (left unmarked, will be retried)"
    tail -8 /tmp/tonne_authority.log | sed 's/^/#   /' | tee -a "$OUT"
  fi
  sleep $COOLDOWN
fi

# ---- phase 2: the trained policies, graded in mass -------------------------------
say ""
say "=========== phase 2: active policy vs payload $(date '+%F %T') ==========="
for m in 100 250 500 750 1000; do
  for s in 1 2 3; do
    one "p${m}_s$s" "$(runname $s)" --seed $s --payload $m $m --floor-friction 0.6 1.0
  done
done

# ---- phase 3: which constraint binds at one tonne --------------------------------
say ""
say "=========== phase 3: binding constraint at one tonne $(date '+%F %T') ==========="
for s in 1 2 3; do
  one "t1000_mu10_s$s" "$(runname $s)" --seed $s --payload 1000 1000 \
      --floor-friction 1.0 1.0
  one "t1000_hitch30_s$s" "$(runname $s)" --seed $s --payload 1000 1000 \
      --floor-friction 0.6 1.0 --hitch-torque 30 30
done

say ""
say "one-tonne study finished $(date '+%F %T')"
touch "$STATE/tonne.done"
