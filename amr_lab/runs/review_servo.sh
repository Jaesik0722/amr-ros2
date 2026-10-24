#!/bin/bash
# Review item: is the hitch servo model too generous?
#
# The task is run with a flat effort limit of 10.6 N.m, which is the XM540-W270's STALL
# torque: the value at zero speed and maximum current. It is neither a continuous rating
# nor available once the joint turns. Measured peak demand is 10.4 N.m, which sits on
# that limit, so the question is whether the policy is asking for stall torque while the
# hitch is moving -- a combination the real servo cannot deliver.
#
# Two steps. First measure where the peak demand falls on the torque-speed curve, which
# needs no change to the dynamics. Then replay the same checkpoints with the servo
# modelled as a DC motor, tau_max(w) = tau_stall * (1 - |w| / w_noload), and see what the
# flat limit was buying. If that costs little the flat model was harmless; if it costs a
# lot the honest configuration is the DC one and the policies have to be retrained in it.
#
# Waits for the search to finish first: two Isaac processes on one GPU is how this
# machine gets into trouble, and the search is the longer job.
RES=$HOME/amr_results/amr_lab
while pgrep -f "[s]earch.py" >/dev/null; do sleep 60; done
cd "$HOME/Repositories/amr-ros2" || exit 1
OUT=$RES/eval_seeds.txt
STATE=$RES/state
PY="env PYTHONUNBUFFERED=1 $HOME/anaconda3/envs/isaaclab/bin/python"
EVAL="--headless --num-envs 256 --episodes 4 --episode-length 20"

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
ev() {
  local tag=$1 name=$2 label=$3; shift 3
  [ -f "$STATE/$name.$label" ] && { echo "# done: $name $label"; return; }
  local ck; ck=$(newest_ckpt "$tag" "$name")
  [ -n "$ck" ] || { say "# $name $label: no checkpoint"; return; }
  say ""
  say "---- $name :: $label"
  timeout 1800 $PY amr_lab/play.py $EVAL "$@" --checkpoint "$ck" \
      > /tmp/servo_${name}_${label}.log 2>&1
  if grep -q '^=== ' /tmp/servo_${name}_${label}.log; then
    sed -n '/^=== /,$p' /tmp/servo_${name}_${label}.log | tee -a "$OUT"
    touch "$STATE/$name.$label"
  else
    say "# EVAL PRODUCED NO TABLE: $name $label"
  fi
  sync
}

say ""
say "=========== hitch servo model  $(date '+%F %T') ==========="
say "flat stall-torque limit vs a DC motor torque-speed curve, same checkpoints"
for spec in ":1" "_s2:2" "_s3:3"; do
  sfx=${spec%%:*}; sd=${spec##*:}
  # where the peak demand sits on the curve, under the model as trained
  rm -f "$STATE/s2r_active$sfx.rev"
  ev caster4 "s2r_active$sfx" rev --cart-wheels caster4 --seed "$sd" \
     --floor-friction 0.6 1.0
  # and the same policy with the servo derated by speed
  ev caster4 "s2r_active$sfx" hitchdc --cart-wheels caster4 --seed "$sd" \
     --floor-friction 0.6 1.0 --hitch-dc
done
say ""
say "=========== servo model finished $(date '+%F %T') ==========="
touch "$STATE/servo.done"
