#!/bin/bash
# Review items B, C and D: three questions the review raised that need no training.
#   B  does the result depend on the integrator or the solver, rather than on the
#      machine being simulated?
#   C  what is the PEAK hitch torque, not just the mean? A mean of 5.7 Nm inside a
#      10.6 Nm stall rating says nothing about the peak.
#   D  the delay conclusion rests on two points, 100 and 200 ms. Sweep it.
# All of it replays checkpoints that already exist; nothing is retrained.
cd "$HOME/Repositories/amr-ros2" || exit 1
RES=$HOME/amr_results/amr_lab
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
ev() {   # tag name label extra-args...
  local tag=$1 name=$2 label=$3; shift 3
  [ -f "$STATE/$name.$label" ] && { echo "# done: $name $label"; return; }
  local ck; ck=$(newest_ckpt "$tag" "$name")
  [ -n "$ck" ] || { say "# $name $label: no checkpoint"; return; }
  say ""
  say "---- $name :: $label"
  timeout 1800 $PY amr_lab/play.py $EVAL "$@" --checkpoint "$ck" \
      > /tmp/bcd_${name}_${label}.log 2>&1
  if grep -q '^=== ' /tmp/bcd_${name}_${label}.log; then
    sed -n '/^=== /,$p' /tmp/bcd_${name}_${label}.log | tee -a "$OUT"
    touch "$STATE/$name.$label"
  else
    say "# EVAL PRODUCED NO TABLE: $name $label"
  fi
  sync
}

say ""
say "=========== review items B, C, D  $(date '+%F %T') ==========="

# C: the trained condition again, now reporting peak hitch torque as well as mean.
# Same seed and same conditions, so the success rates must come out unchanged; the
# only new information is the peak column.
say ""
say "---------------- C: peak hitch torque, trained conditions ----------------"
for spec in ":1" "_s2:2" "_s3:3"; do
  sfx=${spec%%:*}; sd=${spec##*:}
  rm -f "$STATE/s2r_active$sfx.rev" "$STATE/s2r_passive$sfx.rev"
  ev caster4         "s2r_active$sfx"  rev --cart-wheels caster4 --seed "$sd" \
     --floor-friction 0.6 1.0
  ev caster4_passive "s2r_passive$sfx" rev --cart-wheels caster4 --passive-hitch \
     --seed "$sd" --floor-friction 0.6 1.0
done

# B: the same checkpoints under different numerics. The control rate stays 20 Hz.
say ""
say "---------------- B: integrator and solver independence ----------------"
for spec in ":1" "_s2:2" "_s3:3"; do
  sfx=${spec%%:*}; sd=${spec##*:}
  ev caster4 "s2r_active$sfx" dt400 --cart-wheels caster4 --seed "$sd" \
     --floor-friction 0.6 1.0 --physics-hz 400
  ev caster4 "s2r_active$sfx" solver16 --cart-wheels caster4 --seed "$sd" \
     --floor-friction 0.6 1.0 --solver-iters 16 4
done

# D: the delay sweep. 4 steps (200 ms) is already measured.
say ""
say "---------------- D: command delay sweep ----------------"
for spec in ":1" "_s2:2" "_s3:3"; do
  sfx=${spec%%:*}; sd=${spec##*:}
  for L in 0 2 6 8; do
    ev caster4 "s2r_active$sfx" "delay$L" --cart-wheels caster4 --seed "$sd" \
       --floor-friction 0.6 1.0 --latency "$L" "$L"
  done
done

say ""
say "=========== B, C, D finished $(date '+%F %T') ==========="
touch "$STATE/bcd.done"
