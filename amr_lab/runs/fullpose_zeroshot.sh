#!/bin/bash
# Zero-shot evaluation of the trained checkpoints on goals whose heading is drawn from
# the full circle, rather than the +-34 deg window the curriculum actually reached in
# 3000 iterations. No retraining: this asks whether full-pose reaching is within reach
# of the policies we have, which is the difference between a limitation stated from the
# code and one stated from a measurement.
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

say ""
say "========== full-pose zero-shot (heading spread pi) $(date '+%F %T') =========="
for spec in "s2r_active:1" "s2r_active_s2:2" "s2r_active_s3:3"; do
  name=${spec%%:*}; seed=${spec##*:}
  [ -f "$STATE/$name.fullpose" ] && { echo "# done: $name"; continue; }
  ck=$(newest_ckpt caster4 "$name")
  [ -n "$ck" ] || { say "# fullpose($name): no checkpoint"; continue; }
  say ""
  say "---- $name :: fullpose"
  timeout 1800 $PY amr_lab/play.py $EVAL --cart-wheels caster4 --seed "$seed" \
      --floor-friction 0.6 1.0 --heading-spread 3.14159 --checkpoint "$ck" \
      > /tmp/fp_${name}.log 2>&1
  if grep -q '^=== ' /tmp/fp_${name}.log; then
    sed -n '/^=== /,$p' /tmp/fp_${name}.log | tee -a "$OUT"
    touch "$STATE/$name.fullpose"
  else
    say "# FULLPOSE EVAL PRODUCED NO TABLE: $name"
  fi
  sync
done
say "========== full-pose finished $(date '+%F %T') =========="
