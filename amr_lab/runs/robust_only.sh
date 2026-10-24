#!/bin/bash
# The robustness block of campaign3.sh on its own: six evaluations, no training.
# The three training runs it needs are finished, so this only replays their checkpoints
# under conditions outside the training distribution. Each evaluation is about five
# minutes, so a crash costs one evaluation rather than an hour of training.
cd "$HOME/Repositories/amr-ros2" || exit 1
RES=$HOME/amr_results/amr_lab
OUT=$RES/eval_seeds.txt
STATE=$RES/state
PY="env PYTHONUNBUFFERED=1 $HOME/anaconda3/envs/isaaclab/bin/python"
EVAL="--headless --num-envs 256 --episodes 4 --episode-length 20"
ITERS=3000

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

robust() {
  local tag=$1 name=$2 label=$3; shift 3
  [ -f "$STATE/$name.$label" ] && { echo "# already done: $name $label"; return; }
  local ck; ck=$(newest_ckpt "$tag" "$name")
  [ -n "$ck" ] || { say "# robust($name,$label): no checkpoint"; return; }
  say ""
  say "---- $name :: $label"
  timeout 1800 $PY amr_lab/play.py $EVAL "$@" --checkpoint "$ck" \
      > /tmp/rob_${name}_${label}.log 2>&1
  [ $? -eq 124 ] && say "# ROBUST EVAL TIMED OUT: $name $label"
  if grep -q '^=== ' /tmp/rob_${name}_${label}.log; then
    sed -n '/^=== /,$p' /tmp/rob_${name}_${label}.log | tee -a "$OUT"
    touch "$STATE/$name.$label"
  else
    say "# ROBUST EVAL PRODUCED NO TABLE: $name $label (left unmarked, will be retried)"
  fi
  sync
}

say ""
say "========== robustness, seeds 2 and 3, started $(date '+%F %T') =========="
for s in 2 3; do
  robust caster4 s2r_active_s$s perfect  --cart-wheels caster4 --seed $s \
         --floor-friction 0.6 1.0 --perfect-sensors
  robust caster4 s2r_active_s$s lowmu    --cart-wheels caster4 --seed $s \
         --floor-friction 0.4 0.6
  robust caster4 s2r_active_s$s delay200 --cart-wheels caster4 --seed $s \
         --floor-friction 0.6 1.0 --latency 4 4
done
say "========== robustness finished $(date '+%F %T') =========="
