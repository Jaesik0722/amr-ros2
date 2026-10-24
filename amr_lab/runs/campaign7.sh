#!/bin/bash
# Paper 5, review round 4. EVERYTHING is retrained, because the joint-friction model
# was wrong.
#
# Isaac Lab picks its joint-friction model on the Isaac Sim major version. Below 5.0 the
# figure written is a dimensionless coefficient and PhysX scales it by the force the
# joint transmits; from 5.0 it is a resisting effort in newton metres and nothing scales
# it. This install is 5.1.0. The task was written for the first model, so the carts ran
# with 0.005-0.012 N.m of caster swivel friction and 0.0006-0.002 N.m at the wheels --
# no friction at all. Measured, not read: with mu_s = 0.5 on the hitch yaw joint and the
# servo off, the joint breaks away at 0.5 N.m with an empty cart and at 0.5 N.m with
# 100 kg on it.
#
# Corrected, the open-loop authority sweep changes answer: the hitch used to reach the
# whole commanded 69 deg on 1.1-3.3 N.m, and now saturates its 10.6 N.m and reaches
# 23-51 deg. Every learned result therefore has to be produced again.
#
# Order is by decisiveness: the four free-space conditions at seed 1 first, so the
# headline table exists in five hours; then the bay at seed 1; then seeds 2 and 3; then
# the trailer pair for the two-cart comparison. Same crash-tolerant protocol as before.
set -u
PY="env PYTHONUNBUFFERED=1 $HOME/anaconda3/envs/isaaclab/bin/python"
REPO=$HOME/Repositories/amr-ros2
RES=$HOME/amr_results/amr_lab
OUT=$RES/review4.txt
STATE=$RES/state
ITERS=3000
COMMON="--headless --num-envs 2048 --episode-length 20 --floor-friction 0.6 1.0"
EVAL="--headless --num-envs 256 --episodes 4 --episode-length 20"
STALL_MIN=10
STARTUP_MIN=15
MAX_ATTEMPTS=12
COOLDOWN=120
mkdir -p "$STATE"
cd "$REPO" || exit 1

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
ckpt_iter() { local n; n=$(basename "${1:-}" 2>/dev/null); n=${n#model_}; n=${n%.pt}
              [[ "$n" =~ ^[0-9]+$ ]] && echo "$n" || echo -1; }
ckpt_mtime() { local c; c=$(newest_ckpt "$1" "$2"); [ -n "$c" ] && stat -c %Y "$c" || echo 0; }

train_until_done() {
  local tag=$1 name=$2; shift 2
  local flags="$*" a ck it rem res pid quiet limit last now
  for a in $(seq 1 $MAX_ATTEMPTS); do
    ck=$(newest_ckpt "$tag" "$name"); it=$(ckpt_iter "$ck"); [ "$it" -lt 0 ] && it=0
    if [ "$it" -ge $((ITERS - 1)) ]; then
      say "# $name: already at iteration $it, training complete"; return 0
    fi
    rem=$((ITERS - it))
    res=""; [ -n "$ck" ] && res="--resume $ck"
    say ""
    say "######## $name   attempt $a   [$(date '+%F %T')]"
    say "# resume from iteration $it, $rem iterations to go"
    say "# train: $flags $res"
    $PY amr_lab/train.py $COMMON --max-iterations $rem $flags $res --run-name "$name" \
        > /tmp/train_${name}_a$a.log 2>&1 &
    pid=$!
    quiet=0; limit=$STARTUP_MIN; last=$(ckpt_mtime "$tag" "$name")
    while kill -0 $pid 2>/dev/null; do
      sleep 60
      now=$(ckpt_mtime "$tag" "$name")
      if [ "$now" != "$last" ]; then last=$now; quiet=0; limit=$STALL_MIN
      else quiet=$((quiet + 1)); fi
      if [ $quiet -ge $limit ]; then
        say "# STALLED: no checkpoint for $quiet min, killing pid $pid"
        kill -9 $pid 2>/dev/null; sleep 30; break
      fi
    done
    wait $pid 2>/dev/null
    grep -E 'training time' /tmp/train_${name}_a$a.log 2>/dev/null | tee -a "$OUT"
    ck=$(newest_ckpt "$tag" "$name"); it=$(ckpt_iter "$ck"); [ "$it" -lt 0 ] && it=0
    say "# attempt $a ended at iteration $it"
    [ "$it" -ge $((ITERS - 1)) ] && return 0
    say "# cooling down ${COOLDOWN}s before the next attempt"
    sleep $COOLDOWN
  done
  say "# TRAIN INCOMPLETE after $MAX_ATTEMPTS attempts: $name"
  return 1
}

run() {
  local tag=$1 name=$2; shift 2
  train_until_done "$tag" "$name" "$@" || return
  [ -f "$STATE/$name.eval" ] && { say "# $name: evaluation already recorded"; return; }
  local ck; ck=$(newest_ckpt "$tag" "$name")
  say "# checkpoint: $ck"
  timeout 1800 $PY amr_lab/play.py $EVAL "$@" --floor-friction 0.6 1.0 \
      --checkpoint "$ck" > /tmp/play_$name.log 2>&1
  [ $? -eq 124 ] && say "# EVAL TIMED OUT: $name"
  if grep -q '^=== ' /tmp/play_$name.log; then
    sed -n '/^=== /,$p' /tmp/play_$name.log | tee -a "$OUT"
    touch "$STATE/$name.eval"
  else
    say "# EVAL PRODUCED NO TABLE: $name (left unmarked, will be retried)"
  fi
  sleep $COOLDOWN
}

# evaluate an existing checkpoint under a changed condition
ev() {   # tag run-name label extra-args...
  local tag=$1 name=$2 label=$3; shift 3
  [ -f "$STATE/$name.r4$label" ] && { say "# done: $name $label"; return; }
  local ck; ck=$(newest_ckpt "$tag" "$name")
  [ -n "$ck" ] || { say "# $name $label: no checkpoint"; return; }
  say ""
  say "---- $name :: $label"
  say "# checkpoint: $ck"
  timeout 1800 $PY amr_lab/play.py $EVAL "$@" --checkpoint "$ck" \
      > /tmp/r3_${name}_${label}.log 2>&1
  if grep -q '^=== ' /tmp/r3_${name}_${label}.log; then
    sed -n '/^=== /,$p' /tmp/r3_${name}_${label}.log | tee -a "$OUT"
    touch "$STATE/$name.r4$label"
  else
    say "# EVAL PRODUCED NO TABLE: $name $label"
    tail -5 /tmp/r3_${name}_${label}.log | tee -a "$OUT"
  fi
  sync
}

say ""
say "================ review round 4: corrected joint friction  [$(date '+%F %T')] ================"

# ---------------------------------------------------------------- free space, seed 1
run caster4         fr_active_s1 --cart-wheels caster4 --seed 1
run caster4_passive fr_passive_s1 --cart-wheels caster4 --passive-hitch --seed 1
run caster4_passive fr_servo_s1 --cart-wheels caster4 --passive-hitch --lock-hitch --seed 1
run caster4_passive fr_rigid_s1 --cart-wheels caster4 --rigid-lock --seed 1

# ---------------------------------------------------------------- the bay, seed 1
run caster4         by_active_s1 --cart-wheels caster4 --bay --seed 1
run caster4_passive by_passive_s1 --cart-wheels caster4 --bay --passive-hitch --seed 1
run caster4_passive by_servo_s1 --cart-wheels caster4 --bay --passive-hitch --lock-hitch --seed 1
run caster4_passive by_rigid_s1 --cart-wheels caster4 --bay --rigid-lock --seed 1

# ---------------------------------------------------------------- seeds 2 and 3
for s in 2 3; do
  run caster4         fr_active_s$s --cart-wheels caster4 --seed $s
  run caster4_passive fr_passive_s$s --cart-wheels caster4 --passive-hitch --seed $s
  run caster4_passive fr_servo_s$s --cart-wheels caster4 --passive-hitch --lock-hitch --seed $s
  run caster4_passive fr_rigid_s$s --cart-wheels caster4 --rigid-lock --seed $s
  run caster4         by_active_s$s --cart-wheels caster4 --bay --seed $s
  run caster4_passive by_passive_s$s --cart-wheels caster4 --bay --passive-hitch --seed $s
  run caster4_passive by_servo_s$s --cart-wheels caster4 --bay --passive-hitch --lock-hitch --seed $s
  run caster4_passive by_rigid_s$s --cart-wheels caster4 --bay --rigid-lock --seed $s
done

# ---------------------------------------------------------------- the trailer pair
run trailer         fr_active_tr_s1 --cart-wheels trailer --seed 1
run trailer_passive fr_passive_tr_s1 --cart-wheels trailer --passive-hitch --seed 1

# ---------------------------------------------------------------- evaluations
# The MDH300 is rated 11 N.m; Table I randomises 11-30. The headline must survive the
# rating. And the 10 mm support-caster clearance is a modelling choice, so the result's
# dependence on it is measured rather than disclosed.
for spec in "caster4:fr_active_s1:" "caster4_passive:fr_passive_s1:--passive-hitch" \
            "caster4_passive:fr_servo_s1:--passive-hitch --lock-hitch" \
            "caster4_passive:fr_rigid_s1:--rigid-lock"; do
  tag=${spec%%:*}; rest=${spec#*:}; name=${rest%%:*}; flags=${rest#*:}
  ev "$tag" "$name" wheel11 --cart-wheels caster4 --wheel-torque 11 11 $flags
  ev "$tag" "$name" caster3mm --cart-wheels caster4 --caster-clearance 0.003 $flags
  ev "$tag" "$name" phys400 --cart-wheels caster4 --physics-hz 400 $flags
  ev "$tag" "$name" solver16 --cart-wheels caster4 --solver-iters 16 4 $flags
done

say ""
say "================ review round 4 finished  [$(date '+%F %T')] ================"
