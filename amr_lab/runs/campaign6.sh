#!/bin/bash
# Paper 5, review round 3. Two things the review asked for that need training, and a
# set of evaluations that do not.
#
#   1. RIGID LOCK as a fourth condition. The "locked" condition in the paper is a servo
#      commanded to zero on a 10.6 N.m limit, and the load drags it to a peak of 33 deg.
#      That measures what this servo can hold, not what a welded drawbar does, and the
#      paper must not read as a claim about mechanically fixed hitches. So: the same
#      task with the joint pinned by the solver.
#   2. THE BAY. The free-space goal is a pose on an empty floor, and the precision it
#      demands is a tolerance the author chose. A 0.80 m slot for a 0.56 m deck puts the
#      precision in the geometry, and a cart can only be in it nose-out, so every
#      episode ends in a reverse. Four conditions, three seeds.
#   3. Evaluations on checkpoints that already exist: the headline comparison at the
#      MDH300's rated wheel torque rather than the 11-30 N.m randomisation range, the
#      support-caster clearance the result is built on, and the alternative numerics for
#      the passive and locked conditions (Table VII evaluated the active policy alone,
#      so the paper cannot yet attribute the gap).
#
# Same crash-tolerant protocol as campaign3.sh: every run resumes from its highest
# checkpoint, the script is idempotent and re-entrant, and a stalled run is killed and
# retried. This machine has died mid-run twice (split lock), so that is not optional.
set -u
PY="env PYTHONUNBUFFERED=1 $HOME/anaconda3/envs/isaaclab/bin/python"
REPO=$HOME/Repositories/amr-ros2
RES=$HOME/amr_results/amr_lab
OUT=$RES/review3.txt
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
  [ -f "$STATE/$name.r3$label" ] && { say "# done: $name $label"; return; }
  local ck; ck=$(newest_ckpt "$tag" "$name")
  [ -n "$ck" ] || { say "# $name $label: no checkpoint"; return; }
  say ""
  say "---- $name :: $label"
  say "# checkpoint: $ck"
  timeout 1800 $PY amr_lab/play.py $EVAL "$@" --checkpoint "$ck" \
      > /tmp/r3_${name}_${label}.log 2>&1
  if grep -q '^=== ' /tmp/r3_${name}_${label}.log; then
    sed -n '/^=== /,$p' /tmp/r3_${name}_${label}.log | tee -a "$OUT"
    touch "$STATE/$name.r3$label"
  else
    say "# EVAL PRODUCED NO TABLE: $name $label"
    tail -5 /tmp/r3_${name}_${label}.log | tee -a "$OUT"
  fi
  sync
}

say ""
say "================ review round 3  [$(date '+%F %T')] ================"

# ---------------------------------------------------------------- 1. rigid lock
# Free-space task, so it drops straight into the headline table beside the other three.
for s in 1 2 3; do
  run caster4_passive fs_rigid_s$s --cart-wheels caster4 --rigid-lock --seed $s
done

# ---------------------------------------------------------------- 2. the bay, seed 1
# Seed 1 of all four conditions first: one complete column of the new table exists
# before any second seed is spent.
run caster4         bay_active_s1 --cart-wheels caster4 --bay --seed 1
run caster4_passive bay_passive_s1 --cart-wheels caster4 --bay --passive-hitch --seed 1
run caster4_passive bay_servo_s1 --cart-wheels caster4 --bay --passive-hitch --lock-hitch --seed 1
run caster4_passive bay_rigid_s1 --cart-wheels caster4 --bay --rigid-lock --seed 1

# ---------------------------------------------------------------- 3. evaluations
# (a) The MDH300 is rated 11 N.m. Table I randomises the wheel limit 11-30 N.m, which a
#     reader is right to question: the headline numbers must survive at the rating.
for spec in "caster4:s2r_active" "caster4_passive:s2r_locked_s1" \
            "caster4_passive:s2r_passive"; do
  ev "${spec%%:*}" "${spec##*:}" wheel11 --cart-wheels caster4 --wheel-torque 11 11 \
     $( [ "${spec##*:}" = s2r_active ] || echo --passive-hitch ) \
     $( case "${spec##*:}" in *locked*) echo --lock-hitch;; esac )
done

# (b) The support-caster clearance is a modelling choice (10 mm). How much of the result
#     rests on it is a question, not a disclosure.
for spec in "caster4:s2r_active" "caster4_passive:s2r_locked_s1" \
            "caster4_passive:s2r_passive"; do
  ev "${spec%%:*}" "${spec##*:}" caster3mm --cart-wheels caster4 --caster-clearance 0.003 \
     $( [ "${spec##*:}" = s2r_active ] || echo --passive-hitch ) \
     $( case "${spec##*:}" in *locked*) echo --lock-hitch;; esac )
done

# (c) Table VII changed the integrator for the active policy only, so the paper cannot
#     yet say the gap is not an artefact of the numerics. Same two settings, other two
#     conditions.
for spec in "caster4_passive:s2r_locked_s1" "caster4_passive:s2r_passive"; do
  ev "${spec%%:*}" "${spec##*:}" phys400 --cart-wheels caster4 --physics-hz 400 \
     --passive-hitch $( case "${spec##*:}" in *locked*) echo --lock-hitch;; esac )
  ev "${spec%%:*}" "${spec##*:}" solver16 --cart-wheels caster4 --solver-iters 16 4 \
     --passive-hitch $( case "${spec##*:}" in *locked*) echo --lock-hitch;; esac )
done

# ---------------------------------------------------------------- 4. the bay, seeds 2-3
for s in 2 3; do
  run caster4         bay_active_s$s --cart-wheels caster4 --bay --seed $s
  run caster4_passive bay_passive_s$s --cart-wheels caster4 --bay --passive-hitch --seed $s
  run caster4_passive bay_servo_s$s --cart-wheels caster4 --bay --passive-hitch --lock-hitch --seed $s
  run caster4_passive bay_rigid_s$s --cart-wheels caster4 --bay --rigid-lock --seed $s
done

say ""
say "================ review round 3 finished  [$(date '+%F %T')] ================"
