#!/bin/bash
# Paper 5, review round 4, re-queued after the bay task was corrected.
#
# What changed since campaign7. The bay measured wall clearance and then did nothing
# with it: `min_clearance` reached neither the reward nor the termination, so success was
# still the pose tolerance and the walls set nothing, although the task was introduced on
# the grounds that "the walls set the precision, not a tolerance chosen by the author".
# The policy did what that objective asked. Of 693 reached poses at least 255 (36.8%,
# a lower bound from 586 contacts against 331 failures) were reached with the deck in
# contact with a wall; mean clearance over the episodes that entered the slot was -11 mm.
#
# Corrected: contact ends the episode and counts as a failure, a shaping term gives a
# gradient over the last 60 mm rather than a cliff, and the clear span goes from 0.80 m
# to 1.00 m -- 220 mm a side against the 0.56 m deck. 0.80 m is narrower than the spread
# of the manoeuvre (s.d. 144 mm), so with contact terminating it would have failed every
# condition alike and measured nothing. The 0.80 m runs are in logs/amr_lab/stale_bay080
# and are not used.
#
# The free-space seed-1 block from campaign7 stands: it is unaffected and its four
# evaluations are recorded. Order here is by decisiveness. The corrected bay at seed 1
# comes first, because if contact makes the task unlearnable that has to surface in five
# hours rather than in thirty.
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
    say "######## $name -> $ITERS   attempt $a   [$(date '+%F %T')]"
    say "# resume from iteration $it, $rem iterations to go"
    say "# train: $flags $res"
    $PY amr_lab/train.py $COMMON --max-iterations $rem $flags $res --run-name "$name" \
        > /tmp/train9_${name}_a$a.log 2>&1 &
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
    grep -E 'training time' /tmp/train9_${name}_a$a.log 2>/dev/null | tee -a "$OUT"
    ck=$(newest_ckpt "$tag" "$name"); it=$(ckpt_iter "$ck"); [ "$it" -lt 0 ] && it=0
    say "# attempt $a ended at iteration $it"
    [ "$it" -ge $((ITERS - 1)) ] && return 0
    say "# cooling down ${COOLDOWN}s before the next attempt"
    sleep $COOLDOWN
  done
  say "# TRAIN INCOMPLETE after $MAX_ATTEMPTS attempts: $name"
  return 1
}

# train to $ITERS, then evaluate once and mark it with $MARK
run() {
  local tag=$1 name=$2; shift 2
  train_until_done "$tag" "$name" "$@" || return
  [ -f "$STATE/$name.$MARK" ] && { say "# $name: $MARK already recorded"; return; }
  local ck; ck=$(newest_ckpt "$tag" "$name")
  say "# checkpoint: $ck"
  timeout 1800 $PY amr_lab/play.py $EVAL "$@" --floor-friction 0.6 1.0 \
      --checkpoint "$ck" > /tmp/play9_$name.log 2>&1
  [ $? -eq 124 ] && say "# EVAL TIMED OUT: $name"
  if grep -q '^=== ' /tmp/play9_$name.log; then
    sed -n '/^=== /,$p' /tmp/play9_$name.log | tee -a "$OUT"
    touch "$STATE/$name.$MARK"
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
      > /tmp/r4_${name}_${label}.log 2>&1
  if grep -q '^=== ' /tmp/r4_${name}_${label}.log; then
    sed -n '/^=== /,$p' /tmp/r4_${name}_${label}.log | tee -a "$OUT"
    touch "$STATE/$name.r4$label"
  else
    say "# EVAL PRODUCED NO TABLE: $name $label"
    tail -5 /tmp/r4_${name}_${label}.log | tee -a "$OUT"
  fi
  sync
}

say ""
say "======== round 4 re-queued: corrected bay, 1.00 m span, contact terminates  [$(date '+%F %T')] ========"

MARK=eval
ITERS=3000

# ------------------------------------------------- the corrected bay, seed 1 (decisive)
run caster4         by_active_s1 --cart-wheels caster4 --bay --seed 1
run caster4_passive by_passive_s1 --cart-wheels caster4 --bay --passive-hitch --seed 1
run caster4_passive by_servo_s1 --cart-wheels caster4 --bay --passive-hitch --lock-hitch --seed 1
run caster4_passive by_rigid_s1 --cart-wheels caster4 --bay --rigid-lock --seed 1

# ------------------------------------------------- free space, seeds 2 and 3 (error bars)
for s in 2 3; do
  run caster4         fr_active_s$s --cart-wheels caster4 --seed $s
  run caster4_passive fr_passive_s$s --cart-wheels caster4 --passive-hitch --seed $s
  run caster4_passive fr_servo_s$s --cart-wheels caster4 --passive-hitch --lock-hitch --seed $s
  run caster4_passive fr_rigid_s$s --cart-wheels caster4 --rigid-lock --seed $s
done

# ------------------------------------------------- the bay, seeds 2 and 3
for s in 2 3; do
  run caster4         by_active_s$s --cart-wheels caster4 --bay --seed $s
  run caster4_passive by_passive_s$s --cart-wheels caster4 --bay --passive-hitch --seed $s
  run caster4_passive by_servo_s$s --cart-wheels caster4 --bay --passive-hitch --lock-hitch --seed $s
  run caster4_passive by_rigid_s$s --cart-wheels caster4 --bay --rigid-lock --seed $s
done

# ------------------------------------------------- the trailer pair
run trailer         fr_active_tr_s1 --cart-wheels trailer --seed 1
run trailer_passive fr_passive_tr_s1 --cart-wheels trailer --passive-hitch --seed 1

# ------------------------------------------------- robustness evaluations
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

# ------------------------------------------------- how much torque does the task need?
# The corrected friction makes the fitted hitch the binding constraint: the active policy
# uses 6.3 +- 2.6 N.m, peaks at 10.1 +- 1.6 against a 10.6 N.m stall, and spends 39.8% of
# each episode above 90% of it -- fully saturated (10.6 +- 0.3) on the goals behind the
# cart. "The servo this robot already carries is enough" was a conclusion of the
# frictionless runs and is now false, so the useful question is what a servo would have
# to supply. Two further training runs answer it as a sizing requirement rather than as
# an observation that the present one saturates.
for t in 20 30; do
  run caster4 fr_active_t${t}_s1 --cart-wheels caster4 --hitch-torque $t $t --seed 1
done

# ------------------------------------------------- is 3000 iterations enough?
# The pre-fix active policy was flat from iteration 2250 (68.8, 70.3, 70.3 on the
# training-time success metric); the corrected one is still climbing at the end (34.3,
# 32.9, 40.2). The fixed budget is the controlled comparison and stays, but what it costs
# has to be measured rather than assumed. Seed 1 of each condition carries on to 6000
# from its own checkpoint, so nothing already recorded is discarded.
MARK=eval6k
ITERS=6000
run caster4         fr_active_s1 --cart-wheels caster4 --seed 1
run caster4_passive fr_passive_s1 --cart-wheels caster4 --passive-hitch --seed 1
run caster4_passive fr_servo_s1 --cart-wheels caster4 --passive-hitch --lock-hitch --seed 1
run caster4_passive fr_rigid_s1 --cart-wheels caster4 --rigid-lock --seed 1

say ""
say "======== round 4 re-queued: finished  [$(date '+%F %T')] ========"
