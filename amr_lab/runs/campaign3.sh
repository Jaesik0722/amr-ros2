#!/bin/bash
# Paper 5 campaign, crash-tolerant.
#
# This machine cannot finish a 3000-iteration run in one piece. It died 61 min in on
# 2026-10-17 23:14 (kernel "Oops: Split lock detected", task wedged alive holding the
# GPU) and 50 min in on 2026-10-18 08:20 (silent hard death: journal ends mid-sample,
# no shutdown sequence, machine stayed powered off). CPU load was 5% both times, so the
# load is GPU-only and the cause is still unproven - telemetry.sh now records thermals
# every 10 s so the next crash leaves evidence.
#
# So: every run RESUMES from its highest checkpoint rather than restarting, the script is
# idempotent and re-entrant (safe to run again after a reboot; finished work is skipped),
# and a systemd user unit starts it on login. The resume policy is applied uniformly to
# every run, including the ones that never crash, so the training protocol stays the same
# across seeds - that uniformity is what the paper reports.
set -u
PY="env PYTHONUNBUFFERED=1 $HOME/anaconda3/envs/isaaclab/bin/python"
REPO=$HOME/Repositories/amr-ros2
RES=$HOME/amr_results/amr_lab
OUT=$RES/eval_seeds.txt
STATE=$RES/state
SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ITERS=3000
COMMON="--headless --num-envs 2048 --episode-length 20 --floor-friction 0.6 1.0"
EVAL="--headless --num-envs 256 --episodes 4 --episode-length 20"
STALL_MIN=10      # minutes with no new checkpoint while the process lives: wedged
STARTUP_MIN=15    # grace for the Isaac app launch and asset build
MAX_ATTEMPTS=12   # a crash costs one attempt; 12 covers a run dying repeatedly
COOLDOWN=120      # seconds idle between runs, to let the laptop shed heat
mkdir -p "$STATE"
cd "$REPO" || exit 1

say() { echo "$@" | tee -a "$OUT"; }

# keep the thermal log running whatever starts us
pgrep -f "[t]elemetry.sh" >/dev/null || \
  setsid nohup bash "$SELF_DIR/telemetry.sh" >/dev/null 2>&1 < /dev/null &

run_dirs() { ls -d logs/amr_lab/"$1"/*_"$2" 2>/dev/null; }

# highest-iteration checkpoint for a run, across every (resumed) directory it has
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

# mtime of the newest checkpoint, for the stall watchdog
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

# train to completion, then evaluate once (marker file keeps it once across re-entries)
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

robust() {
  local tag=$1 name=$2 label=$3; shift 3
  [ -f "$STATE/$name.$label" ] && return
  local ck; ck=$(newest_ckpt "$tag" "$name")
  local it; it=$(ckpt_iter "$ck")
  [ "$it" -ge $((ITERS - 1)) ] || { say "# robust($name,$label): run not finished"; return; }
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
}

say "================================================================"
say "paper 5 campaign (crash-tolerant), entered $(date '+%F %T')"
say "3000 iterations, 2048 envs, 20 s episodes, floor mu 0.6-1.0,"
say "payload 0-100 kg, 20 Hz control, full sensor model unless stated."
say "Runs resume from their newest checkpoint; the policy is uniform across runs."
say "seed 1 of the caster4 pair is the existing run (eval_caster4_s2r.txt)."
say "================================================================"

for s in 2 3; do
  run caster4         s2r_active_s$s  --cart-wheels caster4 --seed $s
  run caster4_passive s2r_passive_s$s --cart-wheels caster4 --passive-hitch --seed $s
done

run caster4 s2r_truecart_s1  --cart-wheels caster4 --seed 1 --true-cart-obs
run caster4 s2r_legacyobs_s1 --cart-wheels caster4 --seed 1 --legacy-obs

run trailer         s2r_active_tr_s1  --cart-wheels trailer --seed 1
run trailer_passive s2r_passive_tr_s1 --cart-wheels trailer --passive-hitch --seed 1

say ""
say "================ robustness, same checkpoints, OOD conditions ================"
for s in 2 3; do
  robust caster4 s2r_active_s$s perfect --cart-wheels caster4 --seed $s \
         --floor-friction 0.6 1.0 --perfect-sensors
  robust caster4 s2r_active_s$s lowmu   --cart-wheels caster4 --seed $s \
         --floor-friction 0.4 0.6
  robust caster4 s2r_active_s$s delay200 --cart-wheels caster4 --seed $s \
         --floor-friction 0.6 1.0 --latency 4 4
done

say ""
say "campaign finished $(date '+%F %T')"
touch "$STATE/campaign.done"
