#!/usr/bin/env bash
# The blocks the campaign added after the protocol was corrected, re-read on fixed
# scenarios with one row per scenario. The campaign's own numbers for these come from
# `play.py`, which collects episodes in the order they finish and therefore over-reports
# by an amount that depends on the condition (measured: +15.0 points for the locked bay
# policy against +8.8 for the active one). Nothing from `play.py` goes in the paper.
#
# Checkpoints are given as explicit paths. The 6000-iteration block created a second
# directory per run name, so a glob that picks the highest iteration would silently hand
# the robustness block the 6000-iteration policy instead of the 3000-iteration one the
# headline is built on.
set -u
PY="env PYTHONUNBUFFERED=1 $HOME/anaconda3/envs/isaaclab/bin/python"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1   # the repository root
OUT=$HOME/amr_results/amr_lab/pairs
LOG=$OUT/pairs7.log
COMMON="--headless --num-envs 256 --episodes 4 --episode-length 20 \
  --floor-friction 0.6 1.0 --scenario-seed 7 --curriculum 0.024"
say() { echo "$*" | tee -a "$LOG"; }

one() {  # checkpoint label extra...
  local c=$1 label=$2; shift 2
  if [ ! -f "$c" ]; then say "# missing checkpoint: $c"; return; fi
  if [ -s "$OUT/$label.csv" ]; then say "# $label already done"; return; fi
  say ""; say "---- $label  [$(date '+%F %T')]"
  say "# $c"
  timeout 2400 $PY amr_lab/eval_pairs.py $COMMON \
      --checkpoint "$c" --label "$label" --out "$OUT/$label.csv" "$@" \
      > /tmp/pairs7_$label.log 2>&1
  if [ -s "$OUT/$label.csv" ]; then
    sed -n '/^=== /,$p' /tmp/pairs7_$label.log | tee -a "$LOG"
  else
    say "# FAILED: $label"; tail -5 /tmp/pairs7_$label.log | tee -a "$LOG"
  fi
  sleep 20
}

A3=logs/amr_lab/caster4/2026-10-20_21-17-35_fr_active_s1/model_2999.pt
P3=logs/amr_lab/caster4_passive/2026-10-20_22-43-35_fr_passive_s1/model_2999.pt
S3=logs/amr_lab/caster4_passive/2026-10-21_00-08-41_fr_servo_s1/model_2999.pt
R3=logs/amr_lab/caster4_passive/2026-10-21_01-33-51_fr_rigid_s1/model_2999.pt
A6=logs/amr_lab/caster4/2026-10-23_02-38-16_fr_active_s1/model_5999.pt
P6=logs/amr_lab/caster4_passive/2026-10-23_04-04-13_fr_passive_s1/model_5999.pt
S6=logs/amr_lab/caster4_passive/2026-10-23_05-29-18_fr_servo_s1/model_5999.pt
R6=logs/amr_lab/caster4_passive/2026-10-23_06-54-32_fr_rigid_s1/model_5999.pt
T20=logs/amr_lab/caster4/2026-10-22_23-42-29_fr_active_t20_s1/model_2999.pt
T30=logs/amr_lab/caster4/2026-10-23_01-13-20_fr_active_t30_s1/model_2999.pt
TRA=logs/amr_lab/trailer/2026-10-22_20-23-30_fr_active_tr_s1/model_2999.pt
TRP=logs/amr_lab/trailer_passive/2026-10-22_21-47-14_fr_passive_tr_s1/model_2999.pt

say "======== post-campaign re-evaluation  [$(date '+%F %T')] ========"

say ""; say "######## is 3000 iterations enough: the same four conditions at 6000"
one "$A6" k6_free_active  --cart-wheels caster4
one "$P6" k6_free_passive --cart-wheels caster4 --passive-hitch
one "$S6" k6_free_servo   --cart-wheels caster4 --passive-hitch --lock-hitch
one "$R6" k6_free_rigid   --cart-wheels caster4 --rigid-lock

say ""; say "######## how much hitch torque the task wants"
one "$T20" t20_free_active --cart-wheels caster4 --hitch-torque 20 20
one "$T30" t30_free_active --cart-wheels caster4 --hitch-torque 30 30

say ""; say "######## the trailer cart (different support model, reference only)"
one "$TRA" tr_free_active  --cart-wheels trailer
one "$TRP" tr_free_passive --cart-wheels trailer --passive-hitch

say ""; say "######## robustness: the 3000-iteration policies outside their training box"
for spec in "$A3::active" "$P3:--passive-hitch:passive" \
            "$S3:--passive-hitch --lock-hitch:servo" "$R3:--rigid-lock:rigid"; do
  ckpt=${spec%%:*}; rest=${spec#*:}; flags=${rest%%:*}; nm=${rest##*:}
  one "$ckpt" "rb_wheel11_$nm"   --cart-wheels caster4 --wheel-torque 11 11     $flags
  one "$ckpt" "rb_caster3mm_$nm" --cart-wheels caster4 --caster-clearance 0.003 $flags
  one "$ckpt" "rb_phys400_$nm"   --cart-wheels caster4 --physics-hz 400         $flags
  one "$ckpt" "rb_solver16_$nm"  --cart-wheels caster4 --solver-iters 16 4      $flags
done

say ""; say "PAIRS7 DONE $(date '+%F %T')"
