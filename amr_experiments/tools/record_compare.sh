#!/usr/bin/env bash
# Record one experiment twice, once with each cart construction, for a side-by-side clip.
#
#   ./amr_experiments/tools/record_compare.sh e5 --no-baseline --repeats 1
#   ./amr_experiments/tools/record_compare.sh e4 --width 1.00 --repeats 1 --modes on
#
# Both runs get the same options, the same cameras and the same capture length, which is
# the only way the two halves of the finished clip mean anything next to each other. The
# captures land in $AMR_VIDEOS/compare/<trailer|caster4>/<exp>/; compose_compare.py joins
# them, with the per-side time offsets passed there rather than guessed here, because the
# two runs reach the interesting part at slightly different times.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="${1:?usage: record_compare.sh e1|e2|e3|e4|e5|e6 [runner options]}"
shift
ROOT="${AMR_VIDEOS:-$HOME/amr_results/videos}/compare"
SECS="${HL_SECONDS:-200}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-100}"

for WHEELS in trailer caster4; do
  echo "[compare] $EXP with the $WHEELS cart"
  AMR_CART_WHEELS="$WHEELS" AMR_VIDEOS="$ROOT/$WHEELS" HL_SECONDS="$SECS" \
    "$REPO/amr_experiments/tools/record_highlight.sh" "$EXP" "$@" \
    || { echo "[compare] the $WHEELS capture failed" >&2; exit 1; }
done
echo "[compare] captures in $ROOT/{trailer,caster4}/$EXP"
