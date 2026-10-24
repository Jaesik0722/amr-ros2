#!/usr/bin/env bash
# The robot is being assembled with the FRONT support caster about 10 mm up and the REAR
# one flush with the tyres; this model lifts both by 10 mm. The difference matters in a
# specific way: flush at the rear hands part of the chassis load to a frictionless sphere,
# so the drive wheels carry less than the model gives them, and it removes the rear half
# of the fore-and-aft rocking the clearance allows.
#
# Two clips of ONE scenario -- the free-space seed 17 episode the highlight video uses --
# recorded from a low side-on camera where a pitch of a degree or two is actually visible,
# plus the measurement, because a degree of body pitch is easier to measure than to see.
set -u
PY="env PYTHONUNBUFFERED=1 $HOME/anaconda3/envs/isaaclab/bin/python"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1   # the repository root
OUT=$HOME/amr_results/amr_lab/caster
LOG=$OUT/record.log
SC=$HOME/amr_results/amr_lab/highlight/free/scenario.pt
mkdir -p "$OUT"
say() { echo "$*" >&2; echo "$*" >> "$LOG"; }

A=$(ls -1 logs/amr_lab/caster4/*_fr_active_s3/model_*.pt | tail -1)
# Side-on, low, close: the body's fore-and-aft tilt reads against the horizon, which it
# does not from the chase camera the highlight video uses.
# The same chase camera the highlight video uses, so this check can be compared against
# the clip already watched rather than against a differently framed one.
CAM="--resolution 960 540 --follow --eye 1.5 -1.7 1.0 --lookat -0.55 0.0 0.28"
REC="--headless --enable_cameras --cart-wheels caster4 --episode-length 20 \
     --floor-friction 0.6 1.0 --num-envs 8 --env-spacing 60 \
     --visual-meshes --backdrop --every 1 $CAM"

say ""; say "======== rear-caster comparison  [$(date '+%F %T')] ========"
for spec in "both10:" "rear0:--rear-caster-clearance 0"; do
  name=${spec%%:*}; flags=${spec#*:}
  d="$OUT/$name"; rm -rf "$d"; mkdir -p "$d"
  say ""; say "---- $name  [$(date '+%F %T')]  ${flags:-(model as published)}"
  $PY amr_lab/capture.py $REC --seed 17 --checkpoint "$A" \
      --load-scenario "$SC" --out "$d" $flags > "/tmp/caster_$name.log" 2>&1
  say "   $(ls "$d"/*.png 2>/dev/null | wc -l) frames; $(grep -oE 'episode ended.*' "/tmp/caster_$name.log")"
done
say ""; say "CASTER RECORD DONE $(date '+%F %T')"
