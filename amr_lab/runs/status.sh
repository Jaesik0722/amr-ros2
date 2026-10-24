#!/bin/bash
# One-shot campaign status: progress, thermals, ETA. Safe to run any time.
cd "$HOME/Repositories/amr-ros2" || exit 1
RES=$HOME/amr_results/amr_lab
RUNS="s2r_active_s2 s2r_passive_s2 s2r_active_s3 s2r_passive_s3 s2r_truecart_s1 s2r_legacyobs_s1 s2r_active_tr_s1 s2r_passive_tr_s1"
echo "=========== paper 5 campaign  $(date '+%F %H:%M:%S') ==========="
echo "  service : $(systemctl --user is-active paper5-campaign.service)"
echo "  kernel  : $(uname -r)   split_lock_detect=$(grep -o 'split_lock_detect=[a-z]*' /proc/cmdline | cut -d= -f2 || echo 'NOT SET')"
echo "  up      : $(uptime -p)"
echo
done_n=0; left=0
printf '  %-20s %10s  %s\n' run iteration state
for n in $RUNS; do
  b=$(ls -1 logs/amr_lab/*/*_"$n"/model_*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
  b=${b:-0}
  if [ "$b" -ge 2999 ]; then
    s=done; [ -f "$RES/state/$n.eval" ] && s="done + evaluated"; done_n=$((done_n+1))
  else
    s=$(awk -v b="$b" 'BEGIN{printf "%.0f%%", 100*b/3000}'); left=$((left + 3000 - b))
  fi
  printf '  %-20s %10s  %s\n' "$n" "$b/3000" "$s"
done
echo
echo "  finished $done_n of 8 runs, $left iterations left"
awk -v l="$left" -v d="$((8-done_n))" 'BEGIN{printf "  ETA %.1f h training + %.1f h startup/eval = %.1f h\n", l*1.612/3600, d*7/60, l*1.612/3600 + d*7/60}'
echo
echo "  thermals, last 5 min under load:"
awk -F, -v t="$(date -d '5 min ago' -Is 2>/dev/null)" 'NR>1 && $1>t && $4+0>40 {
    n++; g+=$2; p+=$9; u+=$4; if($2+0>gx)gx=$2; if($9+0>px)px=$9 }
  END{ if(n) printf "    GPU %.0f C (max %.0f)   CPU die %.0f C (max %.0f)   util %.0f%%   n=%d\n", g/n,gx,p/n,px,u/n,n
       else print "    (no samples under load in the last 5 min)" }' "$RES/telemetry.csv"
echo
echo "  stop : systemctl --user stop paper5-campaign.service"
echo "  start: systemctl --user start paper5-campaign.service"
