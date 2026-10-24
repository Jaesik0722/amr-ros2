#!/bin/bash
# Log GPU and CPU thermals every 10 s, line-buffered and fsynced, so that a crash that
# leaves nothing in the journal still leaves evidence. Two crashes (2026-10-17 23:14
# split-lock oops, 2026-10-18 08:20 silent hard death) both landed 50-60 min into
# sustained Isaac Lab training with the CPU at 5%, and neither left a usable trace.
OUT=${1:-$HOME/amr_results/amr_lab/telemetry.csv}
mkdir -p "$(dirname "$OUT")"
[ -s "$OUT" ] || echo "iso,gpu_temp_C,gpu_power_W,gpu_util_pct,gpu_mem_MiB,sm_clock_MHz,throttle_hex,pkg_temp_C,acpitz_C,load1" >> "$OUT"
while :; do
  g=$(nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used,clocks.sm,clocks_throttle_reasons.active \
        --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  pkg=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -rn | head -1)
  acpi=$(for z in /sys/class/thermal/thermal_zone*; do
           [ "$(cat $z/type 2>/dev/null)" = "x86_pkg_temp" ] && cat $z/temp 2>/dev/null; done | head -1)
  l=$(awk '{print $1}' /proc/loadavg)
  printf '%s,%s,%s,%s,%s\n' "$(date -Is)" "${g:-,,,,,}" \
    "$(awk -v v="${pkg:-0}" 'BEGIN{printf "%.1f", v/1000}')" \
    "$(awk -v v="${acpi:-0}" 'BEGIN{printf "%.1f", v/1000}')" "$l" >> "$OUT"
  sync -d "$OUT" 2>/dev/null || sync
  sleep 10
done
