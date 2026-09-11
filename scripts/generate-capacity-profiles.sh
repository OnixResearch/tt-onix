#!/usr/bin/env bash
# Generate per-capacity throughput profiles for the physical capacity sweep.
# Each profile is the shipped baseline with the scheduler slot count and the
# workload request count set to the candidate capacity. The workload streams
# come from `generate-capacity-workloads.sh`, so they stay pinned to the FP32
# CPU reference.
#
# A sweep profile compares a candidate capacity against the current admitted
# capacity. It relaxes ONLY the one-card-versus-two-card scaling bound: the
# scaling claim is not the capacity claim, and the capacity A/B gate in
# `rwkv7-p150x2-throughput capacity-evaluate` sets the admission boundary.
# Every fail-closed rail stays: physical hardware, profiler attribution,
# exact token parity, occupancy, batch steps, and the checked memory plan.
#
# usage: generate-capacity-profiles.sh SHIPPED_PROFILE.json CAPACITY OUTPUT.json
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: generate-capacity-profiles.sh SHIPPED_PROFILE.json CAPACITY OUTPUT.json" >&2
  exit 2
fi

base=$1
capacity=$2
output=$3
case "$capacity" in
  4|16|32|64) ;;
  *) echo "capacity must be one of 4 16 32 64" >&2; exit 2 ;;
esac

jq \
  --argjson capacity "$capacity" \
  --argjson request_count "$((capacity * 2))" \
  '.scheduler.per_card_slot_count = $capacity |
   .scheduler.mode = "deterministic_continuous_batching" |
   .workload.request_count = $request_count' \
  "$base" > "$output"

jq -e \
  --argjson capacity "$capacity" \
  --argjson request_count "$((capacity * 2))" '
  .scheduler.per_card_slot_count == $capacity
  and .workload.request_count == $request_count
  and .workload.request_count <= (.scheduler.per_card_slot_count * 2)
' "$output" >/dev/null
echo "$output"