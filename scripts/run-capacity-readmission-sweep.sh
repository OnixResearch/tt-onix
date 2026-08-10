#!/usr/bin/env bash
# Run the order-balanced B4-vs-B16 re-admission evidence sweep.
#
# Order: B4, B16, B16, B4 (both orders appear). Each run is the strict
# throughput benchmark with the token-balanced workload, so the two-card
# scheduler no longer piles all long prompts onto one card. The gate in
# rwkv7-p150x2-throughput capacity-evaluate uses per-token p99, the honest
# cross-capacity latency metric.
#
# usage: run-capacity-readmission-sweep.sh EVIDENCE_ROOT OUTPUT_ROOT
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: run-capacity-readmission-sweep.sh EVIDENCE_ROOT OUTPUT_ROOT" >&2
  exit 2
fi

EVID=/nix/store/fw2xrqjljfpq3d17q5bynxl80dsmz5g0-rwkv7-p150x2-evidence-0.1.0
ROOT=$2
rm -rf "$ROOT"
mkdir -p "$ROOT"

bench="$EVID/bin/rwkv7-p150x2-throughput-benchmark"

run() {
  local name=$1 profile=$2 workload=$3
  local out="$ROOT/$name"
  "$bench" "$profile" "$workload" "$out" >/dev/null 2>&1
  echo "$name: $(jq -r '.two_card.aggregate_tokens_per_second' "$out/throughput-receipt.json") tps, p99pt=$(jq -r '[.two_card_runs[]|.requests[]|(.latency_seconds/.token_count)]|sort|.[((length-1)*0.99)|floor]' "$out/raw-throughput-evidence.json")"
}

run b4-r1 /tmp/balanced-profile-b4.json /tmp/balanced-b4.json
run b16-r1 /tmp/balanced-profile-b16.json /tmp/balanced-b16.json
run b16-r2 /tmp/balanced-profile-b16.json /tmp/balanced-b16.json
run b4-r2 /tmp/balanced-profile-b4.json /tmp/balanced-b4.json

echo "sweep complete in $ROOT"