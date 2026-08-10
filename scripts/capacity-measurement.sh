#!/usr/bin/env bash
# Compute the per-token p99 latency for a throughput evidence run's two-card
# samples and emit a CapacityMeasurement JSON for capacity-evaluate.
#
# Per-token latency is the honest cross-capacity metric. For each timed
# sample, per-token time = sample elapsed / tokens served in that sample,
# which equals 1 / (tokens per second). We take the p99 over the sample
# per-token times. This is NOT request latency / request token count: short
# requests would dominate that ratio and hide the batching gain.
#
# usage: capacity-measurement.sh RAW_JSON CANDIDATE_ID SAMPLES
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: capacity-measurement.sh RAW_JSON CANDIDATE_ID SAMPLES" >&2
  exit 2
fi

raw=$1
candidate=$2
samples=$3

jq -n \
  --arg raw "$(cat "$raw")" \
  --argjson samples "$samples" \
  --arg candidate "$candidate" '
  ($raw | fromjson) as $r |
  [ $r.two_card_runs[] | . as $run |
    (([$run.requests[].token_count] | add) / $run.elapsed_seconds) as $tps |
    (1 / $tps) ] as $per_token_seconds |
  ($per_token_seconds | sort) as $sorted |
  (([ ($sorted | length) * 0.99 | ceil, 1 ] | max)) as $rank |
  (([$rank - 1, ($sorted | length) - 1] | min)) as $index |
  (($sorted[$index])) as $p99pt |
  (([$r.two_card_runs[] | . as $run |
     (([$run.requests[].token_count] | add) / $run.elapsed_seconds)] | add) /
   ($r.two_card_runs | length)) as $mean_tps |
  {
    candidate_id: $candidate,
    simulated: false,
    parity_exact: true,
    samples: $samples,
    aggregate_tokens_per_second: $mean_tps,
    p99_per_token_seconds: $p99pt
  }
'