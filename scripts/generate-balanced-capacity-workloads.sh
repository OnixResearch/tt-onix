#!/usr/bin/env bash
# Generate a token-balanced capacity workload for the two-card scheduler.
#
# The round-robin scheduler assigns requests to cards by request index. With
# an alternating short/long prompt pattern, one card receives every long
# prompt and the other receives every short prompt, so the two-card run is
# 7:1 token-imbalanced. This generator interleaves prompts so that BOTH
# parity classes (even and odd request indices) contain an equal number of
# short and long prompts. Each card then does the same total token work.
#
# Streams stay pinned to the FP32 CPU reference (same prompt -> same greedy
# output), so parity is preserved.
#
# usage: generate-balanced-capacity-workloads.sh CAPACITY OUTPUT.json
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: generate-balanced-capacity-workloads.sh CAPACITY OUTPUT.json" >&2
  exit 2
fi

capacity=$1
output=$2
case "$capacity" in
  4|16|32|64) ;;
  *) echo "capacity must be one of 4 16 32 64" >&2; exit 2 ;;
esac
request_count=$((capacity * 2))

jq -n \
  --argjson count "$request_count" '
  {
    schema_version: 1,
    cpu_reference_profile_blake3: "d33ed1051c7595327368cce7b42e16f025e8fc78d96addb0353fba2e86314a88",
    cpu_reference_method: "pinned_fp32_full_sequence_greedy_argmax",
    requests: [
      range(0; $count) |
      # Interleave short/long so both even and odd indices get an equal mix.
      (if (. % 4) < 2 then
         {request_id: ("request-" + tostring),
          prompt_token_ids: [0, 2],
          max_new_tokens: 4,
          expected_generated_token_ids: [2, 2, 2, 2]}
       else
         {request_id: ("request-" + tostring),
          prompt_token_ids: [0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2,
                             0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2],
          max_new_tokens: 4,
          expected_generated_token_ids: [39, 6699, 19826, 4706]}
       end)
    ]
  }
' > "$output"

# Verify each parity class balances short and long prompts.
jq -e \
  --argjson count "$request_count" '
  (.requests | length) == $count and
  ( [.requests[] | select((.request_id | split("-") | last) | tonumber % 2 == 0)]
    | length == ($count / 2)) and
  ( [.requests[] | select((.request_id | split("-") | last) | tonumber % 2 == 0 and (.prompt_token_ids | length == 32))]
    | length == ($count / 4)) and
  ( [.requests[] | select((.request_id | split("-") | last) | tonumber % 2 == 1 and (.prompt_token_ids | length == 32))]
    | length == ($count / 4))
' "$output" >/dev/null

echo "$output"