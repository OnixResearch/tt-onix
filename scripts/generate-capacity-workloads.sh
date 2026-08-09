#!/usr/bin/env bash
# Generate the exact card-local capacity sweep workloads.
#
# The greedy token streams are pinned by the FP32 CPU reference used by the
# admitted fixtures. Same prompt -> same greedy stream, so each candidate
# workload reuses the exact short and long streams from
# `fixtures/throughput-saturated-workload.json`. The generator writes new
# files only; it never edits a reference stream.
#
# usage: generate-capacity-workloads.sh CAPACITY OUTPUT.json
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: generate-capacity-workloads.sh CAPACITY OUTPUT.json" >&2
  exit 2
fi

capacity=$1
output=$2
case "$capacity" in
  4|16|32|64) ;;
  *) echo "capacity must be one of 4 16 32 64" >&2; exit 2 ;;
esac

repo_root=$(cd "$(dirname "$0")/.." && pwd)
saturated="$repo_root/pkgs/rwkv7-p150x2-evidence/fixtures/throughput-saturated-workload.json"
jq -c '
  .schema_version as $schema |
  .cpu_reference_profile_blake3 as $blake3 |
  .cpu_reference_method as $method |
  (
    [range(0; '$capacity')
     | if . % 2 == 0 then
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
       end]
  ) as $requests |
  {schema_version: $schema,
   cpu_reference_profile_blake3: $blake3,
   cpu_reference_method: $method,
   requests: $requests}
' "$saturated" > "$output"

jq -e \
  --arg capacity "$capacity" '
  ($capacity | tonumber) as $capacity_num |
  (.requests | length) == $capacity_num
  and all(.requests[]; (.expected_generated_token_ids | length) == 4)
  and any(.requests[]; .prompt_token_ids | length == 2)
  and any(.requests[]; .prompt_token_ids | length == 32)
' "$output" >/dev/null

echo "$output"