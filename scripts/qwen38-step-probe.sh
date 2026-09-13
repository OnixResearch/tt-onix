#!/usr/bin/env bash
set -uo pipefail

# Qwen38 batch step-time probe across the instance-per-core boundary.
#
# A SCREEN, not an admission measurement: one fresh server session per width, the
# service's own per-token decode latency list as the timing source, and no receipt.
# It exists because a four-root receipt costs about thirty minutes of device time
# while this costs about three, so candidates can be screened before a receipt is
# spent on them. Nothing here may be quoted as admission evidence.
#
# Model under test: step(W) = C(W) + 48 * k(W) * x(k), where k = ceil(24W/110) is
# the GDN head-instances per core. With 24 heads per layer and at least 110 usable
# compute cores, k is 1 for W <= 4 and 2 for 5 <= W <= 9. So:
#
#   W=3 -> W=4  (both k=1)  difference = C's per-slot cost
#   W=4 -> W=5  (k 1 -> 2)  difference = C's per-slot cost + 48 * (2*x2 - x1)
#
# The second difference minus the first isolates the GDN instance loop's marginal
# cost at the boundary, with only a one-slot confound in the non-GDN path.
#
# Timing source is the service's own per-token decode latency list, whose width-8
# median (79.2 ms) matches the width-8 admission receipt (79.4 ms). The probe
# makes no admission claim: a fresh session is used per width, and only the
# median interior token latency is read.

package=${1:?usage: probe.sh <package-path> [widths...]}
shift
widths=("$@")
if (( ${#widths[@]} == 0 )); then widths=(3 4 5 8); fi

model=/home/brittonr/models/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
corpus=/home/brittonr/git/tenstorrent.nix/.pi/evidence/qwen38/b8-admission-20260826
work=/tmp/gdn-k5
listen_url=http://127.0.0.1:8000
maximum_sequence_length=2048
maximum_generation_tokens=64
startup_attempts=60
startup_interval_seconds=10
health_timeout_seconds=2
request_timeout_seconds=360
group_retry_maximum=4
group_retry_interval_seconds=2
active_server_pid=

export PYTHONPATH="$package/lib/python3.13/site-packages:/nix/store/6ccynlvv46p54pmp2lb7dbs65hkyj5w9-tt-metal-0.77.0/lib/python3.13/site-packages"
export TT_METAL_HOME="$package/libexec/tt-metalium"
export TT_METAL_RUNTIME_ROOT="$package/libexec/tt-metalium"
export MESH_DEVICE=P150x2
export HF_HUB_OFFLINE=1
export QWEN35_TEMP=0
export QWEN35_GDN_DECODE_FUSION=full
export TT_METAL_DISABLE_SFPLOADMACRO=1
export HOME="$model"
export TT_CACHE_PATH="$model"
export MPLCONFIGDIR="$work/mplcache"
mkdir -p "$MPLCONFIGDIR" "$work"

stop_server() {
  if [[ -n "$active_server_pid" ]]; then
    kill -TERM "$active_server_pid" 2>/dev/null || true
    local waited=0
    while kill -0 "$active_server_pid" 2>/dev/null && (( waited < 30 )); do sleep 1; waited=$((waited + 1)); done
    kill -9 "$active_server_pid" 2>/dev/null || true
    wait "$active_server_pid" 2>/dev/null || true
    active_server_pid=
  fi
}
restore() { stop_server; ttwkv7-owner-control restore; }
trap restore EXIT

ttwkv7-owner-control isolate

for width in "${widths[@]}"; do
  root="$work/w$width"
  rm -rf "$root"
  mkdir -p "$root"
  nohup "$package/bin/qwen38-p150x2-b8-admission" \
    --host 127.0.0.1 --port 8000 \
    --model-path "$model" --model-alias Qwen3.8-27B \
    --max-sequence-length "$maximum_sequence_length" \
    --max-generation-tokens "$maximum_generation_tokens" \
    > "$root/serve.log" 2>&1 &
  active_server_pid=$!

  attempt=1
  while (( attempt <= startup_attempts )); do
    if curl -fsS --max-time "$health_timeout_seconds" "$listen_url/healthz" >/dev/null 2>&1; then break; fi
    if ! kill -0 "$active_server_pid" 2>/dev/null; then break; fi
    sleep "$startup_interval_seconds"
    attempt=$((attempt + 1))
  done
  if (( attempt > startup_attempts )); then
    echo "width $width: server not ready"
    stop_server
    continue
  fi

  barrier="$root/.start"
  # The first group after startup can be rejected (the four-root admission
  # harness carries the same retry for the same reason), so re-issue the group
  # before reading anything as a result.
  attempt=1
  while (( attempt <= group_retry_maximum )); do
    rm -f "$barrier"
    pids=()
    for prompt in $(seq 0 $((width - 1))); do
      request="$corpus/request-$(printf '%02d' "$prompt").json"
      output="$root/response-$prompt.json"
      rm -f "$output"
      (
        while [[ ! -e "$barrier" ]]; do sleep 0.01; done
        curl -fsS --max-time "$request_timeout_seconds" -H 'Content-Type: application/json' \
          --data-binary "@$request" "$listen_url/v1/completions" > "$output" || true
      ) &
      pids+=("$!")
    done
    touch "$barrier"
    for pid in "${pids[@]}"; do wait "$pid"; done
    rm -f "$barrier"
    if python3 -c "import json,sys;[json.load(open(f'$root/response-{p}.json')) for p in range($width)]" 2>/dev/null; then
      break
    fi
    echo "width $width: group attempt $attempt produced no usable responses, retrying"
    sleep "$group_retry_interval_seconds"
    attempt=$((attempt + 1))
  done

  fatals=$(grep -c "exceeds device compute grid" "$root/serve.log" 2>/dev/null || true)
  complete=$(ls "$root"/response-*.json 2>/dev/null | wc -l)
  echo "width $width: $complete/$width responses, ${fatals:-0} core-range fatals"
  stop_server
  sleep 2
done

python3 - "$work" "${widths[@]}" <<'PY'
import json
import statistics
import sys

work = sys.argv[1]
widths = [int(w) for w in sys.argv[2:]]
print()
print(f"{'width':>5} {'steps':>5} {'median step ms':>15} {'mean step ms':>13} {'per-user t/s':>13} {'aggregate t/s':>14} {'k=ceil(24W/110)':>17}")
for width in widths:
    samples = []
    aggregate = []
    for prompt in range(width):
        path = f"{work}/w{width}/response-{prompt}.json"
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        latencies = payload.get("x_onix_decode_evidence", {}).get("token_latency_seconds") or []
        # Drop the first sample (prefill-adjacent) and the last (tail spike).
        interior = latencies[1:-1] if len(latencies) > 3 else latencies
        samples.extend(interior)
        evidence = payload.get("x_onix_decode_evidence", {})
        if isinstance(evidence.get("aggregate_decode_tokens_per_second"), (int, float)):
            aggregate.append(evidence["aggregate_decode_tokens_per_second"])
    if not samples:
        print(f"{width:>5} {'-':>5} {'no samples':>15}")
        continue
    median = statistics.median(samples)
    mean = statistics.fmean(samples)
    k = -(-24 * width // 110)
    per_user = 1.0 / median if median else 0.0
    agg = statistics.median(aggregate) if aggregate else 0.0
    print(f"{width:>5} {len(samples):>5} {median * 1e3:>15.3f} {mean * 1e3:>13.3f} {per_user:>13.3f} {agg:>14.3f} {k:>17}")
PY
