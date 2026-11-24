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
# Timing source is the service's own per-token decode latency list. Pooled token
# observations across requests in a shared batch are repeated views of shared batch
# steps, not independent experimental replications.

usage() {
  cat >&2 <<'EOF'
usage: qwen38-step-probe.sh --gdn-route <step|full|off> [--port PORT] [--work-dir DIR] [--model-alias NAME] <package-path> [widths...]

The intended GDN decode route must be explicit via --gdn-route or QWEN35_GDN_DECODE_FUSION
(step = served fused decode step, full = fused recurrence candidate, off = composite).
EOF
}

gdn_route="${QWEN35_GDN_DECODE_FUSION:-}"
listen_host="${QWEN38_PROBE_HOST:-127.0.0.1}"
listen_port="${QWEN38_PROBE_PORT:-8000}"
work="${QWEN38_PROBE_WORK_DIR:-/tmp/gdn-k5}"
model_alias="${QWEN38_PROBE_MODEL_ALIAS:-Qwen3.8-27B}"

while (( $# > 0 )); do
  case "$1" in
    --gdn-route)
      if (( $# < 2 )); then
        usage
        exit 2
      fi
      gdn_route="$2"
      shift 2
      ;;
    --port)
      if (( $# < 2 )); then
        usage
        exit 2
      fi
      listen_port="$2"
      shift 2
      ;;
    --work-dir)
      if (( $# < 2 )); then
        usage
        exit 2
      fi
      work="$2"
      shift 2
      ;;
    --model-alias)
      if (( $# < 2 )); then
        usage
        exit 2
      fi
      model_alias="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "unknown option: $1" >&2
      usage
      exit 2
      ;;
    *)
      break
      ;;
  esac
done

if (( $# < 1 )); then
  usage
  exit 2
fi

package="$1"
shift
widths=("$@")
if (( ${#widths[@]} == 0 )); then
  widths=(3 4 5 8)
fi

for width in "${widths[@]}"; do
  if [[ ! "$width" =~ ^[1-9][0-9]*$ ]]; then
    echo "width must be a positive integer, got: $width" >&2
    exit 2
  fi
done

if [[ -z "$gdn_route" ]]; then
  echo "error: explicit GDN route required (--gdn-route <step|full|off> or QWEN35_GDN_DECODE_FUSION)" >&2
  usage
  exit 2
fi

python_bin="${QWEN38_PROBE_PYTHON:-python3}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
core_module="${QWEN38_PROBE_CORE:-}"
if [[ -z "$core_module" ]]; then
  if [[ -f "$script_dir/../pkgs/qwen36/step_probe_core.py" ]]; then
    core_module="$script_dir/../pkgs/qwen36/step_probe_core.py"
  elif [[ -f "$package/share/qwen36/step_probe_core.py" ]]; then
    core_module="$package/share/qwen36/step_probe_core.py"
  else
    echo "error: unable to locate step_probe_core.py" >&2
    exit 2
  fi
fi

if ! "$python_bin" "$core_module" validate-route "$gdn_route"; then
  exit 2
fi

# Host paths stay outside the repository. Supply the exact admitted model,
# request corpus and package-matched TT-Metal Python directory before probing.
model=${QWEN38_PROBE_MODEL:?set QWEN38_PROBE_MODEL to the model snapshot path}
corpus=${QWEN38_PROBE_CORPUS:?set QWEN38_PROBE_CORPUS to the request corpus path}
tt_metal_python=${QWEN38_PROBE_TT_METAL_SITE_PACKAGES:?set QWEN38_PROBE_TT_METAL_SITE_PACKAGES to the matched TT-Metal site-packages path}

owner_control="${QWEN38_PROBE_OWNER_CONTROL:-ttwkv7-owner-control}"
server_bin="${QWEN38_PROBE_SERVER_BIN:-$package/bin/qwen38-p150x2-b8-admission}"
probe_session_id="${QWEN38_PROBE_SESSION_ID:-qwen38-step-probe-$(date -u +%Y%m%dT%H%M%SZ)-$$}"

listen_url="http://${listen_host}:${listen_port}"
maximum_sequence_length=2048
maximum_generation_tokens=64
startup_attempts="${QWEN38_PROBE_STARTUP_ATTEMPTS:-60}"
startup_interval_seconds="${QWEN38_PROBE_STARTUP_INTERVAL_SECONDS:-10}"
health_timeout_seconds="${QWEN38_PROBE_HEALTH_TIMEOUT_SECONDS:-2}"
request_timeout_seconds="${QWEN38_PROBE_REQUEST_TIMEOUT_SECONDS:-360}"
group_retry_maximum="${QWEN38_PROBE_GROUP_RETRY_MAX:-4}"
group_retry_interval_seconds="${QWEN38_PROBE_GROUP_RETRY_INTERVAL_SECONDS:-2}"
stop_delay_seconds="${QWEN38_PROBE_STOP_DELAY_SECONDS:-2}"

active_server_pid=""
isolated=0

export PYTHONPATH="$package/lib/python3.13/site-packages:$tt_metal_python"
export TT_METAL_HOME="$package/libexec/tt-metalium"
export TT_METAL_RUNTIME_ROOT="$package/libexec/tt-metalium"
export MESH_DEVICE=P150x2
export HF_HUB_OFFLINE=1
export QWEN35_TEMP=0
export QWEN35_GDN_DECODE_FUSION="$gdn_route"
export TT_METAL_DISABLE_SFPLOADMACRO=1
export HOME="$model"
export TT_CACHE_PATH="$model"
export MPLCONFIGDIR="$work/mplcache"
export QWEN38_PROBE_SESSION_ID="$probe_session_id"
mkdir -p "$MPLCONFIGDIR" "$work"

stop_server() {
  if [[ -n "$active_server_pid" ]]; then
    kill -TERM "$active_server_pid" 2>/dev/null || true
    local waited=0
    while kill -0 "$active_server_pid" 2>/dev/null && (( waited < 30 )); do
      sleep 1
      waited=$((waited + 1))
    done
    kill -9 "$active_server_pid" 2>/dev/null || true
    wait "$active_server_pid" 2>/dev/null || true
    active_server_pid=""
  fi
}

cleanup() {
  local rc=$?
  trap - EXIT
  stop_server
  if (( isolated == 1 )); then
    "$owner_control" restore || true
  fi
  exit "$rc"
}
trap cleanup EXIT

write_width_meta() {
  local meta_path="$1"
  local pre_failure="$2"
  local fatals="$3"
  "$python_bin" - "$meta_path" "$pre_failure" "$fatals" "$probe_session_id" "$gdn_route" <<'PY'
import json
import sys
from pathlib import Path

meta_path, pre_failure, fatals, session_id, gdn_route = sys.argv[1:6]
payload = {
    "probe_session_id": session_id,
    "requested_gdn_route": gdn_route,
    "pre_dispatch_failure": pre_failure if pre_failure else None,
    "core_range_fatals": int(fatals),
}
Path(meta_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

# Fail closed before any server launch or dispatch if ownership isolation fails.
if ! "$owner_control" isolate; then
  echo "error: ownership isolation failed via $owner_control; aborting before dispatch" >&2
  exit 2
fi
isolated=1

for width in "${widths[@]}"; do
  root="$work/w$width"
  rm -rf "$root"
  mkdir -p "$root/attempts"

  if ! "$python_bin" "$core_module" check-port-free --host "$listen_host" --port "$listen_port" >/dev/null 2>&1; then
    echo "width $width: unrelated server already listening on ${listen_host}:${listen_port}; aborting width before launch" >&2
    write_width_meta "$root/meta.json" "unrelated_server_already_listening" 0
    continue
  fi

  nohup "$server_bin" \
    --host "$listen_host" --port "$listen_port" \
    --model-path "$model" --model-alias "$model_alias" \
    --max-sequence-length "$maximum_sequence_length" \
    --max-generation-tokens "$maximum_generation_tokens" \
    > "$root/serve.log" 2>&1 &
  active_server_pid=$!

  ready=0
  readiness_failure=""
  attempt=1
  while (( attempt <= startup_attempts )); do
    "$python_bin" "$core_module" check-readiness \
      --pid "$active_server_pid" \
      --url "$listen_url" \
      --host "$listen_host" \
      --port "$listen_port" \
      --expected-model "$model_alias" \
      --probe-session-id "$probe_session_id" \
      --timeout "$health_timeout_seconds" >/dev/null 2>&1
    rc=$?
    if (( rc == 0 )); then
      ready=1
      break
    elif (( rc == 2 )); then
      readiness_failure="server_exited_before_readiness"
      break
    elif (( rc == 3 )); then
      readiness_failure="unrelated_server_detected"
      break
    elif (( rc == 10 )); then
      sleep "$startup_interval_seconds"
      attempt=$((attempt + 1))
    else
      readiness_failure="readiness_check_error"
      break
    fi
  done

  if (( ready != 1 )); then
    if [[ -z "$readiness_failure" ]]; then
      readiness_failure="readiness_timeout"
    fi
    echo "width $width: server not ready ($readiness_failure); skipping dispatch" >&2
    fatals="$(grep -c "exceeds device compute grid" "$root/serve.log" 2>/dev/null || true)"
    write_width_meta "$root/meta.json" "$readiness_failure" "${fatals:-0}"
    stop_server
    continue
  fi

  # The first group after startup can be rejected (the four-root admission
  # harness carries the same retry for the same reason), so re-issue the group
  # while retaining every failed attempt directory on disk.
  attempt=1
  while (( attempt <= group_retry_maximum )); do
    attempt_dir="$root/attempts/attempt-$(printf '%02d' "$attempt")"
    mkdir -p "$attempt_dir"
    barrier="$attempt_dir/.start"
    rm -f "$barrier"
    pids=()
    for (( prompt = 0; prompt < width; prompt++ )); do
      request="$corpus/request-$(printf '%02d' "$prompt").json"
      output="$attempt_dir/response-$prompt.json"
      status_file="$attempt_dir/response-$prompt.http_status"
      rm -f "$output" "$status_file"
      (
        while [[ ! -e "$barrier" ]]; do sleep 0.01; done
        http_code="$(
          curl -sS -o "$output" -w '%{http_code}' \
            --max-time "$request_timeout_seconds" \
            -H 'Content-Type: application/json' \
            -H "X-Onix-Probe-Session: $probe_session_id" \
            -H "X-Onix-Probe-Request: w${width}-a${attempt}-p${prompt}" \
            --data-binary "@$request" \
            "$listen_url/v1/completions" 2>/dev/null
        )" || true
        if [[ -z "$http_code" ]]; then
          http_code="000"
        fi
        printf '%s\n' "$http_code" > "$status_file"
      ) &
      pids+=("$!")
    done
    touch "$barrier"
    for pid in "${pids[@]}"; do wait "$pid"; done
    rm -f "$barrier"

    if "$python_bin" "$core_module" validate-attempt \
      --attempt-dir "$attempt_dir" \
      --attempt-index "$attempt" \
      --width "$width" \
      --gdn-route "$gdn_route" \
      --package-path "$package"; then
      break
    fi

    echo "width $width: group attempt $attempt failed validation (retained in $attempt_dir)" >&2
    if (( attempt < group_retry_maximum )); then
      sleep "$group_retry_interval_seconds"
    fi
    attempt=$((attempt + 1))
  done

  fatals="$(grep -c "exceeds device compute grid" "$root/serve.log" 2>/dev/null || true)"
  write_width_meta "$root/meta.json" "" "${fatals:-0}"
  stop_server
  sleep "$stop_delay_seconds"
done

"$python_bin" "$core_module" reduce \
  --work-dir "$work" \
  --gdn-route "$gdn_route" \
  --package-path "$package" \
  --model-path "$model" \
  --model-alias "$model_alias" \
  --corpus-path "$corpus" \
  --probe-session-id "$probe_session_id" \
  --output-json "$work/report.json" \
  "${widths[@]}"
PY_RC=$?
exit "$PY_RC"
