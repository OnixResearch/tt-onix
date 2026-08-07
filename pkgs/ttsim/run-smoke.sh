#!/usr/bin/env bash
set -euo pipefail

SMOKE_TIMEOUT_SECONDS=3600
TIMEOUT_KILL_GRACE_SECONDS=30
TIMEOUT_EXIT_CODE=124
EXPECTED_ARGUMENT_COUNT=2

if [[ "$#" -ne "$EXPECTED_ARGUMENT_COUNT" || "$1" != "--receipt" ]]; then
  echo "usage: ttsim-blackhole-smoke --receipt <path>" >&2
  exit 1
fi

receipt_path="$2"
if [[ "$receipt_path" != /* ]]; then
  receipt_path="$PWD/$receipt_path"
fi
work_root="$(mktemp -d)"
cleanup() {
  rm -rf "$work_root"
}
trap cleanup EXIT

metal_template="@ttMetal@/libexec/tt-metalium"
for entry in "$metal_template"/*; do
  ln -s "$entry" "$work_root/$(basename "$entry")"
done
mkdir -p "$work_root/build" "$work_root/generated" "$work_root/home"

export HOME="$work_root/home"
export XDG_CACHE_HOME="$work_root/home/.cache"
export TT_METAL_HOME="$work_root"
export TT_METAL_RUNTIME_ROOT="$work_root"
export TT_METAL_PROFILER_DIR="$work_root/generated/profiler"
export PYTHONPATH="@ttMetal@/@pythonSitePackages@"
unset TT_MESH_GRAPH_DESC_PATH

stdout_log="$work_root/smoke.stdout"
stderr_log="$work_root/smoke.stderr"
cd "$work_root"
set +e
"@ttsim@/bin/ttsim-blackhole" \
  timeout --signal=TERM --kill-after="${TIMEOUT_KILL_GRACE_SECONDS}s" \
  "$SMOKE_TIMEOUT_SECONDS" \
  "@python@/bin/python" "@smokeScript@" \
  >"$stdout_log" 2>"$stderr_log"
child_exit_code="$?"
set -e

cat "$stdout_log"
cat "$stderr_log" >&2

timed_out=false
if [[ "$child_exit_code" -eq "$TIMEOUT_EXIT_CODE" ]]; then
  timed_out=true
fi

mkdir -p "$(dirname "$receipt_path")"
"@validator@/bin/ttsim-runtime-validator" emit \
  "$stdout_log" \
  "$stderr_log" \
  "$child_exit_code" \
  "$timed_out" \
  "@ttsim@/lib/ttsim/runtimes/blackhole-x1/libttsim.so" \
  "@ttsim@/lib/ttsim/runtimes/blackhole-x1/soc_descriptor.yaml" \
  "@hostSystem@" \
  "tt-metal-@ttMetalVersion@:@ttMetal@" \
  "$SMOKE_TIMEOUT_SECONDS" \
  "$receipt_path"
