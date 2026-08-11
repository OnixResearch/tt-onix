#!/usr/bin/env bash
set -euo pipefail

usage_exit_code=2
if [[ $# -ne 1 ]]; then
  echo "usage: rwkv7-p150x2-persistent-decode-simulator-smoke NEW_ABSOLUTE_OUTPUT_ROOT" >&2
  exit "$usage_exit_code"
fi

output_root=$1
if [[ $output_root != /* || $output_root == /nix/store/* || -e $output_root ]]; then
  echo "output root must be a new absolute non-store path" >&2
  exit "$usage_exit_code"
fi

simulator_timeout=60m
mkdir -p "$output_root/profiler"

# Prove bounded completion, cancellation, timeout, failure, and cleanup before
# opening the simulator. The same runtime owns both checks.
@runtime@/bin/rwkv7-p150x2-runtime --self-test

# Metal Trace is unsupported by slow dispatch. This candidate-only smoke uses
# simulator fast dispatch and does not claim determinism or performance.
TT_METAL_SIMULATOR=@ttsim@/lib/ttsim/runtimes/blackhole-x1/libttsim_bh.so \
TT_METAL_DISABLE_SFPLOADMACRO=1 \
TT_METAL_PROFILER_DIR="$output_root/profiler" \
@timeout@ "$simulator_timeout" \
  @runtime@/bin/rwkv7-p150x2-runtime \
  --persistent-decode-simulator-smoke \
  --output "$output_root/persistent-decode-simulator-receipt.json"

echo "$output_root"
