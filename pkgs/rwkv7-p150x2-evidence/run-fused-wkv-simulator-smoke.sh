#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: rwkv7-p150x2-fused-wkv-simulator-smoke NEW_ABSOLUTE_OUTPUT_ROOT" >&2
  exit 2
fi

output_root=$1
if [[ $output_root != /* || $output_root == /nix/store/* || -e $output_root ]]; then
  echo "output root must be a new absolute non-store path" >&2
  exit 2
fi

fused_probe_timeout=10m
mkdir -p "$output_root/profiler"

TT_METAL_SIMULATOR=@ttsim@/lib/ttsim/runtimes/blackhole-x1/libttsim_bh.so \
TT_METAL_SLOW_DISPATCH_MODE=1 \
TT_METAL_DISABLE_SFPLOADMACRO=1 \
TT_METAL_PROFILER_DIR="$output_root/profiler" \
@timeout@ "$fused_probe_timeout" \
  @runtime@/bin/rwkv7-p150x2-runtime \
  --fused-wkv-probe --output "$output_root/fused-wkv-simulator-receipt.json"

echo "$output_root"
