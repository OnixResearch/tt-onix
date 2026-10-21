#!/bin/bash
# usage: run_bw.sh <log> [dram_read_bw args...]   -- tt-metal v077, PCIe card 1 only, fast dispatch
LOG=$1; shift
: "${TT_METAL_HOME:?set TT_METAL_HOME to the tt-metal v0.77 checkout}"
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$TT_METAL_HOME" || exit 1
source python_env/bin/activate
TT_METAL_HOME=$(pwd)
PYTHONPATH=$(pwd)
TT_VISIBLE_DEVICES=1
READER_KERNEL=$HERE/reader.cpp
export TT_METAL_HOME PYTHONPATH TT_VISIBLE_DEVICES READER_KERNEL
timeout "${BW_TIMEOUT:-1500}" "$HERE/dram_read_bw" "$@" > "$LOG" 2>&1
echo "exit $?" >> "$LOG"
