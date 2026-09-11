#!/usr/bin/env bash
# r[impl tenstorrent.training_platform.preflight]
# Fail closed unless every selected device has no KMD-reported owner.
set -euo pipefail

USAGE_EXIT_CODE=2

usage() {
  echo "usage: ownership-probe DEVICE_ROOT PROCFS_ROOT FUSER HUGEPAGE_MOUNT" >&2
  exit "$USAGE_EXIT_CODE"
}

[ "$#" -eq 4 ] || usage

DEVICE_ROOT="$1"
PROCFS_ROOT="$2"
FUSER="$3"
HUGEPAGE_MOUNT="$4"
FOUND_DEVICE=false

if "$FUSER" "$HUGEPAGE_MOUNT" >/dev/null 2>&1; then
  exit 1
fi

for device_node in "$DEVICE_ROOT"/[0-9]*; do
  [ -e "$device_node" ] || continue
  FOUND_DEVICE=true
  device_ordinal=${device_node##*/}
  device_pids="$PROCFS_ROOT/$device_ordinal/pids"

  [ -r "$device_pids" ] || exit 1
  if IFS= read -r _owner_pid < "$device_pids"; then
    exit 1
  fi

  if "$FUSER" "$device_node" >/dev/null 2>&1; then
    exit 1
  fi
done

[ "$FOUND_DEVICE" = true ] || exit 1
