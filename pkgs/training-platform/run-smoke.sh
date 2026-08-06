#!/usr/bin/env bash
# r[impl tenstorrent.training_platform.preflight]
# r[impl tenstorrent.training_platform.launch]
# r[impl tenstorrent.training_platform.evidence]
# The smoke uses the same package-owned production launch operation.
set -euo pipefail

usage() {
  echo "usage: training-platform-smoke --evidence-root PATH" >&2
  exit 2
}

EVIDENCE_ROOT=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --evidence-root) EVIDENCE_ROOT="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[ -n "$EVIDENCE_ROOT" ] || usage

exec @launcher@ --evidence-root "$EVIDENCE_ROOT" -- \
  bash /platform/container-smoke.sh
