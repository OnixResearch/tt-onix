#!/usr/bin/env bash
# r[impl tenstorrent.training_platform.launch]
# Nix-owned launcher for the admitted TT-XLA runtime image. Fails closed on
# ambient runtime overrides and launches only the digest-pinned image.
set -euo pipefail

usage() {
  echo "usage: training-platform-launch --manifest PATH --config PATH --descriptor PATH -- COMMAND [ARGS...]" >&2
  exit 2
}

MANIFEST=""
CONFIG=""
DESCRIPTOR=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --manifest) MANIFEST="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --descriptor) DESCRIPTOR="$2"; shift 2 ;;
    --) shift; break ;;
    *) usage ;;
  esac
done
[ -n "$MANIFEST" ] && [ -n "$CONFIG" ] && [ -n "$DESCRIPTOR" ] && [ "$#" -gt 0 ] || usage

# Bind the launch plan to the canonical descriptor path.
DESCRIPTOR=$(readlink -f "$DESCRIPTOR")

# r[verify tenstorrent.training_platform.launch.isolated]
# Ambient Python and TT-Metal paths must not alter the selected bundle.
for variable in PYTHONPATH PYTHONHOME TT_METAL_HOME TT_METAL_RUNTIME_ROOT; do
  if [ -n "${!variable:-}" ]; then
    echo "training-platform-launch: ambient $variable is set and would override the admitted bundle" >&2
    exit 1
  fi
done

PLAN=$(@core@/bin/training-platform-core launch-plan "$MANIFEST" "$CONFIG" "$DESCRIPTOR")

IMAGE=$(printf '%s' "$PLAN" | @jq@ -r '.plan.image')
DEVICE_ROOT=$(printf '%s' "$PLAN" | @jq@ -r '.plan.devices[0]')
HUGEPAGE_MOUNT=$(printf '%s' "$PLAN" | @jq@ -r '.plan.mounts[0].host')
DESCRIPTOR_CONTAINER=$(printf '%s' "$PLAN" | @jq@ -r '.plan.mounts[1].container')
# The environment map has one admitted key. Extract it by name: a
# `.plan.environment[keys[0]]` lookup evaluates `keys` against the root
# object and silently yields null, which the runtime then reports as a
# missing mesh graph descriptor.
MESH_ENV_VALUE=$(printf '%s' "$PLAN" | @jq@ -r '.plan.environment.TT_MESH_GRAPH_DESC_PATH')
if [ -z "$MESH_ENV_VALUE" ] || [ "$MESH_ENV_VALUE" = "null" ]; then
  echo "training-platform-launch: launch plan carries no TT_MESH_GRAPH_DESC_PATH" >&2
  exit 1
fi

exec @docker@ run --rm \
  --device "$DEVICE_ROOT" \
  -v "$HUGEPAGE_MOUNT:$HUGEPAGE_MOUNT" \
  -v "$DESCRIPTOR:$DESCRIPTOR_CONTAINER:ro" \
  -e "TT_MESH_GRAPH_DESC_PATH=$MESH_ENV_VALUE" \
  "$IMAGE" \
  "$@"
