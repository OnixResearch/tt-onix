# r[impl tenstorrent.training_platform.preflight]
# r[impl tenstorrent.training_platform.launch]
# r[impl tenstorrent.training_platform.evidence]
# Host-side orchestrator for the bounded dual-P150A platform smoke.
# Thin shell: all admission decisions live in training-platform-core.
set -euo pipefail

CORE=@core@/bin/training-platform-core
JQ=@jq@
DOCKER=@docker@
TTSMI=@ttsmi@
TIMEOUT_SECONDS=900

usage() {
  echo "usage: training-platform-smoke --manifest PATH --config PATH --descriptor PATH --evidence-root PATH" >&2
  exit 2
}

MANIFEST=""
CONFIG=""
DESCRIPTOR=""
EVIDENCE_ROOT=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --manifest) MANIFEST="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --descriptor) DESCRIPTOR="$2"; shift 2 ;;
    --evidence-root) EVIDENCE_ROOT="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[ -n "$MANIFEST" ] && [ -n "$CONFIG" ] && [ -n "$DESCRIPTOR" ] && [ -n "$EVIDENCE_ROOT" ] || usage

for variable in PYTHONPATH PYTHONHOME TT_METAL_HOME TT_METAL_RUNTIME_ROOT; do
  if [ -n "${!variable:-}" ]; then
    echo "training-platform-smoke: ambient $variable is set and would override the admitted bundle" >&2
    exit 1
  fi
done

mkdir -p "$EVIDENCE_ROOT"
PLAN=$("$CORE" launch-plan "$MANIFEST" "$CONFIG" "$DESCRIPTOR")
IMAGE=$(printf '%s' "$PLAN" | "$JQ" -r '.plan.image')
HUGEPAGE_MOUNT=$(printf '%s' "$PLAN" | "$JQ" -r '.plan.mounts[0].host')
DESCRIPTOR_CONTAINER=$(printf '%s' "$PLAN" | "$JQ" -r '.plan.mounts[1].container')
TIMEOUT_SECONDS=$(printf '%s' "$PLAN" | "$JQ" -r '.plan.smoke_timeout_seconds')
BUNDLE_IDENTITY=$(printf '%s' "$PLAN" | "$JQ" -r '.plan.bundle_identity')
CONFIG_IDENTITY=$(printf '%s' "$PLAN" | "$JQ" -r '.plan.config_identity')
PLAN_IDENTITY=$(printf '%s' "$PLAN" | "$JQ" -r '.identity_blake3')

stage_status() { printf '{"name": "%s", "status": "%s"}' "$1" "$2"; }

finish_receipt() {
  # $1 = overall exit code, remaining args = stage JSON objects
  local overall="$1"; shift
  local stages
  stages=$(printf '%s\n' "$@" | "$JQ" -s '.')
  local artifacts
  artifacts=$(cd "$EVIDENCE_ROOT" && for f in *.json; do
    [ "$f" = "receipt.json" ] && continue
    printf '%s %s\n' "$(@b3sum@ < "$f" | cut -d' ' -f1)" "$f"
  done | "$JQ" -R -s 'split("\n") | map(select(length > 0) | split(" ")) | map({key: .[1], value: .[0]}) | from_entries')
  "$JQ" -n \
    --argjson schema_version 1 \
    --arg kind "dual_p150a_platform_smoke" \
    --arg bundle_identity "$BUNDLE_IDENTITY" \
    --arg config_identity "$CONFIG_IDENTITY" \
    --arg launch_plan_identity "$PLAN_IDENTITY" \
    --argjson stages "$stages" \
    --argjson artifacts "$artifacts" \
    '{
      schema_version: $schema_version,
      kind: $kind,
      bundle_identity: $bundle_identity,
      config_identity: $config_identity,
      launch_plan_identity: $launch_plan_identity,
      stages: $stages,
      artifacts: $artifacts,
      non_claims: [
        "no training correctness claim",
        "no gradient parity claim",
        "no convergence claim",
        "no speedup claim",
        "no thermal endurance claim"
      ]
    }' > "$EVIDENCE_ROOT/receipt.json"
  if [ "$overall" -eq 0 ]; then
    "$CORE" validate-receipt "$EVIDENCE_ROOT/receipt.json" "$MANIFEST" "$CONFIG" "$DESCRIPTOR"
  fi
  exit "$overall"
}

fail_at() {
  # $1 = failed stage name, $2.. = completed stage JSON
  local failed="$1"; shift
  echo "training-platform-smoke: stage $failed failed" >&2
  finish_receipt 1 "$@" "$(stage_status "$failed" fail)"
}

# Stage health_before
"$TTSMI" -s > "$EVIDENCE_ROOT/health-before.json"
S_HEALTH_BEFORE=$(stage_status health_before pass)

# Stage preflight admission (device-free admission of observed facts)
DRIVER_VERSION=$("$JQ" -r '.host_info.Driver' "$EVIDENCE_ROOT/health-before.json" | sed 's/^TT-KMD //')
FREE_HUGEPAGES=$(cat /sys/kernel/mm/hugepages/hugepages-1048576kB/free_hugepages)
if @fuser@ "$HUGEPAGE_MOUNT" >/dev/null 2>&1; then
  DEVICES_FREE=false
else
  DEVICES_FREE=true
fi
for device_node in /dev/tenstorrent/[0-9]*; do
  if @fuser@ "$device_node" >/dev/null 2>&1; then
    DEVICES_FREE=false
  fi
done

"$JQ" -n \
  --arg driver_version "$DRIVER_VERSION" \
  --arg mount "$HUGEPAGE_MOUNT" \
  --argjson free_pages "$FREE_HUGEPAGES" \
  --argjson devices_free "$DEVICES_FREE" \
  --slurpfile health "$EVIDENCE_ROOT/health-before.json" \
  '{
    driver_version: $driver_version,
    devices: [$health[0].device_info[] | {
      board_type: .board_info.board_type,
      board_id: .board_info.board_id,
      dram_status: .board_info.dram_status,
      fw_bundle_version: .firmwares.fw_bundle_version
    }],
    hugepages: {mount: $mount, free_pages: $free_pages},
    ownership: {devices_free: $devices_free}
  }' > "$EVIDENCE_ROOT/preflight-snapshot.json"

if ! "$CORE" admit-preflight "$EVIDENCE_ROOT/preflight-snapshot.json" "$MANIFEST" "$CONFIG" \
  > "$EVIDENCE_ROOT/preflight-admission.json"; then
  echo "training-platform-smoke: preflight admission failed; see preflight-admission.json and stderr" >&2
  exit 1
fi

# Stages mesh_open + smoke_execute (one bounded container run covers both)
set +e
timeout "$TIMEOUT_SECONDS" "$DOCKER" run --rm \
  --device /dev/tenstorrent \
  -v "$HUGEPAGE_MOUNT:$HUGEPAGE_MOUNT" \
  -v "$DESCRIPTOR:$DESCRIPTOR_CONTAINER:ro" \
  -v @smoke_py@:/platform/smoke.py:ro \
  -v @container_smoke@:/platform/container-smoke.sh:ro \
  -e "TT_MESH_GRAPH_DESC_PATH=$DESCRIPTOR_CONTAINER" \
  "$IMAGE" \
  bash /platform/container-smoke.sh \
  > "$EVIDENCE_ROOT/smoke-output.json" 2> "$EVIDENCE_ROOT/smoke-stderr.log"
SMOKE_STATUS=$?
set -e

if [ "$SMOKE_STATUS" -ne 0 ]; then
  fail_at mesh_open "$S_HEALTH_BEFORE"
fi
if ! grep -F '"status": "pass"' "$EVIDENCE_ROOT/smoke-output.json" >/dev/null; then
  fail_at smoke_execute "$S_HEALTH_BEFORE" "$(stage_status mesh_open pass)"
fi
S_MESH_OPEN=$(stage_status mesh_open pass)
S_SMOKE=$(stage_status smoke_execute pass)
# The container process exited; the runtime closed the mesh.
S_MESH_CLOSE=$(stage_status mesh_close pass)

# Stage restore: no process may own a selected device after the run.
for device_node in /dev/tenstorrent/[0-9]*; do
  if @fuser@ "$device_node" >/dev/null 2>&1; then
    fail_at restore "$S_HEALTH_BEFORE" "$S_MESH_OPEN" "$S_SMOKE" "$S_MESH_CLOSE"
  fi
done
S_RESTORE=$(stage_status restore pass)

# Stage health_after
"$TTSMI" -s > "$EVIDENCE_ROOT/health-after.json"
S_HEALTH_AFTER=$(stage_status health_after pass)

finish_receipt 0 \
  "$S_HEALTH_BEFORE" "$S_MESH_OPEN" "$S_SMOKE" "$S_MESH_CLOSE" "$S_RESTORE" "$S_HEALTH_AFTER"
