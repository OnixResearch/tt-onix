#!/usr/bin/env bash
# shellcheck disable=SC2016 # jq filters use their own $variables.
# r[impl tenstorrent.training_platform.preflight]
# r[impl tenstorrent.training_platform.launch]
# r[impl tenstorrent.training_platform.evidence]
# Package-owned production launcher with same-process preflight and receipts.
set -euo pipefail

CORE=@core@/bin/training-platform-core
JQ=@jq@
DOCKER=@docker@
TTSMI=@ttsmi@
FUSER=@fuser@
B3SUM=@b3sum@
AUTHORITY=@authority@

usage() {
  echo "usage: training-platform-launch --evidence-root PATH -- COMMAND [ARGS...]" >&2
  exit 2
}

EVIDENCE_ROOT=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --evidence-root) EVIDENCE_ROOT="$2"; shift 2 ;;
    --) shift; break ;;
    *) usage ;;
  esac
done
[ -n "$EVIDENCE_ROOT" ] && [ "$#" -gt 0 ] || usage
case "$EVIDENCE_ROOT" in
  /*) ;;
  *) echo "training-platform-launch: evidence root must be absolute" >&2; exit 2 ;;
esac
case "$EVIDENCE_ROOT" in
  *..*) echo "training-platform-launch: evidence root must not contain parent traversal" >&2; exit 2 ;;
esac

PLAN=$($CORE authorized-launch-plan "$AUTHORITY")
IMAGE=$(printf '%s' "$PLAN" | "$JQ" -r '.plan.image')
DEVICE_ROOT=$(printf '%s' "$PLAN" | "$JQ" -r '.plan.devices[0]')
HUGEPAGE_MOUNT=$(printf '%s' "$PLAN" | "$JQ" -r '.plan.mounts[0].host')
DESCRIPTOR=$(printf '%s' "$PLAN" | "$JQ" -r '.plan.mounts[1].host')
DESCRIPTOR_CONTAINER=$(printf '%s' "$PLAN" | "$JQ" -r '.plan.mounts[1].container')
WHEELHOUSE=$(printf '%s' "$PLAN" | "$JQ" -r '.plan.mounts[2].host')
WHEELHOUSE_CONTAINER=$(printf '%s' "$PLAN" | "$JQ" -r '.plan.mounts[2].container')
MESH_ENV_VALUE=$(printf '%s' "$PLAN" | "$JQ" -r '.plan.environment.TT_MESH_GRAPH_DESC_PATH')
BUNDLE_IDENTITY=$(printf '%s' "$PLAN" | "$JQ" -r '.plan.bundle_identity')
CONFIG_IDENTITY=$(printf '%s' "$PLAN" | "$JQ" -r '.plan.config_identity')
PLAN_IDENTITY=$(printf '%s' "$PLAN" | "$JQ" -r '.identity_blake3')
EVIDENCE_BASE=$(printf '%s' "$PLAN" | "$JQ" -r '.plan.evidence_root')

case "$EVIDENCE_ROOT" in
  "$EVIDENCE_BASE"/*) ;;
  *)
    echo "training-platform-launch: evidence root is outside package authority" >&2
    exit 2
    ;;
esac
RUN_NAME=${EVIDENCE_ROOT#"$EVIDENCE_BASE"/}
case "$RUN_NAME" in
  ""|*/*)
    echo "training-platform-launch: evidence root must name one fresh run" >&2
    exit 2
    ;;
esac
if ! mkdir "$EVIDENCE_ROOT"; then
  echo "training-platform-launch: evidence root must be fresh" >&2
  exit 2
fi

EVIDENCE_ROOT_MODE=0770
PRIVATE_ROOT_MODE=0700
PORTABLE_ROOT_MODE=0750
PRIVATE_ROOT="$EVIDENCE_ROOT/private"
PORTABLE_ROOT="$EVIDENCE_ROOT/portable"
mkdir "$PRIVATE_ROOT" "$PORTABLE_ROOT"
chmod "$EVIDENCE_ROOT_MODE" "$EVIDENCE_ROOT"
chmod "$PRIVATE_ROOT_MODE" "$PRIVATE_ROOT"
chmod "$PORTABLE_ROOT_MODE" "$PORTABLE_ROOT"

CURRENT_STAGE=health_before
TERMINAL_WRITTEN=false

artifact_map() {
  for artifact in "$PORTABLE_ROOT"/*.json; do
    [ -e "$artifact" ] || continue
    [ "$(basename "$artifact")" = "receipt.json" ] && continue
    $CORE validate-portable-artifact "$artifact" > /dev/null
    hash=$($B3SUM "$artifact" | cut -d' ' -f1)
    printf '%s %s\n' "$(basename "$artifact")" "$hash"
  done | "$JQ" -R -s '
    split("\n")
    | map(select(length > 0) | split(" "))
    | map({key: .[0], value: .[1]})
    | from_entries'
}

write_receipt() {
  failed_stage="$1"
  if [ "$failed_stage" = "success" ]; then
    stages=$($CORE stage-states success)
  else
    stages=$($CORE stage-states "$failed_stage")
  fi
  artifacts=$(artifact_map)
  "$JQ" -n \
    --argjson schema_version 1 \
    --arg kind dual_p150a_platform_smoke \
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
    }' > "$PORTABLE_ROOT/receipt.json"
  $CORE validate-terminal-receipt "$PORTABLE_ROOT/receipt.json" "$AUTHORITY" \
    > "$PORTABLE_ROOT/receipt-admission.json"
  TERMINAL_WRITTEN=true
}

finish() {
  status="$?"
  trap - EXIT
  if [ "$TERMINAL_WRITTEN" != true ]; then
    if [ "$status" -eq 0 ]; then
      write_receipt success || status=1
    else
      write_receipt "$CURRENT_STAGE" || status=1
    fi
  fi
  exit "$status"
}
trap finish EXIT

for variable in PYTHONPATH PYTHONHOME TT_METAL_HOME TT_METAL_RUNTIME_ROOT; do
  if [ -n "${!variable:-}" ]; then
    echo "training-platform-launch: ambient $variable is not admitted" >&2
    exit 1
  fi
done

"$TTSMI" -s > "$PRIVATE_ROOT/health-before.json"
CURRENT_STAGE=preflight
DRIVER_VERSION=$("$JQ" -r '.host_info.Driver' "$PRIVATE_ROOT/health-before.json" | sed 's/^TT-KMD //')
FREE_HUGEPAGES=$(cat /sys/kernel/mm/hugepages/hugepages-1048576kB/free_hugepages)
DEVICES_FREE=true
if "$FUSER" "$HUGEPAGE_MOUNT" >/dev/null 2>&1; then
  DEVICES_FREE=false
fi
for device_node in "$DEVICE_ROOT"/[0-9]*; do
  [ -e "$device_node" ] || continue
  if "$FUSER" "$device_node" >/dev/null 2>&1; then
    DEVICES_FREE=false
  fi
done

"$JQ" -n \
  --arg driver_version "$DRIVER_VERSION" \
  --arg mount "$HUGEPAGE_MOUNT" \
  --argjson free_pages "$FREE_HUGEPAGES" \
  --argjson devices_free "$DEVICES_FREE" \
  --slurpfile health "$PRIVATE_ROOT/health-before.json" \
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
  }' > "$PRIVATE_ROOT/preflight-snapshot.json"

if $CORE admit-authorized-preflight \
  "$PRIVATE_ROOT/preflight-snapshot.json" "$AUTHORITY" \
  > "$PRIVATE_ROOT/preflight-admission.json" \
  2> "$PRIVATE_ROOT/preflight-error.log"; then
  cp "$PRIVATE_ROOT/preflight-admission.json" \
    "$PORTABLE_ROOT/preflight-admission.json"
else
  echo "training-platform-launch: preflight admission failed" >&2
  exit 1
fi

CURRENT_STAGE=mesh_open
set +e
"$DOCKER" run --rm \
  --device "$DEVICE_ROOT" \
  -v "$HUGEPAGE_MOUNT:$HUGEPAGE_MOUNT" \
  -v "$DESCRIPTOR:$DESCRIPTOR_CONTAINER:ro" \
  -v "$WHEELHOUSE:$WHEELHOUSE_CONTAINER:ro" \
  -v @smoke_py@:/platform/smoke.py:ro \
  -v @container_smoke@:/platform/container-smoke.sh:ro \
  -e "TT_MESH_GRAPH_DESC_PATH=$MESH_ENV_VALUE" \
  "$IMAGE" \
  "$@" \
  > "$PRIVATE_ROOT/workload-stdout.log" \
  2> "$PRIVATE_ROOT/workload-stderr.log"
WORKLOAD_STATUS=$?
set -e
if [ "$WORKLOAD_STATUS" -ne 0 ]; then
  CURRENT_STAGE=$($CORE classify-smoke-failure "$PRIVATE_ROOT/workload-stdout.log")
  "$JQ" -n \
    --arg stage "$CURRENT_STAGE" \
    --argjson workload_exit_code "$WORKLOAD_STATUS" \
    '{status: "fail", stage: $stage, workload_exit_code: $workload_exit_code}' \
    > "$PORTABLE_ROOT/workload-summary.json"
  exit "$WORKLOAD_STATUS"
fi

CURRENT_STAGE=restore
for device_node in "$DEVICE_ROOT"/[0-9]*; do
  [ -e "$device_node" ] || continue
  if "$FUSER" "$device_node" >/dev/null 2>&1; then
    echo "training-platform-launch: device remains busy after container exit" >&2
    exit 1
  fi
done

CURRENT_STAGE=health_after
"$TTSMI" -s > "$PRIVATE_ROOT/health-after.json"
"$JQ" -n '{status: "pass", workload_exit_code: 0}' > "$PORTABLE_ROOT/workload-summary.json"
write_receipt success
