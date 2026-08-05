# r[impl tenstorrent.rwkv7_dual_p150a.collectives]
# Host-side orchestrator for the dual-P150A collective probe receipt.
# Runs the focused mesh/all-reduce/barrier probe in the pinned container,
# wraps the result into a receipt, and validates it with the admission
# core. Thin shell: admission decisions live in the core.
set -euo pipefail

CORE=@core@/bin/rwkv7-lora-dual-p150a-core
PLATFORM_CORE=@platform_core@/bin/training-platform-core
JQ=@jq@
DOCKER=@docker@
TTSMI=@ttsmi@
B3SUM=@b3sum@
FUSER=@fuser@

usage() {
  echo "usage: rwkv7-lora-dual-p150a-collective --profile PATH --manifest PATH --config PATH --descriptor PATH --wheels-dir PATH --evidence-root PATH" >&2
  exit 2
}

PROFILE=""
MANIFEST=""
CONFIG=""
DESCRIPTOR=""
WHEELS_DIR=""
EVIDENCE_ROOT=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --descriptor) DESCRIPTOR="$2"; shift 2 ;;
    --wheels-dir) WHEELS_DIR="$2"; shift 2 ;;
    --evidence-root) EVIDENCE_ROOT="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[ -n "$PROFILE" ] && [ -n "$MANIFEST" ] && [ -n "$CONFIG" ] && [ -n "$DESCRIPTOR" ] \
  && [ -n "$WHEELS_DIR" ] && [ -n "$EVIDENCE_ROOT" ] || usage

DESCRIPTOR=$(readlink -f "$DESCRIPTOR")

for variable in PYTHONPATH PYTHONHOME TT_METAL_HOME TT_METAL_RUNTIME_ROOT; do
  if [ -n "${!variable:-}" ]; then
    echo "rwkv7-lora-dual-p150a-collective: ambient $variable is set and would override the admitted bundle" >&2
    exit 1
  fi
done

mkdir -p "$EVIDENCE_ROOT"

"$CORE" admit-profile "$PROFILE" > "$EVIDENCE_ROOT/profile-admission.json"
PROFILE_IDENTITY=$("$JQ" -r '.identity_blake3' "$EVIDENCE_ROOT/profile-admission.json")
TIMEOUT_SECONDS=$("$JQ" -r '.collective_timeout_seconds' "$EVIDENCE_ROOT/profile-admission.json")

PLAN=$("$PLATFORM_CORE" launch-plan "$MANIFEST" "$CONFIG" "$DESCRIPTOR")
IMAGE=$(printf '%s' "$PLAN" | "$JQ" -r '.plan.image')
HUGEPAGE_MOUNT=$(printf '%s' "$PLAN" | "$JQ" -r '.plan.mounts[0].host')
DESCRIPTOR_CONTAINER=$(printf '%s' "$PLAN" | "$JQ" -r '.plan.mounts[1].container')
MESH_ENV_VALUE=$(printf '%s' "$PLAN" | "$JQ" -r '.plan.environment.TT_MESH_GRAPH_DESC_PATH')
BUNDLE_IDENTITY=$(printf '%s' "$PLAN" | "$JQ" -r '.plan.bundle_identity')
CONFIG_IDENTITY=$(printf '%s' "$PLAN" | "$JQ" -r '.plan.config_identity')
PLAN_IDENTITY=$(printf '%s' "$PLAN" | "$JQ" -r '.identity_blake3')

stage_status() { printf '{"name": "%s", "status": "%s"}' "$1" "$2"; }

finish_receipt() {
  local overall="$1"; shift
  local stages
  stages=$(printf '%s\n' "$@" | "$JQ" -s '.')
  local probe_result="{}"
  if [ -f "$EVIDENCE_ROOT/collective-result.json" ]; then
    probe_result=$(cat "$EVIDENCE_ROOT/collective-result.json")
  fi
  if ! printf '%s' "$probe_result" | "$JQ" -e . >/dev/null 2>&1; then
    probe_result="{}"
  fi
  local hlo_blake3=""
  if [ -f "$EVIDENCE_ROOT/collective.hlo" ]; then
    hlo_blake3=$("$B3SUM" "$EVIDENCE_ROOT/collective.hlo" | cut -d' ' -f1)
  fi
  "$JQ" -n \
    --argjson schema_version 1 \
    --arg kind "rwkv7_lora_dual_p150a_collective" \
    --arg profile_identity "$PROFILE_IDENTITY" \
    --arg bundle_identity "$BUNDLE_IDENTITY" \
    --arg config_identity "$CONFIG_IDENTITY" \
    --arg launch_plan_identity "$PLAN_IDENTITY" \
    --argjson stages "$stages" \
    --argjson probe_result "$probe_result" \
    --arg hlo_blake3 "$hlo_blake3" \
    '{
      schema_version: $schema_version,
      kind: $kind,
      profile_identity: $profile_identity,
      bundle_identity: $bundle_identity,
      config_identity: $config_identity,
      launch_plan_identity: $launch_plan_identity,
      stages: $stages,
      probe_result: $probe_result,
      hlo_blake3: $hlo_blake3,
      non_claims: [
        "no speedup claim",
        "no scaling efficiency claim",
        "no long-run stability claim",
        "no useful generalization claim",
        "no support claim for other RWKV models"
      ]
    }' > "$EVIDENCE_ROOT/receipt.json"
  if [ "$overall" -eq 0 ]; then
    "$CORE" admit-collective "$EVIDENCE_ROOT/receipt.json" "$PROFILE" \
      > "$EVIDENCE_ROOT/collective-admission.json"
  fi
  exit "$overall"
}

fail_at() {
  local failed="$1"; shift
  echo "rwkv7-lora-dual-p150a-collective: stage $failed failed" >&2
  finish_receipt 1 "$@" "$(stage_status "$failed" fail)"
}

"$TTSMI" -s > "$EVIDENCE_ROOT/health-before.json"
S_HEALTH_BEFORE=$(stage_status health_before pass)

for device_node in /dev/tenstorrent/[0-9]*; do
  if "$FUSER" "$device_node" >/dev/null 2>&1; then
    echo "rwkv7-lora-dual-p150a-collective: device $device_node is owned by another process" >&2
    fail_at ownership "$S_HEALTH_BEFORE"
  fi
done
if "$FUSER" "$HUGEPAGE_MOUNT" >/dev/null 2>&1; then
  echo "rwkv7-lora-dual-p150a-collective: hugepage mount $HUGEPAGE_MOUNT is in use" >&2
  fail_at ownership "$S_HEALTH_BEFORE"
fi
S_OWNERSHIP=$(stage_status ownership pass)

set +e
timeout "$TIMEOUT_SECONDS" "$DOCKER" run --rm \
  --device /dev/tenstorrent \
  -v "$HUGEPAGE_MOUNT:$HUGEPAGE_MOUNT" \
  -v "$WHEELS_DIR:/wheels:ro" \
  -v @collective_probe_py@:/probe/collective_probe.py:ro \
  -v @install_deps_py@:/probe/install_deps.py:ro \
  -v @container_collective@:/container-collective.sh:ro \
  -v "$PROFILE:/profile/profile.json:ro" \
  -v "$DESCRIPTOR:$DESCRIPTOR_CONTAINER:ro" \
  -e "TT_MESH_GRAPH_DESC_PATH=$MESH_ENV_VALUE" \
  -v "$EVIDENCE_ROOT:/evidence" \
  "$IMAGE" \
  bash /container-collective.sh \
    --profile /profile/profile.json \
  > "$EVIDENCE_ROOT/probe-stdout.log" 2> "$EVIDENCE_ROOT/probe-stderr.log"
PROBE_STATUS=$?
set -e

grep -E '^\{' "$EVIDENCE_ROOT/probe-stdout.log" | tail -1 > "$EVIDENCE_ROOT/collective-result.json" || true
if [ "$PROBE_STATUS" -ne 0 ] || ! grep -F '"status": "pass"' "$EVIDENCE_ROOT/collective-result.json" >/dev/null; then
  fail_at probe_execute "$S_HEALTH_BEFORE" "$S_OWNERSHIP"
fi
S_PROBE=$(stage_status probe_execute pass)

# Cleanup: the container is gone and no device node stays owned.
if "$DOCKER" ps -q --filter "ancestor=$IMAGE" | grep -q .; then
  fail_at cleanup "$S_HEALTH_BEFORE" "$S_OWNERSHIP" "$S_PROBE"
fi
for device_node in /dev/tenstorrent/[0-9]*; do
  if "$FUSER" "$device_node" >/dev/null 2>&1; then
    fail_at cleanup "$S_HEALTH_BEFORE" "$S_OWNERSHIP" "$S_PROBE"
  fi
done
S_CLEANUP=$(stage_status cleanup pass)

# Restoration: both devices answer a fresh topology snapshot.
if ! "$TTSMI" -s > "$EVIDENCE_ROOT/health-after.json"; then
  fail_at restore "$S_HEALTH_BEFORE" "$S_OWNERSHIP" "$S_PROBE" "$S_CLEANUP"
fi
S_RESTORE=$(stage_status restore pass)
S_HEALTH_AFTER=$(stage_status health_after pass)

finish_receipt 0 "$S_HEALTH_BEFORE" "$S_OWNERSHIP" "$S_PROBE" "$S_CLEANUP" "$S_RESTORE" "$S_HEALTH_AFTER"
