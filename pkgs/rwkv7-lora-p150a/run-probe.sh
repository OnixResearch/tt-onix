# r[impl tenstorrent.rwkv7_p150a_finetuning.compiler]
# r[impl tenstorrent.rwkv7_p150a_finetuning.device]
# Host-side orchestrator for one focused single-device compile probe.
# Thin shell: admission decisions live in rwkv7-lora-p150a-core and the
# platform core. This script captures evidence and enforces timeouts.
set -euo pipefail

CORE=@core@/bin/rwkv7-lora-p150a-core
PLATFORM_CORE=@platform_core@/bin/training-platform-core
WHEELS_JSON=@wheels_json@
JQ=@jq@
DOCKER=@docker@
TTSMI=@ttsmi@
B3SUM=@b3sum@
FUSER=@fuser@

usage() {
  echo "usage: rwkv7-lora-p150a-probe --profile PATH --manifest PATH --config PATH --descriptor PATH --wheels-dir PATH --evidence-root PATH --probe NAME --shape-kind KIND" >&2
  exit 2
}

PROFILE=""
MANIFEST=""
WHEELS_JSON_OVERRIDE=""
CONFIG=""
DESCRIPTOR=""
WHEELS_DIR=""
EVIDENCE_ROOT=""
PROBE=""
SHAPE_KIND="focused"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --descriptor) DESCRIPTOR="$2"; shift 2 ;;
    --wheels-dir) WHEELS_DIR="$2"; shift 2 ;;
    --evidence-root) EVIDENCE_ROOT="$2"; shift 2 ;;
    --probe) PROBE="$2"; shift 2 ;;
    --shape-kind) SHAPE_KIND="$2"; shift 2 ;;
    --wheels-json) WHEELS_JSON_OVERRIDE="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[ -n "$PROFILE" ] && [ -n "$MANIFEST" ] && [ -n "$CONFIG" ] && [ -n "$DESCRIPTOR" ] \
  && [ -n "$WHEELS_DIR" ] && [ -n "$EVIDENCE_ROOT" ] && [ -n "$PROBE" ] || usage
if [ -n "$WHEELS_JSON_OVERRIDE" ]; then
  WHEELS_JSON="$WHEELS_JSON_OVERRIDE"
fi

case "$PROBE" in
  wkv7-forward|wkv7-backward|training-ops) ;;
  *) usage ;;
esac
case "$SHAPE_KIND" in
  focused|representative) ;;
  *) usage ;;
esac

DESCRIPTOR=$(readlink -f "$DESCRIPTOR")

for variable in PYTHONPATH PYTHONHOME TT_METAL_HOME TT_METAL_RUNTIME_ROOT; do
  if [ -n "${!variable:-}" ]; then
    echo "rwkv7-lora-p150a-probe: ambient $variable is set and would override the admitted bundle" >&2
    exit 1
  fi
done

mkdir -p "$EVIDENCE_ROOT"

# Admission gates before any device access.
"$CORE" admit-profile "$PROFILE" > "$EVIDENCE_ROOT/profile-admission.json"
PROFILE_IDENTITY=$("$JQ" -r '.identity_blake3' "$EVIDENCE_ROOT/profile-admission.json")
TIMEOUT_SECONDS=$("$JQ" -r '.probe_compile_timeout_seconds' "$EVIDENCE_ROOT/profile-admission.json")
MAX_STABLEHLO_OPERATIONS=$("$JQ" -r '.max_stablehlo_operations' "$EVIDENCE_ROOT/profile-admission.json")

# The launch plan binds the admitted image identity. The single-device
# profile deliberately does not export TT_MESH_GRAPH_DESC_PATH: one logical
# 1x1 device per card is the admitted selection, and a 1x2 mesh descriptor
# would fuse both cards into one logical device.
PLAN=$("$PLATFORM_CORE" launch-plan "$MANIFEST" "$CONFIG" "$DESCRIPTOR")
IMAGE=$(printf '%s' "$PLAN" | "$JQ" -r '.plan.image')
HUGEPAGE_MOUNT=$(printf '%s' "$PLAN" | "$JQ" -r '.plan.mounts[0].host')
BUNDLE_IDENTITY=$(printf '%s' "$PLAN" | "$JQ" -r '.plan.bundle_identity')
CONFIG_IDENTITY=$(printf '%s' "$PLAN" | "$JQ" -r '.plan.config_identity')
PLAN_IDENTITY=$(printf '%s' "$PLAN" | "$JQ" -r '.identity_blake3')

# Wheel admission: every wheel file must match a manifest sha256.
check_wheel() {
  local package="$1"; local version="$2"; local sha="$3"
  local match
  match=$(find "$WHEELS_DIR" -maxdepth 1 -name "${package}-${version}-*.whl" | head -1)
  if [ -z "$match" ]; then
    echo "rwkv7-lora-p150a-probe: missing wheel for ${package} ${version}" >&2
    exit 1
  fi
  echo "${sha}  ${match}" | sha256sum -c --quiet || {
    echo "rwkv7-lora-p150a-probe: wheel ${match} fails the manifest sha256" >&2
    exit 1
  }
}
check_wheel pjrt_plugin_tt "$("$JQ" -r '.pjrt_plugin.version' "$MANIFEST")" \
  "$("$JQ" -r '.pjrt_plugin.wheel_sha256' "$MANIFEST")"
check_wheel torch "$("$JQ" -r '.pytorch.version' "$MANIFEST")" \
  "$("$JQ" -r '.pytorch.wheel_sha256' "$MANIFEST")"
check_wheel torch_xla "$("$JQ" -r '.pytorch_xla.version' "$MANIFEST")" \
  "$("$JQ" -r '.pytorch_xla.wheel_sha256' "$MANIFEST")"
check_wheel tt_lang "$("$JQ" -r '.tt_lang.version' "$WHEELS_JSON")" \
  "$("$JQ" -r '.tt_lang.wheel_sha256' "$WHEELS_JSON")"

stage_status() { printf '{"name": "%s", "status": "%s"}' "$1" "$2"; }

finish_receipt() {
  local overall="$1"; shift
  local stages
  stages=$(printf '%s\n' "$@" | "$JQ" -s '.')
  local probe_result="{}"
  if [ -f "$EVIDENCE_ROOT/probe-result.json" ]; then
    probe_result=$(cat "$EVIDENCE_ROOT/probe-result.json")
  fi
  if ! printf '%s' "$probe_result" | "$JQ" -e . >/dev/null 2>&1; then
    probe_result="{}"
  fi
  local hlo_blake3=""
  if [ -f "$EVIDENCE_ROOT/probe.hlo" ]; then
    hlo_blake3=$("$B3SUM" "$EVIDENCE_ROOT/probe.hlo" | cut -d' ' -f1)
  fi
  local pip_freeze_blake3=""
  if [ -f "$EVIDENCE_ROOT/pip-freeze.txt" ]; then
    pip_freeze_blake3=$("$B3SUM" "$EVIDENCE_ROOT/pip-freeze.txt" | cut -d' ' -f1)
  fi
  "$JQ" -n \
    --argjson schema_version 1 \
    --arg kind "rwkv7_lora_p150a_probe" \
    --arg probe "$PROBE" \
    --arg shape_kind "$SHAPE_KIND" \
    --arg profile_identity "$PROFILE_IDENTITY" \
    --arg bundle_identity "$BUNDLE_IDENTITY" \
    --arg config_identity "$CONFIG_IDENTITY" \
    --arg launch_plan_identity "$PLAN_IDENTITY" \
    --argjson stages "$stages" \
    --argjson probe_result "$probe_result" \
    --arg hlo_blake3 "$hlo_blake3" \
    --arg pip_freeze_blake3 "$pip_freeze_blake3" \
    --argjson max_stablehlo_operations "$MAX_STABLEHLO_OPERATIONS" \
    '{
      schema_version: $schema_version,
      kind: $kind,
      probe: $probe,
      shape_kind: $shape_kind,
      profile_identity: $profile_identity,
      bundle_identity: $bundle_identity,
      config_identity: $config_identity,
      launch_plan_identity: $launch_plan_identity,
      stages: $stages,
      probe_result: $probe_result,
      hlo_blake3: $hlo_blake3,
      pip_freeze_blake3: $pip_freeze_blake3,
      max_stablehlo_operations: $max_stablehlo_operations,
      non_claims: [
        "no full model support claim",
        "no dual-card scaling claim",
        "no speedup claim"
      ]
    }' > "$EVIDENCE_ROOT/receipt.json"
  if [ "$overall" -eq 0 ]; then
    # A passing probe still needs core admission before it becomes evidence.
    "$CORE" admit-probe "$EVIDENCE_ROOT/receipt.json" "$PROFILE" \
      > "$EVIDENCE_ROOT/probe-admission.json"
  fi
  exit "$overall"
}

fail_at() {
  local failed="$1"; shift
  echo "rwkv7-lora-p150a-probe: stage $failed failed" >&2
  finish_receipt 1 "$@" "$(stage_status "$failed" fail)"
}

# Stage health_before and ownership preflight.
"$TTSMI" -s > "$EVIDENCE_ROOT/health-before.json"
S_HEALTH_BEFORE=$(stage_status health_before pass)

for device_node in /dev/tenstorrent/[0-9]*; do
  if "$FUSER" "$device_node" >/dev/null 2>&1; then
    echo "rwkv7-lora-p150a-probe: device $device_node is owned by another process" >&2
    fail_at ownership "$S_HEALTH_BEFORE"
  fi
done
if "$FUSER" "$HUGEPAGE_MOUNT" >/dev/null 2>&1; then
  echo "rwkv7-lora-p150a-probe: hugepage mount $HUGEPAGE_MOUNT is in use" >&2
  fail_at ownership "$S_HEALTH_BEFORE"
fi
S_OWNERSHIP=$(stage_status ownership pass)

# Stage probe_execute: one bounded container run.
set +e
timeout "$TIMEOUT_SECONDS" "$DOCKER" run --rm \
  --device /dev/tenstorrent \
  -v "$HUGEPAGE_MOUNT:$HUGEPAGE_MOUNT" \
  -v "$WHEELS_DIR:/wheels:ro" \
  -v @probe_py@:/probe/wkv7_probe.py:ro \
  -v @install_deps_py@:/probe/install_deps.py:ro \
  -v @container_probe@:/probe/container-probe.sh:ro \
  -v "$PROFILE:/probe/profile.json:ro" \
  -v "$EVIDENCE_ROOT:/evidence" \
  "$IMAGE" \
  bash /probe/container-probe.sh \
    --probe "$PROBE" \
    --shape-kind "$SHAPE_KIND" \
    --profile /probe/profile.json \
    --hlo /evidence/probe.hlo \
  > "$EVIDENCE_ROOT/probe-stdout.log" 2> "$EVIDENCE_ROOT/probe-stderr.log"
PROBE_STATUS=$?
set -e

# The probe prints exactly one JSON result line.
grep -E '^\{' "$EVIDENCE_ROOT/probe-stdout.log" | tail -1 > "$EVIDENCE_ROOT/probe-result.json" || true
if [ "$PROBE_STATUS" -ne 0 ] || ! grep -F '"status": "pass"' "$EVIDENCE_ROOT/probe-result.json" >/dev/null; then
  fail_at probe_execute "$S_HEALTH_BEFORE" "$S_OWNERSHIP"
fi
S_PROBE=$(stage_status probe_execute pass)

# Stage restore: no process may own a device after the run.
for device_node in /dev/tenstorrent/[0-9]*; do
  if "$FUSER" "$device_node" >/dev/null 2>&1; then
    fail_at restore "$S_HEALTH_BEFORE" "$S_OWNERSHIP" "$S_PROBE"
  fi
done
S_RESTORE=$(stage_status restore pass)

"$TTSMI" -s > "$EVIDENCE_ROOT/health-after.json"
S_HEALTH_AFTER=$(stage_status health_after pass)

finish_receipt 0 "$S_HEALTH_BEFORE" "$S_OWNERSHIP" "$S_PROBE" "$S_RESTORE" "$S_HEALTH_AFTER"
