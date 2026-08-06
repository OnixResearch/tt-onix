#!/usr/bin/env bash
# shellcheck disable=SC2016 # jq filters use their own $variables.
# r[impl tenstorrent.rwkv7_p150a_finetuning.device]
# r[impl tenstorrent.rwkv7_p150a_finetuning.placement]
# r[impl tenstorrent.rwkv7_p150a_finetuning.parity]
# Host-side orchestrator for the full-model single-P150A step receipt.
# Runs the full RWKV7 LoRA graph in the pinned container, captures placement
# and parity evidence, wraps it into a receipt, and validates it with the
# admission core. Thin shell: admission decisions live in the core.
set -euo pipefail

CORE=@core@/bin/rwkv7-lora-p150a-core
PLATFORM_CORE=@platform_core@/bin/training-platform-core
JQ=@jq@
DOCKER=@docker@
TTSMI=@ttsmi@
B3SUM=@b3sum@
FUSER=@fuser@
PROBE_PY=@probe_py@
MODEL_DIR=@model_dir@
CONTAINER_SCRIPT=@container_script@
PLATFORM_AUTHORITY=@platform_authority@
WHEELS_DIR=@platform_wheelhouse@

usage() {
  echo "usage: rwkv7-lora-p150a-full-step --profile PATH --checkpoint PATH --evidence-root PATH [--dtype bf16|fp32]" >&2
  exit 2
}

PROFILE=""
CHECKPOINT=""
EVIDENCE_ROOT=""
DTYPE=""
MODE="receipt"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --evidence-root) EVIDENCE_ROOT="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[ -n "$PROFILE" ] && [ -n "$CHECKPOINT" ] && [ -n "$EVIDENCE_ROOT" ] || usage

DTYPE_ARGS=()
if [ -n "$DTYPE" ]; then
  case "$DTYPE" in
    bf16|fp32) DTYPE_ARGS=(--dtype "$DTYPE") ;;
    *) usage ;;
  esac
fi
case "$MODE" in
  receipt|multistep|checkpoint) ;;
  *) usage ;;
esac

CHECKPOINT=$(readlink -f "$CHECKPOINT")

for variable in PYTHONPATH PYTHONHOME TT_METAL_HOME TT_METAL_RUNTIME_ROOT; do
  if [ -n "${!variable:-}" ]; then
    echo "rwkv7-lora-p150a-full-step: ambient $variable is set and would override the admitted bundle" >&2
    exit 1
  fi
done

mkdir -p "$EVIDENCE_ROOT"

"$CORE" admit-profile "$PROFILE" > "$EVIDENCE_ROOT/profile-admission.json"
PROFILE_IDENTITY=$("$JQ" -r '.identity_blake3' "$EVIDENCE_ROOT/profile-admission.json")
TIMEOUT_SECONDS=$("$JQ" -r '.full_compile_timeout_seconds' "$EVIDENCE_ROOT/profile-admission.json")

PLAN=$("$PLATFORM_CORE" authorized-launch-plan "$PLATFORM_AUTHORITY")
IMAGE=$(printf '%s' "$PLAN" | "$JQ" -r '.plan.image')
HUGEPAGE_MOUNT=$(printf '%s' "$PLAN" | "$JQ" -r '.plan.mounts[0].host')
BUNDLE_IDENTITY=$(printf '%s' "$PLAN" | "$JQ" -r '.plan.bundle_identity')
CONFIG_IDENTITY=$(printf '%s' "$PLAN" | "$JQ" -r '.plan.config_identity')
PLAN_IDENTITY=$(printf '%s' "$PLAN" | "$JQ" -r '.identity_blake3')

stage_status() { printf '{"name": "%s", "status": "%s"}' "$1" "$2"; }

finish_receipt() {
  local overall="$1"; shift
  local stages
  stages=$(printf '%s\n' "$@" | "$JQ" -s '.')
  local probe_result="{}"
  if [ -f "$EVIDENCE_ROOT/step-result.json" ]; then
    probe_result=$(cat "$EVIDENCE_ROOT/step-result.json")
  fi
  if ! printf '%s' "$probe_result" | "$JQ" -e . >/dev/null 2>&1; then
    probe_result="{}"
  fi
  local placement_hlo_blake3=""
  if [ -f "$EVIDENCE_ROOT/placement.hlo" ]; then
    placement_hlo_blake3=$("$B3SUM" "$EVIDENCE_ROOT/placement.hlo" | cut -d' ' -f1)
  fi
  if [ "$MODE" = "multistep" ]; then
    local run
    run=$(printf '%s' "$probe_result" | "$JQ" '{steps, finite, base_unchanged, improved, initial_loss, final_loss, cpu_final_loss, improvement_ratio, losses, cpu_losses}')
    "$JQ" -n \
      --argjson schema_version 1 \
      --arg kind "rwkv7_lora_p150a_run" \
      --arg profile_identity "$PROFILE_IDENTITY" \
      --arg bundle_identity "$BUNDLE_IDENTITY" \
      --arg config_identity "$CONFIG_IDENTITY" \
      --arg launch_plan_identity "$PLAN_IDENTITY" \
      --argjson stages "$stages" \
      --argjson run "$run" \
      '{
        schema_version: $schema_version,
        kind: $kind,
        profile_identity: $profile_identity,
        bundle_identity: $bundle_identity,
        config_identity: $config_identity,
        launch_plan_identity: $launch_plan_identity,
        stages: $stages,
        run: $run,
        non_claims: [
          "no dual-card scaling claim",
          "no useful generalization claim",
          "no speedup claim",
          "no support claim for other RWKV models"
        ]
      }' > "$EVIDENCE_ROOT/receipt.json"
    if [ "$overall" -eq 0 ]; then
      "$CORE" admit-run "$EVIDENCE_ROOT/receipt.json" "$PROFILE" \
        > "$EVIDENCE_ROOT/run-admission.json"
    fi
  elif [ "$MODE" = "checkpoint" ]; then
    local checkpoint adapter_blake3
    checkpoint=$(printf '%s' "$probe_result" | "$JQ" '{pre_steps, post_steps, adapter_load_max_diff, final_loss_interrupted, final_loss_resumed, resume_loss_dev}')
    adapter_blake3=""
    if [ -f "$EVIDENCE_ROOT/adapter.safetensors" ]; then
      adapter_blake3=$("$B3SUM" "$EVIDENCE_ROOT/adapter.safetensors" | cut -d' ' -f1)
    fi
    "$JQ" -n \
      --argjson schema_version 1 \
      --arg kind "rwkv7_lora_p150a_checkpoint" \
      --arg profile_identity "$PROFILE_IDENTITY" \
      --arg bundle_identity "$BUNDLE_IDENTITY" \
      --arg config_identity "$CONFIG_IDENTITY" \
      --arg launch_plan_identity "$PLAN_IDENTITY" \
      --argjson stages "$stages" \
      --argjson checkpoint "$checkpoint" \
      --arg adapter_blake3 "$adapter_blake3" \
      '{
        schema_version: $schema_version,
        kind: $kind,
        profile_identity: $profile_identity,
        bundle_identity: $bundle_identity,
        config_identity: $config_identity,
        launch_plan_identity: $launch_plan_identity,
        stages: $stages,
        checkpoint: $checkpoint,
        adapter_blake3: $adapter_blake3,
        non_claims: [
          "no dual-card scaling claim",
          "no useful generalization claim",
          "no speedup claim",
          "no support claim for other RWKV models"
        ]
      }' > "$EVIDENCE_ROOT/receipt.json"
    if [ "$overall" -eq 0 ]; then
      "$CORE" admit-checkpoint "$EVIDENCE_ROOT/receipt.json" "$PROFILE" \
        > "$EVIDENCE_ROOT/checkpoint-admission.json"
    fi
  else
    local placement step parity
    placement=$(printf '%s' "$probe_result" | "$JQ" '.placement // {}')
    step=$(printf '%s' "$probe_result" | "$JQ" '.step // {}')
    parity=$(printf '%s' "$probe_result" | "$JQ" '.parity // {}')
    "$JQ" -n \
      --argjson schema_version 1 \
      --arg kind "rwkv7_lora_p150a_step" \
      --arg profile_identity "$PROFILE_IDENTITY" \
      --arg bundle_identity "$BUNDLE_IDENTITY" \
      --arg config_identity "$CONFIG_IDENTITY" \
      --arg launch_plan_identity "$PLAN_IDENTITY" \
      --argjson stages "$stages" \
      --argjson placement "$placement" \
      --argjson step "$step" \
      --argjson parity "$parity" \
      --arg placement_hlo_blake3 "$placement_hlo_blake3" \
      '{
        schema_version: $schema_version,
        kind: $kind,
        profile_identity: $profile_identity,
        bundle_identity: $bundle_identity,
        config_identity: $config_identity,
        launch_plan_identity: $launch_plan_identity,
        stages: $stages,
        placement: $placement,
        step: $step,
        parity: $parity,
        placement_hlo_blake3: $placement_hlo_blake3,
        non_claims: [
          "no dual-card scaling claim",
          "no useful generalization claim",
          "no speedup claim",
          "no support claim for other RWKV models"
        ]
      }' > "$EVIDENCE_ROOT/receipt.json"
    if [ "$overall" -eq 0 ]; then
      "$CORE" admit-step "$EVIDENCE_ROOT/receipt.json" "$PROFILE" \
        > "$EVIDENCE_ROOT/step-admission.json"
    fi
  fi
  exit "$overall"
}

fail_at() {
  local failed="$1"; shift
  echo "rwkv7-lora-p150a-full-step: stage $failed failed" >&2
  finish_receipt 1 "$@" "$(stage_status "$failed" fail)"
}

"$TTSMI" -s > "$EVIDENCE_ROOT/health-before.json"
S_HEALTH_BEFORE=$(stage_status health_before pass)

for device_node in /dev/tenstorrent/[0-9]*; do
  if "$FUSER" "$device_node" >/dev/null 2>&1; then
    echo "rwkv7-lora-p150a-full-step: device $device_node is owned by another process" >&2
    fail_at ownership "$S_HEALTH_BEFORE"
  fi
done
if "$FUSER" "$HUGEPAGE_MOUNT" >/dev/null 2>&1; then
  echo "rwkv7-lora-p150a-full-step: hugepage mount $HUGEPAGE_MOUNT is in use" >&2
  fail_at ownership "$S_HEALTH_BEFORE"
fi
S_OWNERSHIP=$(stage_status ownership pass)

set +e
timeout "$TIMEOUT_SECONDS" "$DOCKER" run --rm \
  --device /dev/tenstorrent \
  -v "$HUGEPAGE_MOUNT:$HUGEPAGE_MOUNT" \
  -v "$WHEELS_DIR:/wheels:ro" \
  -v "$MODEL_DIR:/model:ro" \
  -v "$PROBE_PY:/probe/full_model_probe.py:ro" \
  -v @install_deps_py@:/probe/install_deps.py:ro \
  -v "$CONTAINER_SCRIPT:/container-full-model.sh:ro" \
  -v "$CHECKPOINT:/ckpt/model.safetensors:ro" \
  -v "$PROFILE:/profile/profile.json:ro" \
  -v "$EVIDENCE_ROOT:/evidence" \
  "$IMAGE" \
  bash /container-full-model.sh \
    --profile /profile/profile.json \
    --checkpoint /ckpt/model.safetensors \
    --mode "$MODE" "${DTYPE_ARGS[@]}" \
  > "$EVIDENCE_ROOT/step-stdout.log" 2> "$EVIDENCE_ROOT/step-stderr.log"
STEP_STATUS=$?
set -e

grep -E '^\{' "$EVIDENCE_ROOT/step-stdout.log" | tail -1 > "$EVIDENCE_ROOT/step-result.json" || true
if [ "$STEP_STATUS" -ne 0 ] || ! grep -F '"status": "pass"' "$EVIDENCE_ROOT/step-result.json" >/dev/null; then
  fail_at step_execute "$S_HEALTH_BEFORE" "$S_OWNERSHIP"
fi
S_STEP=$(stage_status step_execute pass)

for device_node in /dev/tenstorrent/[0-9]*; do
  if "$FUSER" "$device_node" >/dev/null 2>&1; then
    fail_at restore "$S_HEALTH_BEFORE" "$S_OWNERSHIP" "$S_STEP"
  fi
done
S_RESTORE=$(stage_status restore pass)

"$TTSMI" -s > "$EVIDENCE_ROOT/health-after.json"
S_HEALTH_AFTER=$(stage_status health_after pass)

finish_receipt 0 "$S_HEALTH_BEFORE" "$S_OWNERSHIP" "$S_STEP" "$S_RESTORE" "$S_HEALTH_AFTER"
