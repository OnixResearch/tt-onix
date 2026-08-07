#!/usr/bin/env bash
# r[impl tenstorrent.rwkv7_dual_p150a.graph]
# r[impl tenstorrent.rwkv7_dual_p150a.sharding]
# r[impl tenstorrent.rwkv7_dual_p150a.parity]
# Host-side orchestrator for the dual-P150A data-parallel receipts. Runs the
# dual-replica graph in the pinned container, plans the global-batch shards
# with the admission core, wraps placement, shard, optimizer, replica, and
# parity evidence into a receipt, and validates it with the core. Thin
# shell: admission decisions live in the core.
set -euo pipefail

CORE=@core@/bin/rwkv7-lora-dual-p150a-core
PLATFORM_CORE=@platform_core@/bin/training-platform-core
JQ=@jq@
DOCKER=@docker@
TTSMI=@ttsmi@
B3SUM=@b3sum@
FUSER=@fuser@

usage() {
  echo "usage: rwkv7-lora-dual-p150a-full-step --profile PATH --manifest PATH --config PATH --descriptor PATH --checkpoint PATH --wheels-dir PATH --evidence-root PATH [--dtype bf16|fp32] [--mode receipt|multistep|checkpoint] [--parity-receipt PATH]" >&2
  exit 2
}

PROFILE=""
MANIFEST=""
CONFIG=""
DESCRIPTOR=""
CHECKPOINT=""
WHEELS_DIR=""
EVIDENCE_ROOT=""
DTYPE=""
MODE="receipt"
PARITY_RECEIPT=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --descriptor) DESCRIPTOR="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --wheels-dir) WHEELS_DIR="$2"; shift 2 ;;
    --evidence-root) EVIDENCE_ROOT="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --parity-receipt) PARITY_RECEIPT="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[ -n "$PROFILE" ] && [ -n "$MANIFEST" ] && [ -n "$CONFIG" ] && [ -n "$DESCRIPTOR" ] \
  && [ -n "$CHECKPOINT" ] && [ -n "$WHEELS_DIR" ] && [ -n "$EVIDENCE_ROOT" ] || usage

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
if [ "$MODE" = "multistep" ] && [ -z "$PARITY_RECEIPT" ]; then
  echo "rwkv7-lora-dual-p150a-full-step: multistep mode requires --parity-receipt from a passing receipt-mode run" >&2
  usage
fi

DESCRIPTOR=$(readlink -f "$DESCRIPTOR")
CHECKPOINT=$(readlink -f "$CHECKPOINT")

for variable in PYTHONPATH PYTHONHOME TT_METAL_HOME TT_METAL_RUNTIME_ROOT; do
  if [ -n "${!variable:-}" ]; then
    echo "rwkv7-lora-dual-p150a-full-step: ambient $variable is set and would override the admitted bundle" >&2
    exit 1
  fi
done

mkdir -p "$EVIDENCE_ROOT"

"$CORE" admit-profile "$PROFILE" > "$EVIDENCE_ROOT/profile-admission.json"
PROFILE_IDENTITY=$("$JQ" -r '.identity_blake3' "$EVIDENCE_ROOT/profile-admission.json")
TIMEOUT_SECONDS=$("$JQ" -r '.full_compile_timeout_seconds' "$EVIDENCE_ROOT/profile-admission.json")
GLOBAL_BATCH=$("$JQ" -r '.global_batch' "$EVIDENCE_ROOT/profile-admission.json")
SEQUENCE_LENGTH=$("$JQ" -r '.sequence_length' "$EVIDENCE_ROOT/profile-admission.json")

# The global batch is a pure function of the admitted profile: even
# contiguous shards, sample-i in global order, sequence_length active
# labels per sample. The container writes the same document; the runner
# compares them after the run.
"$JQ" -n \
  --argjson samples "$GLOBAL_BATCH" \
  --argjson tokens "$SEQUENCE_LENGTH" \
  '{
    global_batch_index: 0,
    samples: [range(0; $samples) | {sample_id: "sample-\(.)", active_labels: $tokens}]
  }' > "$EVIDENCE_ROOT/batch-expected.json"
"$CORE" plan-shards "$PROFILE" "$EVIDENCE_ROOT/batch-expected.json" \
  > "$EVIDENCE_ROOT/shard-plan.json"

# The bounded run is admitted only after the parity gate passes. Admit the
# parity step receipt and bind its admission identity.
STEP_IDENTITY=""
if [ "$MODE" = "multistep" ]; then
  "$CORE" admit-step "$PARITY_RECEIPT" "$PROFILE" \
    > "$EVIDENCE_ROOT/parity-admission.json"
  STEP_IDENTITY=$("$JQ" -r '.identity_blake3' "$EVIDENCE_ROOT/parity-admission.json")
fi

PLAN=$("$PLATFORM_CORE" launch-plan "$MANIFEST" "$CONFIG" "$DESCRIPTOR")
IMAGE=$(printf '%s' "$PLAN" | "$JQ" -r '.plan.image')
HUGEPAGE_MOUNT=$(printf '%s' "$PLAN" | "$JQ" -r '.plan.mounts[0].host')
DESCRIPTOR_CONTAINER=$(printf '%s' "$PLAN" | "$JQ" -r '.plan.mounts[1].container')
MESH_ENV_VALUE=$(printf '%s' "$PLAN" | "$JQ" -r '.plan.environment.TT_MESH_GRAPH_DESC_PATH')
BUNDLE_IDENTITY=$(printf '%s' "$PLAN" | "$JQ" -r '.plan.bundle_identity')
CONFIG_IDENTITY=$(printf '%s' "$PLAN" | "$JQ" -r '.plan.config_identity')
PLAN_IDENTITY=$(printf '%s' "$PLAN" | "$JQ" -r '.identity_blake3')

stage_status() { printf '{"name": "%s", "status": "%s"}' "$1" "$2"; }

# The receipt shard plan: the core-computed plan document plus its
# identity, with the shard kind from the admitted profile.
shard_plan_block() {
  "$JQ" -n \
    --arg identity "$("$JQ" -r '.identity_blake3' "$EVIDENCE_ROOT/shard-plan.json")" \
    --arg kind "even_contiguous" \
    --argjson replicas "$("$JQ" '.replicas' "$EVIDENCE_ROOT/shard-plan.json")" \
    '{identity_blake3: $identity, shard_kind: $kind, replicas: $replicas}'
}

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
  local non_claims='[
    "no speedup claim",
    "no scaling efficiency claim",
    "no long-run stability claim",
    "no useful generalization claim",
    "no support claim for other RWKV models"
  ]'
  if [ "$MODE" = "multistep" ]; then
    local run shard_plan
    run=$(printf '%s' "$probe_result" | "$JQ" '{steps, finite, base_unchanged, improved, initial_loss, final_loss, cpu_final_loss, improvement_ratio, losses, cpu_losses, gradient_reductions, per_device, replica_boundaries}')
    shard_plan=$(shard_plan_block)
    "$JQ" -n \
      --argjson schema_version 1 \
      --arg kind "rwkv7_lora_dual_p150a_run" \
      --arg profile_identity "$PROFILE_IDENTITY" \
      --arg bundle_identity "$BUNDLE_IDENTITY" \
      --arg config_identity "$CONFIG_IDENTITY" \
      --arg launch_plan_identity "$PLAN_IDENTITY" \
      --argjson stages "$stages" \
      --argjson shard_plan "$shard_plan" \
      --arg step_identity "$STEP_IDENTITY" \
      --argjson run "$run" \
      --argjson non_claims "$non_claims" \
      '{
        schema_version: $schema_version,
        kind: $kind,
        profile_identity: $profile_identity,
        bundle_identity: $bundle_identity,
        config_identity: $config_identity,
        launch_plan_identity: $launch_plan_identity,
        stages: $stages,
        shard_plan: $shard_plan,
        step_identity: $step_identity,
        run: $run,
        non_claims: $non_claims
      }' > "$EVIDENCE_ROOT/receipt.json"
    if [ "$overall" -eq 0 ]; then
      "$CORE" admit-run "$EVIDENCE_ROOT/receipt.json" "$PROFILE" \
        > "$EVIDENCE_ROOT/run-admission.json"
    fi
  elif [ "$MODE" = "checkpoint" ]; then
    local checkpoint adapter_blake3
    checkpoint=$(printf '%s' "$probe_result" | "$JQ" '{pre_steps, post_steps, replica_equality_before_save, canonical_single_artifact, adapter_load_max_diff, replicas_loaded, post_load_replica_equality, final_loss_interrupted, final_loss_resumed, resume_loss_dev}')
    adapter_blake3=""
    if [ -f "$EVIDENCE_ROOT/adapter.safetensors" ]; then
      adapter_blake3=$("$B3SUM" "$EVIDENCE_ROOT/adapter.safetensors" | cut -d' ' -f1)
    fi
    "$JQ" -n \
      --argjson schema_version 1 \
      --arg kind "rwkv7_lora_dual_p150a_checkpoint" \
      --arg profile_identity "$PROFILE_IDENTITY" \
      --arg bundle_identity "$BUNDLE_IDENTITY" \
      --arg config_identity "$CONFIG_IDENTITY" \
      --arg launch_plan_identity "$PLAN_IDENTITY" \
      --argjson stages "$stages" \
      --argjson checkpoint "$checkpoint" \
      --arg adapter_blake3 "$adapter_blake3" \
      --argjson non_claims "$non_claims" \
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
        non_claims: $non_claims
      }' > "$EVIDENCE_ROOT/receipt.json"
    if [ "$overall" -eq 0 ]; then
      "$CORE" admit-checkpoint "$EVIDENCE_ROOT/receipt.json" "$PROFILE" \
        > "$EVIDENCE_ROOT/checkpoint-admission.json"
    fi
  else
    local placement step parity optimizer replicas shard_plan
    placement=$(printf '%s' "$probe_result" | "$JQ" '.placement // {}')
    step=$(printf '%s' "$probe_result" | "$JQ" '.step // {}')
    parity=$(printf '%s' "$probe_result" | "$JQ" '.parity // {}')
    optimizer=$(printf '%s' "$probe_result" | "$JQ" '.optimizer // {}')
    replicas=$(printf '%s' "$probe_result" | "$JQ" '.replicas // {}')
    shard_plan=$(shard_plan_block)
    "$JQ" -n \
      --argjson schema_version 1 \
      --arg kind "rwkv7_lora_dual_p150a_step" \
      --arg profile_identity "$PROFILE_IDENTITY" \
      --arg bundle_identity "$BUNDLE_IDENTITY" \
      --arg config_identity "$CONFIG_IDENTITY" \
      --arg launch_plan_identity "$PLAN_IDENTITY" \
      --argjson stages "$stages" \
      --argjson placement "$placement" \
      --argjson shard_plan "$shard_plan" \
      --argjson optimizer "$optimizer" \
      --argjson replicas "$replicas" \
      --argjson step "$step" \
      --argjson parity "$parity" \
      --arg placement_hlo_blake3 "$placement_hlo_blake3" \
      --argjson non_claims "$non_claims" \
      '{
        schema_version: $schema_version,
        kind: $kind,
        profile_identity: $profile_identity,
        bundle_identity: $bundle_identity,
        config_identity: $config_identity,
        launch_plan_identity: $launch_plan_identity,
        stages: $stages,
        placement: $placement,
        shard_plan: $shard_plan,
        optimizer: $optimizer,
        replicas: $replicas,
        step: $step,
        parity: $parity,
        placement_hlo_blake3: $placement_hlo_blake3,
        non_claims: $non_claims
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
  echo "rwkv7-lora-dual-p150a-full-step: stage $failed failed" >&2
  finish_receipt 1 "$@" "$(stage_status "$failed" fail)"
}

"$TTSMI" -s > "$EVIDENCE_ROOT/health-before.json"
S_HEALTH_BEFORE=$(stage_status health_before pass)

for device_node in /dev/tenstorrent/[0-9]*; do
  if "$FUSER" "$device_node" >/dev/null 2>&1; then
    echo "rwkv7-lora-dual-p150a-full-step: device $device_node is owned by another process" >&2
    fail_at ownership "$S_HEALTH_BEFORE"
  fi
done
if "$FUSER" "$HUGEPAGE_MOUNT" >/dev/null 2>&1; then
  echo "rwkv7-lora-dual-p150a-full-step: hugepage mount $HUGEPAGE_MOUNT is in use" >&2
  fail_at ownership "$S_HEALTH_BEFORE"
fi
S_OWNERSHIP=$(stage_status ownership pass)

set +e
timeout "$TIMEOUT_SECONDS" "$DOCKER" run --rm \
  --device /dev/tenstorrent \
  -v "$HUGEPAGE_MOUNT:$HUGEPAGE_MOUNT" \
  -v "$WHEELS_DIR:/wheels:ro" \
  -v @model_dir@:/model:ro \
  -v @dual_probe_py@:/probe/full_model_dual.py:ro \
  -v @install_deps_py@:/probe/install_deps.py:ro \
  -v @container_script@:/container-dual-model.sh:ro \
  -v "$CHECKPOINT:/ckpt/model.safetensors:ro" \
  -v "$PROFILE:/profile/profile.json:ro" \
  -v "$DESCRIPTOR:$DESCRIPTOR_CONTAINER:ro" \
  -e "TT_MESH_GRAPH_DESC_PATH=$MESH_ENV_VALUE" \
  -v "$EVIDENCE_ROOT:/evidence" \
  "$IMAGE" \
  bash /container-dual-model.sh \
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
# The container's shard-plan input must equal the runner's planned batch.
if [ -f "$EVIDENCE_ROOT/batch.json" ]; then
  if ! cmp -s <("$JQ" -S . "$EVIDENCE_ROOT/batch.json") <("$JQ" -S . "$EVIDENCE_ROOT/batch-expected.json"); then
    echo "rwkv7-lora-dual-p150a-full-step: container batch differs from the admitted shard plan" >&2
    fail_at step_execute "$S_HEALTH_BEFORE" "$S_OWNERSHIP"
  fi
fi
S_STEP=$(stage_status step_execute pass)

# Cleanup: the container is gone and no device node stays owned.
if "$DOCKER" ps -q --filter "ancestor=$IMAGE" | grep -q .; then
  fail_at cleanup "$S_HEALTH_BEFORE" "$S_OWNERSHIP" "$S_STEP"
fi
for device_node in /dev/tenstorrent/[0-9]*; do
  if "$FUSER" "$device_node" >/dev/null 2>&1; then
    fail_at cleanup "$S_HEALTH_BEFORE" "$S_OWNERSHIP" "$S_STEP"
  fi
done
S_CLEANUP=$(stage_status cleanup pass)

# Restoration: both devices answer a fresh topology snapshot.
if ! "$TTSMI" -s > "$EVIDENCE_ROOT/health-after.json"; then
  fail_at restore "$S_HEALTH_BEFORE" "$S_OWNERSHIP" "$S_STEP" "$S_CLEANUP"
fi
S_RESTORE=$(stage_status restore pass)
S_HEALTH_AFTER=$(stage_status health_after pass)

finish_receipt 0 "$S_HEALTH_BEFORE" "$S_OWNERSHIP" "$S_STEP" "$S_CLEANUP" "$S_RESTORE" "$S_HEALTH_AFTER"
