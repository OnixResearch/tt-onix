#!/usr/bin/env bash
# shellcheck disable=SC2016 # jq filters use their own $variables.
# r[impl tenstorrent.rwkv7_p150a_finetuning.device]
# r[impl tenstorrent.rwkv7_p150a_finetuning.placement]
# r[impl tenstorrent.rwkv7_p150a_finetuning.parity]
# r[impl tenstorrent.harness.regression_gate]
# r[impl tenstorrent.harness.determinism]
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
DATE=@date@
MV=@mv@
PROBE_PY=@probe_py@
MODEL_DIR=@model_dir@
CONTAINER_SCRIPT=@container_script@
PLATFORM_AUTHORITY=@platform_authority@
WHEELS_DIR=@platform_wheelhouse@
NANOSECONDS_PER_SECOND=1000000000
REGRESSION_INPUT_TOTAL=3

usage() {
  echo "usage: rwkv7-lora-p150a-full-step --profile PATH --checkpoint PATH --evidence-root PATH [--dtype bf16|fp32] [--mode receipt|measurement|multistep|checkpoint] [--machine-facts PATH] [--device-telemetry PATH] [--regression-baseline PATH --regression-policy PATH --determinism-receipt PATH]" >&2
  exit 2
}

PROFILE=""
CHECKPOINT=""
EVIDENCE_ROOT=""
DTYPE=""
MODE="receipt"
MACHINE_FACTS=""
DEVICE_TELEMETRY=""
REGRESSION_BASELINE=""
REGRESSION_POLICY=""
DETERMINISM_RECEIPT=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --evidence-root) EVIDENCE_ROOT="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --machine-facts) MACHINE_FACTS="$2"; shift 2 ;;
    --device-telemetry) DEVICE_TELEMETRY="$2"; shift 2 ;;
    --regression-baseline) REGRESSION_BASELINE="$2"; shift 2 ;;
    --regression-policy) REGRESSION_POLICY="$2"; shift 2 ;;
    --determinism-receipt) DETERMINISM_RECEIPT="$2"; shift 2 ;;
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
  receipt|measurement|multistep|checkpoint) ;;
  *) usage ;;
esac
if [ "$MODE" = "measurement" ] && { [ -z "$MACHINE_FACTS" ] || [ -z "$DEVICE_TELEMETRY" ]; }; then
  echo "rwkv7-lora-p150a-full-step: measurement mode requires --machine-facts and --device-telemetry" >&2
  usage
fi
REGRESSION_INPUT_COUNT=0
for regression_input in "$REGRESSION_BASELINE" "$REGRESSION_POLICY" "$DETERMINISM_RECEIPT"; do
  if [ -n "$regression_input" ]; then
    REGRESSION_INPUT_COUNT=$((REGRESSION_INPUT_COUNT + 1))
  fi
done
if [ "$REGRESSION_INPUT_COUNT" -ne 0 ] && [ "$REGRESSION_INPUT_COUNT" -ne "$REGRESSION_INPUT_TOTAL" ]; then
  echo "rwkv7-lora-p150a-full-step: regression gating requires baseline, policy, and determinism receipt" >&2
  usage
fi
if [ "$REGRESSION_INPUT_COUNT" -eq "$REGRESSION_INPUT_TOTAL" ] && [ "$MODE" != "measurement" ]; then
  echo "rwkv7-lora-p150a-full-step: regression gating is available only in measurement mode" >&2
  usage
fi

CHECKPOINT=$(readlink -f "$CHECKPOINT")
if [ -n "$MACHINE_FACTS" ]; then
  MACHINE_FACTS=$(readlink -f "$MACHINE_FACTS")
fi
if [ -n "$DEVICE_TELEMETRY" ]; then
  DEVICE_TELEMETRY=$(readlink -f "$DEVICE_TELEMETRY")
fi
if [ "$REGRESSION_INPUT_COUNT" -eq "$REGRESSION_INPUT_TOTAL" ]; then
  REGRESSION_BASELINE=$(readlink -f "$REGRESSION_BASELINE")
  REGRESSION_POLICY=$(readlink -f "$REGRESSION_POLICY")
  DETERMINISM_RECEIPT=$(readlink -f "$DETERMINISM_RECEIPT")
fi

for variable in PYTHONPATH PYTHONHOME TT_METAL_HOME TT_METAL_RUNTIME_ROOT; do
  if [ -n "${!variable:-}" ]; then
    echo "rwkv7-lora-p150a-full-step: ambient $variable is set and would override the admitted bundle" >&2
    exit 1
  fi
done

mkdir -p "$EVIDENCE_ROOT"

"$CORE" admit-profile "$PROFILE" > "$EVIDENCE_ROOT/profile-admission.json"
if [ "$MODE" = "measurement" ]; then
  "$CORE" admit-measurement-profile "$PROFILE" \
    > "$EVIDENCE_ROOT/measurement-profile-admission.json"
  "$CORE" bind-machine-facts "$MACHINE_FACTS" > "$EVIDENCE_ROOT/machine-facts.json"
fi
PROFILE_IDENTITY=$("$JQ" -r '.identity_blake3' "$EVIDENCE_ROOT/profile-admission.json")
TIMEOUT_SECONDS=$("$JQ" -r '.full_compile_timeout_seconds' "$EVIDENCE_ROOT/profile-admission.json")

PLAN=$("$PLATFORM_CORE" authorized-launch-plan "$PLATFORM_AUTHORITY")
IMAGE=$(printf '%s' "$PLAN" | "$JQ" -r '.plan.image')
HUGEPAGE_MOUNT=$(printf '%s' "$PLAN" | "$JQ" -r '.plan.mounts[0].host')
BUNDLE_IDENTITY=$(printf '%s' "$PLAN" | "$JQ" -r '.plan.bundle_identity')
CONFIG_IDENTITY=$(printf '%s' "$PLAN" | "$JQ" -r '.plan.config_identity')
PLAN_IDENTITY=$(printf '%s' "$PLAN" | "$JQ" -r '.identity_blake3')

stage_status() { printf '{"name": "%s", "status": "%s"}' "$1" "$2"; }

finalize_measurement_stage_times() {
  local elapsed_nanoseconds elapsed_seconds
  elapsed_nanoseconds=$((RUN_ENDED_NANOSECONDS - RUN_STARTED_NANOSECONDS))
  elapsed_seconds=$("$JQ" -n \
    --argjson nanoseconds "$elapsed_nanoseconds" \
    --argjson nanoseconds_per_second "$NANOSECONDS_PER_SECOND" \
    '$nanoseconds / $nanoseconds_per_second')
  "$JQ" --argjson elapsed_seconds "$elapsed_seconds" '
    .stage_times_seconds as $stages
    | ($elapsed_seconds - $stages.launch - $stages.compile - $stages.step - $stages.checkpoint) as $teardown
    | if $teardown < 0 then error("measured stage times exceed host elapsed time") else . end
    | .stage_times_seconds.teardown = $teardown
    | .stage_times_seconds.end_to_end = $elapsed_seconds
  ' "$EVIDENCE_ROOT/step-result.json" > "$EVIDENCE_ROOT/step-result-final.json"
  "$MV" "$EVIDENCE_ROOT/step-result-final.json" "$EVIDENCE_ROOT/step-result.json"
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
  if [ "$MODE" = "measurement" ]; then
    if [ "$overall" -ne 0 ]; then
      "$JQ" -n \
        --argjson schema_version 1 \
        --arg kind "rwkv7_lora_p150a_measurement_failure" \
        --arg profile_identity "$PROFILE_IDENTITY" \
        --argjson stages "$stages" \
        --argjson probe_result "$probe_result" \
        '{
          schema_version: $schema_version,
          kind: $kind,
          profile_identity: $profile_identity,
          stages: $stages,
          probe_result: $probe_result
        }' > "$EVIDENCE_ROOT/receipt.json"
    else
      local cpu_baseline_identity cpu_step_seconds machine_facts raw_measurement runs stage_times
      printf '%s' "$probe_result" | "$JQ" '{runs: .cpu_runs, step_seconds: .cpu_step_seconds}' \
        > "$EVIDENCE_ROOT/cpu-baseline.json"
      cpu_baseline_identity=$("$B3SUM" "$EVIDENCE_ROOT/cpu-baseline.json" | cut -d' ' -f1)
      cpu_step_seconds=$(printf '%s' "$probe_result" | "$JQ" '.cpu_step_seconds')
      machine_facts=$(cat "$EVIDENCE_ROOT/machine-facts.json")
      runs=$(printf '%s' "$probe_result" | "$JQ" '.runs')
      stage_times=$(printf '%s' "$probe_result" | "$JQ" '.stage_times_seconds')
      raw_measurement=$("$JQ" -n \
        --argjson stage_times_seconds "$stage_times" \
        --arg cpu_baseline_identity "$cpu_baseline_identity" \
        --argjson cpu_step_seconds "$cpu_step_seconds" \
        --argjson machine_facts "$machine_facts" \
        --argjson runs "$runs" \
        '{
          stage_times_seconds: $stage_times_seconds,
          cpu_baseline: {
            identity_blake3: $cpu_baseline_identity,
            step_seconds: $cpu_step_seconds
          },
          machine_facts: $machine_facts,
          runs: $runs
        }')
      printf '%s\n' "$raw_measurement" > "$EVIDENCE_ROOT/measurement-raw.json"
      "$CORE" summarize-measurement "$EVIDENCE_ROOT/measurement-raw.json" "$PROFILE" \
        > "$EVIDENCE_ROOT/receipt.json"
      "$CORE" admit-measurement "$EVIDENCE_ROOT/receipt.json" "$PROFILE" \
        > "$EVIDENCE_ROOT/measurement-admission.json"
      if [ "$REGRESSION_INPUT_COUNT" -eq "$REGRESSION_INPUT_TOTAL" ]; then
        "$CORE" gate-regression "$EVIDENCE_ROOT/receipt.json" "$PROFILE" \
          "$REGRESSION_BASELINE" "$REGRESSION_POLICY" \
          > "$EVIDENCE_ROOT/regression-receipt.json"
        "$CORE" compare-determinism "$EVIDENCE_ROOT/receipt.json" "$DETERMINISM_RECEIPT" \
          "$PROFILE" "$REGRESSION_POLICY" > "$EVIDENCE_ROOT/determinism-receipt.json"
      fi
      "$JQ" -n \
        --slurpfile measurement_receipt "$EVIDENCE_ROOT/receipt.json" \
        --slurpfile telemetry "$DEVICE_TELEMETRY" \
        '{
          measurement_receipt: $measurement_receipt[0],
          telemetry: $telemetry[0],
          scaling_baseline: null
        }' > "$EVIDENCE_ROOT/utilization-raw.json"
      "$CORE" summarize-utilization "$EVIDENCE_ROOT/utilization-raw.json" "$PROFILE" \
        > "$EVIDENCE_ROOT/utilization-receipt.json"
      "$CORE" admit-utilization "$EVIDENCE_ROOT/utilization-receipt.json" \
        "$EVIDENCE_ROOT/utilization-raw.json" "$PROFILE" \
        > "$EVIDENCE_ROOT/utilization-admission.json"
    fi
  elif [ "$MODE" = "multistep" ]; then
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

RUN_STARTED_NANOSECONDS=$("$DATE" +%s%N)
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
  -e "MEASUREMENT_HOST_START_NS=$RUN_STARTED_NANOSECONDS" \
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
RUN_ENDED_NANOSECONDS=$("$DATE" +%s%N)
if [ "$MODE" = "measurement" ]; then
  finalize_measurement_stage_times
fi

finish_receipt 0 "$S_HEALTH_BEFORE" "$S_OWNERSHIP" "$S_STEP" "$S_RESTORE" "$S_HEALTH_AFTER"
