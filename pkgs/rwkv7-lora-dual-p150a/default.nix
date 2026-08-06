{
  lib,
  rustPlatform,
  runCommand,
  nickel,
  jq,
  b3sum,
  coreutils,
  gnugrep,
  docker,
  psmisc,
  tt-smi,
  trainingPlatform,
  cpuReference,
  rwkv7LoraP150a,
}:
let
  # r[impl tenstorrent.rwkv7_dual_p150a.configuration]
  # r[impl tenstorrent.rwkv7_dual_p150a.authority]
  # The dual core binds the single-P150A receipts through the single core
  # crate, so both crate sources must live in one sandbox tree at the same
  # relative path the Cargo.toml path dependency names.
  coreSrc = runCommand "rwkv7-lora-dual-p150a-core-src" { } ''
    mkdir -p "$out/rwkv7-lora-dual-p150a" "$out/rwkv7-lora-p150a"
    cp -r ${./core} "$out/rwkv7-lora-dual-p150a/core"
    cp -r ${../rwkv7-lora-p150a/core} "$out/rwkv7-lora-p150a/core"
  '';

  core = rustPlatform.buildRustPackage {
    pname = "rwkv7-lora-dual-p150a-core";
    version = "0.1.0";
    src = coreSrc;
    sourceRoot = "${coreSrc.name}/rwkv7-lora-dual-p150a/core";
    cargoLock.lockFile = ./core/Cargo.lock;
    doCheck = true;
    meta = {
      description = "Pure dual-P150A RWKV-7 LoRA data-parallel admission core";
      license = lib.licenses.mit;
      mainProgram = "rwkv7-lora-dual-p150a-core";
      platforms = lib.platforms.linux;
    };
  };

  platformCore = "${trainingPlatform.passthru.core}/bin/training-platform-core";
  platformAuthority = trainingPlatform.passthru.authorityJson;
  platformWheelhouse = trainingPlatform.passthru.wheelhouse;

  singleProfileJson = rwkv7LoraP150a.passthru.profileJson;
  singleStepReceipt = ../rwkv7-lora-p150a/evidence/2026-08-05-full-model-step/receipt.json;
  singleRunReceipt = ../rwkv7-lora-p150a/evidence/2026-08-05-multistep-run/receipt.json;
  singleCheckpointReceipt = ../rwkv7-lora-p150a/evidence/2026-08-05-checkpoint-resume/receipt.json;

  profileJson = runCommand "rwkv7-lora-dual-p150a-profile.json" { nativeBuildInputs = [ nickel ]; } ''
    cp ${./contracts.ncl} contracts.ncl
    cp ${./profile.ncl} profile.ncl
    mkdir fixtures
    cp ${./fixtures/base-profile.ncl} fixtures/base-profile.ncl
    nickel export --format json profile.ncl > "$out"
  '';

  wheelsJson = runCommand "rwkv7-lora-dual-p150a-wheels.json" { nativeBuildInputs = [ nickel ]; } ''
    cp ${./contracts.ncl} contracts.ncl
    cp ${./wheels.ncl} wheels.ncl
    nickel export --format json wheels.ncl > "$out"
  '';

  # r[impl tenstorrent.rwkv7_dual_p150a.collectives]
  collectiveRunner = runCommand "rwkv7-lora-dual-p150a-collective" { } ''
    substitute ${./run-collective.sh} "$out" \
      --subst-var-by core "${core}" \
      --subst-var-by platform_core "${trainingPlatform.passthru.core}" \
      --subst-var-by jq "${jq}/bin/jq" \
      --subst-var-by docker "${docker}/bin/docker" \
      --subst-var-by ttsmi "${tt-smi}/bin/tt-smi" \
      --subst-var-by b3sum "${b3sum}/bin/b3sum" \
      --subst-var-by fuser "${psmisc}/bin/fuser" \
      --subst-var-by collective_probe_py "${./probes/collective_probe.py}" \
      --subst-var-by install_deps_py "${../rwkv7-lora-p150a/probes/install_deps.py}" \
      --subst-var-by container_collective "${./container-collective.sh}" \
      --subst-var-by platform_authority "${platformAuthority}" \
      --subst-var-by platform_wheelhouse "${platformWheelhouse}"
    chmod +x "$out"
    patchShebangs "$out"
  '';

  # r[impl tenstorrent.rwkv7_dual_p150a.graph]
  # r[impl tenstorrent.rwkv7_dual_p150a.parity]
  fullStepRunner = runCommand "rwkv7-lora-dual-p150a-full-step" { } ''
    substitute ${./run-dual-model.sh} "$out" \
      --subst-var-by core "${core}" \
      --subst-var-by platform_core "${trainingPlatform.passthru.core}" \
      --subst-var-by jq "${jq}/bin/jq" \
      --subst-var-by docker "${docker}/bin/docker" \
      --subst-var-by ttsmi "${tt-smi}/bin/tt-smi" \
      --subst-var-by b3sum "${b3sum}/bin/b3sum" \
      --subst-var-by fuser "${psmisc}/bin/fuser" \
      --subst-var-by dual_probe_py "${./probes/full_model_dual.py}" \
      --subst-var-by model_dir "${../rwkv7-lora-p150a/model}" \
      --subst-var-by install_deps_py "${../rwkv7-lora-p150a/probes/install_deps.py}" \
      --subst-var-by container_script "${./container-dual-model.sh}" \
      --subst-var-by platform_authority "${platformAuthority}" \
      --subst-var-by platform_wheelhouse "${platformWheelhouse}"
    chmod +x "$out"
    patchShebangs "$out"
  '';

  tools = runCommand "rwkv7-lora-dual-p150a-tools" { } ''
    mkdir -p "$out/bin" "$out/share/rwkv7-lora-dual-p150a"
    ln -s ${core}/bin/rwkv7-lora-dual-p150a-core "$out/bin/rwkv7-lora-dual-p150a-core"
    ln -s ${collectiveRunner} "$out/bin/rwkv7-lora-dual-p150a-collective"
    ln -s ${fullStepRunner} "$out/bin/rwkv7-lora-dual-p150a-full-step"
    ln -s ${profileJson} "$out/share/rwkv7-lora-dual-p150a/profile.json"
    ln -s ${wheelsJson} "$out/share/rwkv7-lora-dual-p150a/wheels.json"
  '';
in
# r[verify tenstorrent.rwkv7_dual_p150a.configuration.valid]
# r[verify tenstorrent.rwkv7_dual_p150a.configuration.invalid]
# r[verify tenstorrent.rwkv7_dual_p150a.authority.valid]
# r[verify tenstorrent.rwkv7_dual_p150a.authority.invalid]
runCommand "rwkv7-lora-dual-p150a-check"
  {
    nativeBuildInputs = [
      nickel
      jq
      b3sum
      coreutils
      gnugrep
    ];
    CORE = "${core}/bin/rwkv7-lora-dual-p150a-core";
    PLATFORM_CORE = platformCore;
    COLLECTIVE_RUNNER = "${collectiveRunner}";
    FULL_STEP_RUNNER = "${fullStepRunner}";
    PLATFORM_AUTHORITY = "${platformAuthority}";
    # Device-free prerequisite fixture generated by the current platform check.
    # It is not hardware evidence.
    PLATFORM_RECEIPT = "${trainingPlatform}/receipt-valid.json";
    CPU_RECEIPT = "${cpuReference}/receipt.json";
    SINGLE_PROFILE = "${singleProfileJson}";
    SINGLE_STEP_RECEIPT = "${singleStepReceipt}";
    SINGLE_RUN_RECEIPT = "${singleRunReceipt}";
    SINGLE_CHECKPOINT_RECEIPT = "${singleCheckpointReceipt}";
    passthru = {
      inherit core tools profileJson;
    };
  }
  ''
    set -euo pipefail
    cp -r ${./.} "$PWD/source"
    chmod -R u+w "$PWD/source"
    cd "$PWD/source"

    # Valid profile exports deterministically and carries a stable identity.
    nickel export --format json profile.ncl >profile-first.json
    nickel export --format json profile.ncl >profile-second.json
    cmp profile-first.json profile-second.json
    first_identity=$(b3sum profile-first.json | cut -d' ' -f1)
    second_identity=$(b3sum profile-second.json | cut -d' ' -f1)
    test "$first_identity" = "$second_identity"
    grep -F '"profile_id": "rwkv7-lora-dual-p150a"' profile-first.json
    grep -F '"bundle_id": "tt-xla-training-2026-08-04"' profile-first.json

    # Negative Nickel fixtures must fail with contract violations.
    for fixture in fixtures/invalid-*.ncl; do
      if nickel export --format json "$fixture" >fixture.out 2>fixture.err; then
        echo "negative profile fixture exported: $fixture" >&2
        exit 1
      fi
      cat fixture.out fixture.err >fixture.log
      grep -F "contract" fixture.log
    done

    # Core self-test exercises built-in positive and negative fixtures.
    "$CORE" self-test | grep -F "self-test: PASS"

    # Core re-checks every Nickel contract before admitting the profile.
    "$CORE" admit-profile profile-first.json >profile-admission.json
    grep -F '"status":"admitted"' profile-admission.json
    grep -F '"global_batch":2' profile-admission.json

    # r[verify tenstorrent.rwkv7_dual_p150a.sharding.valid]
    # r[verify tenstorrent.rwkv7_dual_p150a.sharding.invalid]
    # Positive: the admitted global batch shards exactly once.
    cat >batch.json <<'BATCH'
    {"global_batch_index": 0, "samples": [
      {"sample_id": "sample-0", "active_labels": 64},
      {"sample_id": "sample-1", "active_labels": 64}]}
    BATCH
    "$CORE" plan-shards profile-first.json batch.json >shard-plan.json
    grep -F '"status":"admitted"' shard-plan.json
    "$CORE" plan-shards profile-first.json batch.json >shard-plan-again.json
    cmp shard-plan.json shard-plan-again.json

    # Negative: an odd batch cannot shard evenly over two replicas.
    jq '.samples += [{"sample_id": "sample-2", "active_labels": 64}]' batch.json >odd-batch.json
    if "$CORE" plan-shards profile-first.json odd-batch.json > /dev/null 2> rejected.log; then
      echo "odd batch was admitted" >&2
      exit 1
    fi
    grep -F "uneven_batch" rejected.log

    # Negative: a duplicate sample is rejected.
    jq '.samples[1].sample_id = "sample-0"' batch.json >duplicate-batch.json
    if "$CORE" plan-shards profile-first.json duplicate-batch.json > /dev/null 2> rejected.log; then
      echo "duplicate sample was admitted" >&2
      exit 1
    fi
    grep -F "duplicate_sample" rejected.log

    # The current device-free fixture must re-validate against the package-owned
    # authority before prerequisite-admission logic can bind it.
    "$PLATFORM_CORE" validate-terminal-receipt "$PLATFORM_RECEIPT" \
      "$PLATFORM_AUTHORITY" >platform-validation.json
    grep -F '"outcome":"success"' platform-validation.json
    jq -n '{status: "admitted"}' >platform-admission.json

    # r[verify tenstorrent.rwkv7_dual_p150a.authority.valid]
    # Prerequisite admission binds platform, CPU, and single-P150A receipts
    # to the dual profile.
    cpu_receipt_blake3=$(b3sum "$CPU_RECEIPT" | cut -d' ' -f1)
    "$CORE" admit-prerequisites \
      --profile profile-first.json \
      --platform-admission platform-admission.json \
      --platform-receipt "$PLATFORM_RECEIPT" \
      --cpu-receipt "$CPU_RECEIPT" \
      --cpu-receipt-blake3 "$cpu_receipt_blake3" \
      --single-profile "$SINGLE_PROFILE" \
      --single-step-receipt "$SINGLE_STEP_RECEIPT" \
      --single-run-receipt "$SINGLE_RUN_RECEIPT" \
      --single-checkpoint-receipt "$SINGLE_CHECKPOINT_RECEIPT" \
      >prerequisite-admission.json
    grep -F '"status":"admitted"' prerequisite-admission.json
    grep -F '"prerequisite_identity":' prerequisite-admission.json
    grep -F '"single_step_identity":' prerequisite-admission.json
    "$CORE" admit-prerequisites \
      --profile profile-first.json \
      --platform-admission platform-admission.json \
      --platform-receipt "$PLATFORM_RECEIPT" \
      --cpu-receipt "$CPU_RECEIPT" \
      --cpu-receipt-blake3 "$cpu_receipt_blake3" \
      --single-profile "$SINGLE_PROFILE" \
      --single-step-receipt "$SINGLE_STEP_RECEIPT" \
      --single-run-receipt "$SINGLE_RUN_RECEIPT" \
      --single-checkpoint-receipt "$SINGLE_CHECKPOINT_RECEIPT" \
      >prerequisite-admission-again.json
    cmp prerequisite-admission.json prerequisite-admission-again.json

    # r[verify tenstorrent.rwkv7_dual_p150a.authority.invalid]
    # Negative: a stale platform admission blocks dual launch.
    jq '.status = "rejected"' platform-admission.json >stale-admission.json
    if "$CORE" admit-prerequisites \
      --profile profile-first.json \
      --platform-admission stale-admission.json \
      --platform-receipt "$PLATFORM_RECEIPT" \
      --cpu-receipt "$CPU_RECEIPT" \
      --cpu-receipt-blake3 "$cpu_receipt_blake3" \
      --single-profile "$SINGLE_PROFILE" \
      --single-step-receipt "$SINGLE_STEP_RECEIPT" \
      --single-run-receipt "$SINGLE_RUN_RECEIPT" \
      --single-checkpoint-receipt "$SINGLE_CHECKPOINT_RECEIPT" \
      > /dev/null 2> rejected.log; then
      echo "stale platform admission was admitted" >&2
      exit 1
    fi
    grep -F "stale_prerequisite" rejected.log

    # Negative: a failing single-P150A step receipt blocks dual launch.
    jq '.step.finite = false' "$SINGLE_STEP_RECEIPT" >failing-single-step.json
    if "$CORE" admit-prerequisites \
      --profile profile-first.json \
      --platform-admission platform-admission.json \
      --platform-receipt "$PLATFORM_RECEIPT" \
      --cpu-receipt "$CPU_RECEIPT" \
      --cpu-receipt-blake3 "$cpu_receipt_blake3" \
      --single-profile "$SINGLE_PROFILE" \
      --single-step-receipt failing-single-step.json \
      --single-run-receipt "$SINGLE_RUN_RECEIPT" \
      --single-checkpoint-receipt "$SINGLE_CHECKPOINT_RECEIPT" \
      > /dev/null 2> rejected.log; then
      echo "failing single-device receipt was admitted" >&2
      exit 1
    fi
    grep -F "stale_prerequisite" rejected.log

    # The runner shells must exist and reject missing arguments even though
    # hardware runs happen outside the sandbox.
    test -x "$COLLECTIVE_RUNNER"
    if "$COLLECTIVE_RUNNER" > /dev/null 2> runner-usage.log; then
      echo "collective runner accepted an empty invocation" >&2
      exit 1
    fi
    grep -F "usage:" runner-usage.log
    test -x "$FULL_STEP_RUNNER"
    if "$FULL_STEP_RUNNER" > /dev/null 2> runner-usage.log; then
      echo "full-step runner accepted an empty invocation" >&2
      exit 1
    fi
    grep -F "usage:" runner-usage.log

    # r[verify tenstorrent.rwkv7_dual_p150a.verification]
    # Checked-in hardware receipts re-validate against the admitted profile.
    # A stale or drifted receipt fails this gate. The glob is empty until
    # the first hardware evidence lands.
    shopt -s nullglob
    for receipt in evidence/*/receipt.json; do
      kind=$(jq -r '.kind' "$receipt")
      case "$kind" in
        rwkv7_lora_dual_p150a_collective)
          "$CORE" admit-collective "$receipt" profile-first.json >checked-in-admission.json
          ;;
        rwkv7_lora_dual_p150a_step)
          "$CORE" admit-step "$receipt" profile-first.json >checked-in-admission.json
          ;;
        rwkv7_lora_dual_p150a_run)
          "$CORE" admit-run "$receipt" profile-first.json >checked-in-admission.json
          ;;
        rwkv7_lora_dual_p150a_checkpoint)
          "$CORE" admit-checkpoint "$receipt" profile-first.json >checked-in-admission.json
          ;;
        *)
          echo "unknown checked-in receipt kind: $kind" >&2
          exit 1
          ;;
      esac
      grep -F '"status":"admitted"' checked-in-admission.json
    done

    mkdir -p "$out"
    cp profile-first.json "$out/profile.json"
    cp profile-admission.json "$out/profile-admission.json"
    cp shard-plan.json "$out/shard-plan.json"
    cp platform-admission.json "$out/platform-admission.json"
    cp prerequisite-admission.json "$out/prerequisite-admission.json"
  ''
