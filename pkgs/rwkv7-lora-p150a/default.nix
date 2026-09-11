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
  platformEvidenceReceipt,
  cpuReference,
  targetProfile,
}:
let
  packageSource = lib.fileset.toSource {
    root = ../..;
    fileset = lib.fileset.unions [
      ./.
      ../../model-targets
    ];
  };

  # r[impl tenstorrent.rwkv7_p150a_finetuning.configuration]
  # r[impl tenstorrent.rwkv7_p150a_finetuning.authority]
  core = rustPlatform.buildRustPackage {
    pname = "rwkv7-lora-p150a-core";
    version = "0.1.0";
    src = ./core;
    cargoLock.lockFile = ./core/Cargo.lock;
    doCheck = true;
    meta = {
      description = "Pure single-P150A RWKV-7 LoRA bring-up admission core";
      license = lib.licenses.mit;
      mainProgram = "rwkv7-lora-p150a-core";
      platforms = lib.platforms.linux;
    };
  };

  platformCore = "${trainingPlatform.passthru.core}/bin/training-platform-core";
  platformAuthority = trainingPlatform.passthru.authorityJson;
  platformWheelhouse = trainingPlatform.passthru.wheelhouse;

  profileJson = runCommand "rwkv7-lora-p150a-profile.json" { nativeBuildInputs = [ nickel ]; } ''
    cp -r ${packageSource} "$PWD/repository"
    chmod -R u+w "$PWD/repository"
    cd "$PWD/repository/pkgs/rwkv7-lora-p150a"
    nickel export --format json profile.ncl > "$out"
  '';

  measurementProfileJson =
    runCommand "rwkv7-lora-p150a-measurement-profile.json" { nativeBuildInputs = [ nickel ]; }
      ''
        cp -r ${packageSource} "$PWD/repository"
        chmod -R u+w "$PWD/repository"
        cd "$PWD/repository/pkgs/rwkv7-lora-p150a"
        nickel export --format json measurement-profile.ncl > "$out"
      '';

  regressionPolicyJson =
    runCommand "rwkv7-lora-p150a-regression-policy.json" { nativeBuildInputs = [ nickel ]; }
      ''
        cp -r ${packageSource} "$PWD/repository"
        chmod -R u+w "$PWD/repository"
        cd "$PWD/repository/pkgs/rwkv7-lora-p150a"
        nickel export --format json regression-policy.ncl > "$out"
      '';

  stabilityPolicyJson =
    runCommand "rwkv7-lora-p150a-stability-policy.json" { nativeBuildInputs = [ nickel ]; }
      ''
        cp -r ${packageSource} "$PWD/repository"
        chmod -R u+w "$PWD/repository"
        cd "$PWD/repository/pkgs/rwkv7-lora-p150a"
        nickel export --format json stability-policy.ncl > "$out"
      '';

  wheelsJson = runCommand "rwkv7-lora-p150a-wheels.json" { nativeBuildInputs = [ nickel ]; } ''
    cp -r ${packageSource} "$PWD/repository"
    chmod -R u+w "$PWD/repository"
    cd "$PWD/repository/pkgs/rwkv7-lora-p150a"
    nickel export --format json wheels.ncl > "$out"
  '';

  # r[impl tenstorrent.rwkv7_p150a_finetuning.device]
  probeRunner = runCommand "rwkv7-lora-p150a-probe" { } ''
    substitute ${./run-probe.sh} "$out" \
      --subst-var-by core "${core}" \
      --subst-var-by platform_core "${trainingPlatform.passthru.core}" \
      --subst-var-by jq "${jq}/bin/jq" \
      --subst-var-by docker "${docker}/bin/docker" \
      --subst-var-by ttsmi "${tt-smi}/bin/tt-smi" \
      --subst-var-by b3sum "${b3sum}/bin/b3sum" \
      --subst-var-by fuser "${psmisc}/bin/fuser" \
      --subst-var-by probe_py "${./probes/wkv7_probe.py}" \
      --subst-var-by install_deps_py "${./probes/install_deps.py}" \
      --subst-var-by container_probe "${./container-probe.sh}" \
      --subst-var-by wheels_json "${wheelsJson}" \
      --subst-var-by platform_authority "${platformAuthority}" \
      --subst-var-by platform_wheelhouse "${platformWheelhouse}"
    chmod +x "$out"
    patchShebangs "$out"
  '';

  # r[impl tenstorrent.rwkv7_p150a_finetuning.placement]
  # r[impl tenstorrent.rwkv7_p150a_finetuning.parity]
  fullStepRunner = runCommand "rwkv7-lora-p150a-full-step" { } ''
    substitute ${./run-full-model.sh} "$out" \
      --subst-var-by core "${core}" \
      --subst-var-by platform_core "${trainingPlatform.passthru.core}" \
      --subst-var-by jq "${jq}/bin/jq" \
      --subst-var-by docker "${docker}/bin/docker" \
      --subst-var-by ttsmi "${tt-smi}/bin/tt-smi" \
      --subst-var-by b3sum "${b3sum}/bin/b3sum" \
      --subst-var-by fuser "${psmisc}/bin/fuser" \
      --subst-var-by date "${coreutils}/bin/date" \
      --subst-var-by mv "${coreutils}/bin/mv" \
      --subst-var-by rm "${coreutils}/bin/rm" \
      --subst-var-by sleep "${coreutils}/bin/sleep" \
      --subst-var-by probe_py "${./probes/full_model_probe.py}" \
      --subst-var-by model_dir "${./model}" \
      --subst-var-by install_deps_py "${./probes/install_deps.py}" \
      --subst-var-by container_script "${./container-full-model.sh}" \
      --subst-var-by platform_authority "${platformAuthority}" \
      --subst-var-by platform_wheelhouse "${platformWheelhouse}"
    chmod +x "$out"
    patchShebangs "$out"
  '';

  tools = runCommand "rwkv7-lora-p150a-tools" { } ''
    mkdir -p "$out/bin" "$out/share/rwkv7-lora-p150a" "$out/share/model-target"
    ln -s ${core}/bin/rwkv7-lora-p150a-core "$out/bin/rwkv7-lora-p150a-core"
    ln -s ${probeRunner} "$out/bin/rwkv7-lora-p150a-probe"
    ln -s ${fullStepRunner} "$out/bin/rwkv7-lora-p150a-full-step"
    ln -s ${profileJson} "$out/share/rwkv7-lora-p150a/profile.json"
    ln -s ${measurementProfileJson} "$out/share/rwkv7-lora-p150a/measurement-profile.json"
    ln -s ${regressionPolicyJson} "$out/share/rwkv7-lora-p150a/regression-policy.json"
    ln -s ${stabilityPolicyJson} "$out/share/rwkv7-lora-p150a/stability-policy.json"
    ln -s ${./baseline-store.json} "$out/share/rwkv7-lora-p150a/baseline-store.json"
    ln -s ${wheelsJson} "$out/share/rwkv7-lora-p150a/wheels.json"
    echo '${targetProfile.profile_identity}' >"$out/share/model-target/profile-blake3"
  '';
in
# r[verify tenstorrent.rwkv7_p150a_finetuning.configuration.valid]
# r[verify tenstorrent.rwkv7_p150a_finetuning.authority.valid]
# r[verify tenstorrent.rwkv7_p150a_finetuning.authority.invalid]
runCommand "rwkv7-lora-p150a-check"
  {
    nativeBuildInputs = [
      nickel
      jq
      b3sum
      coreutils
      gnugrep
    ];
    CORE = "${core}/bin/rwkv7-lora-p150a-core";
    PLATFORM_CORE = platformCore;
    PROBE_RUNNER = "${probeRunner}";
    PLATFORM_AUTHORITY = "${platformAuthority}";
    PLATFORM_RECEIPT = "${platformEvidenceReceipt}";
    CPU_RECEIPT = "${cpuReference}/receipt.json";
    passthru = {
      inherit
        core
        tools
        profileJson
        measurementProfileJson
        regressionPolicyJson
        stabilityPolicyJson
        ;
    };
  }
  ''
    set -euo pipefail
    cp -r ${packageSource} "$PWD/repository"
    chmod -R u+w "$PWD/repository"
    cd "$PWD/repository/pkgs/rwkv7-lora-p150a"

    # Valid profile exports deterministically and carries a stable identity.
    nickel export --format json profile.ncl >profile-first.json
    nickel export --format json profile.ncl >profile-second.json
    cmp profile-first.json profile-second.json
    first_identity=$(b3sum profile-first.json | cut -d' ' -f1)
    second_identity=$(b3sum profile-second.json | cut -d' ' -f1)
    test "$first_identity" = "$second_identity"
    grep -F '"profile_id": "rwkv7-lora-p150a-single"' profile-first.json
    grep -F '"bundle_id": "tt-xla-training-2026-08-04"' profile-first.json
    nickel export --format json measurement-profile.ncl >measurement-profile-first.json
    nickel export --format json measurement-profile.ncl >measurement-profile-second.json
    cmp measurement-profile-first.json measurement-profile-second.json
    grep -F '"profile_id": "rwkv7-lora-p150a-single-measurement"' measurement-profile-first.json
    grep -F '"parity_profile_id": "rwkv7-lora-p150a-single"' measurement-profile-first.json
    "$CORE" admit-measurement-profile measurement-profile-first.json \
      >measurement-profile-admission.json
    grep -F '"repeat_count":5' measurement-profile-admission.json
    jq '.tolerances.loss.rtol = 99.0' measurement-profile-first.json \
      >measurement-profile-relaxed.json
    if "$CORE" admit-measurement-profile measurement-profile-relaxed.json \
      > /dev/null 2> rejected.log; then
      echo "measurement profile relaxed parity" >&2
      exit 1
    fi
    grep -F "parity_profile_mismatch" rejected.log

    # Negative Nickel fixtures must fail with contract violations.
    # Nickel splits the blame message across stdout and stderr, so the
    # check captures both streams before it searches for the contract
    # violation marker.
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
    grep -F '"identity_blake3":' profile-admission.json

    # r[verify tenstorrent.harness.measurement_profile.valid]
    # r[verify tenstorrent.harness.stage_breakdown.valid]
    # r[verify tenstorrent.harness.speedup_ratio.valid]
    # r[verify tenstorrent.harness.machine_facts.valid]
    # r[verify tenstorrent.harness.repeat_stats.valid]
    cat >machine-fact-values.json <<'FACTS'
    {
      "board_identity": "p150a:device-0",
      "firmware_version": "18.8.0.0",
      "mesh_descriptor_identity": "single-device-1x1",
      "pcie_link": "gen4-x16",
      "power_state": "performance"
    }
    FACTS
    "$CORE" bind-machine-facts machine-fact-values.json >machine-facts.json
    cpu_baseline_identity=$(printf '%s' 'single-cpu-baseline' | b3sum | cut -d' ' -f1)
    jq -n \
      --arg cpu_baseline_identity "$cpu_baseline_identity" \
      --slurpfile machine_facts machine-facts.json \
      '{
        stage_times_seconds: {
          launch: 1.0,
          compile: 2.0,
          step: 5.0,
          checkpoint: 0.0,
          teardown: 1.0,
          end_to_end: 9.0
        },
        cpu_baseline: {
          identity_blake3: $cpu_baseline_identity,
          step_seconds: 10.0
        },
        machine_facts: $machine_facts[0],
        runs: [
          {time_seconds: 1.0, loss: 5.0},
          {time_seconds: 1.1, loss: 4.9},
          {time_seconds: 0.9, loss: 4.8},
          {time_seconds: 1.0, loss: 4.7},
          {time_seconds: 1.0, loss: 4.6}
        ]
      }' >measurement-raw.json
    "$CORE" summarize-measurement measurement-raw.json measurement-profile-first.json >measurement-receipt.json
    "$CORE" summarize-measurement measurement-raw.json measurement-profile-first.json >measurement-receipt-again.json
    cmp measurement-receipt.json measurement-receipt-again.json
    "$CORE" admit-measurement measurement-receipt.json measurement-profile-first.json >measurement-admission.json
    grep -F '"status":"admitted"' measurement-admission.json
    grep -F '"repeats":5' measurement-admission.json

    # r[verify tenstorrent.harness.baseline_store.valid]
    # r[verify tenstorrent.harness.regression_gate.valid]
    # r[verify tenstorrent.harness.determinism.valid]
    # r[verify tenstorrent.harness.metric_sanity.valid]
    nickel export --format json regression-policy.ncl >regression-policy.json
    "$CORE" create-baseline measurement-receipt.json measurement-profile-first.json \
      regression-policy.json >baseline-record.json
    jq -c -S '{schema_version: 1, records: [.]}' baseline-record.json >baseline-store-generated.json
    jq -c -S . baseline-store.json >baseline-store-checked.json
    cmp baseline-store-generated.json baseline-store-checked.json
    "$CORE" admit-baseline baseline-store.json measurement-profile-first.json \
      measurement-receipt.json >baseline-admission.json
    "$CORE" gate-regression measurement-receipt.json measurement-profile-first.json \
      baseline-store.json regression-policy.json >regression-receipt.json
    grep -F '"verdict":"pass"' regression-receipt.json
    "$CORE" compare-determinism measurement-receipt.json measurement-receipt-again.json \
      measurement-profile-first.json regression-policy.json >determinism-receipt.json
    grep -F '"verdict":"pass"' determinism-receipt.json

    # r[verify tenstorrent.harness.baseline_store.invalid]
    # r[verify tenstorrent.harness.regression_gate.invalid]
    # r[verify tenstorrent.harness.determinism.invalid]
    # r[verify tenstorrent.harness.metric_sanity.invalid]
    jq '.firmware_version = "other-firmware"' machine-fact-values.json \
      >other-machine-fact-values.json
    "$CORE" bind-machine-facts other-machine-fact-values.json >other-machine-facts.json
    jq --slurpfile facts other-machine-facts.json '.machine_facts = $facts[0]' \
      measurement-raw.json >measurement-other-machine-bound.json
    "$CORE" summarize-measurement measurement-other-machine-bound.json \
      measurement-profile-first.json >measurement-other-machine.json
    if "$CORE" admit-baseline baseline-store.json measurement-profile-first.json \
      measurement-other-machine.json > /dev/null 2> rejected.log; then
      echo "mismatched machine baseline was admitted" >&2
      exit 1
    fi
    grep -F "baseline_key_mismatch" rejected.log
    jq '.runs |= map(.time_seconds = (.time_seconds * 1.25))' measurement-raw.json \
      >measurement-regressed-raw.json
    "$CORE" summarize-measurement measurement-regressed-raw.json measurement-profile-first.json \
      >measurement-regressed.json
    if "$CORE" gate-regression measurement-regressed.json measurement-profile-first.json \
      baseline-store.json regression-policy.json > /dev/null 2> rejected.log; then
      echo "regressed measurement passed its gate" >&2
      exit 1
    fi
    grep -F "regression_gate_failure" rejected.log
    if "$CORE" compare-determinism measurement-receipt.json measurement-regressed.json \
      measurement-profile-first.json regression-policy.json > /dev/null 2> rejected.log; then
      echo "divergent repeated run was admitted" >&2
      exit 1
    fi
    grep -F "determinism_divergence" rejected.log

    # r[verify tenstorrent.harness.long_run.valid]
    # r[verify tenstorrent.harness.thermal.valid]
    # r[verify tenstorrent.harness.heldout_eval.valid]
    nickel export --format json stability-policy.ncl >stability-policy.json
    profile_identity=$(jq -r '.identity_blake3' profile-admission.json)
    eval_split_identity=$(printf '%s' 'single-heldout-eval-split' | b3sum | cut -d' ' -f1)
    reference_batch_identity=$(printf '%s' 'single-periodic-reference-batch' | b3sum | cut -d' ' -f1)
    jq -n \
      --arg profile_identity "$profile_identity" \
      --arg eval_split_identity "$eval_split_identity" \
      --arg reference_batch_identity "$reference_batch_identity" \
      --slurpfile machine_facts machine-facts.json \
      '{
        status: "completed",
        completed_steps: 256,
        checked_steps: 256,
        parity_profile_identity: $profile_identity,
        machine_facts: $machine_facts[0],
        trajectory: [range(1; 17) as $sample | {
          step: ($sample * 16), loss: (5.0 - ($sample * 0.1)), gradient_norm: 2.0
        }],
        thermal: {
          source: "tt-smi-jsonl",
          samples: [range(1; 17) as $sample | {
            step: ($sample * 16),
            devices: [{device_id: 0, temperature_celsius: 70.0, power_watts: 100.0}]
          }]
        },
        heldout_eval: {
          split_identity: $eval_split_identity,
          reference_batch_identity: $reference_batch_identity,
          split_role: "held_out",
          metric: "loss",
          before_loss: 5.0,
          after_loss: 4.0,
          generalization_non_claim: "held-out loss does not prove generalization beyond the bound split"
        }
      }' >stability-raw.json
    "$CORE" summarize-stability stability-raw.json profile-first.json stability-policy.json \
      >stability-receipt.json
    "$CORE" summarize-stability stability-raw.json profile-first.json stability-policy.json \
      >stability-receipt-again.json
    cmp stability-receipt.json stability-receipt-again.json
    "$CORE" admit-stability stability-receipt.json stability-raw.json profile-first.json \
      stability-policy.json >stability-admission.json
    grep -F '"status":"admitted"' stability-admission.json
    grep -F '"completed_steps":256' stability-admission.json

    # r[verify tenstorrent.harness.long_run.invalid]
    # r[verify tenstorrent.harness.thermal.invalid]
    # r[verify tenstorrent.harness.heldout_eval.invalid]
    jq '.trajectory[3].gradient_norm = 1001.0' stability-raw.json >stability-diverged.json
    if "$CORE" summarize-stability stability-diverged.json profile-first.json stability-policy.json \
      > /dev/null 2> rejected.log; then
      echo "divergent long run was admitted" >&2
      exit 1
    fi
    grep -F "step 64" rejected.log
    jq '.thermal.samples[2].devices[0].power_watts = "NaN"' stability-raw.json \
      >stability-nonfinite-thermal.json
    if "$CORE" summarize-stability stability-nonfinite-thermal.json profile-first.json \
      stability-policy.json > /dev/null 2> rejected.log; then
      echo "non-finite thermal sample was admitted" >&2
      exit 1
    fi
    grep -F "step 48, device 0" rejected.log
    jq '.heldout_eval.split_identity = .heldout_eval.reference_batch_identity' \
      stability-raw.json >stability-reference-eval.json
    if "$CORE" summarize-stability stability-reference-eval.json profile-first.json \
      stability-policy.json > /dev/null 2> rejected.log; then
      echo "reference-batch eval was admitted as held out" >&2
      exit 1
    fi
    grep -F "eval_not_held_out" rejected.log

    # r[verify tenstorrent.harness.utilization.valid]
    # r[verify tenstorrent.harness.memory_pressure.valid]
    # r[verify tenstorrent.harness.op_latency.valid]
    # r[verify tenstorrent.harness.scaling_efficiency.invalid]
    jq -n \
      --slurpfile measurement_receipt measurement-receipt.json \
      --slurpfile telemetry fixtures/device-telemetry.json \
      '{
        measurement_receipt: $measurement_receipt[0],
        telemetry: $telemetry[0],
        scaling_baseline: null
      }' >utilization-raw.json
    "$CORE" summarize-utilization utilization-raw.json measurement-profile-first.json \
      >utilization-receipt.json
    "$CORE" summarize-utilization utilization-raw.json measurement-profile-first.json \
      >utilization-receipt-again.json
    cmp utilization-receipt.json utilization-receipt-again.json
    "$CORE" admit-utilization utilization-receipt.json utilization-raw.json \
      measurement-profile-first.json >utilization-admission.json
    grep -F '"status":"admitted"' utilization-admission.json
    grep -F '"device_count":1' utilization-admission.json
    grep -F '"operation_count":2' utilization-admission.json

    # r[verify tenstorrent.harness.utilization.invalid]
    # r[verify tenstorrent.harness.memory_pressure.invalid]
    # r[verify tenstorrent.harness.op_latency.invalid]
    jq '.telemetry.utilization.math_busy_fraction = 1.1' utilization-raw.json \
      >utilization-out-of-range.json
    if "$CORE" summarize-utilization utilization-out-of-range.json \
      measurement-profile-first.json > /dev/null 2> rejected.log; then
      echo "out-of-range utilization was admitted" >&2
      exit 1
    fi
    grep -F "utilization_fraction_out_of_range" rejected.log
    jq '.telemetry.memory[0].l1_peak_bytes = -1' utilization-raw.json \
      >utilization-invalid-memory.json
    if "$CORE" summarize-utilization utilization-invalid-memory.json \
      measurement-profile-first.json > /dev/null 2> rejected.log; then
      echo "negative memory peak was admitted" >&2
      exit 1
    fi
    grep -F "invalid_utilization_metric" rejected.log
    jq 'del(.telemetry.operation_samples[1])' utilization-raw.json \
      >utilization-missing-operation.json
    if "$CORE" summarize-utilization utilization-missing-operation.json \
      measurement-profile-first.json > /dev/null 2> rejected.log; then
      echo "incomplete operation table was admitted" >&2
      exit 1
    fi
    grep -F "incomplete_operation_latency" rejected.log
    jq --slurpfile profile measurement-profile-first.json \
      --slurpfile receipt measurement-receipt.json \
      '.scaling_baseline = {profile: $profile[0], receipt: $receipt[0]}' \
      utilization-raw.json >utilization-false-scaling.json
    if "$CORE" summarize-utilization utilization-false-scaling.json \
      measurement-profile-first.json > /dev/null 2> rejected.log; then
      echo "single-device scaling claim was admitted" >&2
      exit 1
    fi
    grep -F "scaling_claim_without_two_device_run" rejected.log

    # r[verify tenstorrent.harness.measurement_profile.scope]
    # r[verify tenstorrent.harness.stage_breakdown.invalid]
    # r[verify tenstorrent.harness.speedup_ratio.invalid]
    # r[verify tenstorrent.harness.machine_facts.invalid]
    # r[verify tenstorrent.harness.repeat_stats.invalid]
    jq '.stage_times_seconds.compile = -1' measurement-raw.json >measurement-negative-stage.json
    if "$CORE" summarize-measurement measurement-negative-stage.json measurement-profile-first.json \
      > /dev/null 2> rejected.log; then
      echo "negative measurement stage was admitted" >&2
      exit 1
    fi
    grep -F "invalid_metric" rejected.log
    jq '.cpu_baseline.identity_blake3 = "missing"' measurement-raw.json >measurement-missing-cpu.json
    if "$CORE" summarize-measurement measurement-missing-cpu.json measurement-profile-first.json \
      > /dev/null 2> rejected.log; then
      echo "measurement without CPU baseline was admitted" >&2
      exit 1
    fi
    grep -F "missing_cpu_baseline" rejected.log
    jq 'del(.machine_facts.pcie_link)' measurement-raw.json >measurement-missing-fact.json
    if "$CORE" summarize-measurement measurement-missing-fact.json measurement-profile-first.json \
      > /dev/null 2> rejected.log; then
      echo "measurement without PCIe fact was admitted" >&2
      exit 1
    fi
    grep -F "unexpected_measurement_field" rejected.log
    jq '.runs = .runs[0:2]' measurement-raw.json >measurement-low-repeat.json
    if "$CORE" summarize-measurement measurement-low-repeat.json measurement-profile-first.json \
      > /dev/null 2> rejected.log; then
      echo "measurement with too few repeats was admitted" >&2
      exit 1
    fi
    grep -F "repeat_count_mismatch" rejected.log
    jq '.runs[0].time_seconds = "NaN"' measurement-raw.json >measurement-nonfinite.json
    if "$CORE" summarize-measurement measurement-nonfinite.json measurement-profile-first.json \
      > /dev/null 2> rejected.log; then
      echo "measurement with a non-finite metric was admitted" >&2
      exit 1
    fi
    grep -F "invalid_metric" rejected.log

    # The current hardware receipt must re-validate against the package-owned
    # authority before prerequisite-admission logic can bind it.
    "$PLATFORM_CORE" validate-terminal-receipt "$PLATFORM_RECEIPT" \
      "$PLATFORM_AUTHORITY" >platform-validation.json
    grep -F '"outcome":"success"' platform-validation.json
    jq -n '{status: "admitted"}' >platform-admission.json

    # Prerequisite admission binds platform and CPU receipts to the profile.
    cpu_receipt_blake3=$(b3sum "$CPU_RECEIPT" | cut -d' ' -f1)
    "$CORE" admit-prerequisites \
      --profile profile-first.json \
      --platform-admission platform-admission.json \
      --platform-receipt "$PLATFORM_RECEIPT" \
      --cpu-receipt "$CPU_RECEIPT" \
      --cpu-receipt-blake3 "$cpu_receipt_blake3" \
      >prerequisite-admission.json
    grep -F '"status":"admitted"' prerequisite-admission.json
    grep -F '"prerequisite_identity":' prerequisite-admission.json
    "$CORE" admit-prerequisites \
      --profile profile-first.json \
      --platform-admission platform-admission.json \
      --platform-receipt "$PLATFORM_RECEIPT" \
      --cpu-receipt "$CPU_RECEIPT" \
      --cpu-receipt-blake3 "$cpu_receipt_blake3" \
      >prerequisite-admission-again.json
    cmp prerequisite-admission.json prerequisite-admission-again.json

    # Negative: a stale platform admission blocks compilation and device work.
    jq '.status = "rejected"' platform-admission.json >stale-admission.json
    if "$CORE" admit-prerequisites \
      --profile profile-first.json \
      --platform-admission stale-admission.json \
      --platform-receipt "$PLATFORM_RECEIPT" \
      --cpu-receipt "$CPU_RECEIPT" \
      --cpu-receipt-blake3 "$cpu_receipt_blake3" \
      > /dev/null 2> rejected.log; then
      echo "stale platform admission was admitted" >&2
      exit 1
    fi
    grep -F "stale_prerequisite" rejected.log

    # Negative: a failing platform stage blocks prerequisite admission.
    jq '.stages[2].status = "fail"' "$PLATFORM_RECEIPT" >failing-receipt.json
    if "$CORE" admit-prerequisites \
      --profile profile-first.json \
      --platform-admission platform-admission.json \
      --platform-receipt failing-receipt.json \
      --cpu-receipt "$CPU_RECEIPT" \
      --cpu-receipt-blake3 "$cpu_receipt_blake3" \
      > /dev/null 2> rejected.log; then
      echo "failing platform stage was admitted" >&2
      exit 1
    fi
    grep -F "failing_stage" rejected.log

    # Negative: a drifted model identity breaks the CPU authority binding.
    jq '.model.revision = "ffffffffffffffffffffffffffffffffffffffff"' \
      profile-first.json >drifted-profile.json
    if "$CORE" admit-prerequisites \
      --profile drifted-profile.json \
      --platform-admission platform-admission.json \
      --platform-receipt "$PLATFORM_RECEIPT" \
      --cpu-receipt "$CPU_RECEIPT" \
      --cpu-receipt-blake3 "$cpu_receipt_blake3" \
      > /dev/null 2> rejected.log; then
      echo "drifted model revision was admitted" >&2
      exit 1
    fi
    grep -F "model_mismatch" rejected.log

    # The probe runner shell must exist and reject missing arguments even
    # though hardware runs happen outside the sandbox.
    test -x "$PROBE_RUNNER"
    if "$PROBE_RUNNER" > /dev/null 2> runner-usage.log; then
      echo "probe runner accepted an empty invocation" >&2
      exit 1
    fi
    grep -F "usage:" runner-usage.log

    # Positive probe receipt admission against the live profile identity.
    profile_identity=$(jq -r '.identity_blake3' profile-admission.json)
    jq --arg identity "$profile_identity" '.profile_identity = $identity' \
      fixtures/probe-receipt-base.json >probe-receipt.json
    "$CORE" admit-probe probe-receipt.json profile-first.json >probe-admission.json
    grep -F '"status":"admitted"' probe-admission.json

    # Negative: host placement is rejected as forbidden fallback.
    jq '.probe_result.device.type = "CPU"' probe-receipt.json >probe-host.json
    if "$CORE" admit-probe probe-host.json profile-first.json \
      > /dev/null 2> rejected.log; then
      echo "host placement probe was admitted" >&2
      exit 1
    fi
    grep -F "host_fallback" rejected.log

    # Negative: a tolerance failure is rejected.
    jq '.probe_result.tolerances_satisfied = false' probe-receipt.json >probe-tolerance.json
    if "$CORE" admit-probe probe-tolerance.json profile-first.json \
      > /dev/null 2> rejected.log; then
      echo "tolerance-failing probe was admitted" >&2
      exit 1
    fi
    grep -F "tolerance_failure" rejected.log

    # Negative: a stale probe graph is rejected.
    jq '.profile_identity = "0000000000000000000000000000000000000000000000000000000000000000"' \
      probe-receipt.json >probe-stale.json
    if "$CORE" admit-probe probe-stale.json profile-first.json \
      > /dev/null 2> rejected.log; then
      echo "stale probe receipt was admitted" >&2
      exit 1
    fi
    grep -F "stale_probe_receipt" rejected.log

    # The checked-in hardware probe receipts must re-validate against the
    # admitted profile. A stale or drifted receipt fails this gate.
    for receipt in evidence/2026-08-04-*/receipt.json; do
      "$CORE" admit-probe "$receipt" profile-first.json >checked-in-admission.json
      grep -F '"status":"admitted"' checked-in-admission.json
    done

    # Negative: a checked-in receipt that failed its tolerances is rejected.
    representative_receipt=evidence/2026-08-04-wkv7-backward-representative/receipt.json
    jq '.probe_result.tolerances_satisfied = false' "$representative_receipt" >tolerance-drift.json
    if "$CORE" admit-probe tolerance-drift.json profile-first.json \
      > /dev/null 2> rejected.log; then
      echo "tolerance-drifted hardware receipt was admitted" >&2
      exit 1
    fi
    grep -F "tolerance_failure" rejected.log

    # r[verify tenstorrent.rwkv7_p150a_finetuning.placement]
    # r[verify tenstorrent.rwkv7_p150a_finetuning.parity]
    # The checked-in full-model step receipt re-validates against the admitted
    # profile. This is the bounded single-device training evidence.
    step_receipt=evidence/2026-08-05-full-model-step/receipt.json
    "$CORE" admit-step "$step_receipt" profile-first.json >step-admission.json
    grep -F '"status":"admitted"' step-admission.json
    grep -F '"stablehlo_op_count":25594' step-admission.json

    # Negative: a step receipt with a low delta pass fraction is rejected.
    jq '.step.delta_pass_fraction = 0.4' "$step_receipt" >step-low-delta.json
    if "$CORE" admit-step step-low-delta.json profile-first.json \
      > /dev/null 2> rejected.log; then
      echo "low-delta step receipt was admitted" >&2
      exit 1
    fi
    grep -F "tolerance_failure" rejected.log

    # Negative: a step receipt with host placement is rejected.
    jq '.placement.device = "CPU"' "$step_receipt" >step-host.json
    if "$CORE" admit-step step-host.json profile-first.json \
      > /dev/null 2> rejected.log; then
      echo "host-placement step receipt was admitted" >&2
      exit 1
    fi
    grep -F "host_fallback" rejected.log

    # r[verify tenstorrent.rwkv7_p150a_finetuning.run]
    # The checked-in bounded multi-step run receipt re-validates against the
    # admitted profile: finite values, adapter-only mutation, and the loss
    # improvement criterion.
    run_receipt=evidence/2026-08-05-multistep-run/receipt.json
    "$CORE" admit-run "$run_receipt" profile-first.json >run-admission.json
    grep -F '"status":"admitted"' run-admission.json
    grep -F '"steps":8' run-admission.json

    # Negative: a run receipt with a base tensor mutation is rejected.
    jq '.run.base_unchanged = false' "$run_receipt" >run-mutated.json
    if "$CORE" admit-run run-mutated.json profile-first.json \
      > /dev/null 2> rejected.log; then
      echo "base-mutating run receipt was admitted" >&2
      exit 1
    fi
    grep -F "base_mutated" rejected.log

    # Negative: a run receipt that fails the loss criterion is rejected.
    jq '.run.final_loss = 13.0' "$run_receipt" >run-flat.json
    if "$CORE" admit-run run-flat.json profile-first.json \
      > /dev/null 2> rejected.log; then
      echo "non-improving run receipt was admitted" >&2
      exit 1
    fi
    grep -F "loss_criterion" rejected.log

    # r[verify tenstorrent.rwkv7_p150a_finetuning.checkpoint]
    # The checked-in checkpoint receipt re-validates: byte-identical adapter
    # load and an exact resumed result.
    checkpoint_receipt=evidence/2026-08-05-checkpoint-resume/receipt.json
    "$CORE" admit-checkpoint "$checkpoint_receipt" profile-first.json >checkpoint-admission.json
    grep -F '"status":"admitted"' checkpoint-admission.json
    grep -F '"resume_loss_dev":0' checkpoint-admission.json

    # Negative: a checkpoint receipt with adapter load drift is rejected.
    jq '.checkpoint.adapter_load_max_diff = 0.5' "$checkpoint_receipt" >checkpoint-drift.json
    if "$CORE" admit-checkpoint checkpoint-drift.json profile-first.json \
      > /dev/null 2> rejected.log; then
      echo "adapter-drift checkpoint receipt was admitted" >&2
      exit 1
    fi
    grep -F "adapter_load_drift" rejected.log

    # Negative: a checkpoint receipt with resume drift is rejected.
    jq '.checkpoint.resume_loss_dev = 5.0' "$checkpoint_receipt" >checkpoint-resume-drift.json
    if "$CORE" admit-checkpoint checkpoint-resume-drift.json profile-first.json \
      > /dev/null 2> rejected.log; then
      echo "resume-drift checkpoint receipt was admitted" >&2
      exit 1
    fi
    grep -F "resume_drift" rejected.log

    # r[verify tenstorrent.rwkv7_p150a_finetuning.verification]
    # Negative: a probe receipt with nonfinite values is rejected.
    probe_receipt=evidence/2026-08-04-wkv7-forward-focused/receipt.json
    jq '.probe_result.finite = false' "$probe_receipt" >probe-nonfinite.json
    if "$CORE" admit-probe probe-nonfinite.json profile-first.json \
      > /dev/null 2> rejected.log; then
      echo "nonfinite probe receipt was admitted" >&2
      exit 1
    fi
    grep -F "nonfinite_values" rejected.log

    # Negative: a probe receipt with no lowered operations is rejected as
    # fallback (an unsupported operation produces no TT graph).
    jq '.probe_result.hlo.stablehlo_op_count = 0' "$probe_receipt" >probe-no-ops.json
    if "$CORE" admit-probe probe-no-ops.json profile-first.json \
      > /dev/null 2> rejected.log; then
      echo "zero-op probe receipt was admitted" >&2
      exit 1
    fi
    grep -F "host_fallback" rejected.log

    # Negative: a step receipt that produced no adapter gradients is rejected
    # through its low pass fraction.
    step_receipt=evidence/2026-08-05-full-model-step/receipt.json
    jq '.step.grad_pass_fraction = 0.1' "$step_receipt" >step-missing-grads.json
    if "$CORE" admit-step step-missing-grads.json profile-first.json \
      > /dev/null 2> rejected.log; then
      echo "missing-gradient step receipt was admitted" >&2
      exit 1
    fi
    grep -F "tolerance_failure" rejected.log

    # Negative: a step receipt whose device restore failed is rejected.
    jq '(.stages[] | select(.name == "restore") | .status) = "fail"' "$step_receipt" >step-restore-fail.json
    if "$CORE" admit-step step-restore-fail.json profile-first.json \
      > /dev/null 2> rejected.log; then
      echo "restore-failed step receipt was admitted" >&2
      exit 1
    fi
    grep -F "failing_stage" rejected.log

    # Negative: a malformed CPU receipt identity blocks admission.
    if "$CORE" admit-prerequisites \
      --profile profile-first.json \
      --platform-admission platform-admission.json \
      --platform-receipt "$PLATFORM_RECEIPT" \
      --cpu-receipt "$CPU_RECEIPT" \
      --cpu-receipt-blake3 "not-a-hash" \
      > /dev/null 2> rejected.log; then
      echo "malformed CPU receipt identity was admitted" >&2
      exit 1
    fi
    grep -F "malformed_receipt" rejected.log

    mkdir -p "$out/share/model-target"
    echo '${targetProfile.profile_identity}' >"$out/share/model-target/profile-blake3"
    cp profile-first.json "$out/profile.json"
    cp profile-admission.json "$out/profile-admission.json"
    cp platform-admission.json "$out/platform-admission.json"
    cp prerequisite-admission.json "$out/prerequisite-admission.json"
    cp probe-admission.json "$out/probe-admission-fixture.json"
    cp step-admission.json "$out/step-admission.json"
    cp run-admission.json "$out/run-admission.json"
    cp checkpoint-admission.json "$out/checkpoint-admission.json"
  ''
