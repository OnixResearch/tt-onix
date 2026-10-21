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
  rwkv7LoraP150a,
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
      license = lib.licenses.lgpl21Only;
      mainProgram = "rwkv7-lora-dual-p150a-core";
      platforms = lib.platforms.linux;
    };
  };

  platformCore = "${trainingPlatform.passthru.core}/bin/training-platform-core";
  platformAuthority = trainingPlatform.passthru.authorityJson;
  platformWheelhouse = trainingPlatform.passthru.wheelhouse;

  singleProfileJson = rwkv7LoraP150a.passthru.profileJson;
  singleMeasurementProfileJson = rwkv7LoraP150a.passthru.measurementProfileJson;
  singleStepReceipt = ../rwkv7-lora-p150a/evidence/2026-08-05-full-model-step/receipt.json;
  singleRunReceipt = ../rwkv7-lora-p150a/evidence/2026-08-05-multistep-run/receipt.json;
  singleCheckpointReceipt = ../rwkv7-lora-p150a/evidence/2026-08-05-checkpoint-resume/receipt.json;

  profileJson = runCommand "rwkv7-lora-dual-p150a-profile.json" { nativeBuildInputs = [ nickel ]; } ''
    cp -r ${packageSource} "$PWD/repository"
    chmod -R u+w "$PWD/repository"
    cd "$PWD/repository/pkgs/rwkv7-lora-dual-p150a"
    nickel export --format json profile.ncl > "$out"
  '';

  measurementProfileJson =
    runCommand "rwkv7-lora-dual-p150a-measurement-profile.json" { nativeBuildInputs = [ nickel ]; }
      ''
        cp -r ${packageSource} "$PWD/repository"
        chmod -R u+w "$PWD/repository"
        cd "$PWD/repository/pkgs/rwkv7-lora-dual-p150a"
        nickel export --format json measurement-profile.ncl > "$out"
      '';

  regressionPolicyJson =
    runCommand "rwkv7-lora-dual-p150a-regression-policy.json" { nativeBuildInputs = [ nickel ]; }
      ''
        cp -r ${packageSource} "$PWD/repository"
        chmod -R u+w "$PWD/repository"
        cd "$PWD/repository/pkgs/rwkv7-lora-dual-p150a"
        nickel export --format json regression-policy.ncl > "$out"
      '';

  stabilityPolicyJson =
    runCommand "rwkv7-lora-dual-p150a-stability-policy.json" { nativeBuildInputs = [ nickel ]; }
      ''
        cp -r ${packageSource} "$PWD/repository"
        chmod -R u+w "$PWD/repository"
        cd "$PWD/repository/pkgs/rwkv7-lora-dual-p150a"
        nickel export --format json stability-policy.ncl > "$out"
      '';

  wheelsJson = runCommand "rwkv7-lora-dual-p150a-wheels.json" { nativeBuildInputs = [ nickel ]; } ''
    cp -r ${packageSource} "$PWD/repository"
    chmod -R u+w "$PWD/repository"
    cd "$PWD/repository/pkgs/rwkv7-lora-dual-p150a"
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
      --subst-var-by date "${coreutils}/bin/date" \
      --subst-var-by mv "${coreutils}/bin/mv" \
      --subst-var-by rm "${coreutils}/bin/rm" \
      --subst-var-by sleep "${coreutils}/bin/sleep" \
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
    mkdir -p "$out/bin" "$out/share/rwkv7-lora-dual-p150a" "$out/share/model-target"
    ln -s ${core}/bin/rwkv7-lora-dual-p150a-core "$out/bin/rwkv7-lora-dual-p150a-core"
    ln -s ${collectiveRunner} "$out/bin/rwkv7-lora-dual-p150a-collective"
    ln -s ${fullStepRunner} "$out/bin/rwkv7-lora-dual-p150a-full-step"
    ln -s ${profileJson} "$out/share/rwkv7-lora-dual-p150a/profile.json"
    ln -s ${measurementProfileJson} "$out/share/rwkv7-lora-dual-p150a/measurement-profile.json"
    ln -s ${regressionPolicyJson} "$out/share/rwkv7-lora-dual-p150a/regression-policy.json"
    ln -s ${stabilityPolicyJson} "$out/share/rwkv7-lora-dual-p150a/stability-policy.json"
    ln -s ${./baseline-store.json} "$out/share/rwkv7-lora-dual-p150a/baseline-store.json"
    ln -s ${wheelsJson} "$out/share/rwkv7-lora-dual-p150a/wheels.json"
    echo '${targetProfile.profile_identity}' >"$out/share/model-target/profile-blake3"
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
    PLATFORM_RECEIPT = "${platformEvidenceReceipt}";
    CPU_RECEIPT = "${cpuReference}/receipt.json";
    SINGLE_PROFILE = "${singleProfileJson}";
    SINGLE_MEASUREMENT_PROFILE = "${singleMeasurementProfileJson}";
    SINGLE_STEP_RECEIPT = "${singleStepReceipt}";
    SINGLE_RUN_RECEIPT = "${singleRunReceipt}";
    SINGLE_CHECKPOINT_RECEIPT = "${singleCheckpointReceipt}";
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
    cd "$PWD/repository/pkgs/rwkv7-lora-dual-p150a"

    # Valid profile exports deterministically and carries a stable identity.
    nickel export --format json profile.ncl >profile-first.json
    nickel export --format json profile.ncl >profile-second.json
    cmp profile-first.json profile-second.json
    first_identity=$(b3sum profile-first.json | cut -d' ' -f1)
    second_identity=$(b3sum profile-second.json | cut -d' ' -f1)
    test "$first_identity" = "$second_identity"
    grep -F '"profile_id": "rwkv7-lora-dual-p150a"' profile-first.json
    grep -F '"bundle_id": "tt-xla-training-2026-08-04"' profile-first.json
    nickel export --format json measurement-profile.ncl >measurement-profile-first.json
    nickel export --format json measurement-profile.ncl >measurement-profile-second.json
    cmp measurement-profile-first.json measurement-profile-second.json
    grep -F '"profile_id": "rwkv7-lora-dual-p150a-measurement"' measurement-profile-first.json
    grep -F '"parity_profile_id": "rwkv7-lora-dual-p150a"' measurement-profile-first.json
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

    # r[verify tenstorrent.harness.measurement_profile.valid]
    # r[verify tenstorrent.harness.stage_breakdown.valid]
    # r[verify tenstorrent.harness.speedup_ratio.valid]
    # r[verify tenstorrent.harness.machine_facts.valid]
    # r[verify tenstorrent.harness.repeat_stats.valid]
    cat >machine-fact-values.json <<'FACTS'
    {
      "board_identity": "p150a:device-0,p150a:device-1",
      "firmware_version": "18.8.0.0",
      "mesh_descriptor_identity": "p150_x2:1x2",
      "pcie_link": "device-0=gen4-x16,device-1=gen4-x16",
      "power_state": "performance"
    }
    FACTS
    "$CORE" bind-machine-facts machine-fact-values.json >machine-facts.json
    cpu_baseline_identity=$(printf '%s' 'dual-cpu-baseline' | b3sum | cut -d' ' -f1)
    jq -n \
      --arg cpu_baseline_identity "$cpu_baseline_identity" \
      --slurpfile machine_facts machine-facts.json \
      '{
        stage_times_seconds: {
          launch: 1.0,
          compile: 2.0,
          step: 3.0,
          checkpoint: 0.0,
          teardown: 1.0,
          end_to_end: 7.0
        },
        cpu_baseline: {
          identity_blake3: $cpu_baseline_identity,
          step_seconds: 10.0
        },
        machine_facts: $machine_facts[0],
        runs: [
          {time_seconds: 0.6, loss: 5.0},
          {time_seconds: 0.7, loss: 4.9},
          {time_seconds: 0.5, loss: 4.8},
          {time_seconds: 0.6, loss: 4.7},
          {time_seconds: 0.6, loss: 4.6}
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
    eval_split_identity=$(printf '%s' 'dual-heldout-eval-split' | b3sum | cut -d' ' -f1)
    reference_batch_identity=$(printf '%s' 'dual-periodic-reference-batch' | b3sum | cut -d' ' -f1)
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
            devices: [
              {device_id: 0, temperature_celsius: 70.0, power_watts: 100.0},
              {device_id: 1, temperature_celsius: 71.0, power_watts: 101.0}
            ]
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
    jq '.thermal.samples[2].devices[1].power_watts = "NaN"' stability-raw.json \
      >stability-nonfinite-thermal.json
    if "$CORE" summarize-stability stability-nonfinite-thermal.json profile-first.json \
      stability-policy.json > /dev/null 2> rejected.log; then
      echo "non-finite thermal sample was admitted" >&2
      exit 1
    fi
    grep -F "step 48, device 1" rejected.log
    jq '.heldout_eval.split_identity = .heldout_eval.reference_batch_identity' \
      stability-raw.json >stability-reference-eval.json
    if "$CORE" summarize-stability stability-reference-eval.json profile-first.json \
      stability-policy.json > /dev/null 2> rejected.log; then
      echo "reference-batch eval was admitted as held out" >&2
      exit 1
    fi
    grep -F "eval_not_held_out" rejected.log

    # Build a compatible one-device measurement from the same harness profile.
    cp "$SINGLE_MEASUREMENT_PROFILE" single-measurement-profile.json
    single_cpu_baseline_identity=$(printf '%s' 'scaling-single-cpu-baseline' | b3sum | cut -d' ' -f1)
    jq -n \
      --arg cpu_baseline_identity "$single_cpu_baseline_identity" \
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
          {time_seconds: 1.0, loss: 4.9},
          {time_seconds: 1.0, loss: 4.8},
          {time_seconds: 1.0, loss: 4.7},
          {time_seconds: 1.0, loss: 4.6}
        ]
      }' >single-measurement-raw.json
    "$CORE" summarize-measurement single-measurement-raw.json \
      single-measurement-profile.json >single-measurement-receipt.json

    # r[verify tenstorrent.harness.utilization.valid]
    # r[verify tenstorrent.harness.memory_pressure.valid]
    # r[verify tenstorrent.harness.op_latency.valid]
    # r[verify tenstorrent.harness.scaling_efficiency.valid]
    jq -n \
      --slurpfile measurement_receipt measurement-receipt.json \
      --slurpfile telemetry fixtures/device-telemetry.json \
      --slurpfile single_profile single-measurement-profile.json \
      --slurpfile single_receipt single-measurement-receipt.json \
      '{
        measurement_receipt: $measurement_receipt[0],
        telemetry: $telemetry[0],
        scaling_baseline: {
          profile: $single_profile[0],
          receipt: $single_receipt[0]
        }
      }' >utilization-raw.json
    "$CORE" summarize-utilization utilization-raw.json measurement-profile-first.json \
      >utilization-receipt.json
    "$CORE" summarize-utilization utilization-raw.json measurement-profile-first.json \
      >utilization-receipt-again.json
    cmp utilization-receipt.json utilization-receipt-again.json
    "$CORE" admit-utilization utilization-receipt.json utilization-raw.json \
      measurement-profile-first.json >utilization-admission.json
    grep -F '"status":"admitted"' utilization-admission.json
    grep -F '"device_count":2' utilization-admission.json
    grep -F '"operation_count":2' utilization-admission.json
    grep -F '"scaling_efficiency":' utilization-admission.json

    # r[verify tenstorrent.harness.scaling_efficiency.invalid]
    jq '.scaling_baseline = null' utilization-raw.json >utilization-missing-scaling.json
    if "$CORE" summarize-utilization utilization-missing-scaling.json \
      measurement-profile-first.json > /dev/null 2> rejected.log; then
      echo "dual-device scaling without a baseline was admitted" >&2
      exit 1
    fi
    grep -F "missing_scaling_baseline" rejected.log

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
    jq 'del(.machine_facts.power_state)' measurement-raw.json >measurement-missing-fact.json
    if "$CORE" summarize-measurement measurement-missing-fact.json measurement-profile-first.json \
      > /dev/null 2> rejected.log; then
      echo "measurement without power fact was admitted" >&2
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
    jq '.runs[0].loss = "NaN"' measurement-raw.json >measurement-nonfinite.json
    if "$CORE" summarize-measurement measurement-nonfinite.json measurement-profile-first.json \
      > /dev/null 2> rejected.log; then
      echo "measurement with a non-finite metric was admitted" >&2
      exit 1
    fi
    grep -F "invalid_metric" rejected.log

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

    # The current hardware receipt must re-validate against the package-owned
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

    mkdir -p "$out/share/model-target"
    echo '${targetProfile.profile_identity}' >"$out/share/model-target/profile-blake3"
    cp profile-first.json "$out/profile.json"
    cp profile-admission.json "$out/profile-admission.json"
    cp shard-plan.json "$out/shard-plan.json"
    cp platform-admission.json "$out/platform-admission.json"
    cp prerequisite-admission.json "$out/prerequisite-admission.json"
  ''
