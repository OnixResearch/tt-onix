{
  lib,
  rustPlatform,
  nickel,
  jq,
  coreutils,
  b3sum,
  python3,
  ttsim,
  tt-smi,
  targetContext,
  deviceRuntime,
  diagnosticDeviceRuntime,
  packageName ? "rwkv7-p150x2-evidence",
  persistentDecodeProfile ? ../rwkv7-p150x2/persistent-decode-profile.ncl,
  persistentDecodeProfileBlake3 ? "",
  monitoringPolicy ? ./persistent-decode-monitoring-policy.ncl,
  canaryPolicy ? ./persistent-decode-canary-policy.ncl,
  throughputProfile ? ./persistent-decode-throughput-profile.ncl,
  canaryThroughputProfile ? ./persistent-decode-canary-throughput-profile.ncl,
  deploymentPolicy ? null,
  observationPolicy ? null,
}:
let
  pythonEnvironment = python3.withPackages (packages: [
    packages.blake3
    packages.numpy
    packages.safetensors
    packages.torch
  ]);
  profile = targetContext.profile;
  artifacts = targetContext.artifacts;
  artifactBlake3 = lib.mapAttrs (_role: artifact: artifact.blake3) artifacts;
  persistentDecodeProfileFile = builtins.baseNameOf (toString persistentDecodeProfile);
  monitoringPolicyFile = builtins.baseNameOf (toString monitoringPolicy);
  canaryPolicyFile = builtins.baseNameOf (toString canaryPolicy);
  throughputProfileFile = builtins.baseNameOf (toString throughputProfile);
  canaryThroughputProfileFile = builtins.baseNameOf (toString canaryThroughputProfile);
  deploymentPolicyFile =
    if deploymentPolicy == null then null else builtins.baseNameOf (toString deploymentPolicy);
  observationPolicyFile =
    if observationPolicy == null then null else builtins.baseNameOf (toString observationPolicy);
  packageSource = lib.fileset.toSource {
    root = ../.;
    fileset = lib.fileset.unions [
      ./.
      ../model-target-core
      ../rwkv7-p150x2/kernel
      ../rwkv7-p150x2/src/fused_wkv_program.cpp
      ../rwkv7-p150x2/src/hardware_counter_diagnostics.cpp
      ../rwkv7-p150x2/src/hardware_counter_diagnostics.hpp
      ../rwkv7-p150x2/src/main.cpp
      ../rwkv7-p150x2/src/persistent_decode.cpp
      ../rwkv7-p150x2/src/persistent_decode.hpp
      ../rwkv7-p150x2/src/persistent_decode_self_test.cpp
      ../rwkv7-p150x2/src/persistent_decode_self_test.hpp
      ../rwkv7-p150x2/persistent-decode-contracts.ncl
      ../rwkv7-p150x2/persistent-decode-profile.ncl
      ../rwkv7-p150x2/persistent-decode-window8-contracts.ncl
      ../rwkv7-p150x2/persistent-decode-window8-candidate-profile.ncl
      ../rwkv7-p150x2/persistent-decode-window16-contracts.ncl
      ../rwkv7-p150x2/persistent-decode-window16-candidate-profile.ncl
      ../rwkv7-p150x2/persistent-decode-window16-candidate-v2-contracts.ncl
      ../rwkv7-p150x2/persistent-decode-window16-candidate-v2-profile.ncl
    ];
  };
in
rustPlatform.buildRustPackage {
  pname = packageName;
  version = "0.1.0";

  src = packageSource;
  sourceRoot = "source/rwkv7-p150x2-evidence";
  cargoLock.lockFile = ./Cargo.lock;
  nativeBuildInputs = [
    b3sum
    nickel
    pythonEnvironment
  ];

  MODEL_TARGET_PROFILE_BLAKE3 = profile.profile_identity;
  MODEL_TARGET_NAME = profile.target_name;
  MODEL_TARGET_MODEL_ID = profile.model.id;
  MODEL_TARGET_REVISION = profile.model.revision;
  MODEL_TARGET_ARCHITECTURE_FAMILY = profile.model.architecture_family;
  MODEL_TARGET_STATE_KIND = profile.state.kind;
  MODEL_TARGET_PRECISION = profile.model.precision;
  MODEL_TARGET_ARCHITECTURE_ADAPTER = profile.adapters.architecture;
  MODEL_TARGET_EVIDENCE_ADAPTER = profile.adapters.evidence;
  MODEL_TARGET_ARTIFACT_BLAKE3_JSON = builtins.toJSON artifactBlake3;
  MODEL_TARGET_TOPOLOGY_ID = profile.topology.id;
  MODEL_TARGET_MESH_SHAPE_JSON = builtins.toJSON profile.topology.mesh_shape;
  MODEL_TARGET_PHYSICAL_DEVICE_IDS_JSON = builtins.toJSON profile.topology.physical_device_ids;
  MODEL_TARGET_REPLICATION = profile.topology.replication;
  MODEL_TARGET_REQUIRED_OPERATIONS_JSON = builtins.toJSON targetContext.architectureAdapter.requiredOperations;
  MODEL_TARGET_CHECKPOINT_BLAKE3 = artifacts.checkpoint.blake3;
  MODEL_TARGET_TOKENIZER_VOCABULARY_BLAKE3 = artifacts.tokenizer_vocabulary.blake3;
  RWKV7_TOKENIZER_VOCABULARY = artifacts.tokenizer_vocabulary.path;
  RWKV7_CPU_REFERENCE_MODEL = artifacts.checkpoint.path;
  RWKV7_DEVICE_RUNTIME = deviceRuntime;
  RWKV7_DIAGNOSTIC_DEVICE_RUNTIME = diagnosticDeviceRuntime;
  RWKV7_PERSISTENT_DECODE_PROFILE_BLAKE3 = persistentDecodeProfileBlake3;
  TT_SMI_PATH = tt-smi;

  postInstall = ''
    mkdir -p "$out/share/model-target" "$out/share/rwkv7-p150x2-throughput"
    echo '${profile.profile_identity}' >"$out/share/model-target/profile-blake3"
    nickel export --format json \
      ../rwkv7-p150x2/${lib.escapeShellArg persistentDecodeProfileFile} \
      >"$out/share/rwkv7-p150x2-throughput/persistent-decode-profile.json"
    ${lib.optionalString (persistentDecodeProfileBlake3 != "") ''
      actual_profile_blake3="$(${lib.getExe b3sum} "$out/share/rwkv7-p150x2-throughput/persistent-decode-profile.json" | cut -d' ' -f1)"
      test "$actual_profile_blake3" = ${lib.escapeShellArg persistentDecodeProfileBlake3}
      echo "$actual_profile_blake3" \
        >"$out/share/rwkv7-p150x2-throughput/persistent-decode-profile-blake3"
    ''}
    cp throughput-contracts.ncl throughput-profile.ncl \
      persistent-decode-throughput-profile.ncl \
      persistent-decode-monitoring-contracts.ncl \
      persistent-decode-monitoring-policy.ncl \
      persistent-decode-canary-contracts.ncl \
      persistent-decode-canary-policy.ncl \
      persistent-decode-canary-throughput-profile.ncl \
      persistent-decode-window8-monitoring-contracts.ncl \
      persistent-decode-window8-monitoring-policy.ncl \
      persistent-decode-window8-canary-contracts.ncl \
      persistent-decode-window8-canary-policy.ncl \
      persistent-decode-window8-throughput-profile.ncl \
      persistent-decode-window8-canary-throughput-profile.ncl \
      persistent-decode-window16-monitoring-contracts.ncl \
      persistent-decode-window16-monitoring-policy.ncl \
      persistent-decode-window16-canary-contracts.ncl \
      persistent-decode-window16-canary-policy.ncl \
      persistent-decode-window16-throughput-profile.ncl \
      persistent-decode-window16-canary-throughput-profile.ncl \
      persistent-decode-window16-candidate-v2-monitoring-contracts.ncl \
      persistent-decode-window16-candidate-v2-monitoring-policy.ncl \
      persistent-decode-window16-candidate-v2-canary-contracts.ncl \
      persistent-decode-window16-candidate-v2-canary-policy.ncl \
      persistent-decode-window16-candidate-v2-throughput-profile.ncl \
      persistent-decode-window16-candidate-v2-canary-throughput-profile.ncl \
      persistent-decode-window16-candidate-v2-deployment-contracts.ncl \
      persistent-decode-window16-candidate-v2-deployment-policy.ncl \
      persistent-decode-window16-candidate-v2-observation-contracts.ncl \
      persistent-decode-window16-candidate-v2-observation-policy.ncl \
      throughput-saturated-profile.ncl throughput-capacity-profile.ncl \
      fused-wkv-physical-profile.ncl fused-wkv-stall-profile.ncl \
      fixtures/throughput-workload.json \
      fixtures/persistent-decode-workload.json \
      fixtures/persistent-decode-lifecycle-workload.json \
      fixtures/persistent-decode-window8-workload.json \
      fixtures/persistent-decode-window8-lifecycle-workload.json \
      fixtures/persistent-decode-window16-workload.json \
      fixtures/persistent-decode-window16-lifecycle-workload.json \
      fixtures/invalid-persistent-decode-window16-candidate-v2-throughput-profile.ncl \
      fixtures/invalid-persistent-decode-window16-candidate-v2-deployment-policy.ncl \
      fixtures/invalid-persistent-decode-window16-candidate-v2-observation-policy.ncl \
      fixtures/throughput-saturated-workload.json \
      fixtures/fused-wkv-physical-workload.json \
      fixtures/fused-wkv-stall-workload.json \
      fixtures/model-extent.json \
      "$out/share/rwkv7-p150x2-throughput/"
    nickel export --format json throughput-profile.ncl \
      >"$out/share/rwkv7-p150x2-throughput/throughput-profile.json"
    nickel export --format json ${lib.escapeShellArg throughputProfileFile} \
      >"$out/share/rwkv7-p150x2-throughput/persistent-decode-throughput-profile.json"
    nickel export --format json ${lib.escapeShellArg monitoringPolicyFile} \
      >"$out/share/rwkv7-p150x2-throughput/persistent-decode-monitoring-policy.json"
    nickel export --format json ${lib.escapeShellArg canaryPolicyFile} \
      >"$out/share/rwkv7-p150x2-throughput/persistent-decode-canary-policy.json"
    nickel export --format json ${lib.escapeShellArg canaryThroughputProfileFile} \
      >"$out/share/rwkv7-p150x2-throughput/persistent-decode-canary-throughput-profile.json"
    ${lib.optionalString (deploymentPolicy != null) ''
      nickel export --format json ${lib.escapeShellArg deploymentPolicyFile} \
        >"$out/share/rwkv7-p150x2-throughput/persistent-decode-deployment-policy.json"
    ''}
    ${lib.optionalString (observationPolicy != null) ''
      nickel export --format json ${lib.escapeShellArg observationPolicyFile} \
        >"$out/share/rwkv7-p150x2-throughput/persistent-decode-observation-policy.json"
    ''}
    nickel export --format json persistent-decode-window8-monitoring-policy.ncl \
      >"$out/share/rwkv7-p150x2-throughput/persistent-decode-window8-monitoring-policy.json"
    nickel export --format json persistent-decode-window8-canary-policy.ncl \
      >"$out/share/rwkv7-p150x2-throughput/persistent-decode-window8-canary-policy.json"
    nickel export --format json persistent-decode-window8-throughput-profile.ncl \
      >"$out/share/rwkv7-p150x2-throughput/persistent-decode-window8-throughput-profile.json"
    nickel export --format json persistent-decode-window8-canary-throughput-profile.ncl \
      >"$out/share/rwkv7-p150x2-throughput/persistent-decode-window8-canary-throughput-profile.json"
    nickel export --format json persistent-decode-window16-monitoring-policy.ncl \
      >"$out/share/rwkv7-p150x2-throughput/persistent-decode-window16-monitoring-policy.json"
    nickel export --format json persistent-decode-window16-canary-policy.ncl \
      >"$out/share/rwkv7-p150x2-throughput/persistent-decode-window16-canary-policy.json"
    nickel export --format json persistent-decode-window16-throughput-profile.ncl \
      >"$out/share/rwkv7-p150x2-throughput/persistent-decode-window16-throughput-profile.json"
    nickel export --format json persistent-decode-window16-canary-throughput-profile.ncl \
      >"$out/share/rwkv7-p150x2-throughput/persistent-decode-window16-canary-throughput-profile.json"
    nickel export --format json throughput-saturated-profile.ncl \
      >"$out/share/rwkv7-p150x2-throughput/throughput-saturated-profile.json"
    nickel export --format json throughput-capacity-profile.ncl \
      >"$out/share/rwkv7-p150x2-throughput/throughput-capacity-profile.json"
    nickel export --format json fused-wkv-physical-profile.ncl \
      >"$out/share/rwkv7-p150x2-throughput/fused-wkv-physical-profile.json"
    nickel export --format json fused-wkv-stall-profile.ncl \
      >"$out/share/rwkv7-p150x2-throughput/fused-wkv-stall-profile.json"
    python reference/rwkv7_cpu_continuation.py --self-test \
      >"$out/share/rwkv7-p150x2-throughput/persistent-decode-window8-cpu-reference-self-test.json"
    python reference/rwkv7_cpu_continuation.py \
      --mode validate \
      --model "$RWKV7_CPU_REFERENCE_MODEL" \
      --workload "$PWD/fixtures/persistent-decode-window8-workload.json" \
      --output "$out/share/rwkv7-p150x2-throughput/persistent-decode-window8-cpu-reference-receipt.json"
    python reference/rwkv7_cpu_continuation_window16.py --self-test \
      >"$out/share/rwkv7-p150x2-throughput/persistent-decode-window16-cpu-reference-self-test.json"
    python reference/rwkv7_cpu_continuation_window16.py \
      --mode validate \
      --model "$RWKV7_CPU_REFERENCE_MODEL" \
      --workload "$PWD/fixtures/persistent-decode-window16-workload.json" \
      --output "$out/share/rwkv7-p150x2-throughput/persistent-decode-window16-cpu-reference-receipt.json"
    cp reference/rwkv7_cpu_continuation.py \
      reference/rwkv7_cpu_continuation_window16.py \
      "$out/share/rwkv7-p150x2-throughput/"
    substitute run-simulator-smoke.sh \
      "$out/bin/rwkv7-p150x2-simulator-smoke" \
      --replace-fail '@ttsim@' ${lib.escapeShellArg (toString ttsim)} \
      --replace-fail '@timeout@' ${lib.escapeShellArg "${coreutils}/bin/timeout"} \
      --replace-fail '@runtime@' ${lib.escapeShellArg (toString deviceRuntime)}
    substitute run-fused-wkv-simulator-smoke.sh \
      "$out/bin/rwkv7-p150x2-fused-wkv-simulator-smoke" \
      --replace-fail '@ttsim@' ${lib.escapeShellArg (toString ttsim)} \
      --replace-fail '@timeout@' ${lib.escapeShellArg "${coreutils}/bin/timeout"} \
      --replace-fail '@runtime@' ${lib.escapeShellArg (toString deviceRuntime)}
    substitute run-fused-wkv-layer-simulator-smoke.sh \
      "$out/bin/rwkv7-p150x2-fused-wkv-layer-simulator-smoke" \
      --replace-fail '@ttsim@' ${lib.escapeShellArg (toString ttsim)} \
      --replace-fail '@timeout@' ${lib.escapeShellArg "${coreutils}/bin/timeout"} \
      --replace-fail '@runtime@' ${lib.escapeShellArg (toString deviceRuntime)}
    substitute run-persistent-decode-simulator-smoke.sh \
      "$out/bin/rwkv7-p150x2-persistent-decode-simulator-smoke" \
      --replace-fail '@ttsim@' ${lib.escapeShellArg (toString ttsim)} \
      --replace-fail '@timeout@' ${lib.escapeShellArg "${coreutils}/bin/timeout"} \
      --replace-fail '@runtime@' ${lib.escapeShellArg (toString deviceRuntime)}
    chmod +x \
      "$out/bin/rwkv7-p150x2-simulator-smoke" \
      "$out/bin/rwkv7-p150x2-fused-wkv-simulator-smoke" \
      "$out/bin/rwkv7-p150x2-fused-wkv-layer-simulator-smoke" \
      "$out/bin/rwkv7-p150x2-persistent-decode-simulator-smoke"
  '';

  doCheck = true;
  doInstallCheck = true;
  nativeInstallCheckInputs = [
    nickel
    jq
  ];
  installCheckPhase = ''
    runHook preInstallCheck
    "$out/bin/rwkv7-p150x2-evidence" --self-test
    "$out/bin/rwkv7-p150x2-command" --self-test
    cpu_reference_root="$out/share/rwkv7-p150x2-throughput"
    cpu_reference_receipt="$cpu_reference_root/persistent-decode-window8-cpu-reference-receipt.json"
    cpu_reference_self_test="$cpu_reference_root/persistent-decode-window8-cpu-reference-self-test.json"
    jq -e '
      .outcome == "validated_cpu_continuation" and
      .known_prefix_matched == true and
      .audit.generated_tokens_equal == true and
      .generated_token_ids == [36786, 34, 308, 459, 39743, 4811, 31063, 32227]
    ' "$cpu_reference_receipt" >/dev/null
    jq -e '
      .positive_orientation_equivalence == true and
      .negative_shape_rejected == true
    ' "$cpu_reference_self_test" >/dev/null
    grep -F 'No production window-8 admission is established.' \
      "$cpu_reference_receipt"
    window_sixteen_cpu_receipt="$cpu_reference_root/persistent-decode-window16-cpu-reference-receipt.json"
    window_sixteen_cpu_self_test="$cpu_reference_root/persistent-decode-window16-cpu-reference-self-test.json"
    jq -e '
      .outcome == "validated_cpu_continuation" and
      .known_prefix_matched == true and
      .audit.generated_tokens_equal == true and
      .generated_token_ids == [36786, 34, 308, 459, 39743, 4811, 31063, 32227, 22799, 5230, 46170, 53200, 32487, 32515, 55809, 47]
    ' "$window_sixteen_cpu_receipt" >/dev/null
    jq -e '
      .positive_orientation_equivalence == true and
      .negative_shape_rejected == true
    ' "$window_sixteen_cpu_self_test" >/dev/null
    grep -F 'No production window-16 admission is established.' \
      "$window_sixteen_cpu_receipt"
    bash -n "$out/bin/rwkv7-p150x2-simulator-smoke"
    bash -n "$out/bin/rwkv7-p150x2-fused-wkv-simulator-smoke"
    bash -n "$out/bin/rwkv7-p150x2-fused-wkv-layer-simulator-smoke"
    bash -n "$out/bin/rwkv7-p150x2-persistent-decode-simulator-smoke"
    if "$out/bin/rwkv7-p150x2-throughput-benchmark" \
      >missing-benchmark-args.out 2>missing-benchmark-args.err; then
      echo "throughput benchmark accepted missing arguments" >&2
      exit 1
    fi
    grep -F 'usage:' missing-benchmark-args.err

    # r[verify tenstorrent.rwkv7_p150x2.persistent_decode_monitoring.policy]
    nickel export --format json persistent-decode-monitoring-policy.ncl \
      >monitoring-policy-first.json
    nickel export --format json persistent-decode-monitoring-policy.ncl \
      >monitoring-policy-second.json
    cmp monitoring-policy-first.json monitoring-policy-second.json
    if nickel export --format json \
      fixtures/invalid-persistent-decode-monitoring-policy.ncl \
      >invalid-monitoring-policy.out 2>invalid-monitoring-policy.err; then
      echo "invalid monitoring policy passed its Nickel contract" >&2
      exit 1
    fi
    test -s invalid-monitoring-policy.err
    nickel export --format json persistent-decode-window8-monitoring-policy.ncl \
      >monitoring-window8-policy-first.json
    nickel export --format json persistent-decode-window8-monitoring-policy.ncl \
      >monitoring-window8-policy-second.json
    cmp monitoring-window8-policy-first.json monitoring-window8-policy-second.json
    test "$(jq -c '.admitted_windows' monitoring-window8-policy-first.json)" = '[2,4,8]'
    nickel export --format json persistent-decode-window16-monitoring-policy.ncl \
      >monitoring-window16-policy-first.json
    nickel export --format json persistent-decode-window16-monitoring-policy.ncl \
      >monitoring-window16-policy-second.json
    cmp monitoring-window16-policy-first.json monitoring-window16-policy-second.json
    test "$(jq -c '.admitted_windows' monitoring-window16-policy-first.json)" = '[2,4,8,16]'
    selected_monitoring_policy="$out/share/rwkv7-p150x2-throughput/persistent-decode-monitoring-policy.json"
    nickel export --format json ${lib.escapeShellArg monitoringPolicyFile} \
      >selected-monitoring-policy-check.json
    cmp selected-monitoring-policy-check.json "$selected_monitoring_policy"

    monitoring="$out/bin/rwkv7-p150x2-persistent-decode-monitor"
    persistent_fixture=fixtures/monitoring/physical-persistent-telemetry-receipt.json
    fallback_fixture=fixtures/monitoring/physical-fallback-telemetry-receipt.json
    critical_status=3
    warning_status=2
    input_error_status=64

    # r[verify tenstorrent.rwkv7_p150x2.persistent_decode_monitoring.collect]
    "$monitoring" monitoring-policy-first.json \
      "$persistent_fixture" "$fallback_fixture" >monitoring-clean.json
    grep -F '"verdict": "clean"' monitoring-clean.json
    grep -F '"persistent_admission_recommendation": "retain"' monitoring-clean.json
    if grep -F 'fixtures/monitoring' monitoring-clean.json; then
      echo "monitoring receipt disclosed an input path" >&2
      exit 1
    fi

    # r[verify tenstorrent.rwkv7_p150x2.persistent_decode_monitoring.critical]
    if "$monitoring" monitoring-policy-first.json \
      "$persistent_fixture" "$persistent_fixture" >monitoring-critical.json; then
      echo "duplicate monitoring event returned a clean status" >&2
      exit 1
    else
      test "$?" -eq "$critical_status"
    fi
    grep -F '"code": "duplicate_event"' monitoring-critical.json
    grep -F '"persistent_admission_recommendation": "disable"' \
      monitoring-critical.json

    # r[verify tenstorrent.rwkv7_p150x2.persistent_decode_monitoring.fallback]
    minimum_samples="$(jq -r '.minimum_fallback_samples' monitoring-policy-first.json)"
    warning_inputs=""
    for ordinal in $(seq 1 "$minimum_samples"); do
      output="monitoring-fallback-$ordinal.json"
      jq --argjson ordinal "$ordinal" '.monitoring_fixture_ordinal = $ordinal' \
        "$fallback_fixture" >"$output"
      warning_inputs="$warning_inputs $output"
    done
    if "$monitoring" monitoring-policy-first.json $warning_inputs \
      >monitoring-warning.json; then
      echo "high fallback rate returned a clean status" >&2
      exit 1
    else
      test "$?" -eq "$warning_status"
    fi
    grep -F '"code": "fallback_rate_high"' monitoring-warning.json
    grep -F '"persistent_admission_recommendation": "retain"' \
      monitoring-warning.json

    # r[verify tenstorrent.rwkv7_p150x2.persistent_decode_monitoring.invalid]
    if "$monitoring" >monitoring-missing.out 2>monitoring-missing.err; then
      echo "monitoring command accepted missing arguments" >&2
      exit 1
    else
      test "$?" -eq "$input_error_status"
    fi
    grep -F 'monitoring_usage' monitoring-missing.err

    # r[verify tenstorrent.rwkv7_p150x2.persistent_decode_canary.policy]
    nickel export --format json persistent-decode-canary-policy.ncl \
      >canary-policy-first.json
    nickel export --format json persistent-decode-canary-policy.ncl \
      >canary-policy-second.json
    cmp canary-policy-first.json canary-policy-second.json
    nickel export --format json persistent-decode-canary-throughput-profile.ncl \
      >canary-throughput-first.json
    nickel export --format json persistent-decode-canary-throughput-profile.ncl \
      >canary-throughput-second.json
    cmp canary-throughput-first.json canary-throughput-second.json
    test "$(jq -r '.sample_count' canary-policy-first.json)" \
      -eq "$(jq -r '.repeat_count' canary-throughput-first.json)"
    expected_canary_request_executions=104
    test "$(jq -r '.maximum_request_executions' canary-policy-first.json)" \
      -eq "$expected_canary_request_executions"
    if nickel export --format json \
      fixtures/invalid-persistent-decode-canary-policy.ncl \
      >invalid-canary-policy.out 2>invalid-canary-policy.err; then
      echo "invalid canary policy passed its Nickel contract" >&2
      exit 1
    fi
    test -s invalid-canary-policy.err
    window_eight_generated_tokens=8
    window_eight_worker_warmup_tokens=4
    nickel export --format json persistent-decode-window8-canary-policy.ncl \
      >canary-window8-policy-first.json
    nickel export --format json persistent-decode-window8-canary-policy.ncl \
      >canary-window8-policy-second.json
    cmp canary-window8-policy-first.json canary-window8-policy-second.json
    nickel export --format json persistent-decode-window8-throughput-profile.ncl \
      >window8-throughput-first.json
    nickel export --format json persistent-decode-window8-throughput-profile.ncl \
      >window8-throughput-second.json
    cmp window8-throughput-first.json window8-throughput-second.json
    nickel export --format json persistent-decode-window8-canary-throughput-profile.ncl \
      >canary-window8-throughput-first.json
    nickel export --format json persistent-decode-window8-canary-throughput-profile.ncl \
      >canary-window8-throughput-second.json
    cmp canary-window8-throughput-first.json canary-window8-throughput-second.json
    test "$(jq -r '.workload.generated_token_count' window8-throughput-first.json)" \
      -eq "$window_eight_generated_tokens"
    test "$(jq -r '.timing.warmup_token_count' window8-throughput-first.json)" \
      -eq "$window_eight_worker_warmup_tokens"
    test "$(jq -r '.workload.generated_token_count' canary-window8-throughput-first.json)" \
      -eq "$window_eight_generated_tokens"
    test "$(jq -r '.timing.warmup_token_count' canary-window8-throughput-first.json)" \
      -eq "$window_eight_worker_warmup_tokens"
    window_sixteen_generated_tokens=16
    window_sixteen_worker_warmup_tokens=4
    nickel export --format json persistent-decode-window16-canary-policy.ncl \
      >canary-window16-policy-first.json
    nickel export --format json persistent-decode-window16-canary-policy.ncl \
      >canary-window16-policy-second.json
    cmp canary-window16-policy-first.json canary-window16-policy-second.json
    nickel export --format json persistent-decode-window16-throughput-profile.ncl \
      >window16-throughput-first.json
    nickel export --format json persistent-decode-window16-throughput-profile.ncl \
      >window16-throughput-second.json
    cmp window16-throughput-first.json window16-throughput-second.json
    nickel export --format json persistent-decode-window16-canary-throughput-profile.ncl \
      >canary-window16-throughput-first.json
    nickel export --format json persistent-decode-window16-canary-throughput-profile.ncl \
      >canary-window16-throughput-second.json
    cmp canary-window16-throughput-first.json canary-window16-throughput-second.json
    test "$(jq -r '.workload.generated_token_count' window16-throughput-first.json)" \
      -eq "$window_sixteen_generated_tokens"
    test "$(jq -r '.timing.warmup_token_count' window16-throughput-first.json)" \
      -eq "$window_sixteen_worker_warmup_tokens"
    test "$(jq -r '.workload.generated_token_count' canary-window16-throughput-first.json)" \
      -eq "$window_sixteen_generated_tokens"
    test "$(jq -r '.timing.warmup_token_count' canary-window16-throughput-first.json)" \
      -eq "$window_sixteen_worker_warmup_tokens"
    selected_canary_policy="$out/share/rwkv7-p150x2-throughput/persistent-decode-canary-policy.json"
    selected_throughput_profile="$out/share/rwkv7-p150x2-throughput/persistent-decode-throughput-profile.json"
    selected_canary_throughput_profile="$out/share/rwkv7-p150x2-throughput/persistent-decode-canary-throughput-profile.json"
    nickel export --format json ${lib.escapeShellArg canaryPolicyFile} \
      >selected-canary-policy-check.json
    nickel export --format json ${lib.escapeShellArg throughputProfileFile} \
      >selected-throughput-profile-check.json
    nickel export --format json ${lib.escapeShellArg canaryThroughputProfileFile} \
      >selected-canary-throughput-profile-check.json
    cmp selected-canary-policy-check.json "$selected_canary_policy"
    cmp selected-throughput-profile-check.json "$selected_throughput_profile"
    cmp selected-canary-throughput-profile-check.json \
      "$selected_canary_throughput_profile"
    nickel export --format json \
      persistent-decode-window16-candidate-v2-deployment-policy.ncl \
      >window16-v2-deployment-policy-first.json
    nickel export --format json \
      persistent-decode-window16-candidate-v2-deployment-policy.ncl \
      >window16-v2-deployment-policy-second.json
    cmp window16-v2-deployment-policy-first.json \
      window16-v2-deployment-policy-second.json
    nickel export --format json \
      persistent-decode-window16-candidate-v2-observation-policy.ncl \
      >window16-v2-observation-policy-first.json
    nickel export --format json \
      persistent-decode-window16-candidate-v2-observation-policy.ncl \
      >window16-v2-observation-policy-second.json
    cmp window16-v2-observation-policy-first.json \
      window16-v2-observation-policy-second.json
    if nickel export --format json \
      fixtures/invalid-persistent-decode-window16-candidate-v2-deployment-policy.ncl \
      >invalid-window16-v2-deployment.json \
      2>invalid-window16-v2-deployment.log; then
      echo "invalid window-16 v2 deployment policy passed validation" >&2
      exit 1
    fi
    test -s invalid-window16-v2-deployment.log
    if nickel export --format json \
      fixtures/invalid-persistent-decode-window16-candidate-v2-observation-policy.ncl \
      >invalid-window16-v2-observation.json \
      2>invalid-window16-v2-observation.log; then
      echo "invalid window-16 v2 observation policy passed validation" >&2
      exit 1
    fi
    test -s invalid-window16-v2-observation.log
    ${lib.optionalString (deploymentPolicy != null) ''
      selected_deployment_policy="$out/share/rwkv7-p150x2-throughput/persistent-decode-deployment-policy.json"
      nickel export --format json ${lib.escapeShellArg deploymentPolicyFile} \
        >selected-deployment-policy-check.json
      cmp selected-deployment-policy-check.json "$selected_deployment_policy"
    ''}
    ${lib.optionalString (observationPolicy != null) ''
      selected_observation_policy="$out/share/rwkv7-p150x2-throughput/persistent-decode-observation-policy.json"
      nickel export --format json ${lib.escapeShellArg observationPolicyFile} \
        >selected-observation-policy-check.json
      cmp selected-observation-policy-check.json "$selected_observation_policy"
    ''}

    canary="$out/bin/rwkv7-p150x2-persistent-decode-canary"
    if "$canary" >canary-missing.out 2>canary-missing.err; then
      echo "canary command accepted missing arguments" >&2
      exit 1
    else
      test "$?" -eq "$input_error_status"
    fi
    grep -F 'canary_usage' canary-missing.err

    # r[verify tenstorrent.rwkv7_p150x2.persistent_decode_canary.invalid]
    canary_failure_root="$PWD/canary-failure-root"
    if "$canary" \
      canary-policy-first.json \
      canary-throughput-first.json \
      fixtures/invalid-persistent-decode-canary-workload.json \
      monitoring-policy-first.json \
      "$canary_failure_root"; then
      echo "failed benchmark produced a retained canary" >&2
      exit 1
    else
      test "$?" -eq "$critical_status"
    fi
    jq -e '.state == "disabled" and .reason == "benchmark_failed"' \
      "$canary_failure_root/canary-interlock.json" >/dev/null
    jq -e '.decision.state == "disabled" and .decision.reason == "benchmark_failed"' \
      "$canary_failure_root/canary-receipt.json" >/dev/null
    expected_canary_planned_executions=72
    test "$(jq -r '.budget.planned_request_executions' \
      "$canary_failure_root/canary-receipt.json")" \
      -eq "$expected_canary_planned_executions"
    cmp "$canary_failure_root/canary-interlock.json" \
      "$canary_failure_root.canary-interlock.json"

    # r[verify tenstorrent.rwkv7_p150x2.throughput_profile.valid]
    nickel export --format json throughput-profile.ncl >throughput-profile-first.json
    nickel export --format json throughput-profile.ncl >throughput-profile-second.json
    cmp throughput-profile-first.json throughput-profile-second.json
    nickel export --format json fused-wkv-physical-profile.ncl \
      >fused-wkv-physical-profile-first.json
    nickel export --format json fused-wkv-physical-profile.ncl \
      >fused-wkv-physical-profile-second.json
    cmp fused-wkv-physical-profile-first.json fused-wkv-physical-profile-second.json
    grep -F '"per_card_slot_count": 16' fused-wkv-physical-profile-first.json
    grep -F '"request_count": 32' fused-wkv-physical-profile-first.json
    fused_request_count="$(jq -r '.workload.request_count' fused-wkv-physical-profile-first.json)"
    test "$(jq -r '.requests | length' fixtures/fused-wkv-physical-workload.json)" \
      -eq "$fused_request_count"
    test "$(jq -r '[.requests[].request_id] | unique | length' \
      fixtures/fused-wkv-physical-workload.json)" -eq "$fused_request_count"
    nickel export --format json fused-wkv-stall-profile.ncl \
      >fused-wkv-stall-profile-first.json
    nickel export --format json fused-wkv-stall-profile.ncl \
      >fused-wkv-stall-profile-second.json
    cmp fused-wkv-stall-profile-first.json fused-wkv-stall-profile-second.json
    grep -F '"generated_token_count": 1' fused-wkv-stall-profile-first.json
    grep -F '"warmup_token_count": 1' fused-wkv-stall-profile-first.json
    test "$(jq -r '.requests | length' fixtures/fused-wkv-stall-workload.json)" \
      -eq "$fused_request_count"
    test "$(jq -c '[.requests[].max_new_tokens] | unique' \
      fixtures/fused-wkv-stall-workload.json)" = '[1]'
    if nickel export --format json fixtures/invalid-throughput-lockstep.ncl \
      >invalid-profile.json 2>invalid-profile.log; then
      echo "lockstep throughput profile passed its typed contract" >&2
      exit 1
    fi
    test -s invalid-profile.log
    if nickel export --format json \
      fixtures/invalid-persistent-decode-window16-candidate-v2-throughput-profile.ncl \
      >invalid-window16-v2-throughput.json \
      2>invalid-window16-v2-throughput.log; then
      echo "invalid window-16 v2 throughput profile passed validation" >&2
      exit 1
    fi
    test -s invalid-window16-v2-throughput.log

    throughput_core="$out/bin/rwkv7-p150x2-throughput"
    profile_identity="$($throughput_core profile-identity throughput-profile-first.json)"
    jq --arg identity "$profile_identity" \
      '.profile_identity_blake3 = $identity' \
      fixtures/throughput-valid.json >throughput-raw.json

    # r[verify tenstorrent.rwkv7_p150x2.throughput_admission.valid]
    "$throughput_core" summarize throughput-profile-first.json throughput-raw.json \
      >throughput-receipt.json
    "$throughput_core" admit throughput-profile-first.json throughput-raw.json \
      throughput-receipt.json >throughput-admission.json
    grep -F '"status": "admitted"' throughput-admission.json

    # r[verify tenstorrent.rwkv7_p150x2.throughput_admission.estimated]
    jq '.two_card_runs[0].physical_hardware = false' throughput-raw.json \
      >throughput-invalid-estimated.json
    if "$throughput_core" summarize throughput-profile-first.json \
      throughput-invalid-estimated.json >invalid-estimated.out 2>invalid-estimated.err; then
      echo "estimated throughput evidence passed admission" >&2
      exit 1
    fi
    grep -F 'unattributed_throughput_run' invalid-estimated.out

    # r[verify tenstorrent.rwkv7_p150x2.scalable_card_local_batching.valid]
    capacity_profile="$out/share/rwkv7-p150x2-throughput/throughput-capacity-profile.json"
    extent="$out/share/rwkv7-p150x2-throughput/model-extent.json"
    "$throughput_core" capacity-plan "$capacity_profile" "$extent" >capacity-plans.json
    grep -F '"candidate_id": "B4"' capacity-plans.json
    grep -F '"candidate_id": "B16"' capacity-plans.json
    grep -F '"candidate_id": "B32"' capacity-plans.json
    grep -F '"candidate_id": "B64"' capacity-plans.json

    # r[verify tenstorrent.rwkv7_p150x2.scalable_card_local_batching.invalid_memory]
    if nickel export --format json fixtures/invalid-capacity-unsupported.ncl \
      >invalid-capacity.json 2>invalid-capacity.log; then
      echo "zero-slot capacity candidate passed its typed contract" >&2
      exit 1
    fi
    test -s invalid-capacity.log
    if "$throughput_core" capacity-plan \
      fixtures/invalid-capacity-plan.json "$extent" >invalid-capacity-plan.out 2>invalid-capacity-plan.err; then
      echo "unsafe capacity plan passed device-free admission" >&2
      exit 1
    fi
    grep -F 'insufficient_device_memory' invalid-capacity-plan.out
    # r[verify tenstorrent.rwkv7_p150x2.scalable_card_local_batching.target_miss]
    sweep_root="$out/share/rwkv7-p150x2-capacity-sweep-retained"
    mkdir -p "$sweep_root"
    cp -R fixtures/physical-capacity-sweep-retained/. "$sweep_root/"
    jq -e '
      ("b4-baseline/throughput-receipt.json" | input_filename),
      .two_card.aggregate_tokens_per_second > 0
    ' "$sweep_root/b4-baseline/throughput-receipt.json" >/dev/null
    # The retained sweep keeps B4 admitted and records the measured B16 blocker.
    grep -F 'B4 remains the admitted card-local capacity' \
      "$sweep_root/README.md"
    # The B16 re-admission evidence binds the per-token re-admission decision.
    b16_root="$out/share/rwkv7-p150x2-capacity-b16-admitted"
    mkdir -p "$b16_root"
    cp -R fixtures/physical-capacity-b16-admitted/. "$b16_root/"
    jq -e '.two_card.aggregate_tokens_per_second > 300' \
      "$b16_root/b16-r2/throughput-receipt.json" >/dev/null
    grep -F 'B16 is promoted in every order-balanced comparison' \
      "$b16_root/README.md"
    runHook postInstallCheck
  '';

  passthru = {
    inherit deviceRuntime profile;
    targetIdentity = {
      profileBlake3 = profile.profile_identity;
      targetName = profile.target_name;
      modelId = profile.model.id;
      revision = profile.model.revision;
      artifactBlake3 = artifactBlake3;
    };
  };

  meta = {
    description = "Fail-closed raw evidence classifier for RWKV-7 generation on P150x2";
    license = lib.licenses.mit;
    platforms = [ "x86_64-linux" ];
    mainProgram = "rwkv7-p150x2-evidence";
  };
}
