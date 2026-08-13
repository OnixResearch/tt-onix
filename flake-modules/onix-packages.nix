{
  self,
  inputs,
  lib,
  ...
}:
{
  perSystem =
    { system, pkgs, ... }:
    let
      supportedSystem = "x86_64-linux";
      isSupportedSystem = system == supportedSystem;
      ttwkv7Pkgs = import inputs.nixpkgs {
        inherit system;
        overlays = [ self.overlays.default ];
        config.allowUnfreePredicate = package: lib.getName package == "ttwkv7";
      };
      rwkvLab = pkgs.callPackage ../pkgs/rwkv-lab { };
      modelTargetConsumerGenerator = pkgs.callPackage ../pkgs/model-target-consumer-generator { };
      modelTargets = pkgs.callPackage ../model-targets { inherit pkgs; };
      rwkvProfile = modelTargets.getProfile "rwkv7-goose-world2.8-0.1b-p150x2";
      qwenCandidateProfile = modelTargets.getProfile "qwen3-6-35b-a3b-p150x1-candidate";
      rwkvArtifacts = modelTargets.fetchProfileArtifacts rwkvProfile;
      rwkvArchitectureAdapter = modelTargets.mkArchitectureAdapter rwkvProfile;
      rwkvEvidenceAdapter = modelTargets.mkEvidenceAdapter rwkvProfile;
      rwkvTopology = modelTargets.mkTopology rwkvProfile;
      rwkvTargetContext = {
        profile = rwkvProfile;
        artifacts = rwkvArtifacts;
        architectureAdapter = rwkvArchitectureAdapter;
        evidenceAdapter = rwkvEvidenceAdapter;
        topology = rwkvTopology;
      };
      ttMetalFetchQueueDiagnostics = pkgs.callPackage ../pkgs/tt-metal/fetch-queue-diagnostics.nix { };
      rwkv7P150x2DiagnosticRuntime = pkgs.callPackage ../pkgs/rwkv7-p150x2 {
        targetContext = rwkvTargetContext;
        tt-metal = ttMetalFetchQueueDiagnostics;
        fetchQueueDiagnostics = true;
      };
      rwkvTarget = modelTargets.mkModelTarget {
        profile = rwkvProfile;
        artifacts = rwkvArtifacts;
        architectureAdapter = rwkvArchitectureAdapter;
        evidenceAdapter = rwkvEvidenceAdapter;
        topology = rwkvTopology;
        builders = {
          runtime = targetContext: pkgs.callPackage ../pkgs/rwkv7-p150x2 { inherit targetContext; };
          evidence =
            context:
            pkgs.callPackage ../pkgs/rwkv7-p150x2-evidence {
              targetContext = context;
              deviceRuntime = context.runtime;
              diagnosticDeviceRuntime = rwkv7P150x2DiagnosticRuntime;
            };
          command =
            context:
            pkgs.runCommand "rwkv7-p150x2-command" { } ''
              mkdir -p "$out/bin" "$out/share/model-target"
              ln -s ${rwkv7P150x2Evidence}/bin/rwkv7-p150x2-command \
                "$out/bin/rwkv7-p150x2-command"
              echo '${context.profile.profile_identity}' \
                >"$out/share/model-target/profile-blake3"
            '';
          sourceBoundary = _context: rwkv7P150x2SourceCheck;
          closure = _context: rwkv7P150x2ProductionClosure;
        };
      };
      rwkvLayerHarness = pkgs.callPackage ../pkgs/rwkv-layer-harness {
        targetProfile = rwkvProfile;
        targetArtifacts = rwkvArtifacts;
      };
      modelTargetReport = pkgs.callPackage ../pkgs/model-target-report {
        targetProfile = qwenCandidateProfile;
      };
      rwkv7LoraReference = pkgs.callPackage ../pkgs/rwkv7-lora-reference {
        model = rwkvArtifacts.checkpoint.path;
        modelBlake3 = rwkvArtifacts.checkpoint.blake3;
        targetProfile = rwkvProfile;
      };
      # r[impl tenstorrent.training_platform.verification]
      trainingPlatform = pkgs.callPackage ../pkgs/training-platform { };
      # r[impl tenstorrent.rwkv7_p150a_finetuning.configuration]
      # r[impl tenstorrent.rwkv7_p150a_finetuning.authority]
      rwkv7LoraP150a = pkgs.callPackage ../pkgs/rwkv7-lora-p150a {
        inherit trainingPlatform;
        targetProfile = rwkvProfile;
        platformEvidenceReceipt = ../pkgs/training-platform/evidence/2026-08-08-dual-p150a-smoke/receipt.json;
        cpuReference = rwkv7LoraReference;
      };
      # r[impl tenstorrent.rwkv7_dual_p150a.configuration]
      # r[impl tenstorrent.rwkv7_dual_p150a.authority]
      rwkv7LoraDualP150a = pkgs.callPackage ../pkgs/rwkv7-lora-dual-p150a {
        inherit trainingPlatform rwkv7LoraP150a;
        targetProfile = rwkvProfile;
        platformEvidenceReceipt = ../pkgs/training-platform/evidence/2026-08-08-dual-p150a-smoke/receipt.json;
        cpuReference = rwkv7LoraReference;
      };
      # r[verify tenstorrent.training_platform.configuration.valid]
      trainingPlatformEval =
        extraConfig:
        import (inputs.nixpkgs + "/nixos/lib/eval-config.nix") {
          inherit system;
          modules = [
            self.nixosModules.default
            self.nixosModules."dual-p150a-training"
            {
              config = {
                nixpkgs.hostPlatform = system;
                system.stateVersion = "26.11";
                fileSystems."/".device = "nodev";
                fileSystems."/".fsType = "tmpfs";
                boot.loader.grub.enable = false;
              }
              // extraConfig;
            }
          ];
        };
      trainingPlatformExampleEval = trainingPlatformEval { };
      incompatibleTrainingDriver = pkgs.runCommand "tt-kmd-incompatible-fixture" {
        pname = "tt-kmd";
        version = "2.7.0";
      } "mkdir -p $out";
      incompatibleDescriptorPackage =
        pkgs.runCommand "training-descriptor-incompatible-fixture" { }
          "mkdir -p $out";
      trainingPlatformInvalidDriverEval = trainingPlatformEval {
        hardware.tenstorrent.driverPackage = lib.mkForce incompatibleTrainingDriver;
      };
      trainingPlatformInvalidDescriptorEval = trainingPlatformEval {
        hardware.tenstorrent.descriptorPackage = lib.mkForce incompatibleDescriptorPackage;
      };
      trainingPlatformInvalidHugepageEval = trainingPlatformEval {
        hardware.tenstorrent.training.hugepageMount = lib.mkForce "/tmp/unapproved-hugepages";
      };
      hasFailedAssertion =
        message: evaluation:
        lib.any (
          assertion: !assertion.assertion && lib.hasInfix message assertion.message
        ) evaluation.config.assertions;
      invalidDriverRejected = hasFailedAssertion "does not match admitted training driver" trainingPlatformInvalidDriverEval;
      invalidDescriptorRejected = hasFailedAssertion "package-owned TT-Metal descriptor" trainingPlatformInvalidDescriptorEval;
      invalidHugepageRejected = hasFailedAssertion "requires hugepageMount" trainingPlatformInvalidHugepageEval;
      trainingPlatformExampleCheck =
        assert lib.assertMsg invalidDriverRejected
          "the training example must reject an incompatible TT-KMD package";
        assert lib.assertMsg invalidDescriptorRejected
          "the training example must reject an incompatible descriptor package";
        assert lib.assertMsg invalidHugepageRejected
          "the training example must reject an incompatible hugepage mount";
        assert lib.assertMsg (
          trainingPlatformExampleEval.config.hardware.tenstorrent.meshName == "p150_x2"
        ) "the training example must select the p150_x2 mesh";
        assert lib.assertMsg (trainingPlatformExampleEval.config.virtualisation.docker.enable
        ) "the training example must enable docker";
        assert lib.assertMsg (
          lib.getVersion trainingPlatformExampleEval.config.hardware.tenstorrent.driverPackage == "2.10.0"
        ) "the training example must select TT-KMD 2.10.0";
        assert lib.assertMsg (
          trainingPlatformExampleEval.config.hardware.tenstorrent.accessPolicy == "group-restricted"
        ) "the training example must use group-restricted device access";
        assert lib.assertMsg (trainingPlatformExampleEval.config.hardware.tenstorrent.training.enable
        ) "the training example must enable the admitted host profile";
        pkgs.runCommand "training-platform-nixos-example" { } ''
          set -euo pipefail
          descriptor=${trainingPlatformExampleEval.config.environment.variables.TT_MESH_GRAPH_DESC_PATH}
          case "$descriptor" in
            */p150_x2_mesh_graph_descriptor.textproto) ;;
            *)
              echo "training example does not export the p150_x2 descriptor path" >&2
              exit 1
              ;;
          esac
          test -e "$descriptor"
          kernel_params=${pkgs.writeText "training-example-kernel-params" (lib.concatStringsSep "\n" trainingPlatformExampleEval.config.boot.kernelParams)}
          grep -Fx 'hugepagesz=1G' "$kernel_params"
          grep -Fx 'hugepages=8' "$kernel_params"
          mount_options=${
            pkgs.writeText "training-example-mount-options" (
              lib.concatStringsSep "\n" (
                map (mount: mount.options) trainingPlatformExampleEval.config.systemd.mounts
              )
            )
          }
          grep -F 'mode=0770' "$mount_options"
          grep -F 'gid=tenstorrent' "$mount_options"
          tmpfiles_rules=${pkgs.writeText "training-example-tmpfiles-rules" (lib.concatStringsSep "\n" trainingPlatformExampleEval.config.systemd.tmpfiles.rules)}
          grep -Fx 'd /var/lib/tenstorrent-training 0750 root tenstorrent - -' "$tmpfiles_rules"
          grep -Fx 'd /var/lib/tenstorrent-training/evidence 0770 root tenstorrent - -' "$tmpfiles_rules"
          echo '${trainingPlatformExampleEval.config.hardware.tenstorrent.meshName}' > "$out"
        '';
      modelTargetFactoryCheck = pkgs.callPackage ../model-targets/check.nix {
        inherit (pkgs) nickel;
        inherit modelTargetConsumerGenerator modelTargets;
      };
      rwkv7P150x2Runtime = rwkvTarget.runtime;
      windowEightMaximumGeneratedTokens = 8;
      # r[impl tenstorrent.rwkv7_p150x2.persistent_decode_window_8]
      rwkv7P150x2Window8CandidateRuntime = pkgs.callPackage ../pkgs/rwkv7-p150x2 {
        targetContext = rwkvTargetContext;
        persistentDecodeProfile = ../pkgs/rwkv7-p150x2/persistent-decode-window8-candidate-profile.ncl;
        maximumGeneratedTokens = windowEightMaximumGeneratedTokens;
        runtimeName = "rwkv7-p150x2-window8-candidate-runtime";
      };
      windowSixteenMaximumGeneratedTokens = 16;
      windowSixteenCandidateV2ProfileBlake3 = "acd6d609397beee32b6dbe83bd00d042661ce1961c2353dbfd5794870839fee6";
      # r[impl tenstorrent.rwkv7_p150x2.persistent_decode_window_16]
      rwkv7P150x2Window16CandidateRuntime = pkgs.callPackage ../pkgs/rwkv7-p150x2 {
        targetContext = rwkvTargetContext;
        persistentDecodeProfile = ../pkgs/rwkv7-p150x2/persistent-decode-window16-candidate-profile.ncl;
        maximumGeneratedTokens = windowSixteenMaximumGeneratedTokens;
        runtimeName = "rwkv7-p150x2-window16-candidate-runtime";
      };
      rwkv7P150x2Window16ReadoutDiagnosticRuntime = pkgs.callPackage ../pkgs/rwkv7-p150x2 {
        targetContext = rwkvTargetContext;
        persistentDecodeProfile = ../pkgs/rwkv7-p150x2/persistent-decode-window16-candidate-profile.ncl;
        maximumGeneratedTokens = windowSixteenMaximumGeneratedTokens;
        fusedWkvBf16ReadoutBoundary = true;
        fusedWkvFp32RecurrenceVectors = true;
        runtimeName = "rwkv7-p150x2-window16-readout-diagnostic-runtime";
      };
      rwkv7P150x2Window16CandidateV2Runtime = pkgs.callPackage ../pkgs/rwkv7-p150x2 {
        targetContext = rwkvTargetContext;
        persistentDecodeProfile = ../pkgs/rwkv7-p150x2/persistent-decode-window16-candidate-v2-profile.ncl;
        expectedPersistentDecodeProfileBlake3 = windowSixteenCandidateV2ProfileBlake3;
        maximumGeneratedTokens = windowSixteenMaximumGeneratedTokens;
        fusedWkvBf16ReadoutBoundary = true;
        fusedWkvFp32RecurrenceVectors = true;
        runtimeName = "rwkv7-p150x2-window16-candidate-v2-runtime";
      };
      # r[impl tenstorrent.rwkv7_p150x2.evidence_classifier]
      rwkv7P150x2Evidence = rwkvTarget.evidence;
      rwkv7P150x2Window8CandidateEvidence = pkgs.callPackage ../pkgs/rwkv7-p150x2-evidence {
        targetContext = rwkvTargetContext;
        deviceRuntime = rwkv7P150x2Window8CandidateRuntime;
        diagnosticDeviceRuntime = rwkv7P150x2DiagnosticRuntime;
        packageName = "rwkv7-p150x2-window8-candidate-evidence";
      };
      rwkv7P150x2Window16CandidateEvidence = pkgs.callPackage ../pkgs/rwkv7-p150x2-evidence {
        targetContext = rwkvTargetContext;
        deviceRuntime = rwkv7P150x2Window16CandidateRuntime;
        diagnosticDeviceRuntime = rwkv7P150x2DiagnosticRuntime;
        packageName = "rwkv7-p150x2-window16-candidate-evidence";
      };
      rwkv7P150x2Window16CandidateV2Evidence = pkgs.callPackage ../pkgs/rwkv7-p150x2-evidence {
        targetContext = rwkvTargetContext;
        deviceRuntime = rwkv7P150x2Window16CandidateV2Runtime;
        diagnosticDeviceRuntime = rwkv7P150x2Window16CandidateV2Runtime;
        packageName = "rwkv7-p150x2-window16-candidate-v2-evidence";
        persistentDecodeProfile = ../pkgs/rwkv7-p150x2/persistent-decode-window16-candidate-v2-profile.ncl;
        persistentDecodeProfileBlake3 = windowSixteenCandidateV2ProfileBlake3;
        monitoringPolicy = ../pkgs/rwkv7-p150x2-evidence/persistent-decode-window16-candidate-v2-monitoring-policy.ncl;
        canaryPolicy = ../pkgs/rwkv7-p150x2-evidence/persistent-decode-window16-candidate-v2-canary-policy.ncl;
        throughputProfile = ../pkgs/rwkv7-p150x2-evidence/persistent-decode-window16-candidate-v2-throughput-profile.ncl;
        canaryThroughputProfile = ../pkgs/rwkv7-p150x2-evidence/persistent-decode-window16-candidate-v2-canary-throughput-profile.ncl;
        deploymentPolicy = ../pkgs/rwkv7-p150x2-evidence/persistent-decode-window16-candidate-v2-deployment-policy.ncl;
        observationPolicy = ../pkgs/rwkv7-p150x2-evidence/persistent-decode-window16-candidate-v2-observation-policy.ncl;
      };
      rwkv7P150x2FusedWkvArchitectureCheck =
        pkgs.callPackage ../pkgs/rwkv7-p150x2/fused-wkv-architecture-check.nix
          { };
      rwkv7P150x2ProductionClosure = pkgs.closureInfo {
        rootPaths = [ rwkv7P150x2Evidence ];
      };
      rwkv7P150x2Window8CandidateClosure = pkgs.closureInfo {
        rootPaths = [ rwkv7P150x2Window8CandidateEvidence ];
      };
      rwkv7P150x2Window16CandidateClosure = pkgs.closureInfo {
        rootPaths = [ rwkv7P150x2Window16CandidateEvidence ];
      };
      rwkv7P150x2Window16CandidateV2Closure = pkgs.closureInfo {
        rootPaths = [ rwkv7P150x2Window16CandidateV2Evidence ];
      };
      rwkv7P150x2Attempt3Evidence =
        pkgs.callPackage ../pkgs/rwkv7-p150x2-evidence/terminal-attempt-3.nix
          {
            evidencePackage = rwkv7P150x2Evidence;
          };
      # r[impl tenstorrent.rwkv7_p150x2.device_free_validation]
      # r[verify tenstorrent.rwkv7_p150x2.full_device_runtime.complete_path]
      # r[verify tenstorrent.rwkv7_p150x2.device_free_validation.focused]
      rwkv7P150x2SourceCheck =
        pkgs.runCommand "rwkv7-p150x2-source-check"
          {
            nativeBuildInputs = [ pkgs.gnugrep ];
          }
          ''
            set -eu
            runtime_source=${../pkgs/rwkv7-p150x2/src/main.cpp}
            persistent_decode_source=${../pkgs/rwkv7-p150x2/src/persistent_decode.cpp}
            persistent_decode_test_source=${../pkgs/rwkv7-p150x2/src/persistent_decode_self_test.cpp}
            persistent_decode_profile=${../pkgs/rwkv7-p150x2/persistent-decode-profile.ncl}
            persistent_decode_window8_profile=${../pkgs/rwkv7-p150x2/persistent-decode-window8-candidate-profile.ncl}
            persistent_decode_window16_profile=${../pkgs/rwkv7-p150x2/persistent-decode-window16-candidate-profile.ncl}
            persistent_decode_window16_v2_profile=${../pkgs/rwkv7-p150x2/persistent-decode-window16-candidate-v2-profile.ncl}
            persistent_decode_evidence_source=${../pkgs/rwkv7-p150x2-evidence/src/persistent_decode_contract.rs}
            persistent_decode_monitoring_source=${../pkgs/rwkv7-p150x2-evidence/src/persistent_decode_monitoring.rs}
            persistent_decode_monitoring_command=${../pkgs/rwkv7-p150x2-evidence/src/bin/rwkv7-p150x2-persistent-decode-monitor.rs}
            persistent_decode_monitoring_policy=${../pkgs/rwkv7-p150x2-evidence/persistent-decode-monitoring-policy.ncl}
            persistent_decode_canary_source=${../pkgs/rwkv7-p150x2-evidence/src/persistent_decode_canary.rs}
            persistent_decode_canary_command=${../pkgs/rwkv7-p150x2-evidence/src/bin/rwkv7-p150x2-persistent-decode-canary.rs}
            persistent_decode_canary_policy=${../pkgs/rwkv7-p150x2-evidence/persistent-decode-canary-policy.ncl}
            hardware_counter_source=${../pkgs/rwkv7-p150x2/src/hardware_counter_diagnostics.cpp}
            command_source=${../pkgs/rwkv7-p150x2-evidence/src/bin/rwkv7-p150x2-command.rs}
            throughput_source=${../pkgs/rwkv7-p150x2-evidence/src/throughput.rs}
            benchmark_source=${../pkgs/rwkv7-p150x2-evidence/src/bin/rwkv7-p150x2-throughput-benchmark.rs}
            runtime_cmake_source=${../pkgs/rwkv7-p150x2/CMakeLists.txt}
            runtime_package_source=${../pkgs/rwkv7-p150x2/default.nix}
            evidence_package_source=${../pkgs/rwkv7-p150x2-evidence/default.nix}
            production_tt_metal_package_source=${../pkgs/tt-metal/default.nix}
            diagnostic_tt_metal_package_source=${../pkgs/tt-metal/fetch-queue-diagnostics.nix}
            fetch_queue_patch=${../pkgs/tt-metal/fetch-queue-diagnostics.patch}

            grep -F 'MeshShape{kMeshRows, mesh_columns}' "$runtime_source"
            grep -F 'DeviceContext(int physical_device_id)' "$runtime_source"
            grep -F 'physical_device_ids' "$runtime_source"
            grep -F 'replicate_tensor_to_mesh_mapper' "$runtime_source"
            grep -F 'zero_fast_model_state' "$runtime_source"
            grep -F 'ttnn::DataType::FLOAT32' "$runtime_source"
            grep -F 'set_profiler_flush_enabled(false)' "$runtime_source"
            grep -F 'simulated_single_card_layer_state_carry' "$runtime_source"
            grep -F 'simulated_single_card_fused_wkv_layer_state_carry' "$runtime_source"
            grep -F 'WkvExecutionPolicy::RequireFusedCandidate' "$runtime_source"
            grep -F 'ttnn::operations::trace::begin_trace_capture' "$runtime_source"
            grep -F 'ttnn::operations::trace::execute_trace' "$runtime_source"
            grep -F 'trace_capture_outside_timed_region' "$runtime_source"
            grep -F 'selected_token_device_resident_inside_region' "$runtime_source"
            grep -F 'exact_fp32_recurrent_state_parity' "$runtime_source"
            grep -F 'state_zeroed_before_slot_release' "$runtime_source"
            grep -F 'persistent_decode_telemetry' "$runtime_source"
            grep -F 'trace_setup_seconds' "$runtime_source"
            grep -F 'trace_replay_seconds' "$runtime_source"
            grep -F 'cleanup_seconds' "$runtime_source"
            grep -F 'rwkv7-persistent-decode-binding-v1' "$persistent_decode_source"
            grep -F 'summarize_persistent_telemetry' "$persistent_decode_source"
            grep -F 'validate_persistent_telemetry' "$persistent_decode_source"
            grep -F 'test_binding_drift_and_receipts' "$persistent_decode_test_source"
            grep -F 'test_slot_admission_and_recycling' "$persistent_decode_test_source"
            grep -F 'physically_admitted = true' "$persistent_decode_profile"
            grep -F 'physically_admitted = false' "$persistent_decode_window8_profile"
            grep -F 'admitted_decode_windows = contracts.AdmittedDecodeWindows' \
              "$persistent_decode_window8_profile"
            grep -F 'physically_admitted = false' "$persistent_decode_window16_profile"
            grep -F 'admitted_decode_windows = contracts.AdmittedDecodeWindows' \
              "$persistent_decode_window16_profile"
            grep -F 'profile_id = "rwkv7-p150x2-persistent-decode-window16-candidate-v2"' \
              "$persistent_decode_window16_v2_profile"
            grep -F 'predecessor_profile_blake3 = "0fd161f7172276d1efbe51623b1d110947e68de40024109618547cf7a42bb01a"' \
              "$persistent_decode_window16_v2_profile"
            grep -F 'fp32_recurrence_vectors = true' "$persistent_decode_window16_v2_profile"
            grep -F 'bf16_readout_boundary = true' "$persistent_decode_window16_v2_profile"
            grep -F 'validate_persistent_receipt' "$persistent_decode_evidence_source"
            grep -F 'validate_persistent_telemetry_receipt' "$persistent_decode_evidence_source"
            grep -F 'validated_persistent_telemetry' "$persistent_decode_evidence_source"
            grep -F 'deny_unknown_fields' "$persistent_decode_evidence_source"
            grep -F 'collect_persistent_decode_monitoring' "$persistent_decode_monitoring_source"
            grep -F 'ALERT_PARITY_FAILURE' "$persistent_decode_monitoring_source"
            grep -F 'ALERT_CLEANUP_FAILURE' "$persistent_decode_monitoring_source"
            grep -F 'ALERT_REPLAY_TIMEOUT' "$persistent_decode_monitoring_source"
            grep -F 'ALERT_TERMINAL_FAILURE' "$persistent_decode_monitoring_source"
            grep -F 'ALERT_UNSUPPORTED_ADMISSION' "$persistent_decode_monitoring_source"
            grep -F 'PersistentAdmissionRecommendation::Disable' "$persistent_decode_monitoring_source"
            grep -F 'read_bounded' "$persistent_decode_monitoring_command"
            grep -F 'CRITICAL_EXIT_STATUS' "$persistent_decode_monitoring_command"
            grep -F 'maximum_batch_events = 256' "$persistent_decode_monitoring_policy"
            grep -F 'maximum_fallback_rate = 0.25' "$persistent_decode_monitoring_policy"
            grep -F 'validate_canary_interlock' "$persistent_decode_source"
            grep -F 'CanaryInterlockBlocked' "$persistent_decode_source"
            grep -F 'load_canary_interlock' "$runtime_source"
            grep -F 'RWKV7_PERSISTENT_DECODE_CANARY_INTERLOCK' "$runtime_source"
            grep -F 'canary_interlock_allows_persistent' "$runtime_source"
            grep -F 'evaluate_completed_canary' "$persistent_decode_canary_source"
            grep -F 'validate_worker_interlock_receipts' "$persistent_decode_canary_source"
            grep -F 'summarize_balanced_comparative_throughput' "$throughput_source"
            grep -F 'run_balanced_one_card_pair' "$benchmark_source"
            grep -F 'wait_bounded' "$persistent_decode_canary_command"
            grep -F 'atomic_replace' "$persistent_decode_canary_command"
            grep -F 'maximum_request_executions = 104' "$persistent_decode_canary_policy"
            grep -F 'maximum_duration_seconds = 600' "$persistent_decode_canary_policy"
            grep -F 'deterministic_slot_assignments' "$throughput_source"
            grep -F 'minimum_dual_card_speedup' "$throughput_source"
            grep -F 'maximum_p99_latency_regression_fraction' "$throughput_source"
            grep -F 'TT_METAL_PROFILER_MID_RUN_DUMP' "$benchmark_source"
            grep -F '.arg("--wkv-policy")' "$benchmark_source"
            grep -F '.arg("--initialization-ready-barrier")' "$benchmark_source"
            grep -F '.arg("--warmup-start-barrier")' "$benchmark_source"
            grep -F '.arg("--timed-ready-barrier")' "$benchmark_source"
            grep -F '.arg("--timed-start-barrier")' "$benchmark_source"
            grep -F 'wait_for_worker_barriers' "$benchmark_source"
            grep -F 'synchronized_wave_timing' "$benchmark_source"
            grep -F '"synchronized_timed_start": true' "$benchmark_source"
            grep -F '.env("TT_METAL_INSPECTOR", "0")' "$benchmark_source"
            grep -F 'FUSED_CANDIDATE_MODE' "$benchmark_source"
            grep -F 'STALL_CAPTURE_MODE' "$benchmark_source"
            grep -F 'STALL_PROBE_ENVIRONMENT' "$benchmark_source"
            grep -F 'FETCH_QUEUE_DIAGNOSTICS_ENVIRONMENT' "$benchmark_source"
            grep -F 'PRIMARY_THREAD_CPUS_ENVIRONMENT' "$benchmark_source"
            grep -F 'PRIMARY_THREAD_CPU_ENVIRONMENT' "$benchmark_source"
            grep -F 'PRIMARY_THREAD_FIFO_PRIORITY_ENVIRONMENT' "$benchmark_source"
            grep -F 'PROFILER_PROGRAM_SUPPORT_COUNT_ENVIRONMENT' "$benchmark_source"
            grep -F 'STALL_PROBE_PROFILER_PROGRAM_SUPPORT_COUNT' "$benchmark_source"
            grep -F 'MAXIMUM_STALL_CAPTURE_ATTEMPTS_PER_SAMPLE' "$benchmark_source"
            grep -F '"failed_attempt_artifacts_retained": false' "$benchmark_source"
            grep -F 'PROFILER_GENERATED_TOKEN_COUNT' "$benchmark_source"
            grep -F 'PROFILER_WARMUP_TOKEN_COUNT' "$benchmark_source"
            grep -F 'PROFILER_DROPPED_MARKER' "$benchmark_source"
            grep -F 'require_complete_profiler_output' "$benchmark_source"
            grep -F 'compare_stall_capture_samples' "$benchmark_source"
            grep -F 'compare_boundary_timings' "$benchmark_source"
            grep -F 'compare_fetch_queue_diagnostics' "$benchmark_source"
            grep -F 'compare_hardware_counter_diagnostics' "$benchmark_source"
            grep -F 'diagnostic_runtime_executable' "$benchmark_source"
            grep -F 'stall_probe_step_boundary_timings' "$runtime_source"
            grep -F 'blocking_readback_seconds' "$runtime_source"
            grep -F 'model_submission_process_cpu_seconds' "$runtime_source"
            grep -F 'fetch_queue_diagnostics' "$runtime_source"
            grep -F 'hardware_counter_diagnostics' "$runtime_source"
            grep -F 'sched_setaffinity(0, sizeof(cpu_set), &cpu_set)' "$runtime_source"
            grep -F 'sched_setscheduler(0, SCHED_FIFO, &parameters)' "$runtime_source"
            grep -F 'primary_thread_cpu_affinity' "$runtime_source"
            grep -F 'primary_thread_fifo_priority' "$runtime_source"
            grep -F 'SYS_perf_event_open' "$hardware_counter_source"
            grep -F 'PR_TASK_PERF_EVENTS_ENABLE' "$hardware_counter_source"
            grep -F 'PERF_COUNT_HW_INSTRUCTIONS' "$hardware_counter_source"
            grep -F 'PERF_COUNT_HW_CACHE_MISSES' "$hardware_counter_source"
            grep -F 'attributes.pinned = spec.pinned' "$hardware_counter_source"
            grep -F './fetch-queue-diagnostics.patch' "$diagnostic_tt_metal_package_source"
            if grep -F './fetch-queue-diagnostics.patch' "$production_tt_metal_package_source"; then
              echo 'production TT-Metal includes diagnostic-only fetch-queue counters' >&2
              exit 1
            fi
            grep -F 'RWKV7_FETCH_QUEUE_DIAGNOSTICS' "$runtime_cmake_source"
            grep -F 'if(RWKV7_FETCH_QUEUE_DIAGNOSTICS)' "$runtime_cmake_source"
            grep -F 'src/hardware_counter_diagnostics.cpp' "$runtime_cmake_source"
            grep -F 'fetchQueueDiagnostics ? false' "$runtime_package_source"
            grep -F 'diagnosticDeviceRuntime' "$evidence_package_source"
            grep -F 'ZoneScopedN("wait_for_fetch_q_space")' "$fetch_queue_patch"
            grep -F '"dropped_profiler_markers": false' "$benchmark_source"
            grep -F '"raw_profiler_csv_retained": false' "$benchmark_source"
            grep -F 'order_balanced_wkv_policies' "$benchmark_source"
            grep -F '"exact_generated_token_parity": true' "$benchmark_source"
            grep -F '"production_physically_admitted": false' "$benchmark_source"
            grep -F 'profiler-program-signature.json' "$benchmark_source"
            grep -F '.env_remove("TT_METAL_WATCHER")' "$benchmark_source"
            grep -F '.env_remove("TT_METAL_DPRINT_CORES")' "$benchmark_source"
            grep -F 'ttnn::embedding' "$runtime_source"
            grep -F 'ttnn::matmul' "$runtime_source"
            grep -F 'ttnn::argmax' "$runtime_source"
            grep -F 'ReadDeviceProfilerResults' "$runtime_source"
            grep -F 'kStallProbeWarmupTokenCount' "$runtime_source"
            grep -F 'timed_start_unix_ns' "$runtime_source"
            grep -F '!environment.watcher.has_value()' "$runtime_source"
            grep -F '!environment.dprint_cores.has_value()' "$runtime_source"
            grep -F 'remove_environment(WATCHER_ENVIRONMENT)' "$command_source"
            grep -F 'remove_environment(DPRINT_CORES_ENVIRONMENT)' "$command_source"

            for output in \
              ${rwkv7P150x2Runtime} \
              ${rwkvTarget.command} \
              ${rwkv7P150x2Evidence}; do
              grep -Fx '${rwkvProfile.profile_identity}' \
                "$output/share/model-target/profile-blake3"
            done

            if grep -E 'torch|pytorch|cpu-dispatch|test_only_cpu_server|std::system|popen|fork\(' \
              "$runtime_source"; then
              echo 'full-device runtime contains a forbidden CPU-model or process fallback surface' >&2
              exit 1
            fi
            if grep -F 'set_environment(WATCHER_ENVIRONMENT' "$command_source"; then
              echo 'profiler command assigns a value to the presence-based Watcher variable' >&2
              exit 1
            fi
            touch "$out"
          '';
      # r[verify tenstorrent.rwkv7_p150x2.device_free_validation.cpu_fallback]
      rwkv7P150x2ClosureCheck =
        pkgs.runCommand "rwkv7-p150x2-production-closure-check"
          {
            nativeBuildInputs = [ pkgs.gnugrep ];
          }
          ''
            set -eu
            cp ${rwkv7P150x2ProductionClosure}/store-paths closure-paths.txt
            if grep -Ei '/[^/]*(pytorch|torch-|rwkv-layer-harness|python.*safetensors)' \
              closure-paths.txt; then
              echo 'production closure contains a CPU-model or PyTorch fallback dependency' >&2
              exit 1
            fi
            grep -F '${rwkv7P150x2Runtime}' closure-paths.txt
            cp closure-paths.txt "$out"
          '';
      ttwkv7 = ttwkv7Pkgs.callPackage ../pkgs/ttwkv7 {
        inherit (ttwkv7Pkgs) enchantum tt-logger tt-metal;
      };
      ttwkv7OwnerControl = pkgs.callPackage ../pkgs/ttwkv7-owner-control {
        commandName = "ttwkv7-owner-control";
        ownerUnit = "docker-tt-inference-server-llama-3-1-8b-instruct-p150.service";
        devicePath = "/dev/tenstorrent/1";
      };
      rwkvTtwkv7BoundaryDevice = pkgs.callPackage ../pkgs/rwkv-ttwkv7-boundary-device {
        inherit
          rwkvLab
          rwkvLayerHarness
          ttwkv7
          ttwkv7OwnerControl
          ;
      };
      rwkvTtwkv7PersistentDevice = pkgs.callPackage ../pkgs/rwkv-ttwkv7-persistent-device {
        inherit
          rwkvLab
          rwkvLayerHarness
          ttwkv7
          ttwkv7OwnerControl
          ;
      };
      rwkvTtwkv7PersistentReceiptValidator =
        pkgs.callPackage ../pkgs/rwkv-ttwkv7-persistent-receipt-validator
          { };
      rwkvTtwkv7HostLayoutCheck = ttwkv7Pkgs.callPackage ../pkgs/ttwkv7/rwkv-host-layout-check.nix {
        inherit rwkvLayerHarness ttwkv7;
      };
      rwkvTtwkv7DecodeReaderCheck = ttwkv7Pkgs.callPackage ../pkgs/ttwkv7/rwkv-decode-reader-check.nix {
        inherit rwkvLayerHarness ttwkv7;
      };
      rwkvTtwkv7PersistentDispatchTransportCheck =
        ttwkv7Pkgs.callPackage ../pkgs/ttwkv7/rwkv-persistent-dispatch-transport-check.nix
          {
            inherit rwkvLayerHarness ttwkv7;
          };
      rwkvTtwkv7BoundaryDeviceCheck = pkgs.callPackage ../pkgs/rwkv-ttwkv7-boundary-device/check.nix {
        boundaryDevice = rwkvTtwkv7BoundaryDevice;
        inherit ttwkv7;
      };
      rwkvTtwkv7PersistentDeviceCheck = pkgs.callPackage ../pkgs/rwkv-ttwkv7-persistent-device/check.nix {
        persistentDevice = rwkvTtwkv7PersistentDevice;
        inherit rwkvLayerHarness ttwkv7;
      };
      rwkvTtwkv7PersistentDevice4RunbookCheck =
        pkgs.callPackage ../pkgs/rwkv-ttwkv7-persistent-device/runbook-check.nix
          { };
      rwkvTtwkv7PersistentReceiptValidatorCheck =
        pkgs.callPackage ../pkgs/rwkv-ttwkv7-persistent-receipt-validator/check.nix
          { validator = rwkvTtwkv7PersistentReceiptValidator; };
      rwkvTtwkv7PersistentPartialDiagnosticCheck =
        pkgs.callPackage ../pkgs/rwkv-ttwkv7-persistent-evidence
          { };
      rwkvTtwkv7PersistentDevice4EvidenceCheck =
        pkgs.callPackage ../pkgs/rwkv-ttwkv7-persistent-passed-evidence
          { };
      ttVibeThinkerBench = pkgs.callPackage ../pkgs/tt-vibethinker-bench { };
      ttsim = pkgs.ttsim;
      rwkv7Chat = pkgs.callPackage ../pkgs/rwkv7-chat { };
      migrationBoundaryCheck = pkgs.callPackage ../checks/migration-boundary.nix { };
      rwkvArtifactBlake3 = lib.mapAttrs (_role: artifact: artifact.blake3) rwkvArtifacts;
      repositoryQuality = import ../checks/repository-quality.nix {
        inherit lib pkgs;
        cairn = inputs.cairn.packages.${system}.cairn;
        cairnPolicy = "${inputs.cairn}/cairn-policy/generated/cairn-policy.json";
        evidenceEnvironment = {
          MODEL_TARGET_PROFILE_BLAKE3 = rwkvProfile.profile_identity;
          MODEL_TARGET_NAME = rwkvProfile.target_name;
          MODEL_TARGET_MODEL_ID = rwkvProfile.model.id;
          MODEL_TARGET_REVISION = rwkvProfile.model.revision;
          MODEL_TARGET_ARCHITECTURE_FAMILY = rwkvProfile.model.architecture_family;
          MODEL_TARGET_STATE_KIND = rwkvProfile.state.kind;
          MODEL_TARGET_PRECISION = rwkvProfile.model.precision;
          MODEL_TARGET_ARCHITECTURE_ADAPTER = rwkvProfile.adapters.architecture;
          MODEL_TARGET_EVIDENCE_ADAPTER = rwkvProfile.adapters.evidence;
          MODEL_TARGET_ARTIFACT_BLAKE3_JSON = builtins.toJSON rwkvArtifactBlake3;
          MODEL_TARGET_TOPOLOGY_ID = rwkvProfile.topology.id;
          MODEL_TARGET_MESH_SHAPE_JSON = builtins.toJSON rwkvProfile.topology.mesh_shape;
          MODEL_TARGET_PHYSICAL_DEVICE_IDS_JSON = builtins.toJSON rwkvProfile.topology.physical_device_ids;
          MODEL_TARGET_REPLICATION = rwkvProfile.topology.replication;
          MODEL_TARGET_REQUIRED_OPERATIONS_JSON = builtins.toJSON rwkvArchitectureAdapter.requiredOperations;
          MODEL_TARGET_CHECKPOINT_BLAKE3 = rwkvArtifacts.checkpoint.blake3;
          MODEL_TARGET_TOKENIZER_VOCABULARY_BLAKE3 = rwkvArtifacts.tokenizer_vocabulary.blake3;
        };
      };
      repositoryQualityAggregate =
        pkgs.runCommand "tenstorrent-repository-quality"
          {
            qualityInputs = repositoryQuality.cargoChecks ++ [
              repositoryQuality.artifactQuality
              repositoryQuality.cairnQuality
              repositoryQuality.negativeLanguageQuality
              repositoryQuality.nixQuality
              repositoryQuality.policyQuality
              repositoryQuality.pythonQuality
              repositoryQuality.shellQuality
              migrationBoundaryCheck
              modelTargetFactoryCheck
              rwkvLayerHarness.passthru.frameworkParityCheck
              trainingPlatform
            ];
          }
          ''
            set -euo pipefail
            for quality_input in $qualityInputs; do
              test -e "$quality_input"
            done
            if test -e /dev/tenstorrent; then
              echo "device-free quality sandbox exposes /dev/tenstorrent" >&2
              exit 1
            fi
            touch "$out"
          '';
      qualityRunner = pkgs.writeShellApplication {
        name = "tenstorrent-quality";
        runtimeInputs = [
          pkgs.git
          pkgs.nix
        ];
        text = lib.removePrefix "#!/usr/bin/env bash\n" (builtins.readFile ../checks/run-quality.sh);
      };

      developmentAppSpecs = {
        ttwkv7 = {
          package = ttwkv7;
          executable = "wkv7";
          description = "Run the exact ttWKV7 package from this checkout";
        };
        ttwkv7-constant-probe = {
          package = ttwkv7;
          executable = "wkv7-constant-probe";
          description = "Run the ttWKV7 constant-tile probe wrapper from this checkout";
        };
        ttwkv7-data-movement = {
          package = ttwkv7;
          executable = "wkv7-data-movement";
          description = "Run the ttWKV7 data-movement wrapper from this checkout";
        };
        ttwkv7-diagnose = {
          package = ttwkv7;
          executable = "wkv7-diagnose";
          description = "Run the ttWKV7 cross-kernel diagnostic wrapper from this checkout";
        };
        training-platform-core = {
          package = trainingPlatform.core;
          executable = "training-platform-core";
          description = "Run the reviewed training-platform admission core";
        };
        training-platform-smoke = {
          package = trainingPlatform.tools;
          executable = "training-platform-smoke";
          description = "Run the reviewed bounded dual-P150A smoke";
        };
        rwkv7-chat = {
          package = rwkv7Chat;
          executable = "rwkv7-chat";
          description = "Chat with RWKV-7 on the Tenstorrent card via llama.cpp Metalium";
        };
        quality = {
          package = qualityRunner;
          executable = "tenstorrent-quality";
          description = "Run the repository-owned device-free quality gate";
        };
      };
      isSafeExecutableName =
        executable: builtins.match "^[A-Za-z0-9][A-Za-z0-9._+-]*$" executable != null;
      mkDevelopmentApp =
        appName:
        {
          package,
          executable,
          description,
        }:
        if !lib.isDerivation package then
          throw "development app '${appName}' requires a package derivation"
        else if !isSafeExecutableName executable then
          throw "development app '${appName}' has an unsafe executable name"
        else if description == "" then
          throw "development app '${appName}' requires a description"
        else
          {
            type = "app";
            program = "${package}/bin/${executable}";
            meta = { inherit description; };
          };
      developmentApps = lib.mapAttrs mkDevelopmentApp developmentAppSpecs;
      trainingPlatformDevShell = pkgs.mkShell {
        packages = [
          pkgs.b3sum
          pkgs.cargo
          pkgs.clippy
          pkgs.docker
          pkgs.jq
          pkgs.nickel
          pkgs.rustc
          pkgs.rustfmt
          pkgs.shellcheck
          pkgs.tt-smi
          trainingPlatform.core
          trainingPlatform.tools
        ];
        shellHook = ''
          unset PYTHONPATH PYTHONHOME TT_METAL_HOME TT_METAL_RUNTIME_ROOT
        '';
      };
      storeDirectory = builtins.storeDir;
      positiveDevelopmentAppFixture = builtins.tryEval (
        mkDevelopmentApp "positive-fixture" {
          package = ttwkv7;
          executable = "wkv7";
          description = "Positive development app fixture";
        }
      );
      unsafeDevelopmentAppFixture = builtins.tryEval (
        mkDevelopmentApp "unsafe-fixture" {
          package = ttwkv7;
          executable = "../wkv7";
          description = "Unsafe development app fixture";
        }
      );
      missingDescriptionFixture = builtins.tryEval (
        mkDevelopmentApp "missing-description-fixture" {
          package = ttwkv7;
          executable = "wkv7";
          description = "";
        }
      );
      invalidModeStatus = 2;
      shapeTestDiagnostic = "ttWKV7 checkpoint shape self-test: PASS S=64 H=12 Hpad=32 St=2 Ht=1 C=768 chunked_units=6";
      constantProbeDiagnostic = "constant-tile oracle self-test: PASS";
      dataMovementDiagnostic = "data-movement oracle self-test: PASS";
      invalidDiagnosticMode = "invalid-mode";
      invalidDiagnosticUsage = "wkv7-diagnose: usage: wkv7-diagnose validate-runtime|diagnose";
      developmentInterfaceCheck =
        assert lib.assertMsg positiveDevelopmentAppFixture.success
          "the positive development app fixture must evaluate";
        assert lib.assertMsg (
          !unsafeDevelopmentAppFixture.success
        ) "the development app constructor must reject path traversal";
        assert lib.assertMsg (
          !missingDescriptionFixture.success
        ) "the development app constructor must reject an empty description";
        pkgs.runCommand "tenstorrent-development-interface" { } ''
          set -euo pipefail

          for app_program in ${
            lib.escapeShellArgs (map (app: app.program) (builtins.attrValues developmentApps))
          }; do
            case "$app_program" in
              ${storeDirectory}/*/bin/*) ;;
              *)
                echo "development app does not use an exact Nix store executable: $app_program" >&2
                exit 1
                ;;
            esac
            test -x "$app_program"
          done

          ${developmentApps.ttwkv7.program} shape-test >shape-test.log
          grep -F ${lib.escapeShellArg shapeTestDiagnostic} shape-test.log

          ${developmentApps.ttwkv7-constant-probe.program} self-test >constant-probe.log
          grep -F ${lib.escapeShellArg constantProbeDiagnostic} constant-probe.log

          ${developmentApps.ttwkv7-data-movement.program} self-test >data-movement.log
          grep -F ${lib.escapeShellArg dataMovementDiagnostic} data-movement.log

          ${developmentApps.training-platform-core.program} self-test >training-platform.log
          grep -F 'training-platform-core self-test: PASS' training-platform.log

          if ${developmentApps.ttwkv7-diagnose.program} ${lib.escapeShellArg invalidDiagnosticMode} \
            >diagnostic-invalid.log 2>&1; then
            echo "ttwkv7-diagnose accepted an invalid mode" >&2
            exit 1
          else
            diagnostic_status="$?"
          fi
          test "$diagnostic_status" -eq ${toString invalidModeStatus}
          grep -F ${lib.escapeShellArg invalidDiagnosticUsage} diagnostic-invalid.log

          touch "$out"
        '';
    in
    {
      apps = lib.optionalAttrs isSupportedSystem developmentApps;

      devShells = lib.optionalAttrs isSupportedSystem {
        default = trainingPlatformDevShell;
        training-platform = trainingPlatformDevShell;
      };

      packages = lib.optionalAttrs isSupportedSystem {
        inherit ttwkv7;
        rwkv7-chat = rwkv7Chat;
        rwkv-lab = rwkvLab;
        rwkv-layer-harness = rwkvLayerHarness;
        rwkv7-p150x2-runtime = rwkv7P150x2Runtime;
        rwkv7-p150x2-fetch-queue-diagnostic-runtime = rwkv7P150x2DiagnosticRuntime;
        rwkv7-p150x2-evidence = rwkv7P150x2Evidence;
        rwkv7-p150x2-window8-candidate-runtime = rwkv7P150x2Window8CandidateRuntime;
        rwkv7-p150x2-window8-candidate-evidence = rwkv7P150x2Window8CandidateEvidence;
        rwkv7-p150x2-window16-candidate-runtime = rwkv7P150x2Window16CandidateRuntime;
        rwkv7-p150x2-window16-readout-diagnostic-runtime = rwkv7P150x2Window16ReadoutDiagnosticRuntime;
        rwkv7-p150x2-window16-candidate-evidence = rwkv7P150x2Window16CandidateEvidence;
        rwkv7-p150x2-window16-candidate-v2-runtime = rwkv7P150x2Window16CandidateV2Runtime;
        rwkv7-p150x2-window16-candidate-v2-evidence = rwkv7P150x2Window16CandidateV2Evidence;
        model-target-consumer-generator = modelTargetConsumerGenerator;
        model-target-report = modelTargetReport;
        training-platform-tools = trainingPlatform.tools;
        training-platform-core = trainingPlatform.core;
        rwkv7-lora-p150a-tools = rwkv7LoraP150a.passthru.tools;
        rwkv7-lora-p150a-core = rwkv7LoraP150a.passthru.core;
        rwkv7-lora-dual-p150a-tools = rwkv7LoraDualP150a.passthru.tools;
        rwkv7-lora-dual-p150a-core = rwkv7LoraDualP150a.passthru.core;
        rwkv-ttwkv7-boundary-device = rwkvTtwkv7BoundaryDevice;
        rwkv-ttwkv7-persistent-device = rwkvTtwkv7PersistentDevice;
        rwkv-ttwkv7-persistent-receipt-validator = rwkvTtwkv7PersistentReceiptValidator;
        tt-vibethinker-bench = ttVibeThinkerBench;
        ttwkv7-owner-control = ttwkv7OwnerControl;
        repository-quality = repositoryQuality.repositoryQualityPackage;
      };

      checks = lib.optionalAttrs isSupportedSystem {
        development-interface = developmentInterfaceCheck;
        migration-boundary = migrationBoundaryCheck;
        model-target-consumer-generator = modelTargetConsumerGenerator;
        model-target-profile-factory = modelTargetFactoryCheck;
        model-target-candidate-report = modelTargetReport;
        rwkv7-lora-reference = rwkv7LoraReference;
        rwkv7-lora-p150a = rwkv7LoraP150a;
        rwkv7-lora-dual-p150a = rwkv7LoraDualP150a;
        training-platform = trainingPlatform;
        training-platform-nixos-example = trainingPlatformExampleCheck;
        rwkv7-p150x2-runtime = rwkv7P150x2Runtime;
        rwkv7-p150x2-evidence = rwkv7P150x2Evidence;
        rwkv7-p150x2-window8-candidate-runtime = rwkv7P150x2Window8CandidateRuntime;
        rwkv7-p150x2-window8-candidate-evidence = rwkv7P150x2Window8CandidateEvidence;
        rwkv7-p150x2-window8-candidate-closure = rwkv7P150x2Window8CandidateClosure;
        rwkv7-p150x2-window16-candidate-runtime = rwkv7P150x2Window16CandidateRuntime;
        rwkv7-p150x2-window16-readout-diagnostic-runtime = rwkv7P150x2Window16ReadoutDiagnosticRuntime;
        rwkv7-p150x2-window16-candidate-evidence = rwkv7P150x2Window16CandidateEvidence;
        rwkv7-p150x2-window16-candidate-closure = rwkv7P150x2Window16CandidateClosure;
        rwkv7-p150x2-window16-candidate-v2-runtime = rwkv7P150x2Window16CandidateV2Runtime;
        rwkv7-p150x2-window16-candidate-v2-evidence = rwkv7P150x2Window16CandidateV2Evidence;
        rwkv7-p150x2-window16-candidate-v2-closure = rwkv7P150x2Window16CandidateV2Closure;
        rwkv7-p150x2-fused-wkv-architecture = rwkv7P150x2FusedWkvArchitectureCheck;
        rwkv7-p150x2-source-boundary = rwkv7P150x2SourceCheck;
        rwkv7-p150x2-production-closure = rwkv7P150x2ClosureCheck;
        rwkv7-p150x2-attempt-3-evidence = rwkv7P150x2Attempt3Evidence;
        rwkv-layer-framework-parity = rwkvLayerHarness.passthru.frameworkParityCheck;
        rwkv-ttwkv7-observed-layer = rwkvLayerHarness.passthru.observedLayerReplayCheck;
        rwkv-ttwkv7-observed-state-carry = rwkvLayerHarness.passthru.stateCarryCheck;
        rwkv-ttwkv7-observed-model-carry = rwkvLayerHarness.passthru.modelCarryCheck;
        rwkv-ttwkv7-dispatch-abi = rwkvLayerHarness.passthru.dispatchAbiCheck;
        rwkv-ttwkv7-model-dispatch = rwkvLayerHarness.passthru.modelDispatchCheck;
        rwkv-ttwkv7-persistent-model-dispatch = rwkvLayerHarness.passthru.persistentModelDispatchCheck;
        rwkv-ttwkv7-persistent-physical-core = rwkvLayerHarness.passthru.persistentPhysicalCoreCheck;
        rwkv-ttwkv7-persistent-physical-process-shell =
          rwkvLayerHarness.passthru.persistentPhysicalProcessShellCheck;
        rwkv-ttwkv7-persistent-dispatch-transport = rwkvTtwkv7PersistentDispatchTransportCheck;
        rwkv-ttwkv7-persistent-device = rwkvTtwkv7PersistentDeviceCheck;
        rwkv-ttwkv7-persistent-device-4-runbook = rwkvTtwkv7PersistentDevice4RunbookCheck;
        rwkv-ttwkv7-persistent-receipt-validator = rwkvTtwkv7PersistentReceiptValidatorCheck;
        rwkv-ttwkv7-persistent-partial-diagnostic = rwkvTtwkv7PersistentPartialDiagnosticCheck;
        rwkv-ttwkv7-persistent-device-4-evidence = rwkvTtwkv7PersistentDevice4EvidenceCheck;
        ttwkv7-architectures = ttwkv7.passthru.architectureCheck;
        rwkv-ttwkv7-host-layout = rwkvTtwkv7HostLayoutCheck;
        rwkv-ttwkv7-decode-reader = rwkvTtwkv7DecodeReaderCheck;
        rwkv-ttwkv7-boundary-device = rwkvTtwkv7BoundaryDeviceCheck;
        tt-vibethinker-bench-package = ttVibeThinkerBench;
        ttsim-package = ttsim;
        ttwkv7-owner-control-package = ttwkv7OwnerControl;
        repository-quality = repositoryQualityAggregate;
        repository-quality-artifacts = repositoryQuality.artifactQuality;
        repository-quality-cairn = repositoryQuality.cairnQuality;
        repository-quality-language-negative = repositoryQuality.negativeLanguageQuality;
        repository-quality-nix = repositoryQuality.nixQuality;
        repository-quality-policy = repositoryQuality.policyQuality;
        repository-quality-python = repositoryQuality.pythonQuality;
        repository-quality-shell = repositoryQuality.shellQuality;
      };
    };
}
