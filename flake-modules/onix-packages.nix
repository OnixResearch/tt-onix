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
      modelTargets = pkgs.callPackage ../model-targets { inherit pkgs; };
      rwkvProfile = modelTargets.getProfile "rwkv7-goose-world2.8-0.1b-p150x2";
      rwkvArtifacts = modelTargets.fetchProfileArtifacts rwkvProfile;
      rwkvTarget = modelTargets.mkModelTarget {
        profile = rwkvProfile;
        artifacts = rwkvArtifacts;
        architectureAdapter = modelTargets.mkArchitectureAdapter rwkvProfile;
        evidenceAdapter = modelTargets.mkEvidenceAdapter rwkvProfile;
        topology = modelTargets.mkTopology rwkvProfile;
        builders = {
          runtime = targetContext: pkgs.callPackage ../pkgs/rwkv7-p150x2
            { inherit targetContext; };
          evidence = context: pkgs.callPackage ../pkgs/rwkv7-p150x2-evidence
            {
              targetContext = context;
              deviceRuntime = context.runtime;
            };
          command = _context: pkgs.runCommand "rwkv7-p150x2-command" { } 
            "mkdir -p $out/bin; ln -s ${rwkv7P150x2Evidence}/bin/rwkv7-p150x2-command $out/bin/rwkv7-p150x2-command";
          sourceBoundary = _context: rwkv7P150x2SourceCheck;
          closure = _context: rwkv7P150x2ProductionClosure;
        };
      };
      rwkvLayerHarness = pkgs.callPackage ../pkgs/rwkv-layer-harness {
        targetProfile = rwkvProfile;
        targetArtifacts = rwkvArtifacts;
      };
      modelTargetReport = pkgs.callPackage ../pkgs/model-target-report { };
      rwkv7LoraReference = pkgs.callPackage ../pkgs/rwkv7-lora-reference {
        model = rwkvArtifacts.checkpoint.path;
        modelBlake3 = rwkvArtifacts.checkpoint.blake3;
      };
      modelTargetFactoryCheck = pkgs.callPackage ../model-targets/check.nix {
        inherit (pkgs) nickel;
        modelTargets = modelTargets;
      };
      rwkv7P150x2Runtime = rwkvTarget.runtime;
      # r[impl tenstorrent.rwkv7_p150x2.evidence_classifier]
      rwkv7P150x2Evidence = rwkvTarget.evidence;
      rwkv7P150x2ProductionClosure = pkgs.closureInfo {
        rootPaths = [ rwkv7P150x2Evidence ];
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
            command_source=${../pkgs/rwkv7-p150x2-evidence/src/bin/rwkv7-p150x2-command.rs}

            grep -F 'MeshShape{kMeshRows, kMeshColumns}' "$runtime_source"
            grep -F 'physical_device_ids' "$runtime_source"
            grep -F 'replicate_tensor_to_mesh_mapper' "$runtime_source"
            grep -F 'ttnn::embedding' "$runtime_source"
            grep -F 'ttnn::matmul' "$runtime_source"
            grep -F 'ttnn::argmax' "$runtime_source"
            grep -F 'ReadDeviceProfilerResults' "$runtime_source"
            grep -F '!environment.watcher.has_value()' "$runtime_source"
            grep -F '!environment.dprint_cores.has_value()' "$runtime_source"
            grep -F 'remove_environment(WATCHER_ENVIRONMENT)' "$command_source"
            grep -F 'remove_environment(DPRINT_CORES_ENVIRONMENT)' "$command_source"

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
      ttsim = pkgs.callPackage ../pkgs/ttsim { };
      migrationBoundaryCheck = pkgs.callPackage ../checks/migration-boundary.nix { };

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

      packages = lib.optionalAttrs isSupportedSystem {
        inherit ttsim ttwkv7;
        rwkv-lab = rwkvLab;
        rwkv-layer-harness = rwkvLayerHarness;
        rwkv7-p150x2-runtime = rwkv7P150x2Runtime;
        rwkv7-p150x2-evidence = rwkv7P150x2Evidence;
        model-target-report = modelTargetReport;
        rwkv-ttwkv7-boundary-device = rwkvTtwkv7BoundaryDevice;
        rwkv-ttwkv7-persistent-device = rwkvTtwkv7PersistentDevice;
        rwkv-ttwkv7-persistent-receipt-validator = rwkvTtwkv7PersistentReceiptValidator;
        tt-vibethinker-bench = ttVibeThinkerBench;
        ttwkv7-owner-control = ttwkv7OwnerControl;
      };

      checks = lib.optionalAttrs isSupportedSystem {
        development-interface = developmentInterfaceCheck;
        migration-boundary = migrationBoundaryCheck;
        model-target-profile-factory = modelTargetFactoryCheck;
        model-target-candidate-report = modelTargetReport;
        rwkv7-lora-reference = rwkv7LoraReference;
        rwkv7-p150x2-runtime = rwkv7P150x2Runtime;
        rwkv7-p150x2-evidence = rwkv7P150x2Evidence;
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
      };
    };
}
