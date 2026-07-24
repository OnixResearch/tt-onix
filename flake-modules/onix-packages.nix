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
      rwkvLayerHarness = pkgs.callPackage ../pkgs/rwkv-layer-harness { };
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
      rwkvTtwkv7PersistentPartialDiagnosticCheck =
        pkgs.callPackage ../pkgs/rwkv-ttwkv7-persistent-evidence
          { };
      rwkvTtwkv7PersistentDevice4EvidenceCheck =
        pkgs.callPackage ../pkgs/rwkv-ttwkv7-persistent-passed-evidence
          { };
      ttVibeThinkerBench = pkgs.callPackage ../pkgs/tt-vibethinker-bench { };
      ttsim = pkgs.callPackage ../pkgs/ttsim { };
      migrationBoundaryCheck = pkgs.callPackage ../checks/migration-boundary.nix { };
    in
    {
      packages = lib.optionalAttrs isSupportedSystem {
        inherit ttsim ttwkv7;
        rwkv-lab = rwkvLab;
        rwkv-layer-harness = rwkvLayerHarness;
        rwkv-ttwkv7-boundary-device = rwkvTtwkv7BoundaryDevice;
        rwkv-ttwkv7-persistent-device = rwkvTtwkv7PersistentDevice;
        tt-vibethinker-bench = ttVibeThinkerBench;
        ttwkv7-owner-control = ttwkv7OwnerControl;
      };

      checks = lib.optionalAttrs isSupportedSystem {
        migration-boundary = migrationBoundaryCheck;
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
