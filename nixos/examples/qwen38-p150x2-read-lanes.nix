# Candidate profile for the dedicated 4,113-block structured-read lanes.
# Import alongside tenstorrent-nix.nixosModules.default into a reviewed host
# generation only after the same-fixture physical requalification described in
# docs/qwen38-27b-adoption.md. Supply the model path and competing owner units
# from the host configuration; this profile does not activate the host.
{ config, pkgs, ... }:
{
  assertions = [
    {
      assertion = config.systemd.services.qwen38-p150x2.enable;
      message = "The Qwen read-lane profile requires an unmasked qwen38-p150x2.service";
    }
    {
      assertion =
        let
          qwen = config.hardware.tenstorrent.qwen38;
        in
        qwen.enable
        && qwen.maximumSequenceLength == 131072
        && qwen.maximumGenerationTokens == 8192
        && qwen.groupedPrefill
        && qwen.tracedPrefill
        && qwen.sampledFusion
        && qwen.continuousSlotScheduler
        && qwen.prefixCache;
      message = "The Qwen read-lane profile requires the reviewed 131072/8192 grouped, traced, sampled, cached continuous-slot configuration";
    }
    {
      assertion = config.hardware.tenstorrent.qwen38.conflictingUnits != [ ];
      message = "The Qwen read-lane profile requires host-specific conflictingUnits before activation";
    }
    {
      # e3cfded's han2 output passed a same-fixture root-systemd requalification
      # against preserved exact-byte old/y58 responses, not a fresh old baseline.
      assertion =
        config.hardware.tenstorrent.qwen38.package.outPath
        == "/nix/store/han2q1h90mcp8ad1k9qq7bz7sl8r4rf9-qwen38-0.1.0";
      message = "The Qwen read-lane profile requires the han2q1h90mcp8ad1k9qq7bz7sl8r4rf9 output requalified against preserved baseline responses; requalify a changed package output";
    }
  ];

  hardware.tenstorrent = {
    enable = true;
    qwen38 = {
      enable = true;
      package = pkgs.qwen38;
      maximumSequenceLength = 131072;
      maximumGenerationTokens = 8192;
      groupedPrefill = true;
      tracedPrefill = true;
      sampledFusion = true;
      continuousSlotScheduler = true;
      prefixCache = true;
    };
  };
}
