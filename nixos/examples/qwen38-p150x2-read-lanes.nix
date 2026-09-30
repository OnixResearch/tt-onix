# Candidate profile for the dedicated 4,113-block structured-read lanes.
# Import alongside tenstorrent-nix.nixosModules.default into a reviewed host
# generation only after the paired physical admission described in
# docs/qwen38-27b-adoption.md. This profile does not activate the host.
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
      # Only this derivation ran the paired P150x2 root-systemd trial.
      assertion =
        config.hardware.tenstorrent.qwen38.package.outPath
        == "/nix/store/y58n6n12g680m4g9bqmd5n68smhs0ha9-qwen38-0.1.0";
      message = "The Qwen read-lane profile requires the physically paired y58n6n12g680m4g9bqmd5n68smhs0ha9 package; requalify a changed package output";
    }
  ];

  hardware.tenstorrent = {
    enable = true;
    qwen38 = {
      enable = true;
      package = pkgs.qwen38;
      modelPath = "/home/brittonr/models/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0";
      maximumSequenceLength = 131072;
      maximumGenerationTokens = 8192;
      groupedPrefill = true;
      tracedPrefill = true;
      sampledFusion = true;
      continuousSlotScheduler = true;
      prefixCache = true;
      conflictingUnits = [
        "llamacpp-server-vibethinker-britton-desktop.service"
        "docker-tt-inference-server-llama-3-1-8b-instruct-p150.service"
      ];
    };
  };
}
