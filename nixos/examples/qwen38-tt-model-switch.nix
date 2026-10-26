# SPDX-License-Identifier: LGPL-2.1-only
# r[impl tenstorrent.runtime_profile.switch.host]
# Reviewable host fragment for the runtime-profile switch. Importing this module
# in a host generation would enable the served Qwen3.8-27B owner unit and grant
# `tt-model-owner` to one user.
#
# This file is NOT imported by the flake's default module, and no host
# configuration lives in this repository. Activation is a deliberate host action:
#
#   nixos-rebuild switch --flake <host-flake>#<host>
#
# Blast radius: one new unit (`qwen38-p150x2.service`), one sudo rule for the
# owner helper, two tmpfiles rules for the record directory and the helper
# pointer, plus the existing tt-model tooling. No other unit is enabled, disabled,
# masked, or edited.
#
# Rollback: `nixos-rebuild switch --rollback`, then, if the unit must stay down,
# `systemctl mask qwen38-p150x2.service`. The record directory under
# /var/lib/tenstorrent-models is untouched by either action.
{
  config,
  lib,
  ...
}:
{
  assertions = [
    {
      assertion = config.hardware.tenstorrent.qwen38.conflictingUnits != [ ];
      message = "the Qwen owner needs the host's competing owner units before activation";
    }
  ];

  hardware.tenstorrent = {
    enable = true;

    qwen38 = {
      enable = true;
      # The pinned Qwen3.8-27B snapshot the served deployment uses.
      modelPath = "/home/brittonr/models/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0";
      modelAlias = "Qwen3.8-27B";
      maximumSequenceLength = 131072;
      maximumGenerationTokens = 8192;
      # The reviewed production route: grouped, traced, sampled, with the
      # eight-slot continuous scheduler and the hybrid prefix cache.
      groupedPrefill = true;
      tracedPrefill = true;
      sampledFusion = true;
      continuousSlotScheduler = true;
      continuousSlotProfile = "b8";
      prefixCache = true;
      # The two competing owners on this host. Both are inactive today; the
      # declaration is what makes systemd stop them if either is started.
      conflictingUnits = [
        "llamacpp-server-vibethinker-britton-desktop.service"
        "docker-tt-inference-server-llama-3-1-8b-instruct-p150.service"
      ];
    };

    ttModel = {
      enable = true;
      operatorUser = "brittonr";
    };
  };
}
