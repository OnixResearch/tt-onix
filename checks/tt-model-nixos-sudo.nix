# SPDX-License-Identifier: LGPL-2.1-only
# r[verify tenstorrent.runtime_profile.owner_helper.grant]
# Evaluation-time proof that the NixOS module renders exactly one sudo rule, that
# the rule names the helper binary and nothing broader, and that adding the helper
# does not change any existing unit.
{
  lib,
  pkgs,
  self,
}:
let
  baseConfig = {
    nixpkgs.hostPlatform = "x86_64-linux";
    system.stateVersion = "26.11";
    fileSystems."/".device = "nodev";
    fileSystems."/".fsType = "tmpfs";
    boot.loader.grub.enable = false;
    hardware.tenstorrent = {
      enable = true;
      ttModel = {
        enable = true;
        operatorUser = "operator";
      };
    };
  };

  eval =
    extra:
    import (pkgs.path + "/nixos/lib/eval-config.nix") {
      system = "x86_64-linux";
      modules = [
        self.nixosModules.default
        { config = lib.recursiveUpdate baseConfig extra; }
      ];
    };

  qwenConfig = {
    hardware.tenstorrent.qwen38 = {
      enable = true;
      modelPath = "/var/lib/qwen38-p150x2/model";
      conflictingUnits = [ "demo-owner.service" ];
    };
  };

  enabled = eval { };
  qwenWithoutHelper = eval (
    lib.recursiveUpdate qwenConfig {
      hardware.tenstorrent.ttModel.enable = lib.mkForce false;
    }
  );
  qwenWithHelper = eval qwenConfig;

  helper = enabled.config.hardware.tenstorrent.ttModel.ownerPackage;
  helperBin = "${helper}/bin/tt-model-owner";
  sudoRules = enabled.config.security.sudo.extraRules;
  helperRules = lib.filter (
    rule: lib.any (command: command.command == helperBin) rule.commands
  ) sudoRules;
  widerCommands = lib.concatMap (
    rule:
    if !(lib.elem "operator" rule.users) then
      [ ]
    else
      lib.filter (
        command:
        command.command == "${helper}/bin"
        || (lib.hasPrefix "${helper}/" command.command && command.command != helperBin)
      ) rule.commands
  ) sudoRules;

  # Compare only the fields the unit defines. Whole-attribute equality would force
  # systemd options that no module sets, which is not a property of this change.
  qwenUnitProjection =
    evaluation:
    let
      unit = evaluation.config.systemd.services.qwen38-p150x2;
    in
    {
      inherit (unit)
        after
        conflicts
        environment
        wantedBy
        ;
      execStart = unit.serviceConfig.ExecStart;
      type = unit.serviceConfig.Type;
      conditions = unit.unitConfig.ConditionPathExists;
    };

  hasFailedAssertion =
    message: evaluation:
    lib.any (
      assertion: !assertion.assertion && lib.hasInfix message assertion.message
    ) evaluation.config.assertions;

  wideGrant = eval {
    security.sudo.extraRules = [
      {
        users = [ "operator" ];
        commands = [
          {
            command = "${helper}/bin";
            options = [ "NOPASSWD" ];
          }
        ];
      }
    ];
  };
  secondHelperGrant = eval {
    security.sudo.extraRules = [
      {
        users = [ "operator" ];
        commands = [
          {
            command = helperBin;
            options = [ "NOPASSWD" ];
          }
          {
            command = "/run/current-system/sw/bin/true";
            options = [ "NOPASSWD" ];
          }
        ];
      }
    ];
  };
  allUser = eval { hardware.tenstorrent.ttModel.operatorUser = "ALL"; };

  wideRejected = hasFailedAssertion "which is wider than the single helper binary" wideGrant;
  secondRejected = hasFailedAssertion "must render exactly one NOPASSWD rule" secondHelperGrant;
  allUserRejected = hasFailedAssertion "must name one user, not ALL" allUser;
in
assert lib.assertMsg (
  lib.length helperRules == 1
) "the module must render exactly one sudo rule that names the owner helper";
assert lib.assertMsg (
  (lib.head helperRules).users == [ "operator" ]
) "the owner helper rule must name only the configured operator user";
assert lib.assertMsg (
  (lib.head helperRules).commands == [
    {
      command = helperBin;
      options = [ "NOPASSWD" ];
    }
  ]
) "the owner helper rule must grant the helper binary and no other command";
assert lib.assertMsg (
  widerCommands == [ ]
) "no sudo rule for the operator may grant a command wider than the helper binary";
assert lib.assertMsg (lib.any (
  rule: lib.hasPrefix "L+ /var/lib/tenstorrent-models/owner-helper" rule
) enabled.config.systemd.tmpfiles.rules) "the module must record which installed helper may act";
assert lib.assertMsg (
  !(enabled.config.systemd.services ? qwen38-p150x2)
) "enabling the owner helper must not enable the masked Qwen unit";
assert lib.assertMsg (
  qwenUnitProjection qwenWithHelper == qwenUnitProjection qwenWithoutHelper
) "enabling the owner helper must not change the qwen38-p150x2 unit";
assert lib.assertMsg wideRejected "a directory-wide helper grant must be rejected";
assert lib.assertMsg secondRejected "a second helper rule must be rejected";
assert lib.assertMsg allUserRejected "ALL as the operator user must be rejected";
pkgs.runCommand "tt-model-nixos-sudo-check" { } ''
  mkdir -p "$out"
  {
    echo "helper=${helperBin}"
    echo "rules=$(printf '%s\n' ${lib.escapeShellArg (toString (lib.length helperRules))})"
    echo "wide_grant_rejected=true"
    echo "second_grant_rejected=true"
    echo "all_user_rejected=true"
    echo "qwen_unit_unchanged=true"
  } >"$out/summary.txt"
''
