# SPDX-License-Identifier: LGPL-2.1-only
# r[verify tenstorrent.runtime_profile.switch.host]
# Evaluation-time proof for the reviewable host fragment: it renders one enabled
# owner unit, one sudo rule for the owner helper, and no other unit change. The
# negative case proves a wider grant is refused.
{
  lib,
  pkgs,
  self,
}:
let
  base = {
    nixpkgs.hostPlatform = "x86_64-linux";
    system.stateVersion = "26.11";
    fileSystems."/".device = "nodev";
    fileSystems."/".fsType = "tmpfs";
    boot.loader.grub.enable = false;
  };

  eval =
    extra:
    import (pkgs.path + "/nixos/lib/eval-config.nix") {
      system = "x86_64-linux";
      modules = [
        self.nixosModules.default
        ../nixos/examples/qwen38-tt-model-switch.nix
        { config = lib.recursiveUpdate base extra; }
      ];
    };

  # The same host without the example, to prove the fragment adds one unit and
  # nothing else.
  evalWithout =
    extra:
    import (pkgs.path + "/nixos/lib/eval-config.nix") {
      system = "x86_64-linux";
      modules = [
        self.nixosModules.default
        {
          config = lib.recursiveUpdate base (
            {
              hardware.tenstorrent.enable = true;
            }
            // extra
          );
        }
      ];
    };

  withFragment = eval { };
  withoutFragment = evalWithout { };
  unit = withFragment.config.systemd.services.qwen38-p150x2;
  helper = withFragment.config.hardware.tenstorrent.ttModel.ownerPackage;
  helperBin = "${helper}/bin/tt-model-owner";
  helperRules = lib.filter (
    rule: lib.any (command: command.command == helperBin) rule.commands
  ) withFragment.config.security.sudo.extraRules;

  addedUnits = lib.subtractLists (lib.attrNames withoutFragment.config.systemd.services) (
    lib.attrNames withFragment.config.systemd.services
  );

  wideGrant = eval {
    security.sudo.extraRules = [
      {
        users = [ "brittonr" ];
        commands = [
          {
            command = "${helper}/bin";
            options = [ "NOPASSWD" ];
          }
        ];
      }
    ];
  };
  hasFailedAssertion =
    message: evaluation:
    lib.any (
      assertion: !assertion.assertion && lib.hasInfix message assertion.message
    ) evaluation.config.assertions;
in
assert lib.assertMsg (addedUnits == [ "qwen38-p150x2" ])
  "the host fragment must add exactly the qwen38-p150x2 unit, got ${lib.concatStringsSep ", " addedUnits}";
assert lib.assertMsg (
  lib.length helperRules == 1
) "the host fragment must render exactly one sudo rule for the owner helper";
assert lib.assertMsg (
  (lib.head helperRules).users == [ "brittonr" ]
) "the owner helper rule must name only the configured operator";
assert lib.assertMsg (
  (lib.head helperRules).commands == [
    {
      command = helperBin;
      options = [ "NOPASSWD" ];
    }
  ]
) "the owner helper rule must grant the helper binary and no other command";
assert lib.assertMsg
  (lib.hasInfix "--model-path /home/brittonr/models/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0" unit.serviceConfig.ExecStart)
  "the unit must serve the pinned snapshot";
assert lib.assertMsg (lib.hasInfix "--max-sequence-length 131072" unit.serviceConfig.ExecStart)
  "the unit must serve the reviewed 131072-token row";
assert lib.assertMsg (lib.hasInfix "continuous-slot-policy-b8.json" unit.serviceConfig.ExecStart)
  "the unit must use the eight-slot continuous scheduler";
assert lib.assertMsg (
  unit.environment.QWEN36_SAMPLED_FUSION == "1"
) "the unit must keep sampled fusion on";
assert lib.assertMsg (
  unit.conflicts == withFragment.config.hardware.tenstorrent.qwen38.conflictingUnits
) "the unit must keep the declared competing owners";
assert lib.assertMsg (hasFailedAssertion "which is wider than the single helper binary" wideGrant)
  "a directory-wide helper grant must be rejected";
pkgs.runCommand "tt-model-host-config-check" { } ''
  mkdir -p "$out"
  {
    echo "added_units=${lib.concatStringsSep "," addedUnits}"
    echo "helper=${helperBin}"
    echo "helper_rules=${toString (lib.length helperRules)}"
    echo "wide_grant_rejected=true"
  } >"$out/summary.txt"
''
