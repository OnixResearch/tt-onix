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
  reconcile = eval { hardware.tenstorrent.ttModel.leaseReconcile.enable = true; };
  reconcileWithoutHelper = eval {
    hardware.tenstorrent.ttModel.enable = lib.mkForce false;
    hardware.tenstorrent.ttModel.leaseReconcile.enable = true;
  };
  reconcileUnit = reconcile.config.systemd.services.tt-model-lease-reconcile;
  reconcileTimer = reconcile.config.systemd.timers.tt-model-lease-reconcile;
  reconcileHelper = reconcile.config.hardware.tenstorrent.ttModel.ownerPackage;
  reconcileScript = builtins.unsafeDiscardStringContext (
    builtins.readFile reconcileUnit.serviceConfig.ExecStart
  );
  reconcileAddedUnits = lib.subtractLists (lib.attrNames withoutFragment.config.systemd.services) (
    lib.attrNames reconcile.config.systemd.services
  );
  forbiddenReconcileActions = [
    "stop-owner"
    "start-owner"
    "systemctl"
    "tt-smi"
    "reset"
    "record"
    "switch"
    " run "
    "--force"
  ];
  # A record whose advisory holder pid is gone. The timer path must flag it,
  # not clear it, and must not claim the KMD exports were released.
  reconcileStaleRecord = pkgs.writeText "tt-model-reconcile-stale-record.json" (
    builtins.toJSON {
      schema = "onix.tt-model.record.v1";
      profile = "reconcile-fixture";
      recorded_at = "2026-10-28T00:00:00+00:00";
      mesh.physical_device_ids = [ 0 ];
      lease = {
        holder_pid = 99999999;
        holder_start_ticks = null;
        generation = 1;
        expires_at = null;
      };
    }
  );
  reconcileStaleCards = pkgs.writeText "tt-model-reconcile-stale-cards.txt" ''
    owner_name=qwen38-p150x2.service
    owner_pid=4242
    card /dev/tenstorrent/0
    holder /dev/tenstorrent/0 4242 qwen38-p150x2.service qwen38-p150x2-serve
  '';
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
assert lib.assertMsg (
  !withFragment.config.hardware.tenstorrent.ttModel.leaseReconcile.enable
) "lease reconciliation must be off by default";
assert lib.assertMsg (
  !(withFragment.config.systemd.services ? tt-model-lease-reconcile)
) "the default host must not render a reconciliation service";
assert lib.assertMsg (
  !(withFragment.config.systemd.timers ? tt-model-lease-reconcile)
) "the default host must not render a reconciliation timer";
assert lib.assertMsg
  (
    reconcileAddedUnits == [
      "qwen38-p150x2"
      "tt-model-lease-reconcile"
    ]
  )
  "enabling reconciliation must add exactly one service, got ${lib.concatStringsSep ", " reconcileAddedUnits}";
assert lib.assertMsg (
  reconcileTimer.wantedBy == [ "timers.target" ]
) "the reconciliation timer must be pulled in by timers.target only";
assert lib.assertMsg (
  reconcileTimer.timerConfig.Persistent == false
) "the reconciliation timer must not be persistent";
assert lib.assertMsg (
  reconcileTimer.timerConfig.OnUnitActiveSec == "5min"
) "the reconciliation timer must use the fixed default interval";
assert lib.assertMsg (
  reconcileUnit.serviceConfig.Type == "oneshot"
) "the reconciliation service must be oneshot";
assert lib.assertMsg (lib.all (key: !(reconcileUnit.serviceConfig ? ${key})) [
  "ExecStartPre"
  "ExecStartPost"
  "ExecStop"
  "ExecReload"
]) "the reconciliation service must not add a pre, post, stop, or reload action";
assert lib.assertMsg (
  reconcileUnit.environment.TT_MODEL_STATE_ROOT == "/var/lib/tenstorrent-models"
) "the reconciliation service must read the owner helper's fixed state root";
assert lib.assertMsg (
  !(reconcileUnit.serviceConfig ? User)
) "the reconciliation service must run as root, not a configurable user";
assert lib.assertMsg
  (lib.hasInfix (builtins.unsafeDiscardStringContext "${reconcileHelper}/bin/tt-model-owner cards") reconcileScript)
  "the reconciliation script must capture the owner helper's cards view by absolute path";
assert lib.assertMsg (lib.hasInfix "/bin/tt-model reconcile --cards" reconcileScript)
  "the reconciliation script must call tt-model reconcile by absolute path";
assert lib.assertMsg (lib.all (action: !(lib.hasInfix action reconcileScript))
  forbiddenReconcileActions
) "the reconciliation script must not contain a stop, reset, record, switch, or run action";
assert lib.assertMsg (
  reconcile.config.systemd.timers ? tt-model-lease-reconcile
) "enabling reconciliation must render the timer";
assert lib.assertMsg
  (hasFailedAssertion "requires hardware.tenstorrent.ttModel.enable" reconcileWithoutHelper)
  "reconciliation without the owner helper must be refused";
pkgs.runCommand "tt-model-host-config-check" { } ''
  mkdir -p "$out" "$TMPDIR/state"
  cp ${reconcileStaleRecord} "$TMPDIR/state/active.json"
  TT_MODEL_STATE_ROOT="$TMPDIR/state" ${pkgs.tt-model}/bin/tt-model reconcile \
    --cards ${reconcileStaleCards} --json >"$out/reconcile-stale-lock.json"
  grep -F '"state": "stale"' "$out/reconcile-stale-lock.json"
  grep -F '"dma_buf": "unobserved"' "$out/reconcile-stale-lock.json"
  grep -F '"release_claimed": false' "$out/reconcile-stale-lock.json"
  grep -F '"action": "none"' "$out/reconcile-stale-lock.json"
  {
    echo "added_units=${lib.concatStringsSep "," addedUnits}"
    echo "helper=${helperBin}"
    echo "helper_rules=${toString (lib.length helperRules)}"
    echo "wide_grant_rejected=true"
    echo "reconcile_default_off=true"
    echo "reconcile_timer_persistent=false"
    echo "reconcile_stale_lock=flagged"
  } >"$out/summary.txt"
''
