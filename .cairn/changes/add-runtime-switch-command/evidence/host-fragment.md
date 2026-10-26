# Host fragment evidence

Captured 2026-10-26 by evaluation only. The fragment was not imported into any
host configuration, `nixos-rebuild` was not run, and no unit or sudo rule was
installed.

## Command

```text
nix build .#checks.x86_64-linux.tt-model-host-config
```

Exit 0. Summary written by the check:

```text
added_units=qwen38-p150x2
helper=/nix/store/…-tt-model-owner/bin/tt-model-owner
helper_rules=1
wide_grant_rejected=true
```

## What the check proves

- Compared with the same configuration without the example module, exactly one
  unit is added: `qwen38-p150x2`.
- Exactly one sudo rule names `${helper}/bin/tt-model-owner`, for `brittonr`, with
  `NOPASSWD`, and it grants no other command.
- The unit keeps the pinned model path, `--max-sequence-length 131072`, the
  eight-slot `continuous-slot-policy-b8.json`, sampled fusion, and the declared
  competing owners.
- A rule granting `${helper}/bin` is rejected by the module assertion.

## The remaining host action

```text
# in the host flake, import the example
imports = [ tenstorrent.nixosModules."qwen38-tt-model-switch" ];
nixos-rebuild switch --flake <host-flake>#<host>
```

Then, deliberately and separately:

```text
tt-model switch qwen38 --package /nix/store/<admitted-qwen38> --repo-root <checkout>
```

The second command is the live switch. It stops the owner, resets the cards,
starts the new owner, waits for readiness, and reads identity back. It was **not**
run here.

## Blast radius

- One new unit, `qwen38-p150x2.service`, enabled at boot.
- One sudo rule for `tt-model-owner`, for `brittonr`, `NOPASSWD`, single binary.
- Two tmpfiles rules: `/var/lib/tenstorrent-models` (0755) and the
  `owner-helper` pointer into it.
- The unit's `conflicts` list, which stops the two declared competing owners if
  they are started.

## Rollback

```text
nixos-rebuild switch --rollback
systemctl mask qwen38-p150x2.service    # if the unit must stay down
```

`--rollback` returns to the previous generation, which has no unit and no sudo
rule. The record directory is not touched by either action; removing
`/var/lib/tenstorrent-models/active.json` is a separate, explicit decision.

## Non-claims

The check proves the rendered configuration, not that the unit serves. The unit
still needs an admitted package and a live switch.
