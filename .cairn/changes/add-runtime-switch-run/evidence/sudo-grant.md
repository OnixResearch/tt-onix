# Sudo grant evidence

Captured 2026-10-26 by evaluation only. No rule was installed on this host, no
sudoers file was written, and `sudo` was not run.

## Command

```text
nix build .#checks.x86_64-linux.tt-model-nixos-sudo
```

Exit 0. Summary written by the check:

```text
helper=/nix/store/…-tt-model-owner/bin/tt-model-owner
rules=1
wide_grant_rejected=true
second_grant_rejected=true
all_user_rejected=true
qwen_unit_unchanged=true
```

## What the check proves

Positive evaluation (`hardware.tenstorrent.ttModel.enable = true`, operator
`operator`):

- exactly one rule names the helper binary;
- that rule's users are exactly `[ "operator" ]`;
- that rule's commands are exactly `[{ command = "${helper}/bin/tt-model-owner";
  options = [ "NOPASSWD" ]; }]`;
- no rule for the operator grants anything under the helper package other than
  that binary;
- a tmpfiles rule records
  `L+ /var/lib/tenstorrent-models/owner-helper` pointing at the helper package;
- `systemd.services.qwen38-p150x2` is absent in the base configuration, and its
  projection (`ExecStart`, `Type`, environment, conflicts, conditions, `after`,
  `wantedBy`) is identical with and without the ttModel option.

Negative evaluations, each rejected by a named assertion:

| Case | Assertion message |
|---|---|
| a second rule granting `${helper}/bin` | `grants … which is wider than the single helper binary` |
| a second rule naming the helper plus another command | `must render exactly one NOPASSWD rule` |
| `operatorUser = "ALL"` | `must name one user, not ALL or an empty string` |

The unit comparison is a projection rather than whole-attribute equality:
comparing the whole attrset forces systemd options that no module defines
(`startLimitIntervalSec`), which is not a property of this change.

## Module diff scope

`git diff HEAD --numstat -- nixos/module.nix` reports `88 0`: the file gains the
option block and the new `mkIf cfg.ttModel.enable` block and loses nothing. The
`qwen38-p150x2` unit definition, its environment, its conditions, and its
`conflicts` are untouched.

## Non-claims

The rule is staged in the module. It takes effect only when a host generation
enables `hardware.tenstorrent.ttModel`, which this change does not do on this
host. The helper was not run as root at any point.
