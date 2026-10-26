# Host status evidence

Captured 2026-10-26 on britton-desktop with the packaged command, no environment
overrides, no root, and no sudo. Nothing was written, started, stopped, or
masked.

## Command

```text
/nix/store/<tt-model>/bin/tt-model status
/nix/store/<tt-model>/bin/tt-model status --json
```

## Human output

```text
verdict: broken
registry: <tt-model>/share/tt-model/runtime-profiles.json (schema 1, 4 profiles: metalium-rs-rwkv7, nanbeige43, qwen38, rwkv7-ttnn)
active record: absent (/var/lib/tenstorrent-models/active.json)
operator symlink: dangling (/home/brittonr/.local/state/qwen38-p150x2/current) -> /nix/store/wlsl6pnxllv9qhr6yqimnwv3si0qyz9k-qwen38-0.1.0
nix profile: ok (/nix/var/nix/profiles/qwen38-p150x2, generation qwen38-p150x2-5-link) -> /nix/store/k0f3ph7jh4bjd0imh3nnj4wm2a3zr5cf-profile
  packages: /nix/store/inkfvsjqchqp7bki76asdib3b57nw6kj-qwen38-0.1.0
unit: masked (qwen38-p150x2.service, active=inactive, file=masked)
problem: operator symlink: /home/brittonr/.local/state/qwen38-p150x2/current: dangling symlink to /nix/store/wlsl6pnxllv9qhr6yqimnwv3si0qyz9k-qwen38-0.1.0
```

Exit status 1, because a reported source is dangling.

## JSON output

```json
{
  "active": {
    "path": "/var/lib/tenstorrent-models/active.json",
    "state": "absent"
  },
  "legacy": {
    "nix_profile": {
      "generation": "qwen38-p150x2-5-link",
      "packages": [
        "/nix/store/inkfvsjqchqp7bki76asdib3b57nw6kj-qwen38-0.1.0"
      ],
      "path": "/nix/var/nix/profiles/qwen38-p150x2",
      "state": "ok",
      "target": "/nix/store/k0f3ph7jh4bjd0imh3nnj4wm2a3zr5cf-profile"
    },
    "operator_symlink": {
      "path": "/home/brittonr/.local/state/qwen38-p150x2/current",
      "state": "dangling",
      "target": "/nix/store/wlsl6pnxllv9qhr6yqimnwv3si0qyz9k-qwen38-0.1.0"
    },
    "unit": {
      "active_state": "inactive",
      "fragment": "/etc/systemd/system/qwen38-p150x2.service",
      "name": "qwen38-p150x2.service",
      "state": "masked",
      "sub_state": "dead",
      "unit_file_state": "masked"
    }
  },
  "problems": [
    {
      "path": "/home/brittonr/.local/state/qwen38-p150x2/current",
      "reason": "dangling symlink to /nix/store/wlsl6pnxllv9qhr6yqimnwv3si0qyz9k-qwen38-0.1.0",
      "source": "operator symlink"
    }
  ],
  "registry": {
    "path": "<tt-model>/share/tt-model/runtime-profiles.json",
    "profiles": [
      "metalium-rs-rwkv7",
      "nanbeige43",
      "qwen38",
      "rwkv7-ttnn"
    ],
    "schema_version": 1
  },
  "schema": "onix.tt-model.status.v1",
  "unavailable": [],
  "verdict": "broken"
}
```

## What the three records say

| Source | Says | Consistent? |
|---|---|---|
| Active record | absent | — |
| Operator symlink | `/nix/store/wlsl6pnxllv9qhr6yqimnwv3si0qyz9k-qwen38-0.1.0` | **no**: that store path no longer exists |
| Nix profile generation | installs `/nix/store/inkfvsjqchqp7bki76asdib3b57nw6kj-qwen38-0.1.0` | resolves, but no admission receipt in this tree names that package |
| Owner unit | `masked`, fragment symlinked to `/dev/null` | the host generation does not enable the service |

This is the state the registry exists to make legible: two different package
answers, the effective one removed by garbage collection, and no unit able to
start. `status` names all three and fails instead of picking one.

The operator symlink is present but dangling; it is not absent. The earlier
assumption that it had been removed is wrong, and the command reports the
stronger fact: a symlink pointing at a package the store no longer holds.

## Non-claims

This is a read of host state, not repository state. It does not prove the
reported mask or the missing store path will still hold at any later time, and it
does not repair any of it.
