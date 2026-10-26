# Switch evidence

Captured 2026-10-26. Every scenario runs device-free with a fake owner helper, a
self-terminating loopback stub, and temporary state roots under the build
directory. Nothing was applied to the host.

## Commands

```text
nix build .#tt-model
nix build .#checks.x86_64-linux.model-runtime-profiles
nix build .#checks.x86_64-linux.tt-model-owner
```

Exit 0 for all three.

## What the install check asserts

| Scenario | Expectation |
|---|---|
| successful switch | the record is committed, `readiness.observed_seconds > 0`, `identity.package_identity` equals the package's execution identity, `physical_pcie_device_ids` is `[0,1]`, `simulated` is false, `mesh` is recorded, `reset_performed` is true, `admission_bypassed` is false, and a `.verified` history entry exists |
| wrong package identity | `switch refused: identity: … package-identity: … != …`, and `active.json` is byte-identical to the previous record (`cmp`) with `switch-failure.json` at `restored: true` |
| start failure | `switch refused: start-owner`, no `active.json`, `restored: true` |
| rollback failure | the previous owner was running, the restart failed, `active.json` removed, `restored: false` with `rollback_error`, and the withdrawn record kept as `*.rollback-failed.json` |
| readiness timeout | `switch refused: readiness … not ready within 2s`, no record |
| `--force` with an admission failure | exit 0 and `admission_bypassed: true` in both the report and the record |
| `--force` while unattended | `refused when TT_MODEL_UNATTENDED=1` |
| structural failure with `--force` | `switch refused: check:` and no record |
| foreign card holder | `switch refused: card-holder`, and the helper never received `stop-owner` |
| job profile | `is a job profile; use tt-model run` |
| held switch lock | `another switch operation holds the lock` |
| containment failure with `--force` | `switch refused: containment: …: passed is not True` |

The readiness stub serves `/healthz` with a fixed body; the identity scenarios use
the same profile on separate ports so no scenario can inherit a listener.

## Registry side

```text
fixtures_rejected=8
containment_commands=1
```

The eighth fixture rejects a containment argument that expands `@pkg@`, and the
check proves `scripts/qwen38-admission-containment.py` exists and is executable in
the repository.

## Non-claims

The stub is not a model; the helper is fake; no unit, device, or real `tt-smi` was
touched. The switch was not run against the host's real owner, and the live switch
remains a deliberate authorized action.
