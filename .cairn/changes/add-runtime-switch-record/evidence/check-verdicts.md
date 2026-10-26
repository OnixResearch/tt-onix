# Check evidence

Captured 2026-10-26 on britton-desktop. The sandbox cases run inside the
`tt-model` install check; the real-profile cases run on the host with the
packaged command, `--repo-root .`, and no privilege.

## Commands

```text
nix build --no-link --builders '' .#tt-model
/nix/store/<tt-model>/bin/tt-model check qwen38 \
  --package /nix/store/inkfvsjqchqp7bki76asdib3b57nw6kj-qwen38-0.1.0 --repo-root .
/nix/store/<tt-model>/bin/tt-model check nanbeige43 \
  --package /nix/store/kn85q95rgk3py9rb1fyw4c7bvkajdjg2-nanbeige43-0.1.0 --repo-root . --json
/nix/store/<tt-model>/bin/tt-model check nanbeige43 \
  --package /nix/store/kn85q95rgk3py9rb1fyw4c7bvkajdjg2-nanbeige43-0.1.0 --repo-root . --deep
```

## Real verdicts

**qwen38** — exit 1. The package, entry binary, runtime root, execution identity
and checkpoint all resolve; both receipts hash to their recorded digests and
still report admitted, but they name another package:

```text
failure: receipt-package: pkgs/qwen36/evidence/2026-10-21/qwen38-continuous-slot-admission-b8-receipt.json: no receipt names this package (receipt names '/nix/store/wlsl6pnxllv9qhr6yqimnwv3si0qyz9k-qwen38-0.1.0')
failure: receipt-package: pkgs/qwen36/evidence/2026-10-21/qwen38-prefix-cache-abba-admission-receipt.json: no receipt names this package (receipt names '/nix/store/wlsl6pnxllv9qhr6yqimnwv3si0qyz9k-qwen38-0.1.0')
```

That is the true verdict for this host: the package the Nix profile installs is
not named by any admitted receipt, and the package the receipts do name has been
garbage-collected.

**nanbeige43** — exit 0 in 0.72 s. Cheap structural path only:

```json
{"verdict": "pass", "registration": "nix-path-info", "name_matches": true,
 "deep": {"enabled": false, "bytes": null, "files": null,
          "max_deep_bytes": 1073741824, "allow_large": false},
 "failures": []}
```

The authority document hashes to the recorded digest, the checkpoint directory
name carries `c1093c501cc6`, and the runtime root exists. Nothing read the 62 GB
snapshot.

**nanbeige43 --deep** — exit 1 in 0.62 s, before any hashing:

```text
deep: files=76 bytes=66259320199 bound=1073741824 allow-large=False
failure: deep-size-bound: /home/brittonr/models/nanbeige4.3-33b-c1093c501cc6: deep re-hash would read 66259320199 bytes, above the 1073741824-byte bound; pass --allow-large
```

## Sandbox cases (package install check)

The install check builds fixture packages whose store names end with a registry
suffix, a fixture repository holding the two Qwen receipts and the two authority
documents, and a fixture checkpoint. Every case runs with
`TT_MODEL_STATE_ROOT`, `TT_MODEL_GC_ROOTS`, and `TT_MODEL_HOME` under `$TMPDIR`,
`TT_MODEL_SYSTEMCTL=none`, and `TT_MODEL_NIX=none`, so nothing is privileged and
nothing outside the build directory is written.

Every refusal is asserted by its message:

| Case | Message |
|---|---|
| real qwen38 with a fixture package | `no receipt names this package` |
| unknown profile | `no such runtime profile` |
| package outside the store | `not a /nix/store path` |
| missing package | `package is missing` |
| failing registration query | `package is not registered` |
| wrong store-name suffix | `store name must end with -qwen38-0.1.0` |
| no candidate package | `pass --package or record one first` |
| changed receipt | `receipt bytes hash to` |
| receipt no longer admitted | `receipt no longer reports an admitted verdict` |
| changed authority document | `authority document hashes to` |
| missing checkpoint | `checkpoint path is missing` |
| misnamed checkpoint | `checkpoint directory name does not carry revision` |
| changed gate artifact | `gate artifact hashes to` |
| deep above the bound | `above the 1073741824-byte bound` |

Positive sandbox cases: nanbeige43 with a fixture checkpoint passes with
`name_matches: true` and `store-presence`; a gate artifact with the right digest
passes; `--deep` on a three-file checkpoint reports `files: 3` and
`enabled: true`; the same registry without `--deep` reports `enabled: false` and
never walks the tree.
