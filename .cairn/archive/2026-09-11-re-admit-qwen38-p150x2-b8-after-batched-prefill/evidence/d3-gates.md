# D3: repository Cairn, artifact, and Qwen package gates

Recorded 2026-09-11 for change `re-admit-qwen38-p150x2-b8-after-batched-prefill`.

## Results

| Gate | Result |
| --- | --- |
| `repository-quality-cairn` | PASS |
| `qwen38` | PASS |
| `qwen38-authority-negative` | PASS |
| `qwen38-nixos-service` | PASS |
| `qwen36` | PASS |
| `qwen36-nixos-service` | PASS |
| `repository-quality-artifacts` | FAIL, pre-existing |

Also from this change's own verification:

- `nix build .#qwen38 .#qwen38-gdn-grid` exits 0 with the admitted binding, and
  the install phase re-checks export determinism, the installed artifact
  equality, and the receipt-to-binding digest link.
- `./scripts/qwen36-batch-receipt.rs --self-test` exits 0 over both widths.
- The device-free check probe that covers the B=8 profile and negative mappings
  runs inside the package gate.

## The artifact-gate failure is pre-existing

```
repository-quality: artifact manifest is invalid JSON: expected `,` or `}` at line 1719 column 7
```

Same failure, same line, as recorded before this change: the malformed
`quality/generated/artifact-manifest.json` is invalid at `origin/main`, is not
touched by this change, and its last modifying commit is the merge `e0035c3`.
This change does add one evidence artifact
(`pkgs/qwen36/evidence/2026-09-11/qwen38-27b-batch-admission-b8-receipt.json`),
which the manifest does not yet list; once the manifest is repaired, that entry
must be regenerated rather than hand-edited. The repair belongs to the
repository-quality owner, not to this admission change.
