# All three admissions now name the promoted package, 2026-10-26

The promoted package `/nix/store/gzwghi7spl93iv4y6qb680cg8wyfbq6r-qwen38-0.1.0`
and execution identity
`/nix/store/qwnxwcppn16za67i710qm5drnzs57vg9-qwen36-execution-identity-0.1.0`
now carry all three admissions:

| Kind | Receipt | BLAKE3 | Binds |
| --- | --- | --- | --- |
| B=8 interleaved batch admission (v3) | `pkgs/qwen36/evidence/2026-10-26/qwen38-batch-admission-b8-v3-receipt.json` | `cf9fc9d417091a83bad96c71b58015dcab75529146247664039e6de1758f3e55` | identity `qwnxwc…`; the promoted v3 profile names this digest and is `physically_admitted: true` |
| continuous-slot admission (B=8) | `pkgs/qwen36/evidence/2026-10-26/qwen38-continuous-slot-admission-b8-receipt.json` | `e5487ac541cc753ef4d3030f3e3d56bac04a499adb6a4a1051a50c94984d6b01` | package `gzwghi7…` and identity `qwnxwc…` |
| prefix-cache admission | `pkgs/qwen36/evidence/2026-10-26/qwen38-prefix-cache-abba-admission-receipt.json` | `11288cecce8aa47a8308d605ade20de9b4e4295f65d0627b12c6a3ebdae4a109` | package `gzwghi7…` and identity `qwnxwc…` |

This is the end state the receipt-bootstrap change was written to make possible.

`scripts/qwen38-deploy` now accepts the promoted package without `--force`:
its `receipts_for()` scan finds both required kinds
(`continuous-slot-admission` and `prefix-cache-admission`), and its containment
check returns `v3-admitted`, `can_serve: true`, receipt
`cf9fc9d4…`. This was verified by calling the guard functions read-only; the
deploy move itself was **not** executed, and no service was started.

The earlier re-earn result's statement that the package remained a `--force`
case is superseded by this note; the scheduler and prefix-cache admissions were
then re-earned on the same package. Their evidence is in
`.cairn/changes/widen-qwen38-continuous-slots-b8/evidence/re-admission-promoted-2026-10-26.md`
and
`.cairn/changes/readmit-qwen38-prefix-cache-b8/evidence/re-admission-promoted-2026-10-26.md`.
