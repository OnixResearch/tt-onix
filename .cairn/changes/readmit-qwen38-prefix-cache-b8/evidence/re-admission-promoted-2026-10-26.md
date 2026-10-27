# Prefix-cache admission re-earned on the promoted package, 2026-10-26

This is the current admission. The 2026-10-21 prefix-cache receipt and the
`wlsl6pnx…`/`l4arf461…` package and identity it named are superseded for
serving purposes but remain historical evidence and are not deleted.

## Receipt

- Path: `pkgs/qwen36/evidence/2026-10-26/qwen38-prefix-cache-abba-admission-receipt.json`
- BLAKE3: `11288cecce8aa47a8308d605ade20de9b4e4295f65d0627b12c6a3ebdae4a109`
- Receipt kind: `prefix-cache-admission`
- Admission policy: `qwen38-p150x2-prefix-cache-admission-v5`, BLAKE3
  `01ff50affee07c0ac22848408973728a07e837908790c6844a98267662fa188e`
- Prefix-cache profile: `qwen38-p150x2-prefix-cache-v1`, policy BLAKE3
  `f83d2991a0b964449857b49cbaa42de53eeb535a8ef970d44c7dbb81331cf6bb`
- Scheduler profile required on every root: `qwen38-p150x2-continuous-slot-b8-v1`
- Package: `/nix/store/gzwghi7spl93iv4y6qb680cg8wyfbq6r-qwen38-0.1.0`
- Execution identity:
  `/nix/store/qwnxwcppn16za67i710qm5drnzs57vg9-qwen36-execution-identity-0.1.0`
- `admitted: true`, `reasons: []`, `simulated: false`, physical PCIe devices 0 and 1.

## Gates

| Gate | Bound | Measured | Result |
| --- | ---: | ---: | --- |
| matched_identity | required | yes | pass |
| exact_parity | required | yes | pass |
| resumed_hits | required | 9 resumed / 13 missed turns | pass |
| hit_median_ttft_ratio | <= 0.70 | 0.409201 | pass |
| hit_median_wall_ratio | <= 0.85 | 0.573632 | pass |
| miss_median_wall_ratio | <= 1.10 | 0.998594 | pass |
| miss_wall_ratio | <= 1.25 | 1.007258 | pass |
| startup_dram_drop_bytes | <= 536,870,912 | 364,904,448 | pass |

## Campaign

Four fresh served-limit roots on the promoted package in the predeclared order
cold, cached, cached, cold, at physical devices 0 and 1. Every root served the
eight-slot scheduler profile; the cached roots added
`share/qwen36/prefix-cache-policy.json`. Each root ran both declared workloads
(3 conversations x 4 turns and 5 conversations x 2 turns).

- Roots: `/home/brittonr/models/qwen38-readmit-2026-10-26/campaign/` with `a1`, `b1`, `b2`, and `a2`, each holding `w3.json` and `w5.json`
- Provenance: `.cairn/changes/readmit-qwen38-prefix-cache-b8/evidence/readmit-gzwghi7-prefix-provenance.json`
- Containment before/after: `non-v3`, passed, same package/profile/identity.
