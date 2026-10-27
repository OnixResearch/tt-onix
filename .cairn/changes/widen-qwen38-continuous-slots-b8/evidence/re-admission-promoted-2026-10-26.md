# B=8 scheduler admission re-earned on the promoted package, 2026-10-26

This is the current admission. The 2026-10-21 B=8 receipt and the
`wlsl6pnx…`/`l4arf461…` package and identity it named are superseded for
serving purposes but remain historical evidence and are not deleted.

## Receipt

- Path: `pkgs/qwen36/evidence/2026-10-26/qwen38-continuous-slot-admission-b8-receipt.json`
- BLAKE3: `e5487ac541cc753ef4d3030f3e3d56bac04a499adb6a4a1051a50c94984d6b01`
- Receipt kind: `continuous-slot-admission`
- Admission policy: `qwen38-p150x2-continuous-slot-admission-b8-v1`, BLAKE3
  `4a63054752d830bff8075ff2ed7bea354c3ee43bc380815e2fc0ce58e44aa1f2`
- Scheduler profile: `qwen38-p150x2-continuous-slot-b8-v1`, policy BLAKE3
  `43d2110837561cfbe8e46bb9424883683ff06c0f52f6158c438e3d8e224a475d`
- Package: `/nix/store/gzwghi7spl93iv4y6qb680cg8wyfbq6r-qwen38-0.1.0`
- Execution identity:
  `/nix/store/qwnxwcppn16za67i710qm5drnzs57vg9-qwen36-execution-identity-0.1.0`
- `admitted: true`, `reasons: []`, `simulated: false`, physical PCIe devices 0 and 1.

## Gates

| Gate | Bound | Measured | Result |
| --- | ---: | ---: | --- |
| matched_identity | required | yes | pass |
| all_completed | required | yes | pass |
| exact_candidate_parity | required | exact | pass |
| short_median_wall_ratio | <= 0.5 | 0.066147 | pass |
| short_p95_wall_ratio | <= 1.0 | 0.209471 | pass |
| long_p95_wall_ratio | <= 1.3 | 1.250890 | pass |
| slowdown_ratio | <= 1.0 | 0.063229 | pass |
| median_inter_token_gap_ratio | <= 1.25 | 1.072873 | pass |
| inter_token_gap_seconds | <= 6.0 | 5.490118 | pass |
| delivered_throughput_ratio | >= 0.95 | 1.068002 | pass |
| startup_dram_free_ratio | >= 0.99 | 1.0 | pass |

## Campaign

Two fresh served-limit roots on the promoted package, each reset before start and
run at physical devices 0 and 1: the default group coordinator baseline and the
production serve binary with
`share/qwen36/continuous-slot-policy-b8.json`. Each root ran every request alone
and then two measured passes, 1,024-token cap, with production `step`, grouped,
and traced prefill and no `TT_METAL_DISABLE_SFPLOADMACRO`.

- Roots: `/home/brittonr/models/qwen38-readmit-2026-10-26/campaign/scheduler-group/root.json`
  and `/home/brittonr/models/qwen38-readmit-2026-10-26/campaign/scheduler-continuous/root.json`
- Provenance: `.cairn/changes/widen-qwen38-continuous-slots-b8/evidence/readmit-gzwghi7-scheduler-provenance.json`
- Containment before/after: `non-v3`, passed, same package/profile/identity.
