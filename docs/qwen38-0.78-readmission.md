# Qwen 0.78 re-admission receipts

## Status

Both deploy-required receipts were re-earned on the TT-Metal 0.78 package:

| Receipt | Path | BLAKE3 |
| --- | --- | --- |
| B=8 continuous-slot admission | `pkgs/qwen36/evidence/2026-10-28/qwen38-continuous-slot-admission-b8-receipt.json` | `1c92aec7f96c295e5128ae7322157682e7376b087f31de9b697a82324c640390` |
| Prefix-cache admission | `pkgs/qwen36/evidence/2026-10-28/qwen38-prefix-cache-admission-receipt.json` | `2def4d0e128a1118305aedec139fac6469a83af50a8ea45dd18c15bd335ddb42` |

Both name:

- package `/nix/store/aj5wfbg0slndcp7szj9h9jggqczli77c-qwen38-0.1.0`;
- execution identity `/nix/store/wlbzj42qsynbn2isa5xgf988yrad1nxd-qwen36-execution-identity-0.1.0`;
- TT-Metal `/nix/store/lz9hma19v7hfhc53fldq6qhdkyyyjmrh-tt-metal-0.78.0`;
- physical PCIe devices 0 and 1.

The deploy guard accepts both receipts and the package containment:

```text
receipt: continuous-slot-admission .../qwen38-continuous-slot-admission-b8-receipt.json
receipt: prefix-cache-admission .../qwen38-prefix-cache-admission-receipt.json
containment: package .../wlbzj42... mode v3-admitted profile qwen38-p150x2-batched-throughput-b8-v3
```

The move is blocked before any symlink change:

```text
qwen38-profile-service: NixOS owner is missing, disabled or masked
```

The served symlink still resolves to
`/nix/store/wlsl6pnxllv9qhr6yqimnwv3si0qyz9k-qwen38-0.1.0`, and
`qwen38-p150x2.service` is inactive. No service start, deployment, profile
change, or device reset after the campaign is claimed.

## B=8 scheduler gates

The receipt admits `qwen38-p150x2-continuous-slot-b8-v1`:

| Gate | Bound | Measured |
| --- | ---: | ---: |
| short median wall ratio | <= 0.5 | 0.0654 |
| short p95 wall ratio | <= 1.0 | 0.2062 |
| long p95 wall ratio | <= 1.3 | 1.2281 |
| slowdown ratio | <= 1.0 | 0.0505 |
| median inter-token gap ratio | <= 1.25 | 1.0740 |
| max inter-token gap | <= 6.0 s | 5.362 s |
| delivered throughput ratio | >= 0.95 | 1.0870 |
| startup DRAM free ratio | >= 0.99 | 1.0000 |

The two roots used the served limits (131,072 context, 8,192 generation
tokens), candidate decode, `step`, grouped and traced prefill, and the same
device-DRAM-free value after startup. The measured short-request median wall
was 3.12-3.32 s for the scheduler and 50.8-51.3 s for the group coordinator.

## Prefix-cache gates

The receipt admits `qwen38-p150x2-prefix-cache-v1`:

| Gate | Bound | Measured |
| --- | ---: | ---: |
| hit median TTFT ratio | <= 0.70 | 0.4028 |
| hit median wall ratio | <= 0.85 | 0.5767 |
| miss median wall ratio | <= 1.10 | 0.9920 |
| miss wall ratio | <= 1.25 | 0.9993 |
| startup DRAM drop | <= 512 MiB | 348 MiB |

The ABBA campaign covered both declared workloads: three conversations with
four turns, and five conversations with two turns. It recorded 9 resumed turns
and 13 missed turns with exact parity. The cached roots used the B=8
continuous-slot policy plus the prefix-cache policy; the cold roots used the
scheduler policy alone.

## Remaining blocker

The host's NixOS owner unit is masked. The repository deploy tooling cannot
override that mask, and the profile drop-in is not a substitute. Re-enabling
the owner requires a reviewed host generation (or rollback) with root access;
this record does not claim that it was done.

The 0.78 smoke and these receipts are not MTP, fused-GDN, long-context
accuracy, or production-serving claims beyond the gates above.
