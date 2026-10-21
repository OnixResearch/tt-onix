# B=8 continuous-slot admission: every gate passes except the declared startup-DRAM bound

Recorded 2026-10-21 on physical PCIe devices 0 and 1 (two P150a, firmware 19.11.0.0,
driver 2.10.0). Package `39p6vlfkf15jy77lh9x8k2gzxqp7fnwx-qwen38-0.1.0`, runtime
`7hplkynh047dsk98cwqpfsaaq6lwyy0m-tt-metal-0.77.0`.

## Protocol

- Admission policy `continuous-slot-admission-policy-b8.ncl` (profile
  `qwen38-p150x2-continuous-slot-admission-b8-v1`, BLAKE3
  `4a63054752d830bff8075ff2ed7bea354c3ee43bc380815e2fc0ce58e44aa1f2`), predeclared
  before the run: same workload and gates as B4, `scheduler_profile_id` the B8 policy
  (BLAKE3 `43d21108…`).
- Workload `qwen38-staggered-mixed-length-v1` (BLAKE3 `8a0b5883…`): two long requests at
  t=0 to 1,024 tokens, six short ones at 4–24 s, all with the shared 1,024-token cap.
  Filler: the pinned RWKV-7 paper text; the workload hash matches the B4 screen.
- Baseline: the default group coordinator. Candidate: the production serve binary with
  `--continuous-slot-policy …/continuous-slot-policy-b8.json` and the prefix-cache
  policy. Each root ran every request alone first, then two measured passes.
- Receipt: `pkgs/qwen36/evidence/2026-10-21/qwen38-continuous-slot-admission-b8-receipt.json`,
  BLAKE3 `63e8d5494ad8196a65d4ba86d6dd90bae517eff18823460cd269ba817696af81`.

## Result

| Gate | Bound | Measured | Result |
| --- | ---: | ---: | --- |
| matched identity | required | yes | pass |
| every request completed | required | yes | pass |
| candidate tokens equal its solo references | required | exact | pass |
| short median wall ratio | ≤ 0.5 | 0.0493 | pass |
| short p95 wall ratio | ≤ 1.0 | 0.1468 | pass |
| long p95 wall ratio | ≤ 1.3 | 1.0472 | pass |
| worst slowdown ratio | ≤ 1.0 | 0.0429 | pass |
| longest inter-token pause | ≤ 6.0 s | 0.610 s | pass |
| median inter-token pause ratio | ≤ 1.25 | 1.0722 | pass |
| delivered-throughput ratio | ≥ 0.95 | 1.2741 | pass |
| **startup DRAM free ratio** | **≥ 0.99** | **0.9631** | **fail** |

The B8 scheduler does what the change set out to do: short requests stop waiting behind
long ones (median wall 2.54 s against 51.66 s; p95 9.01 s against 61.66 s), long-request
p95 is within 4.7 %, delivered throughput is 27 % higher, the longest pause is 0.61 s,
and every candidate request reproduced its solo tokens exactly. The width-8 trace stayed
resident (zero captures during the measured passes).

## The failing gate and its cause

The candidate left 9,530,484,736 B of device DRAM free after startup against the
baseline's 9,895,389,184 B: a 364,904,448 B cost, ratio 0.9631 against the declared 0.99
floor. The predeclared gate allows 1 % (about 99 MB), so the wider resident trace and
its buffers exceed the declared budget by about 266 MB.

The gate was not changed. The B=8 scheduler profile therefore stays a candidate; the B4
profile remains the served default. Reducing the startup footprint (for example, sizing
the resident trace and its buffers to the profile's slot capacity rather than the
adapter's maximum width, or sharing the masked-prefill scratch) is the next bounded
lever if this profile is to be admitted.

## Non-claims

- The B=8 scheduler profile is **not admitted**. Only the receipt's 10 passing gates are
  claimed; the DRAM gate is a declared failure.
- The group coordinator stays the served default; no NixOS default or deployment changed.
- The comparison covers one workload on one package and identity. It is not a general
  latency or throughput claim.
- The receipt reports cross-root token differences for l1, l2, and s3, and the baseline
  group root differs from its own s3 solo reference. The gated parity is candidate versus
  the candidate's solo references, which is exact.
