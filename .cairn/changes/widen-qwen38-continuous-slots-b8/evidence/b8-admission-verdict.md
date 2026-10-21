# B=8 continuous-slot admission: the profile is admitted for its package and identity

Recorded 2026-10-21 on physical PCIe devices 0 and 1 (two P150a, firmware 19.11.0.0,
driver 2.10.0). The B=8 continuous-slot scheduler profile passes every predeclared
gate against the group coordinator.

## Protocol

- Admission policy `continuous-slot-admission-policy-b8.ncl` (profile
  `qwen38-p150x2-continuous-slot-admission-b8-v1`, BLAKE3
  `4a63054752d830bff8075ff2ed7bea354c3ee43bc380815e2fc0ce58e44aa1f2`), predeclared
  before any measurement: the same workload and gates as B4, with
  `scheduler_profile_id` the B=8 policy (BLAKE3 `43d21108…`).
- Workload `qwen38-staggered-mixed-length-v1` (BLAKE3 `8a0b5883…`): two long requests
  at t=0 running to 1,024 tokens, six short ones arriving at 4–24 s, all with the
  shared 1,024-token cap. Filler: the pinned RWKV-7 paper text.
- Roots: one package, run twice on the same host and port, with the production unit's
  environment and the same configuration on both sides. The baseline is the default
  group coordinator; the candidate is the production serve binary with
  `--continuous-slot-policy …/continuous-slot-policy-b8.json`. Each root ran every
  request alone first, then two measured passes.
- Receipt: `pkgs/qwen36/evidence/2026-10-21/qwen38-continuous-slot-admission-b8-receipt.json`,
  BLAKE3 `2689347235ed11105d39d284cd5f0a3b37e0809ae101bc22c31354f97edc2280`.
  Package `/nix/store/inkfvsjqchqp7bki76asdib3b57nw6kj-qwen38-0.1.0`, execution
  identity `/nix/store/fk8p9n0xdx62yx8hr1jnccgjfyp2j8s0-qwen36-execution-identity-0.1.0`,
  runtime `/nix/store/7hplkynh047dsk98cwqpfsaaq6lwyy0m-tt-metal-0.77.0`.

## Result

| Gate | Bound | Measured | Result |
| --- | ---: | ---: | --- |
| matched identity | required | yes | pass |
| every request completed | required | yes | pass |
| candidate tokens equal its solo references | required | exact | pass |
| short median wall ratio | ≤ 0.5 | 0.0652 | pass |
| short p95 wall ratio | ≤ 1.0 | 0.2066 | pass |
| long p95 wall ratio | ≤ 1.3 | 1.2563 | pass |
| worst slowdown ratio | ≤ 1.0 | 0.0555 | pass |
| longest inter-token pause | ≤ 6.0 s | 5.4750 s | pass |
| median inter-token pause ratio | ≤ 1.25 | 1.0731 | pass |
| delivered-throughput ratio | ≥ 0.95 | 1.0618 | pass |
| startup DRAM free ratio | ≥ 0.99 | 1.0000 | pass |

`admission.admitted = true`, no failure reasons, for scheduler profile
`qwen38-p150x2-continuous-slot-b8-v1` on the package and execution identity above.

| Per pass (baseline → candidate) | Pass 1 | Pass 2 |
| --- | --- | --- |
| Short median wall (s) | 55.49 → 3.30 | 51.08 → 3.33 |
| Short p95 wall (s) | 65.49 → 12.60 | 61.08 → 12.62 |
| Long p95 wall (s) | 51.89 → 60.05 | 48.73 → 61.22 |
| Worst slowdown against solo | 96.61 → 4.97 | 90.10 → 5.00 |
| Longest pause (s) | 3.509 → 5.475 | 0.885 → 5.472 |
| Median pause (s) | 0.0436 → 0.0468 | 0.0436 → 0.0468 |
| Delivered tokens/s | 34.16 → 39.48 | 36.48 → 38.73 |
| Decode traces captured | 8 → 0 | 8 → 0 |

Short requests stop waiting behind the two long ones (median wall 3.3 s against
51–55 s), long-request p95 stays inside the 1.3 bound, delivered throughput is 6 %
higher, the longest pause is 5.475 s against the 6 s bound, and the width-8 trace stayed
resident: zero decode-trace captures during the measured passes.

## The first attempt failed, and why

The first run (`b8-admission-unmatched-receipt.json`, BLAKE3 `63e8d549…`) passed ten
gates and failed the startup-DRAM gate at 0.9631 against 0.99: the candidate had
9,530,484,736 B free against the baseline's 9,895,389,184 B, a 364,904,448 B cost.

The cause was in the campaign, not the scheduler: the candidate root alone ran with
`--prefix-cache-policy`, whose saved carried-state slots are allocated with the masked
prefill traces. The B4 admission campaign ran both roots with the scheduler policy only.
A baseline startup probe on the same package with a warm program cache reported
9,895,389,184 B free, identical to the first baseline, so process order and cache state
were excluded. The gate was not changed, and the unmatched run was not admitted.

The campaign was rerun with configuration-matched roots, the scheduler policy alone on
both sides. That run admitted the profile for the then-current package
(`b8-admission-prewiring-receipt.json`, BLAKE3 `41a08e10…`). It was then rerun once
more on the package the tree now builds, so the shipped receipt binds the current
execution identity.

## Non-claims

- The admission covers only the named scheduler profile, package, and execution
  identity, on one declared workload. It is not a general latency or throughput claim.
- **The served default is unchanged.** The group coordinator stays the served profile
  until a deployment change selects the B=8 policy.
- The prefix-cache configuration is not part of this admission: the measured roots ran
  without it. A deployment that adds the prefix cache must account for its state slots.
- The receipt reports cross-root token differences for l1, l2, and s3, and the baseline
  group root differs from its own s3 solo reference. The gated parity is candidate
  against the candidate's solo references, which is exact.
- The longest pause (5.475 s against a 6 s bound) and the long p95 ratio (1.256 against
  1.3) are close to their bounds; a different workload may fail them.
