# The hybrid prefix cache, re-admitted in the served eight-slot configuration

Recorded 2026-10-21 on physical PCIe devices 0 and 1 (two P150a, firmware 19.11.0.0,
driver 2.10.0). The hybrid prefix cache passes every predeclared gate in the served
continuous-slot configuration, and the eight-slot scheduler admission was measured
again on the same package so that one package carries both receipts.

## What was measured, and against which policy

- Policy: `pkgs/qwen36/prefix-cache-admission-policy-v5.ncl`, schema 5, profile
  `qwen38-p150x2-prefix-cache-admission-v5`. Exported by the package as
  `share/qwen36/prefix-cache-admission-policy-v5.json`, BLAKE3
  `01ff50affee07c0ac22848408973728a07e837908790c6844a98267662fa188e`.
- The policy's `scheduler_profile_id` is read from the packaged eight-slot scheduler
  policy rather than written out, so the bound identifier is the profile the service
  actually reports: `qwen38-p150x2-continuous-slot-b8-v1`.
- Every schema-2 gate stays bound: the same ABBA root order, the same two workloads,
  the same hit, miss, parity, and startup-DRAM bounds.
- The policy was committed as `6726090` at 18:19:39 EDT; the first root's process
  started at 18:19:40 EDT, and the first measured request came after its startup.

## Roots

Four fresh processes per workload, in the predeclared order cold, cached, cached,
cold. Every root ran the eight-slot scheduler profile at the served limits
(`--max-sequence-length 131072`, `--max-generation-tokens 8192`, the production
environment). The cached roots add the prefix-cache policy; the cold roots do not,
so the cache is the only difference between the roots.

| Root | Kind | Scheduler profile | Workload w3 | Workload w5 |
| --- | --- | --- | --- | --- |
| a1 | cold | `qwen38-p150x2-continuous-slot-b8-v1` | 12 turns | 10 turns |
| b1 | cached | same | 12 turns | 10 turns |
| b2 | cached | same | 12 turns | 10 turns |
| a2 | cold | same | 12 turns | 10 turns |

The two workloads are the same bytes the 2026-09-27 campaign used: w3 is
`7bdff9dbc3e4a36b11c7746255484b7d77a378a4bb58dd509c8ff954cfc175b8` and w5 is
`38038f5d8c1cba257d8ff2da181be0c3b4574daf7a3e715a34c7ab6e8236bbb6`.

- Receipt: `pkgs/qwen36/evidence/2026-10-21/qwen38-prefix-cache-abba-admission-receipt.json`,
  BLAKE3 `eb09b2cc9ec785b223200bbe67b11b6e2665dd25214ce20ca41500220b136f91`.
- Roots: `qwen38-prefix-cache-abba-{a1,b1,b2,a2}-{w3,w5}.json.gz`, and the startup
  facts in `qwen38-prefix-cache-abba-startup.json`.
- Package `/nix/store/wlsl6pnxllv9qhr6yqimnwv3si0qyz9k-qwen38-0.1.0`, execution
  identity `/nix/store/l4arf461kxbbns0nnlg2vd98anx943pc-qwen36-execution-identity-0.1.0`,
  runtime `/nix/store/7hplkynh047dsk98cwqpfsaaq6lwyy0m-tt-metal-0.77.0`.

## Result

| Gate | Bound | Measured | Pass |
| --- | ---: | ---: | --- |
| matched identity | required | yes | yes |
| exact token parity across the four roots | required | yes | yes |
| every cached root resumed the prefix | required | yes | yes |
| hit median TTFT ratio | ≤ 0.7 | 0.4118 | yes |
| hit median wall ratio | ≤ 0.85 | 0.5747 | yes |
| miss median wall ratio | ≤ 1.10 | 1.0063 | yes |
| miss wall ratio | ≤ 1.25 | 1.0698 | yes |
| startup DRAM drop | ≤ 536,870,912 B | 364,904,448 B | yes |

`admission.admitted = true`, no failure reasons, for prefix-cache profile
`qwen38-p150x2-prefix-cache-v1` on the package and execution identity above. The
campaign produced 9 resumed hits and 13 misses.

The eight-slot scheduler's own admission was measured on the same package and
identity: `pkgs/qwen36/evidence/2026-10-21/qwen38-continuous-slot-admission-b8-receipt.json`,
BLAKE3 `4573cea118453a339ad86b73fe67f729e38dd1c5187b9bb09a4ba527e66af11e`, all
eleven gates pass (short median wall ratio 0.0624, short p95 0.2001, long p95 1.1874,
worst slowdown 0.0664, longest pause 5.542 s, delivered throughput 1.1118, startup
DRAM free ratio 1.0).

## The admitted package, served in the deployed configuration

The package that carries both receipts was then served in the deployed
configuration: the eight-slot policy and the prefix-cache policy together, at the
served limits, on devices 0 and 1.

| Observation | Value |
| --- | --- |
| Readiness | 466 s |
| Device DRAM after startup | 9,501,091,840 B free of 23,603,897,344 B allocated |
| Requests sent concurrently | 8 |
| Failed requests | 0 |
| Scheduler profile | `qwen38-p150x2-continuous-slot-b8-v1` |
| Decode path | `continuous-slot` |
| Distinct slot indices | 0, 1, 2, 3, 4, 5, 6 (one slot refilled inside the run) |
| Execution identity | `/nix/store/l4arf461kxbbns0nnlg2vd98anx943pc-qwen36-execution-identity-0.1.0` |
| Runtime identity | `/nix/store/7hplkynh047dsk98cwqpfsaaq6lwyy0m-tt-metal-0.77.0` |
| Devices | physical PCIe 0 and 1, `simulated: false` |
| Client-observed walls | 1.389 s to 3.421 s |
| Answers | Paris, Red, 4, Iron, 7, Mars, Two hydrogens one oxygen, Summer |

The instance was stopped afterwards; no process holds either device node.

## The first schema-5 policy was defective, and the receipt said so

The first version of the schema-5 policy named the scheduler profile with an
underscore, `qwen38_p150x2-continuous-slot-b8-v1`, which is not an identifier any
service reports. The campaign on that policy was rejected with
`scheduler-profile-differs` and `gate-failed:matched_identity`, exactly as the new
gate should reject a mismatch. The defect was in the policy, not in the measurement:
the packaged scheduler policy's own `profile_id` is
`qwen38-p150x2-continuous-slot-b8-v1`, which the roots had recorded.

The policy was corrected to take that identifier from the packaged scheduler policy
instead of writing it out, so it cannot drift again, and the campaign was repeated on
the corrected package. No gate was changed at any point: the corrected policy
carries the same eight gates with the same bounds.

## Non-claims

- The admission covers the named prefix-cache profile, package, execution identity,
  and the two workloads. It is not a general cache claim.
- The scheduler profile is bound by the receipt; the scheduler's own admission is the
  separate receipt named above.
- The served profile still points at the earlier package until the switch runs. The
  switch needs root: the profile directory and the managed drop-in are root-owned.
  The package itself was served directly in the deployed configuration above, which
  is the same command line the switch ends up running.
- The host's other work drifts during the campaign. The ABBA order cancels that
  drift, which is what the protocol is for, and the miss ratios above are close to
  one.
