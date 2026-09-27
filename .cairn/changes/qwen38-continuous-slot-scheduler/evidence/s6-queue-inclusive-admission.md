# s6: queue-inclusive comparison against the group coordinator (2026-09-27)

The continuous-slot profile `qwen38-p150x2-continuous-slot-b4-v1` passes every
predeclared gate against the group coordinator, on one package and one workload. The
receipt admits that profile for that package and execution identity only. The group
coordinator stays the served default, because no deployment change selects the profile.

## Protocol

- Gates: `pkgs/qwen36/continuous-slot-admission-policy.ncl`, committed in `65b33de`
  before the measurement. Exported policy BLAKE3:
  `f887c1b353f0ba1863e78305fd5f4bdbca0f5646d7bd14ec7f46e81cfad1c33d`.
- Roots: one package, run twice on britton-desktop (P150x2, devices 0 and 1), port 8011,
  with the production unit's environment.
  - Package: `/nix/store/jxhspibs4ml8sbspj16bynhb1yp887vb-qwen38-0.1.0`.
  - Execution identity:
    `/nix/store/h73yynndm34hkzr3aimlfnif6r3402iy-qwen36-execution-identity-0.1.0`.
  - Runtime: `/nix/store/7hplkynh047dsk98cwqpfsaaq6lwyy0m-tt-metal-0.77.0`.
  - Baseline: the default group coordinator, 16:50–17:05 UTC.
  - Candidate: `--continuous-slot-policy` (scheduler policy BLAKE3
    `8a77067c9b18ad69ad45f465566f496a8f05a38d6b6d0fb03c4a5beb0e2994af`),
    17:05–17:14 UTC.
- Workload `qwen38-staggered-mixed-length-v1` (content BLAKE3
  `8a0b5883ded3058795e230e4ff9ee79d4f732dd7877ddb054faf84d8dc2e3448`): two long
  requests at t=0 that run to 1,024 tokens, and six short ones arriving at 4–24 s. The
  shared `max_tokens` of 1,024 lets the group coordinator batch them. Details are in
  `s5-simulator-blocker-and-physical-screen.md`.
- Per root, each request first ran alone as its solo reference. Then two measured passes
  of the staggered workload were streamed and timed at the client, from each request's
  send to its `[DONE]`.
- Every gate must hold on both matched pairs of passes. The receipt reports the worst
  pair.

## Result

| Gate | Bound | Worst pair | Pass |
| --- | --- | --- | --- |
| Matched identity (package, runtime, devices, workload, not simulated) | required | yes | yes |
| Every request completed | required | yes | yes |
| Candidate tokens equal its solo references | required | 16 of 16 | yes |
| Short median wall ratio | ≤ 0.5 | 0.0741 | yes |
| Short p95 wall ratio | ≤ 1.0 | 0.2046 | yes |
| Long p95 wall ratio | ≤ 1.3 | 1.2043 | yes |
| Worst slowdown ratio (starvation guard) | ≤ 1.0 | 0.0533 | yes |
| Longest inter-token pause | ≤ 6.0 s | 5.4796 s | yes |
| Median inter-token pause ratio | ≤ 1.25 | 1.0245 | yes |
| Delivered-throughput ratio | ≥ 0.95 | 1.1168 | yes |
| Startup DRAM free ratio | ≥ 0.99 | 1.0 | yes |

| Per pass (baseline → candidate) | Pass 1 | Pass 2 |
| --- | --- | --- |
| Short median wall (s) | 51.74 → 3.83 | 51.67 → 3.49 |
| Short p95 wall (s) | 61.74 → 12.63 | 61.67 → 12.35 |
| Long p95 wall (s) | 48.90 → 58.89 | 48.91 → 58.62 |
| Worst slowdown against solo | 52.1 → 2.8 | 52.0 → 2.6 |
| Longest pause (s) | 0.91 → 5.48 | 0.90 → 5.48 |
| Median pause (s) | 0.044 → 0.045 | 0.044 → 0.045 |
| Delivered tokens/s | 36.1 → 40.3 | 36.1 → 40.5 |
| Decode traces captured | 2 → 1 | 2 → 1 |

Per-request walls in the first pass (seconds, baseline → candidate):

| l1 | l2 | s1 | s2 | s3 | m1 | s4 | s5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 48.9 → 58.2 | 48.9 → 58.9 | 61.7 → 2.4 | 57.7 → 4.0 | 53.7 → 10.7 | 49.7 → 12.6 | 45.7 → 3.7 | 41.7 → 2.5 |

### Where the time goes

- **Baseline.** l1 and l2 decode as a width-2 group until 48.9 s. The six short
  requests queue behind them and then run as one width-6 group. Their first tokens
  arrive at 56.8 s and every stream ends at 65.7 s, when the longest of them (m1)
  finishes.
- **Candidate.** Each short request is prefilled at the next window boundary. Its
  queue time is 0.1–2.8 s, and its stream ends at the boundary after its last token.
- **Long requests.** The long requests lose 9–10 s: every refill beside them pauses
  them for its prefill (about 8 s of prefill in total, plus l2's 1.9 s beside l1). The
  median pause is within 2.5 % of the baseline (0.045 s against 0.044 s), so the width-4
  trace decodes at almost the same per-step speed as the baseline's width-2 group.

### Margins

Two gates passed with less than 10 % margin, and both are set by the same prefill:

- The longest pause, 5.48 s against 6.0 s, is s3's 5.42 s prefill of 9,618 tokens
  beside the live rows. The policy comment estimated the largest refill prompt at about
  6.5K tokens; the measured prompt was larger, and no gate was changed.
- The long p95 ratio, 1.204 against 1.3, includes that same pause.

The policy always allows the first refill at a boundary, whatever its length. A longer
refill prompt would therefore stall live streams for longer. [INFERENCE] Because the
measured pauses match the refill prefill times, a 60K-token refill (39 s TTFT measured
earlier) would pause live streams for about 39 s and fail the declared 6 s tail bound.
The admission covers this workload's prompt sizes, not arbitrary ones.

## Receipt

- `pkgs/qwen36/evidence/2026-09-27/qwen38-continuous-slot-admission-receipt.json`,
  BLAKE3 `aabc9b8a560a1ac0fe31935d5d731a4c3a6e8fddd47a17da99c7d9a567545a96`.
  `admission.admitted = true`, with no failure reasons, for scheduler profile
  `qwen38-p150x2-continuous-slot-b4-v1` on the package and execution identity above.
- Roots:
  - `qwen38-continuous-slot-group-root.json.gz`: JSON BLAKE3
    `f83d6650053c6ffd0409458cef1558670486cc214202fc7299cb8ef38a3c7715`.
  - `qwen38-continuous-slot-continuous-root.json.gz`: JSON BLAKE3
    `8ab0e183d40332babe1d8e965d6a08b32228a3a1ec9db292d7b49e1e77b12e43`.

  The receipt binds the decompressed JSON.
- Reproduce with `qwen38-continuous-slot-receipt --baseline-root <group.json>
  --candidate-root <continuous.json> --output <receipt.json>` from the same package.
  Run on the decompressed roots, it reproduced the stored receipt byte for byte.

## Cross-root numerics

The solo references of the two roots agree for five of eight requests. l1, l2 and s3
diverge at tokens 46, 32 and 29: the baseline decodes a solo request at width 1, the
candidate at width 4. The gates compare each root with its own references, as the
design specifies. The baseline's own outputs also change with load: in the width-6
group, s3 diverged from its solo run at token 29. See the s5 evidence.

## Not claimed

- One host, one workload, and two passes per root. The receipt is not a general
  latency or throughput claim.
- Capture counts are reported, never gated.
- Serving the profile by default is a separate deployment decision. It trades
  queue-inclusive latency for short requests against long streams pausing for the
  refills beside them.
