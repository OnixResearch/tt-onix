# Physical result

## Verdict

The B=8 batch-admission candidate is **REJECTED** on its predeclared gates. The
B=4 admission remains the only admitted batch width, and the B=8 binding stays
`physically_admitted = false`.

- Receipt BLAKE3: `e6641e7efa02f384e550e0621fea9ea97f1eaffe5d258fb3634312cc183a3fa5`
- Tracked receipt: `pkgs/qwen36/evidence/2026-08-26/qwen38-27b-batch-admission-b8-receipt.json`
- Candidate package: `/nix/store/k1r12ppf0pzqnqqvdb78sfksn7v37r5r-qwen36-0.1.0`
- Runtime: `/nix/store/mg0ha5nsvnir1y1ysiyyzmm4hk83z3m6-tt-metal-0.74.0`
- Physical devices: `0` and `1`
- Root order: `single,batch,batch,single`
- Corpus: eight prompts, three measured runs per prompt per root

## Measured gates

| Gate | Measured | Bound | Result |
| --- | ---: | ---: | --- |
| Aggregate decode speedup | `5.881509219624968` | at least `6.0` | FAIL |
| Per-user decode retention | `0.735188652453121` | at least `0.75` | FAIL |
| TTFT ratio | `9.079183271882933` | at most `9.0` | FAIL |
| Tail-latency ratio | `1.4461700431804296` | at most `1.5` | pass |
| Exact token parity | pass | required | pass |
| Exact text parity | pass | required | pass |
| State slices 0..7 per group | pass | required | pass |
| Clean reset | pass | required | pass |
| Identity match | pass | required | pass |

- Single-stream median aggregate: `16.88145527595542` tokens/second
- B=8 batch median aggregate: `99.28843484621837` tokens/second

## Interpretation

Served B=8 throughput is real — 99.3 aggregate tokens/second against 16.9 at
B=1 — but per-user efficiency lands at 73.5 percent, just under the 75-percent
floor this profile shares with the admitted B=4 profile (88.7 percent at B=4).
The layer-level curve projected about 6.4x; the served system delivers 5.88x
because eight serialized per-user prefills dominate each request and the extra
four users add decode-step time without proportional decode gain.

This is a near-miss, not a failed mechanism: every correctness gate passed and
the margin is roughly two percentage points of efficiency. The bound, however,
was predeclared and the rejection stands unchanged.

## Run artifacts

- Raw roots, request corpus, server logs, and the exact rejection receipt:
  `.pi/evidence/qwen38/b8-admission-20260826/`
- The runner's bounded group-retry loop fired on `batch-2` when concurrent
  clients straggled past the coordinator's 10 ms collection window, assembling
  a width-6 and a width-2 group. Explicit B=8 mode rejected those groups before
  device work, and each retained measured group completed at width 8 with full
  slice prefixes 0 through 7. No rejected-group sample entered the receipt.

## Non-claims

- This rejection does not falsify B=8 execution; it withholds the throughput claim only.
- It does not measure B=8 under a future runtime with faster or batched prefill.
- It does not weaken or reopen the admitted B=4 receipt.
- The efficiency loss is attributed by phase timing; no device-side profiler ran.
