# Qwen38 speculative serving progress

## Status

Speculation remains off by default. The latest two-draft candidate reached a 0.992 round-only ratio. The three-draft control reached 0.797. Both passed bounded token parity, but neither establishes a serving speedup. General exactness and production safety remain unproven.

An earlier target-KV test failed its bitwise comparison. A follow-up found the zero sign absent before rollback, at mesh upload/readback. The recovery prototype remains diagnostic-only.

Local commits reside on `work/qwen38-mtp-serving-20260908` in `../tenstorrent.nix-mtp-serving-20260908`. No push or main-branch integration ran. Owner review remains open.

## Implemented candidate

- A device-free core owns commit decisions, request bounds, and the sticky economics guard.
- The serving adapter owns request-scoped traces, state snapshots, deadlines, and cleanup.
- The watchdog adapter terminates the process after an abandoned device-worker timeout.
- Rejection restores the GDN prefix and replays the head continuation. Target KV relies on position masks and rewrite-before-read, not complete tail restoration.
- Runtime-scoped head caches prevent CLI and server paths from reusing legacy caches across runtime identities.
- The diagnostic screen compares greedy MTP output with plain traced decode. It does not admit production use.

The cache-regeneration experiment restored 12/15 pilot draft acceptances and reproduced the historical token golden. This isolates a cache-dependent regression. It does not identify a corrupt tensor or prove that 0.77 misreads every 0.74 cache.

The pilot measures one initial window per prompt. Its acceptance average does not predict the acceptance distribution across complete serving requests.

## Five-prompt physical screen (2026-09-08)

Package: `/nix/store/r5rc59ayv38sqmgr1dhl3qpm5jkzcncn-qwen38-0.1.0`.
Runtime: `/nix/store/dr9bj8qwflylq39zl5ajcx42cdrb1bv9-tt-metal-0.77.0`.
Checkpoint: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Mesh: physical P150x2, `1x2`.

Private evidence: `.pi/logs/qwen38-mtp-serving/hw6-preserved/serving-screen.json` in the primary repository. The copy preserves the receipt from 18:25:32. The shared `screen/` path is not an immutable run identifier.

| Prompt | Compared tokens | Accepted drafts | Rounds | Rejected rounds | Speculative tokens emitted | Round time | Plain-equivalent / round time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Deterministic rollback | 64 | 4 | 3 | 2 | 7 | 0.812 s | 0.533 |
| Compiler correctness | 64 | 1 | 2 | 2 | 3 | 0.449 s | 0.393 |
| Capital of France | 64 | 4 | 2 | 1 | 6 | 1.412 s | 0.302 |
| Reliable cache | 64 | 3 | 3 | 2 | 6 | 0.575 s | 0.645 |
| Counting | 64 | 10 | 5 | 2 | 15 | 0.985 s | 0.906 |

All 320 compared tokens matched. Across 15 rounds, the candidate accepted 22 drafts and emitted 37 tokens in 4.232 s. Nine rounds rejected a draft. The mean was 1.467 accepted drafts per round, not 2.4.

The aggregate round-only ratio is 0.543. Its numerator sums each prompt's measured plain-step median times its speculative emitted count. Its denominator sums capture-excluded round durations. This ratio excludes request capture, prefill, fallback capture, and the remaining plain tokens. It is not an end-to-end serving speedup.

The earlier 185-token, 0.50×, and 3.3-draft break-even claims were incorrect. The earlier 20× claim compared an outlier with a later typical duration, not matched distributions.

## Cost attribution

The five-prompt receipt reports verification totals from 126.6 to 201.5 ms. These totals include device execution, readback, host reductions, input preparation, and watchdog overhead. They do not isolate a device bottleneck.

Rollback ranges from 34.6 to 759.2 ms in that receipt. The source performs per-layer concat and copy operations. The receipt does not prove that allocator behavior caused the outlier.

The old draft stage includes trace capture, although round time excludes it. The continuation adds separate capture and warm-draft fields. It also reports the verifier's existing device and readback durations, plus the residual time.

The exact 0.77 source documents `concat(output_tensor=...)` but rejects that argument in `concat.cpp`. A preallocated concat output is therefore not an available fix on this pin.

## Validation and limits

The timing continuation passed the six policy tests and nine adapter tests. The latter include timing partition and capture-exclusion checks. Its package build passed the existing device-free checks. The continuation screens use unique evidence directories.

Earlier package, NixOS off/on/conflict, and pinned Cairn checks passed. The basic simulator check resolves successfully but does not establish a fresh numerical recurrence run. Full GDN simulation remains blocked by the documented unsupported instruction behavior.

Two earlier review workers timed out without reports. No independent review resulted. Host fixtures do not prove arbitrary target-cache rollback correctness.

The repository artifact gate remains blocked by inherited manifest and evidence-root errors. Private MTP receipts remain unregistered. The change stays active, without spec sync or archive.

## Convolution-window comparison (2026-09-08 continuation)

The candidate replaces per-token convolution-state shifts with ordered row references. It copies the final window once and keeps the same arithmetic order. Prefix snapshots retain each row window. Shape and dtype checks reject incompatible rows before a commit.

The baseline and candidate each ran all five prompts with the same checkpoint, runtime, precision, and three-draft profile. Both compared 320 tokens against plain decode. All tokens matched within and across runs. Both recorded the same acceptance pattern across 18 rounds: 25 accepted drafts and 43 emitted tokens. Both receipts report complete cleanup.

| Metric | Instrumented baseline | Row-window candidate |
|---|---:|---:|
| Mean verifier device time | 113.664 ms | 110.303 ms |
| Device-time range | 113.615–113.720 ms | 110.261–110.341 ms |
| Total capture-excluded round time | 2.970 s | 2.884 s |
| Plain-equivalent / round time | 0.825 | 0.849 |
| Counting-prompt round-only ratio | 1.063 | 1.115 |

This pair shows a 2.96% decrease in verifier device time and a 2.89% decrease in total round time. It is one ordered comparison, not a repeated ABBA benchmark or end-to-end serving acceptance. The aggregate remains slower than plain decode. The earlier 759 ms rollback outlier did not recur in either run, so these results do not establish its cause.

Private evidence directories in the primary repository:

- `cost-partition-hw7/`: baseline package `98qhq24b6d5jdqbq1nvyznv5ixdm9pgv`.
- `convolution-window-hw8/`: candidate package `ibbyiqg5h1vi2nfxcpsqsl8y5fy1l5cf`.

Both reside under `.pi/logs/qwen38-mtp-serving/`. Each contains a separate log, receipt, ownership precheck, and scratch-path record. No shared receipt was overwritten.

Validation: five new convolution-window tests, six policy tests, nine serving-adapter tests, package checks, scoped Ruff, Nix formatting, and pinned Cairn validation passed. Negative cases cover wrong widths, missing rows, dtype changes, and partial-commit prevention. These host tests do not model device timing or prove arbitrary-context parity.

The bounded continuation used two physical screens. It keeps the measured row-window reduction and leaves further verifier work open. About 110 ms of device time remains per verification pass. Readback was a smaller term in the instrumented baseline, approximately 6–13 ms.

## Draft width and prefix traces (2026-09-08 continuation)

The two-draft diagnostic profiles change only width and profile ID. The production defaults still select three drafts, with speculation off.

The prefix-trace candidate records the existing GDN restoration operations once per prefix. Each replay reads the current prefix contents through fixed bindings. The adapter rejects changed bindings and invalid indices before device effects. It releases prefix traces before their borrowed buffers and restores original GDN state after capture failure.

All three new physical screens compared 320 tokens against plain controls. All tokens matched. Cross-run token comparisons also passed. The runtime, checkpoint, precision, and admission thresholds stayed unchanged.

| Screen | Drafts | Prefix traces | Rounds | Accepted drafts | Emitted tokens | Mean verifier device time | Mean rejection rollback | Round-only ratio |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| hw8, earlier control | 3 | No | 18 | 25 | 43 | 110.303 ms | 21.568 ms | 0.849 |
| hw9, width screen | 2 | No | 18 | 23 | 41 | 94.493 ms | 81.683 ms | 0.770 |
| hw10, prefix candidate | 2 | Yes | 20 | 24 | 44 | 94.465 ms | 8.836 ms | 0.992 |
| hw11, width control | 3 | Yes | 18 | 25 | 43 | 110.336 ms | 10.030 ms | 0.797 |

The hw9 rollback mean includes a 489 ms outlier. In hw10, rollback ranged from 8.69 to 9.09 ms. These runs executed different round counts, so their aggregate change is not a matched-round speedup.

The three-draft pair retained the same acceptance sequence across all 18 rounds. Mean rejection rollback decreased from 21.568 to 10.030 ms. However, total round time increased from 2.884 to 3.263 s.

Verifier device time stayed near 110.3 ms. Host-side stage times increased. In hw11, verifier readback ranged from 8.31 to 41.85 ms. The cause of that variation remains unknown.

The candidate therefore reduces the measured rollback stage, but does not establish a consistent aggregate improvement. The best round-only ratio, 0.992, still excludes capture and remains below break-even. No end-to-end gain is claimed. Repeated matched controls remain necessary.

Private evidence under `.pi/logs/qwen38-mtp-serving/` in the primary repository:

- `width-k2-hw9/`: package `ibbyiqg5h1vi2nfxcpsqsl8y5fy1l5cf`.
- `prefix-trace-k2-hw10/` and `prefix-trace-k3-hw11/`: package `hs3i1w8xk02gh6w05qmnhcr3q4m0qkwk`.
- `width-prefix-comparison.json`: aggregate metrics and comparison limits.
- `width-sweep-scope.md`: the bounded experiment and its extension.

Each receipt reports complete cleanup. The build passed package checks and 33 scoped host tests: six policy, ten serving-round, five convolution-window, nine prefix-trace, and three profile tests. Ruff, Nix formatting, diff checks, and pinned Cairn validation passed. Host negative cases include capture failure, stale bindings, incomplete capture, invalid widths, and invalid replay indices.

The simulator limitation, missing independent review, and artifact-registration blocker remain. Task 4 is open again because the target path still relies on masking instead of full recovery of rejected KV rows. The delta specification requires rejected-KV restoration. Task 7 remains open for owner review. Neither these measurements nor the host tests authorize spec sync, archive, or production admission.

## Rejected target-KV recovery (2026-09-09 UTC)

The existing serving fixture checked committed rows but permitted stale rejected rows. A new assertion before head continuation exposed that gap at every rejection position for widths two and three. The baseline produced five errors. The clean-tail integration cases pass only with the diagnostic recovery seam enabled.

The prototype reuses the request-entry snapshot instead of copying the entire model before each round. A pure planner retains the seed and accepted input rows. It maps rejected positions through the page table. The adapter restores those rows with `paged_update_cache` and preserves cache allocations.

This route depends on a monotonic committed boundary and an unchanged request-entry tail. The planner rejects backward or skipped boundaries, invalid counts, malformed pages, aliases, and capacity overflow. The adapter checks all cache bindings and bf16 formats before writes. An operation error prevents another attempt through that owner.

### Simulated rejection

Package `048w967wvj88dx8nf894sg5ab7g7vnj1` passed its package and host checks. The model-free operator test then ran on TTsim 1.10.6, with a `1x1` mesh, slow dispatch, and no physical device nodes in its sandbox.

The first comparison failed after zero accepted drafts at position 127. A diagnostic replay located one mismatch in the first compared cache:

| Field | Observed value |
|---|---|
| Cache index | `[0, 0, 1, 0]` |
| Expected bf16 value | `-0.0` |
| Observed bf16 value | `+0.0` |
| Expected signed bits | `-32768` |
| Observed signed bits | `0` |

This is a bitwise failure, not a token divergence or a tolerance failure. The test did not finish the other cache comparisons or its device-negative cases. The stage that changes the zero sign remains unknown. This result does not prove that silicon has the same behavior.

The bitwise gate remains unchanged. `diagnostic_kv_tail_restore` defaults to false, and no server or CLI route enables it. The normal speculative route still has the rejected-KV recovery gap. Task 4 remains open. No physical operator test or hardware serving screen followed this failure.

The host oracle now compares storage bits and includes a mutation that erases negative-zero signs. That mutation must fail. It models the observed loss, not its cause. The final host suites cover the disabled default and reject an invalid diagnostic switch before device effects.

Private evidence under `.pi/logs/qwen38-mtp-serving/`:

- `kv-tail-red.log`: the baseline integration errors.
- `kv-tail-operator-sim/operator.log`: the first simulated failure.
- `kv-tail-operator-sim-detail/`: the diagnostic replay, source copy, and source hash.
- `kv-tail-scope.md`: the bounds and stop decision.
- `build-kv-tail.log` and `build-kv-tail-isolated.log`: the prototype and isolated package builds.

The isolated package `i9c3s50v369fyrrzvp73a0awzyirrdsy` passed its build, 47 scoped host tests, Ruff, Nix formatting, and pinned Cairn validation. Its device gate is not accepted. The two simulated failures came from the earlier prototype package, not a new run of the isolated package.

The follow-up below checks each row-conversion stage and the cache update. Independent review and artifact registration remain incomplete. No speedup or production acceptance follows from this prototype.

### Signed-zero observation boundary

A separate stage audit used the unchanged `i9c3s50v369fyrrzvp73a0awzyirrdsy` package on the simulator. It injected one negative zero into a bf16 tensor of shape `[4,2,128,256]`.

The first upload/readback already reported zero negative-zero elements. The snapshot, aligned slice, row slice, permutation, padding, sharding, and cache update also reported zero. The audit completed and reported cleanup. It was not a passing restoration test: its input control did not preserve the injected bit.

Matched host-only controls used the same tensor shape and values. Both row-major and tile-layout roundtrips preserved the negative-zero bits, `-32768`. Neither control opened a device.

This narrows the first observed loss to the mesh upload/readback path, before recovery. It does not distinguish upload, mesh mapping, simulator storage, or device readback. The earlier assertion therefore does not establish that rollback changed the sign. The original expected bits came from the pre-upload host tensor, not a measured request-entry cache.

The bitwise criterion remains unchanged. A valid restoration test must establish its device-entry baseline and retain a separate upload control. The next discriminating check is an explicit host-tiled transfer compared with direct device construction, with readback controls. No package code, production defaults, or admission gates changed during this audit.

Private evidence: `kv-zero-stage-sim/`, `kv-zero-host-control.log`, and `kv-zero-host-matched-control.log` under `.pi/logs/qwen38-mtp-serving/`. The scope is `kv-zero-stage-scope.md`. No hardware test ran.

## Upstream reference

The README cites `sjettTT/tt-qwen-3.8-flash-next`, a different model and topology. Its near-tie observations are external claims, not local validation. The earlier review did not record an immutable source revision. The reference does not prove the local trace order or rollback contract.
