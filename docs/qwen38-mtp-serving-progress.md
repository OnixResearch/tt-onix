# Qwen38 speculative serving progress

## Status

Speculation remains off by default. The five-prompt screen passed bounded token parity but lost against plain decode. This result does not establish general exactness, production safety, or a speedup.

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

The timing continuation passed the six policy tests and nine adapter tests. The latter include timing partition and capture-exclusion checks. Its package build passed the existing device-free checks. The next physical screen uses a unique evidence directory.

Earlier package, NixOS off/on/conflict, and pinned Cairn checks passed. The basic simulator check resolves successfully but does not establish a fresh numerical recurrence run. Full GDN simulation remains blocked by the documented unsupported instruction behavior.

Two earlier review workers timed out without reports. No independent review resulted. Host fixtures do not prove arbitrary target-cache rollback correctness.

The repository artifact gate remains blocked by inherited manifest and evidence-root errors. Private MTP receipts remain unregistered. The change stays active, without spec sync or archive.

## Upstream reference

The README cites `sjettTT/tt-qwen-3.8-flash-next`, a different model and topology. Its near-tie observations are external claims, not local validation. The earlier review did not record an immutable source revision. The reference does not prove the local trace order or rollback contract.
