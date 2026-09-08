# Qwen38 speculative serving progress

## Status

Speculative serving is not ready for production. The default remains off. The physical exactness screen passed; the economics screen failed, so speculation must stay disabled at this operating point.

The dedicated worktree is `../tenstorrent.nix-mtp-serving-20260908`, on `work/qwen38-mtp-serving-20260908`. It starts from fetched `origin/main` at `51e07b8`, then includes the earlier local fixes through `e205476`. The new changes remain uncommitted because the command runner is now blocked.

## Implemented candidate

- Move the thread watchdog out of the pure policy core. A device-stage timeout requests process termination before cleanup can touch the abandoned worker.
- Replace the ineffective loss counter with an explicit, sticky economics state. Compare round time against supplied plain-step measurements.
- Reject invalid generation limits and expired requests before device effects. Correct the undefined request timer.
- Give one request ownership of its speculative traces and snapshots. Remove the feedback-pool limit from the round loop.
- Restore complete target KV and recurrent state on rejection. Replay only committed target inputs and compare their outputs with the verification result.
- Correct draft-head continuation positions. Consume the final accepted draft before the bonus token on full acceptance.
- Serialize greedy MTP requests. Calibrate against plain traced decode and close the drafter with the engine.
- Add a diagnostic `serving-screen` mode that compares two prompts for 64 tokens through the real serving route.

The screen mode and its CLI changes came after the successful package check. They remain unbuilt and unverified.

## Observed validation

The baseline package and repository Cairn check passed. The baseline stage gates exposed malformed requirement headings and task markers. The corrected proposal, design, and tasks gates report `PASS`.

Two device-free suites passed six tests each. They execute the candidate source with controlled device adapters. The cases cover all draft acceptance positions, exact target-state recovery, repeated zero acceptance, and sticky fallback. Negative cases cover invalid inputs, expired requests, capture errors, head errors, and changed tensor bindings.

The first candidate package check passed; build rounds r2 and r3 then passed the package, package check, NixOS service gate, and repository Cairn quality. The NixOS gate verifies that the default ExecStart is unchanged, that the speculative option adds `--decode-mode eager` plus the packaged profiles, and that combining speculation with sampled fusion fails the module assertion. A fresh single-chip simulator smoke passed before hardware (`ttsim-smoke.log`). Focused Ruff checks pass on every touched file; the six remaining `check.py` findings are pre-existing at `51e07b8` and were left untouched. Two read-only workers reached their eight-minute limits without reports, so the source review has no independent worker review.

## Physical screen result (2026-09-08, P150x2)

Receipt: private log `screen/serving-screen.json` (the repository artifact gate stays blocked, so this evidence is not registered in `docs/evidence`). Package `wj3rbz1rkw362wgx6nv2lvn56arr446g`, runtime `dr9bj8qwflylq39zl5ajcx42cdrb1bv9`, checkpoint `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, mesh 1x2, two prompts, 64 tokens each, greedy, bwrap-sandboxed, 900 s bound.

- Exact token parity: PASS on both prompts against the plain traced control. No divergence occurred, so no near-tie classification was needed.
- Acceptance: 0 of 6 drafted tokens accepted across both requests. Every round rejected at the first draft and emitted only the continuation token.
- Round cost: 0.445 s and 0.430 s (prompt 1), 0.538 s and 0.643 s (prompt 2), against measured plain steps of 0.0565 s and 0.0578 s. The sticky economics guard disengaged after two losing rounds per request (`fallback_reason: round-economics`).
- Trace capture: 13.1 s on the first request lifecycle, 1.9 s on the second. Minimum verification top-two margin: 0.0625 and 0.125.
- Cleanup: `cleanup_complete: true`, cluster destructor completed, both device nodes free afterward. The service was inactive before and after; restoration was not required.

Interpretation: the exactness contract held on hardware. Even with zero acceptance, rollback plus the continuation commit produced the identical stream, and the economics guard contained the losses exactly as designed. Speculation gave no speedup at this operating point, so it must stay disabled.

Open anomaly, resolved and fixed: the diagnostic pilot re-run showed `accepted_prefix_length: 0` on all five prompts because the 0.77 runtime misreads head weight caches written by 0.74. The fix scopes the head cache directory by runtime identity (`mtp_drafter_probe-runtime-<identity>`), so each runtime regenerates once. After regeneration the pilot reports 12/15 accepted drafts with the exact 0.74 golden tokens (prompt 1: `89454, 4384, 58377`), and the only remaining blocker is `production-admission-missing`.

A second hardware screen with the healthy head confirmed exact parity again and cut round costs from up to 3.5 s to a steady 0.17 s by replacing per-round whole-state snapshots with the verifier's per-row prefix rollback. Economics remain near break-even on acceptance-heavy prompts and negative on rejection-heavy ones; the sticky guard still protects every request. Committed as `a611242`.

Remaining levers, in order: fold the bonus-token head advance into the verify replay (removes a replay per round), reduce verification readback latency, and widen the screen prompt set before any enablement reconsideration.

## Five-prompt operating point (final measurement, 2026-09-08)

Package `r5rc59ayv38sqmgr1dhl3qpm5jkzcncn` added per-stage round timing and widened the screen to all five profile prompts, 64 tokens each. Receipt: `screen/serving-screen.json` (private, artifact gate still blocked).

- Exact token parity on all five prompts (185 tokens) against plain traced decode.
- Acceptance: 22 drafts, 4 rejected across 15 rounds; the counting prompt reached 10/15.
- Stage table: verification replay dominates at 124-130 ms per round; draft 10-49 ms; rollback 18-25 ms; head-commit 6-14 ms; plain step 57-71 ms.
- Aggregate speedup 0.50x; the best prompt (counting, 10/15 accepted) reached 0.91x.
- Break-even needs about 3.3 accepted drafts per round at the measured verify cost, against the 2.4 average. Two prompts also showed one stalled round each (up to 1.4 s), which the stage table now isolates.

The remaining lever is device-level: make the width-k backbone replay cheaper (the design assumed 60-90 ms; the runtime delivers 108-130 ms), and chase the stalled rounds. Serving integration itself is complete, exact, and safe; only the operating point is short of break-even.

## Upstream reference consulted

The `sjettTT/tt-qwen-3.8-flash-next` Blackhole MTP serving reference (a different 51.2B checkpoint) was reviewed mid-task and cited in the README. Its pinned numerics show width-k verify rows breaking near-ties differently from 1-row decode on 4 of 12 prompts. That motivated the margin capture in the verifier result and the screen's divergence classification. It also independently validates the local architecture choices: exact acceptance, default-off gating, and measured economics with sticky disengage.

## Remaining work

- Owner review (task 7) records: do not enable speculation on this runtime. Revisit after the 0.77 draft-target agreement is repaired; the serving integration itself passed its exactness screen.
- Keep the Cairn change active. Do not enable production. Do not sync or archive.

The inherited artifact-policy and manifest failures also remain outside this change's acceptance evidence. No push or main-branch integration ran.

Private logs and the search registry remain in the primary worktree at `.pi/logs/qwen38-mtp-serving/`.
