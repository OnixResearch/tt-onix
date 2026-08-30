# Design

## Round shape

One speculative round replaces `draft_token_count` plain decode steps:

1. Draft: the MTP head chains `draft_token_count` drafts from the target
   hidden state (captured draft trace, 9-14 ms total).
2. Verify: the target backbone runs the draft chain teacher-forced at
   width `draft_token_count` (native verifier trace; exact, PCC 1.0).
3. Commit: the longest accepted draft prefix plus the verification bonus
   token. For greedy requests the committed stream is provably identical
   to sequential greedy decode - the verification argmaxes ARE the
   sequential outputs for the accepted prefix, and the bonus token is the
   sequential output at the rejection (or final) position.
4. Recover: on rejection, restore rejected KV/GDN rows via the snapshot
   rollback path (~12 ms measured) before the next round.

## Trace residency

Runtime 0.74 falsified capture-while-resident. The serving adapter
captures three traces once per request lifecycle in hygiene order - no
other trace resident, verify first (largest), then draft, then the
existing width-1 decode trace - and afterwards only replays them. No
per-round recapture is permitted; a forced recapture disengages
speculation for the request and records the reason.

## Exactness contract

The core owns the commit rule. `commit_verified_round` accepts the draft
tokens and the verification argmaxes (both length k) and returns the
committed tokens: `drafts[:accepted] + (argmaxes[accepted],)` on rejection
at `accepted < k`, or `drafts + (argmaxes[k-1],)` on full acceptance. The
emitted stream equals sequential greedy output for any acceptance pattern;
the device-free probe asserts this property against a reference stream.

## Economics guard

Speculation only pays when `emitted_tokens / round_seconds` beats the
measured plain step rate. The adapter records plain-rate calibration from
startup warmup and per-round emitted counts; the pure core owns the
sticky disengage decision (a configurable number of consecutive losing
windows, so one bad window does not flip mode). Re-engagement within one
request is forbidden; a disengaged request finishes on plain steps.

## Gate

`QWEN36_MTP_SERVING` defaults off. Off means the serving loop is
byte-identical to today: no MTP weights load, no traces capture, no
rounds run. The module option mirrors the sampled-fusion gate shape.

## Round-loop state model (derived from the proven components)

Semantics: the verify pass feeds `[last_committed, d0..d(k-1)]` at
positions `[p..p+k]` and produces argmax rows `a0..ak`, where `a_i`
predicts position `p+i+1`. Acceptance: first `i` with `d_i != a_i`;
committed = `d0..d(i-1)` plus `a_i` (bonus on full acceptance: `a_k`).
The pure core's continuation contract is exactly these `k+1` rows.

The draft chain continues ACROSS rounds without reprefill:

- On full acceptance the head's own cache and feedback chain are already
  advanced through `d(k-1)`; the next round's first draft consumes the
  bonus token `a_k` with the recorded feedback. No target reprefill and
  no head rebuild happen.
- On partial acceptance the rollback context restores the head's rejected
  rows and selects the recorded prefix feedback at the accepted count
  (measured: rollback exact 60/60, reprefill fallback median 8.8 ms).
- The target paged KV rows written by the verify pass are exact for the
  accepted prefix and must be restored for rejected rows; the next round
  (or plain step) resumes from position `p + accepted + 1` with seed
  token `a_accepted`.

Round cost (measured pieces, k=3): draft replay ~10-14 ms + verify replay
~60-90 ms + restore ~12 ms on rejection, against 3.4 emitted tokens -
the 1.382x single-round measurement, sustained across rounds.

## Persistent verifier requirement

The diagnostic verifier captures and releases a trace per call (~1 s).
Serving requires a persistent capture-once/replay mode: the op sequence
and every baked buffer (inputs, prefix-snapshot state buffers, logits)
are allocated once at capture; each round refreshes the mutable inputs,
replays, reads back the k+1 argmax rows, and copies the prefix snapshot
at the committed width over the live GDN state on rejection. Per-round
fresh allocations are limited to the safety full-state clones. Capture
order per request lifecycle: release everything, capture verify (largest),
capture draft, then the width-1 decode trace; only replays afterwards.
