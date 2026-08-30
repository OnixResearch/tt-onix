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
