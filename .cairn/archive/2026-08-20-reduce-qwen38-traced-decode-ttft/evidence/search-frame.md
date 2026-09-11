# TTFT portfolio-search frame

Date: 2026-08-20

## Exact goal

Find and validate the cause of the traced candidate's TTFT ratio above `1.05`.
Reduce the ratio to at most `1.05`, or return the strongest checked rejection.

## Completion evidence

- Unchanged total TTFT timing boundaries.
- Phase-attributed eager and candidate responses.
- Exact token and text parity.
- Clean reset and stable trace binding.
- Matched `eager, candidate, candidate, eager` physical roots.
- A deterministic passing or rejected receipt.

## False completions

- Decode-rate gain without TTFT admission.
- Moving trace, reset, or readback work outside the measured interval.
- Static source agreement without physical discrimination.
- TTsim or upstream performance.
- One prompt, one execution order, or unmatched package identities.
- A median that hides missing, negative, or non-finite phase data.

## Audit risks

- Page-table prefix and decode-table mapping can diverge.
- Device addresses can change after trace capture.
- Program compilation or allocator history can dominate short prompts.
- Timing calls can perturb the measured interval.
- Service order can create a false regression or false pass.

## Budgets

- Source and authority reads: at most `16` targeted reads.
- Static implementation rounds: at most `2`.
- Physical diagnostic rounds: at most `1`.
- Physical admission rounds: at most `1`.
- Workload per root: `4` prompts, `1` warm-up, `3` measured runs, `64` generated tokens.
- One bounded production-service outage.
- No subagents were authorized. The labeled lenses are serial and correlated.

Allowed terminal states are `validated`, `blocked`, `exhausted`, and `user-decision-required`.

## Approach registry

| Family | Mechanism | Claim | Evidence | State | Next check |
|---|---|---|---|---|---|
| timing-boundary | Trace capture directly increases TTFT | Capture causes the observed ratio | Source places `ensure_trace` after TTFT stops | falsified | Keep capture outside causal attribution |
| reset-boundary | Request reset directly increases TTFT | Reset causes the observed ratio | Source runs reset before TTFT starts | falsified for direct inclusion | Record reset separately for indirect device history |
| prefill-capacity | Maximum-width page table and cache placement slow short prefill | A request prefix is sufficient to meet TTFT admission | Physical candidate used `2` prefill blocks and `32` decode blocks but measured TTFT ratio `1.1069467490614955` | falsified as sufficient | Retain the rejected result; do not infer the remaining operation |
| program-shape safety | A new prefill width compiles after trace capture | Width changes can clobber a parked trace | Package-matched model comments forbid post-capture compile | mitigated | Release the parked trace before a new width and capture again after TTFT |
| first-token-readback | Synchronize, full-logit readback, or argmax differs | Readback causes the ratio | Candidate and eager medians differed by less than `0.002` seconds | falsified as primary cause | Retain phase evidence |
| run-order | Service and device history caused a false aggregate ratio | Balanced roots remove the apparent regression | Balanced physical order still measured TTFT ratio `1.1069467490614955` | falsified as sufficient | Retain the balanced receipt |

## Current strongest route

The request-bounded prefill page table was the smallest source-backed diagnostic mechanism.
It preserved the stable decode table but did not meet the TTFT bound.

The surviving observation is narrower: candidate device prefill remained slower than eager device prefill.
The aggregate phase result does not identify the responsible prefill operation.
A later change must use operation-attributed physical profiling before it selects another optimization.
