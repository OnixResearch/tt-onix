# Native short-width verifier search frame

## Goal

Find and validate the smallest pinned-runtime implementation that verifies widths 1, 2, and 4 in one hybrid-model forward on physical P150x2.

## Completion evidence

A validated route must:

- batch the weight-bearing projection, MLP, and LM-head work across candidate positions;
- preserve ordered high-precision GDN recurrence and convolution history;
- write and read full-attention paged KV at consecutive positions without alias races;
- return target logits for every position;
- match sequential traced target tokens and bounded final hybrid state for five samples per width;
- beat, or clearly fail to beat, the matched sequential target cost with a stable receipt;
- remain diagnostic-only and leave production eager.

## False completion cases

- Calling several full-model trace replays one verifier.
- Starting GDN or attention state from zero after the prompt.
- Measuring a prefill-only path that does not continue the prompt state.
- Returning only the final logit.
- Ignoring full-attention KV aliasing or GDN convolution state.
- Selecting MTP without a drafter, acceptance, rollback, and end-to-end evidence.
- Updating the runtime pin before proving that the pinned runtime has no bounded route.

## Audit risks

- Pseudo-batch candidates can race when they write distinct positions in one shared paged block.
- The existing chunk path quantizes or reorders operations differently from high-precision decode.
- Width-dependent TTNN program selection can change logits near ties.
- Temporary verifier buffers can invalidate parked trace addresses.
- A correct final state can hide incorrect intermediate logits.
- Static source compatibility does not prove P150x2 memory or program support.

## Search budget

- Source authority: pinned Qwen source `d102e7f4512060e09e580e40cf26a870bfa88c55`, package-matched TT-Metal `0.74.0`, and latest pinned reference `0fa87dd38f087a10383dd320670419428db45979` for comparison only.
- Mechanism families: at most five.
- Retrieval rounds: two targeted source rounds, then one adversarial audit.
- Declared physical budget: one bounded pilot after device-free checks, plus one discriminating fix.
- Used physical budget: two pre-measurement tensor-view ownership fixes, one complete untraced root, two traced confirmation roots, and one final policy-bound traced root.
- Budget amendment: exact untraced correctness introduced a materially new traced mechanism. Two traced roots then isolated diagnostic host-readback jitter from stable replay timing. The final root bound the revised replay-stability policy to one immutable package.
- Allowed outcomes: validated, blocked, or exhausted. A plausible design without deterministic and physical evidence is not success.

## Approach registry

| Family | Mechanism | Claim | State | Exact blocker or next check |
| --- | --- | --- | --- | --- |
| Existing masked chunk | Continue prompt state through `gated_delta_attn_seq` in a 128-token bucket | Reuses one model pass without runtime changes | Falsified | Physical receipt failed tokens, GDN state, stability, and cost. |
| Fused chunk backport | Backport `ttnn.transformer.chunk_gated_delta_rule` from latest source | Replaces the slow sequence adapter with one device operation | Blocked | The missing compiled symbol requires nanobind, host, primitive, program-factory, and device-kernel changes. It is a runtime uplift, not a model patch. |
| Untraced hybrid packed verify | Batch weight-bearing work, serialize ordered GDN recurrence and same-page KV writes | Preserves decode semantics without a runtime update | Falsified for performance | It was exact, but width-four end-to-end execution took `0.7049244520312641` seconds. |
| Traced hybrid packed verify | Capture the exact hybrid packed forward and measure one replay | Removes Python dispatch while amortizing target weights | Validated as target-only base | Final width-four end-to-end amortization was `1.9469233481156285`, with exact logits and state. |
| Multi-step trace | Capture several normal target steps in one trace | Removes host launch overhead | Falsified | It scans target weights once per token and does not amortize target work. |
| Runtime uplift | Move to a runtime that exposes fused GDN chunk support | Uses upstream native operation | Blocked | No immutable uplift candidate is admitted, and the pinned-runtime traced route already establishes target feasibility. |

## Adversarial audit

The surviving route was checked against the false-completion cases:

- It runs one width-specific full-model trace, not several full-model decode replays.
- Prompt GDN and paged-KV state remain live.
- Every position returns a full target-logit row.
- Same-page K/V writes execute in position order.
- Widths `1`, `2`, and `4` matched sequential target tokens, logits, recurrent state, convolution state, and written K/V exactly.
- Replay timing, not diagnostic full-logit host transfer, owns stability. A future acceptance path must remain device-resident.
- The result selects only a target-verification base. It does not select MTP, a drafter, rollback, or production admission.

## Terminal result

Validated target-only base. The final immutable receipt is `pkgs/qwen36/evidence/2026-08-22/qwen38-27b-native-mtp-verifier-receipt.json`, BLAKE3 `2a4f171b36dcf31f4637d48706dfa3e037af100a5ca21bae91f47070b023ac19`.
