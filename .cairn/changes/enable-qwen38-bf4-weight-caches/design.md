## Context

The accepted requirement `tenstorrent.qwen38_p150x2.low_precision_non_admission`
already states the rule this change must satisfy:

> The repository MUST keep BFP8 as the production precision for the Qwen3.8
> weight families covered by the existing opt-in BF4 gates until one named
> candidate has complete numerical and matched physical evidence. An environment
> flag, supported cache format, PCC result, static byte count, simulator result,
> or incomplete receipt MUST NOT imply production admission. Future admission
> MUST use a new validated Cairn change.

This is that new validated change. It adds the admission contract and the
evidence path. It does not edit the existing requirement.

The three gates already exist in the package and already write distinct
`.bfp4` cache names. What does not exist is per-family evidence, an accuracy
floor, and a physical attribution of the byte reduction.

The published MTP acceptance rate for this checkpoint is 92.2 percent at BF16
and 84.8 percent at FP8. A one-prompt token-parity check would not detect that
class of change. The evidence must therefore include an accuracy floor over a
fixed corpus.

## Decisions

### Decision: Admit per family, never as a bundle

**Choice:** Each of `QWEN36_LM_HEAD_BF4`, `QWEN36_MLP_DOWN_BF4`, and
`QWEN36_GDN_BF4` gets its own numerical screen, its own physical timing result,
and its own verdict. A combined run is reported but does not admit a bundle.

**Rationale:** The three families differ in risk. The GDN family feeds the
recurrent state that this repository has already seen drift across 48 layers.
The LM head family sits directly on the argmax. Bundling would hide which family
caused a failure and which family produced a gain.

### Decision: Keep distinct cache names and prove the default is never overwritten

**Choice:** Every BF4 family keeps its `.bfp4` cache name. The screen must prove
the default BFP8 bins are byte-identical after a BF4 run.

**Rationale:** Overwriting a shared cache would silently change the admitted
production path and invalidate every prior receipt. This is a correctness gate,
not a tidiness gate.

### Decision: Require an accuracy floor, not only token parity

**Choice:** Each family must hold an accuracy floor over a fixed corpus in
addition to exact greedy parity on the parity corpus. The profile owns the
floor. Token parity alone is explicitly insufficient.

**Rationale:** The published precision-versus-acceptance gap shows that a
precision change can keep short-prompt parity while moving quality elsewhere.
Parity proves the graph is unchanged; the floor proves the model still answers
correctly.

### Decision: Attribute the byte reduction with a step-time ratio

**Choice:** Each family must report a decode step-time ratio against the BFP8
baseline on matched roots in `control,candidate,candidate,control` order. The
receipt must state the predicted byte saving beside the measured result.

**Rationale:** Weight bytes set the ceiling, so a BF4 family should show an
attributable step-time change. A family that removes bytes without moving the
step time is worth recording, because it falsifies the bandwidth model and
redirects the next optimization.

### Decision: Keep the gate default off and the fallback exact

**Choice:** A missing, unknown, or malformed gate selects BFP8. A missing
`.bfp4` cache builds it once outside the measured window and never substitutes
it silently inside a measured run.

**Rationale:** Matches the existing closed-mode discipline and keeps every
admitted receipt valid.

## Risks / Trade-offs

- A BF4 family may meet parity and fail the accuracy floor. That is a rejection
  and is a valid deliverable.
- The step-time gain may be smaller than the byte saving predicts. The receipt
  must then report the gap rather than the theoretical ceiling.
- Cache generation costs host time on first use. That cost must stay outside
  every measured window.
- The accuracy floors are new policy constants. They must be set before the
  screen runs, not after a result is known.
