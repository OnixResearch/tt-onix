## Context

The admitted P150x2 service decodes one token per traced step. The measured
step is 56.33 ms, which is 17.75 tokens/second. The weight-bandwidth floor for
a 27.49 GiB model split across two 512 GB/s cards is 28.8 ms, which is 34.7
tokens/second. The service therefore runs at 51 percent of the floor.

The per-component budget from the admitted profile is:

| Component | Time | Share |
| --- | ---: | ---: |
| GDN mixer, 48 layers | 26.34 ms | 47 percent |
| MLP, 64 layers | 23.86 ms | 42 percent |
| Full attention, 16 layers | 5.65 ms | 10 percent |
| LM head and argmax | 2.07 ms | 4 percent |
| Host input and readback | 1.17 ms | 2 percent |

Measured upstream on the same 1x2 mesh, the fused T=1 decode op moves the
traced 128-token case from 17.75 to 19.16 tokens/second. Two independent
upstream efforts produce two differently named kernels. This repository must
select on its own measured evidence, not on the upstream claim.

## Decisions

### Decision: Screen both named upstream kernels, then select one

**Choice:** Evaluate `ttnn.transformer.decode_gated_delta_rule` and
`ttnn.transformer.fused_recurrent_gated_delta_rule` in one screen. Select the
kernel that passes exact parity with the larger step-time reduction. Record the
losing kernel and its measured result either way.

**Rationale:** The two kernels come from independent authors and carry different
names, shapes, and maturity. The first has a same-mesh end-to-end receipt at
four context lengths. The second has a per-layer receipt and covers T=K+1
verify. Selection on a measured head-to-head removes the naming ambiguity and
avoids an unmeasured preference.

### Decision: Keep the gate default-off and the composite path byte-identical

**Choice:** The fused op is reachable only through an explicit gate. A missing,
unknown, or malformed gate value selects the composite path. When the gate is
off, the produced decode graph must be byte-identical to the current one.

**Rationale:** This matches the existing closed-mode policy the repository used
for the rejected staged fusion candidate. It keeps every prior receipt valid and
makes the candidate droppable without a rollback.

### Decision: Gate on parity first, then on the step-time ratio

**Choice:** The screen requires exact greedy token parity across the corpus and
recurrent-state parity against the composite reference before it reads any
timing. It then requires a step-time ratio at or below the profile-owned bound.
It rejects on unstable roots exactly as the existing admission discipline does.

**Rationale:** The repository's earlier GDN fusion rejection came from drift
across 48 layers that flipped late argmax margins. Parity over a multi-prompt
corpus is the observable form of that hazard. Timing without parity would admit
a faster wrong model.

### Decision: Re-run MTP economics under the fused verify path

**Choice:** After the decode screen passes, re-run the existing MTP drafter and
rollback probes with the fused verify kernel. Report the emitted-token economics
under the same predeclared bounds that rejected MTP before.

**Rationale:** The MTP rejection was not a correctness failure. The recorded
verdicts were `sample-instability` and `emitted-token-economics`, and the
verification path cost 2.1 ms per candidate at width four. Issue 50475 reports
the fused verify at 2.3 ms against about 18 ms before. The old rejection
therefore does not describe the new kernel.

### Decision: Treat both upstream PRs as unmerged, unreviewed sources

**Choice:** Record the PR numbers, the branch, the commit, and the review state
in the package authority. Require local physical evidence before admission.

**Rationale:** PR 53482 and PR 53587 carry no substantive human review. Their
silicon receipts match this hardware, which is corroboration and not
endorsement. This repository already depends on similarly sourced patches, so
the risk is bounded and must be explicit.

## Risks / Trade-offs

- The two upstream kernels may not be compatible with the pinned runtime. The
  screen must then fall back to the composite path and record the blocker.
- The fused verify may change acceptance outcomes for MTP. The MTP re-run must
  reject on outcome change rather than explain it.
- Vendoring an unreviewed op expands the patch surface. The op must stay behind
  the default-off gate and outside the admitted execution identity until the
  owner reviews the screen.
- Kernel-level projections may overstate the end-to-end gain. The 0.541 to 0.345
  ms per-layer reduction implies a larger step change than the measured 7.9
  percent end-to-end result. The screen must report both numbers.
