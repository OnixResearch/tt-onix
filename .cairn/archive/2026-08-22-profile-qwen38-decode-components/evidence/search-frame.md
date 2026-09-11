# Search frame

## Goal

Identify the largest stable traced-decode component on the exact Qwen3.8 P150x2 package, or return an exact bounded attribution blocker.

## Completion evidence

A valid result must:

- bind the package, runtime, checkpoint, authority, physical device IDs, mesh, prompt, and CPU affinity;
- use real package-selected component code, weights, layouts, collectives, and state;
- run one unmeasured replay and readback before each retained group;
- retain at least five replay samples for every component and full-path group;
- restore prompt-bound recurrent state before each stateful replay;
- release one trace before later device allocations;
- separate synchronized input update and bounded winner readback from device replay;
- make the weighted additive model close against matched full replay within typed bounds;
- select a component only when its weighted share and dominance pass typed bounds; and
- restore eager production health after physical cleanup.

## False completion

The following outputs are not completion:

- static tensor bytes without direct timing;
- eager operation timing instead of trace replay;
- trace capture or compilation time counted as replay cost;
- a single representative layer or one retained sample;
- zero-state GDN timing when the full path uses prompt-bound state;
- full-logit readback used as the bounded winner path;
- component shares from an additive model that does not close; or
- simulator, upstream, TP4, or unmatched-package evidence used as P150x2 attribution.

## Audit risks

- Independent traces can change cache residency and command scheduling.
- GDN replay mutates recurrent and convolution state.
- Full attention mutates paged KV at the decode position.
- A pending host copy can be charged to the next synchronization.
- A parked trace can make later allocations unsafe.
- Representative layers can differ because of memory placement.
- Cleanup failure can leave production unavailable.

## Budget

Use the adopted package source, pinned checkpoint, and existing physical receipts. Permit one pilot outage to find measurement-path defects and one final outage after a bounded correction. Use three representative layers per repeated layer class, five retained samples per group, and one full-model load per attempt. Use no runtime pin update and no precision change. No subagents are authorized, so the search lenses are serial and correlated.

## Approach registry

| Family | Mechanism | Claim | State | Exact blocker or next check |
| --- | --- | --- | --- | --- |
| Static bytes | Sum checkpoint tensor bytes by component | Weight traffic predicts the bottleneck | Falsified | Direct replay placed weighted GDN cost above MLP cost. |
| Full-model subtraction | Compare models with selected layers removed | Slope estimates per-layer cost | Blocked | Model construction always changes loaded weights and trace shape. |
| Program profiler | Sum device program durations by source signature | Kernel records attribute decode cost | Blocked | Existing program profiling targeted prefill and adds diagnostic drain behavior. |
| Direct component trace | Capture coherent real layer stages independently | Direct replay measures stage cost | Rejected | Embedding spread exceeded `1.25`, and GDN-to-MLP dominance was only `1.156087701933436`. |
| Full trace closure | Compare scaled component sum with matched full replay | Independent stages explain the full trace | Validated | Closure ratio `1.1025414225426509` passed the typed bounds. |
| Host boundary | Measure synchronized update and bounded winner readback | Host costs stay separate from device cost | Rejected | Update spread was `1.611457421591108`; readback spread was `38.500578404176125`. |

## Allowed outcomes

- **Validated:** stable component groups and additive closure select one bounded bottleneck.
- **Blocked:** direct groups are valid, but additive closure or dominance fails.
- **Exhausted:** the one-outage budget ends without complete stable evidence.
- **User decision required:** safe physical access or eager restoration fails.

## Terminal result

Blocked. The additive model closed and the full trace was stable, but strict group stability and dominance gates rejected attribution. No optimization is selected.
