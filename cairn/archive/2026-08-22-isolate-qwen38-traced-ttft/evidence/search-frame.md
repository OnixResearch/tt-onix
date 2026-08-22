# Search frame

## Goal

Identify whether request-cache capacity or a parked decode trace causes the traced candidate's TTFT regression on the exact Qwen3.8 P150x2 package. If neither mechanism is stable and material, return the exact bounded blocker.

## Completion evidence

A valid result must:

- bind the package, runtime, checkpoint, authority, devices, mesh, prompt, page table, and CPU affinity;
- use one loaded model and stable request-sized and trace-capacity cache sets;
- use the exact six-root overlapping balanced schedule;
- retain at least five complete samples after warm-up in every root;
- reset GDN state and synchronize before each TTFT clock starts;
- preserve the same first token across all samples;
- retain the existing device-prefill, first-token-readback, and TTFT boundaries;
- keep every within-root and duplicate-root spread no more than `1.25`;
- select a mechanism only when its stable TTFT ratio is more than `1.05`; and
- release traces, close devices, and restore eager production health.

## False completion

The following outputs are not completion:

- a comparison across separately loaded models without the same-process control;
- trace capture time counted as TTFT;
- cache allocation or GDN reset time counted as TTFT;
- one sample, one root, or an unbalanced route order;
- different page-table widths between compared routes;
- a selected mechanism from unstable roots or token mismatch;
- a same-process result used as direct production admission; or
- simulator, upstream, TP4, or unmatched-package evidence used as P150x2 attribution.

## Audit risks

- A cache switch can leave model-layer references on the wrong cache set.
- A parked trace binds full-capacity cache and GDN buffer addresses.
- Pending reset copies can enter the next timing interval without synchronization.
- Stale paged KV values can affect token parity when the prompt does not overwrite a read position.
- Trace release can deallocate output tensors that later code still references.
- Program compilation or cache warming can affect the first retained sample.
- Two cache sets can change DRAM placement and reduce external validity.
- Cleanup failure can leave production unavailable.

## Budget

Use the adopted package source, pinned checkpoint, and current physical devices. Permit one pilot outage and one final outage after one bounded correction. Use six roots, one warm-up per root, and five retained samples per root. Use one model load per attempt. Do not update the runtime pin or precision. No subagents are authorized, so all search lenses are serial and correlated.

## Approach registry

| Family | Mechanism | Claim | State | Exact blocker or next check |
| --- | --- | --- | --- | --- |
| Cache capacity | Compare stable request-sized and trace-capacity paged KV sets | Extra cache capacity causes prefill TTFT regression | Blocked | The observed TTFT ratio was `1.0062416611969702`, but readback instability invalidated the control. |
| Parked trace | Compare full-capacity prefill before, during, and after trace parking | A parked decode trace causes prefill TTFT regression | Blocked | The observed TTFT ratio was `0.9951621585112468`, but readback instability invalidated the control. |
| Process drift | Compare the same routes without separate model loads | Earlier root variance created the apparent regression | Audit | The combined ratio was `1.0013736235407196`; rejected evidence cannot prove drift. |
| Device programs | Attribute the gap to changed device kernels | Device program work causes the wall gap | Falsified | Prior matched profiling explained only `0.22521929455787217%` of the wall gap. |
| GDN lifecycle | Compare fresh GDN allocation with in-place reset | Stable recurrent buffers change prefill cost | Blocked | The control held GDN buffer identity fixed and did not isolate allocation lifecycle. |

## Allowed outcomes

- **Validated:** one stable mechanism exceeds the material ratio and becomes the only selected correction target.
- **Blocked:** the control is valid, but no measured mechanism explains a material same-process regression.
- **Exhausted:** the two-outage budget ends without complete stable evidence.
- **User decision required:** safe physical access or eager restoration fails.

## Adversarial audit

The final run preserved token parity, route order, widths, cleanup, TTFT boundaries, and eager-service restoration. All TTFT spreads passed. Three first-token readback roots exceeded the all-phase `1.25` bound, so the evaluator rejected the complete control. TT-Metal also warned about allocation while a trace was parked. The control did not execute that trace.

## Terminal result

Exhausted. The pilot found one command defect, and the corrected final run consumed the second outage. No stable material mechanism was selected. The exact blockers are readback instability in roots `3`, `4`, and `6`, plus the active-trace allocation warning for any post-prefill correctness claim.
