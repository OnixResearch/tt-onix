# Search frame

## Success contract

The goal is one concrete, exact-P150x2 mechanism that improves single-stream Qwen3.8 generation economics without changing model meaning. Completion requires source-bound causality, matched physical timing, exact greedy outputs, exact hybrid-state rollback, stable duplicate roots, and aggregate emitted-token speedup above one.

False completions include static byte estimates, isolated operation wins, upstream TP4 numbers, greedy text without logit or state checks, BF4 speed without long-context accuracy, median-only economics, unpinned runtime changes, and aggregate batching throughput.

The search uses six mechanism families, two source-retrieval rounds, the pinned package and latest reference trees, one smoke outage, and one final paired outage. Without subagent consent, all lenses are serial and correlated.

## Approach registry

| Family | Mechanism | Evidence | Gap | State | Next discriminating check |
| --- | --- | --- | --- | --- | --- |
| precision | BF4 MLP-down, LM-head, or GDN weights reduce weight traffic | Upstream PR 53543 reports `+3.8%` to `+7.5%`, but 256k PCC `0.847` or `0.737` | accuracy failure is stronger than the local speed goal | falsified | none without a new accuracy-safe quantizer |
| decode-fusion | Fuse row projection with reduce-scatter | Pinned source contains unused decode helpers; source states M=1 collapses to about eight cores and loses | isolated TP2 sweep still absent | blocked | exact TP2 operation sweep only after a stable attributed target |
| runtime-uplift | Reuse a newer fused recurrent operation | Latest reference retains the same recurrent decode graph; fused C++ operation is chunk-prefill only | new decode operation is required | falsified as reuse | design a runtime-owned decode primitive |
| GDN-kernel | Fuse high-precision recurrence, state update, and adjacent elementwise work | GDN is `46.58%` of component replay. Upstream PR 44296 reduced a B=32 composite recurrence from `2916` to `495` microseconds through L1 intermediates and in-place state, within `1.19x` of its `414`-microsecond fused proof | the pinned path already uses L1 intermediates, while a full fused decode operation still needs TTNN host, factory, kernel, binding, and parity work | active-later | split exact P150x2 GDN recurrence from projections and reduce-scatter |
| MTP-selector | Avoid full-vocabulary all-gather and host transfer for each draft | All twenty roots produced identical tokens. Prompt 2 rollback was exact. Observed bounded/full selector ratio was `0.3350445565492493` | duplicate roots were unstable, so selector speedup and end-to-end gain both failed validation | audit | trace the selector with at least five stable repeats per root |
| traced adaptive-MTP | Trace the fixed-shape draft step, restore rejected MTP KV from a snapshot, then choose zero, one, or three drafts | One `0.4443297780235298`-second prefill outlier and one zero-acceptance prompt dominated aggregate speedup `0.8197972276186697`; oracle-gated stable-root opportunity was `1.1657458805489858x` | trace stability, cheap MTP rollback, confidence calibration, and sustained evidence are missing | active-later | trace one MTP draft step and snapshot its KV before collecting multi-round confidence |

## Surviving candidate

The bounded selector is output-equivalent in this control, but its timing is not validated. The complete speculative candidate is rejected. The next bounded candidate combines a persistent MTP-step trace with device-resident MTP KV snapshot restore. Only after compute and rollback stabilize should a confidence gate choose target-verification width.

The longer-term GDN route is now more specific. Tenstorrent issue 42828 names the fast reference as Flash Linear Attention's fused recurrent kernel. At pinned FLA revision `bc3b101dcb713ddc5bd8924b66754eb68b5ccf89`, that kernel keeps one `[K,V]` state block in registers, then fuses Q/K normalization, decay, beta, state update, and query read. The package does not have an equivalent decode primitive. This is a runtime project, not the smallest next model patch.

## Adversarial checks

- A local argmax on one shard is invalid. The selector must compare one maximum from every shard.
- Equal maxima need the existing deterministic global tie policy.
- Selector speed alone is insufficient. End-to-end emitted-token economics must include every rejection cost.
- A full-readback control and bounded candidate must produce identical drafts before timing can be compared.
- One profitable prompt or median cannot override aggregate slowdown.
- The result cannot admit MTP because traced decode still lacks production TTFT admission.
