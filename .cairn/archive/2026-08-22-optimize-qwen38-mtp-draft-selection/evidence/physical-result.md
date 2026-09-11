# Physical result

## Scope

- Package: `/nix/store/0hswsadjwb955srrfy6y78nm60xssvnf-qwen36-0.1.0`
- Runtime: `/nix/store/mg0ha5nsvnir1y1ysiyyzmm4hk83z3m6-tt-metal-0.74.0`
- Checkpoint and tokenizer: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Model source: `d102e7f4512060e09e580e40cf26a870bfa88c55`
- Physical PCIe devices: `0`, `1`
- Mesh: `(1, 2)`
- CPU affinity: `8-31`
- Routes: exact order `full,bounded,bounded,full` for each of five fixed prompts
- Receipt BLAKE3: `b0e0eb8df13fe8db1667cca7d82973cb069596675bdf50a5857b9f67e995d923`

## Mechanical smoke correction

The first outage stopped in target-prefill readback because `_read_token()` had been renamed to `_read_full_token()`. The exact error was:

```text
AttributeError: 'MtpDrafterAdapter' object has no attribute '_read_token'. Did you mean: '_read_full_token'?
```

The correction changed that one call and added a device-free regression check. The owner-control trap restored the eager service.

## Selector result

The bounded selector produced the same draft, target, and continuation tokens as full readback across all twenty measured roots. Prompt 2 also restored recurrent state, convolution state, paged KV, and MTP cache state exactly after rejection.

The direct selector observation was large:

- full-logit selector median: `0.03712464599811938` seconds per three drafts;
- shard-local selector median: `0.01438837700698059` seconds per three drafts;
- bounded/full ratio: `0.3350445565492493`;
- observed selector reduction: `66.49554434507507%`.

The implementation removes full-vocabulary all-gather and host readback by construction. The unstable roots prevent a validated selector-speedup claim, and the result does not prove useful speculative decoding.

## End-to-end rejection

The strict evaluator rejected the candidate:

- aggregate emitted-token speedup: `0.8197972276186697`;
- total bounded/full draft ratio: `1.1031780425039517`;
- accepted prefixes: `3, 0, 3, 3, 3`;
- mean accepted drafts: `2.4`.

Duplicate-root stability failed in most prompt-route groups. The largest unrelated outlier was bounded prompt 4 MTP prefill:

- bounded root 2 MTP prefill: `0.021768614009488374` seconds;
- bounded root 3 MTP prefill: `0.4443297780235298` seconds.

The bounded selector itself stayed near `0.011` to `0.016` seconds on those two roots. The outlier was MTP compute before selection, not bounded winner readback.

Prompt 2 accepted zero drafts. Its bounded round still paid draft, target verification, diagnostic readback, target rollback, and MTP cache rebuild before emitting one token. This rejection cost erased gains from the four full-acceptance prompts.

## Deeper optimization decision

The next useful implementation has three ordered parts:

1. Capture the fixed-shape MTP decode step and bounded selector in one persistent trace. Stable token, position, feedback, selector-index, and selector-value buffers must keep their addresses. Trace capture must snapshot and restore MTP attention KV state.
2. Replace rejected-prefix MTP cache re-prefill with a device-resident KV snapshot restore and one committed-token advance. Full prompt re-prefill cannot scale across many speculative rounds.
3. After trace and rollback stability pass, add a device-side confidence observation for the first draft. Use calibrated confidence to choose zero, one, or three drafts before target verification.

The next physical gate must run multiple speculative rounds per prompt. It must count initial MTP prefill once, preserve rollback after every rejected prefix, and compare at least 64 emitted target tokens per prompt with sequential traced decode. Confidence thresholds need separate calibration and held-out prompt sets.

A diagnostic counterfactual replaced the prompt-4 compute outlier with its stable paired root and skipped the zero-acceptance prompt with an oracle. That calculation gives `1.1657458805489858x`. It is not admission evidence because the oracle is not implementable and the outlier substitution is post hoc. It only sets the approximate opportunity bound for traced MTP plus confidence gating.

## Other researched routes

- BF4 weight compression is not accuracy-safe. Upstream PR 53543 measured up to `7.5%` decode gain, but 256k full-model logits PCC fell to `0.847` or `0.737`.
- Upstream PR 44296 reduced a B=32 composite recurrence from about `2916` to `495` microseconds through L1 intermediates and in-place state. Its fused proof was about `414` microseconds. The pinned package already uses L1 recurrence intermediates, so the headline gain cannot be copied again.
- The package has unused decode matmul-reduce-scatter helpers, but their source states that the M=1 fused path loses because its two-dimensional matmul uses too few cores.
- The latest reference has a fused chunk-prefill GDN operation. It does not have a fused single-token decode operation.
- A true fused decode kernel should follow Flash Linear Attention's fused recurrent structure at revision `bc3b101dcb713ddc5bd8924b66754eb68b5ccf89`. That is a TTNN runtime project, not a model-only patch.

## Decision

Keep shard-local MTP selection as a diagnostic building block. Do not select MTP, traced decode, reduced precision, a GDN kernel, or a production change from this receipt.

Production returned to the eager service. It is active and healthy on port `8000`.

## Evidence

- Receipt: `pkgs/qwen36/evidence/2026-08-22/qwen38-27b-mtp-draft-selection-receipt.json`
- Receipt BLAKE3: `b0e0eb8df13fe8db1667cca7d82973cb069596675bdf50a5857b9f67e995d923`

## Non-claims

- The selector result does not establish sustained multi-round speedup.
- The oracle-gated counterfactual is not a deployable result.
- The result does not apply to TP4, batching, long context, another checkpoint, or another runtime.
- The result does not admit traced decode or change production from eager mode.
