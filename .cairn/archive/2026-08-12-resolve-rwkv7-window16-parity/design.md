# Design: Resolve the RWKV-7 window-16 parity failure

## Goal and completion evidence

The goal is to identify and correct, or exactly bound, the first physical parity failure at generated token index `10`.

Validated completion requires all of these results:

- matching CPU and device comparison points through the first failing step;
- a classified root cause supported by retained evidence;
- positive and negative regression tests for that cause;
- exact generated-token and FP32 recurrent-state parity for a new immutable candidate;
- a complete repeated window-16 simulator and physical admission evaluation before any production change.

A bounded blocker is also a valid result. It must name the first divergent layer, value, and unsupported or incorrect operation. It must retain production windows `[2, 4, 8]`.

False completion includes accepting normal and persistent agreement without CPU parity, changing expected tokens to device output, weakening parity, using simulator performance claims, or admitting the rejected candidate.

## Bounded portfolio search

Subagent consent is absent. These serial lenses are correlated.

| Family | Mechanism | Discriminating check | Initial state |
|---|---|---|---|
| CPU authority | Compare FLA and official RWKV-LM values at each step and layer. | The two CPU paths disagree before the device boundary. | CPU tokens already agree; deeper values remain active. |
| BF16 drift | Compare device top logits and layer boundaries against FP32 CPU values. | The device remains close, but the smallest CPU margin changes rank after BF16 execution. | Active; index `10` has the smallest CPU margin, `0.0925440788269043`. |
| WKV implementation | Run production fused WKV and generic TT-NN WKV on the same bounded request. | Only one WKV policy diverges from CPU or their states diverge first. | Active. |
| Non-WKV device operation | Compare hidden, final normalization, and language-head values after matching WKV state. | WKV state matches while a later operation first diverges. | Active. |
| Orchestration | Repeat a single bounded normal request with fixed card and scheduling. | Token or state output changes between identical runs. | Low probability; four slots and both dispatch modes agreed in the rejected run. |

## Budgets

- Source authorities: the pinned checkpoint, package-matched runtime, FLA `0.3.0`, official RWKV-LM revision `952102498e9ed367ea0a59ee64106916d474d30f`, and package-matched TT-Metal.
- Physical devices: `[0, 1]` only.
- Initial physical diagnostics: one card, one request path, and only enough steps to include generated index `10`.
- Retry: rerun only after an orchestration or diagnostic defect is fixed. Do not retry a valid negative result.
- Temporary files: unique roots under `/var/tmp`.

## Diagnostic structure

The pure comparison core consumes CPU and device records. It locates the first differing token, layer boundary, and value family. It reports maximum absolute error, rank changes for expected and actual tokens, and the exact prior matching boundary.

The runtime shell loads the checkpoint, opens one selected device, performs bounded model steps, and writes artifacts. It does not change production dispatch or admission.

## Admission rule

A source fix can start a new window-16 evaluation only after the bounded diagnostic request has exact CPU generated-token parity. The new evaluation must use a new immutable profile identity and repeat all gates. Otherwise, production remains `[2, 4, 8]`.
