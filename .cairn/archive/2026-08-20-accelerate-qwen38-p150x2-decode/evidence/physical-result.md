# Physical traced-decode result

Date: 2026-08-20

## Verdict

`REJECTED_TTFT_REGRESSION`

The traced-decode candidate stays unadmitted. The production wrapper stays eager.

## Physical closure

- Physical PCIe device IDs: `0`, `1`
- Mesh: exact `(1, 2)`
- Model: `Qwen3.8-27B`
- Checkpoint and tokenizer revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Model source revision: `d102e7f4512060e09e580e40cf26a870bfa88c55`
- Runtime: TT-Metal `0.74.0-rc6`
- Generated tokens per request: `64`
- Prompt count: `4`
- Warm-up runs per prompt: `1`
- Measured runs per prompt: `3`
- Sampling: greedy
- Simulated: `false`

The bounded run used a preliminary `candidate, eager` pair and a final `eager, candidate` pair. The final receipt uses the instrumented matched pair.

## Candidate repair found by physical execution

The first candidate request rejected a single-device GDN state name inside the TP2 path. `TPGatedDeltaNet` owns `rec_state` and `conv_states`, not `recurrent_state`.

The adapter now snapshots and restores every per-device TP state through `ConcatMeshToTensor` and `ShardTensorToMesh`. It also binds their physical buffer addresses into the trace identity.

The repaired request generated:

```text
 Paris.
The capital of Germany is
```

## Final measured result

| Metric | Eager | Candidate | Ratio |
|---|---:|---:|---:|
| Median decode tokens/second | `0.946354139616546` | `15.336118315891143` | `16.20547496321535` |
| Median TTFT seconds | `1.8999489599991648` | `2.311131047001254` | `1.2164174383938555` |
| P99 token latency seconds | `4.450717410996731` | `0.08870580499933567` | `0.01993067562100514` |
| Median cleanup seconds | `0.36499940500289085` | `0.011382804997992935` | not applicable |

The profile allows a maximum TTFT ratio of `1.05`. The observed ratio was `1.2164174383938555`.

All measured requests had exact generated token-ID parity, exact generated text parity, matched identities, and clean reset results.

## Service restoration

The bounded outage ended after the owner restored `qwen38-p150x2.service`.
The service reported `active` and returned `{"model": "Qwen3.8-27B", "status": "ok"}` from `/healthz`.

## Evidence

- Receipt: `pkgs/qwen36/evidence/2026-08-20/qwen38-27b-traced-decode-candidate-receipt.json`
- Receipt BLAKE3: `33c3654bf417f5bf671cb127977d12465abbab2d6d524117d8d32f8b4cd11a37`
- Persistent raw responses: `/home/brittonr/git/tt-onix/.pi/qwen36-decode-20260820/`
- Receipt generator: `scripts/qwen36-decode-receipt.rs`

## Non-claims

- The variable eager rate does not replace the admitted baseline receipt.
- The candidate decode-rate gain does not override the failed TTFT gate.
- No concurrent-request throughput claim is made.
- No prefill speedup claim is made.
- No mesh other than physical P150x2 is covered.
