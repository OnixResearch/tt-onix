# Qwen3.8-27B: local service vs vLLM

## Scope

There is no local vLLM head-to-head. The packaged Qwen service is measured on
TT-Metal 0.78; vLLM figures below are upstream-reported on a different mesh and
runtime. This note records the comparison that is supported, and the comparison
that is not.

## Local measured result

`qwen36-p150x2-trace-candidate` on two physical P150 cards, devices 0 and 1:

- TT-Metal 0.78.0, Qwen package `/nix/store/mpnqg0ans7kzpssxambmr4g7mkgx48sp-qwen38-0.1.0`;
- batch 1, five prompt tokens, eight greedy output tokens;
- candidate traced decode, no MTP, no fused GDN;
- 18.40 and 18.39 decode tokens/s;
- TTFT 0.619 s and 0.568 s;
- identical token IDs and text on both requests.

This is a short-prompt smoke, not a throughput benchmark. The B=8 scheduler
receipt separately measured 39.5-40.0 delivered tokens/s on the same mesh under
its mixed workload; it is not comparable to the single-request vLLM number
either.

## Upstream-reported vLLM result

The Qwen3.8 PR author reported, on P150x4 with the four precision flags and
vLLM `0.10.1.1` text-only:

- traced-4k: 28.69 tokens/s;
- traced-150k: 25.54 tokens/s;
- vLLM `max_model_len=4096`: 22.63 end-to-end tokens/s.

These are upstream claims, not repository evidence. The mesh has twice the
cards, the vLLM version is different, and the workload and sampling settings are
not matched to the local smoke.

A naive raw comparison is 18.4 tokens/s locally versus 22.63 upstream-reported,
with local using half the mesh. That is not a valid speedup or parity claim:
different engines, prompt lengths, batch settings, runtime versions, and
measurement scopes.

## Feature and deployment comparison

| Surface | Local service | Upstream vLLM lane |
| --- | --- | --- |
| Status | Built, smoke-tested, production owner masked | Plugin packaged; vLLM core not built |
| Runtime | TT-Metal 0.78, P150x2 | vLLM 0.26 empty target required by plugin; reported numbers use vLLM 0.10.1.1 |
| Batching | B4/B8/B16 policies, continuous B8 admitted | Paged attention and continuous batching in vLLM |
| Prefix reuse | Process-local hybrid prefix cache admitted | vLLM prefix caching upstream |
| Structured reads | `/v1/systemone` contract implemented locally | Not part of the reported vLLM lane |
| MTP | Experimental, not admitted | Not part of the reported vLLM lane |
| Local head-to-head | None | None |

The official `vllm-tt-plugin` is packaged, but the matching vLLM core is absent.
The preflight records a base vLLM 0.16.0 and concrete dependency differences:
NumPy 2.x, Starlette below the required 1.0.1 floor, and an older
compressed-tensors. The plugin targets vLLM 0.26.0 built with
`VLLM_TARGET_DEVICE=empty`; the PyPI CUDA wheel is not a substitute. No vLLM package build, plugin test, or serving run is claimed.

## What a real comparison requires

1. Build vLLM 0.26.0 with the empty target in an isolated Python profile.
2. Install the pinned plugin and pass its positive and negative tests.
3. Run the same model checkpoint, prompt corpus, generation length, temperature,
   and batch settings on the same P150x2 mesh.
4. Compare TTFT, inter-token latency, decode tokens/s, and memory at matched
   admission state; report cold compile and warm results separately.
5. Keep the local Qwen service and vLLM runs separate; do not relax the
   Starlette security floor or substitute CUDA wheels.

Until those steps exist, the only defensible statement is: the local service
serves the matched regression prompt at 18.4 tokens/s on P150x2, and the
upstream-reported vLLM number is 22.63 tokens/s on P150x4. No local vLLM
comparison is available.
