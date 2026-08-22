## Context

The first target-only probe continued prompt state through the adopted masked-prefill path. It used `gated_delta_attn_seq` because TT-Metal 0.74 does not expose `ttnn.transformer.chunk_gated_delta_rule`. Physical evidence rejected that path.

A useful target verifier must process several candidate positions layer by layer. Large projections, MLPs, and the LM head must run once over the width. GDN recurrence and convolution remain ordered within each layer. Full-attention candidates share one logical sequence and can occupy pseudo-batch rows, but writes to consecutive positions in one physical paged block must be serialized.

The latest pinned reference has two relevant mechanisms. Its Qwen stack has a fused C++ GDN chunk operation that is absent from the compiled 0.74 bindings. Its Gemma4 verifier demonstrates pseudo-batch target verification and explicitly serializes same-page `paged_update_cache` writes. The fused GDN operation is not a model-only backport because it changes compiled TTNN bindings, host operations, program factories, and device kernels.

## Decision

### Decision: implement a diagnostic hybrid packed-decode adapter

**Choice:** Keep the pure profile and evidence evaluator in `mtp_probe_core`. Add one TTNN adapter that runs the existing Qwen layer components in a verifier-specific composition:

1. Embed all width positions together.
2. Run each distributed norm in decode mode over the width.
3. For GDN layers, run input projection once, then apply convolution shifts and high-precision recurrence in token order. Run output projection once over the collected rows.
4. For full-attention layers, run QKV projection once over pseudo-batch rows. Serialize K/V writes by position before one batched paged SDPA decode.
5. Run each MLP once over the width.
6. Capture the complete width-specific hybrid forward and measure one replay, including bounded input refresh.
7. Run final norm and LM head once, returning one full target-logit row per position.
8. Compare exact target tokens, full logits, final GDN recurrent and convolution state, and the written paged-KV positions against sequential traced decode.

**Rationale:** This route reuses package-matched operations and avoids a TT-Metal runtime rebuild. It amortizes the model's large weights while preserving the stateful operations that cannot be parallelized. A width-specific trace removes Python dispatch from the comparison. The route also makes the same-page alias hazard explicit instead of relying on unsupported batched cache semantics.

### Decision: do not backport the fused C++ GDN operation in this change

**Choice:** Record the fused operation as a runtime-uplift candidate, not as a local model patch.

**Rationale:** The operation requires a compiled nanobind export plus host, primitive, program-factory, and device-kernel sources. Treating those files as a Python-only compatibility patch would not produce the missing symbol and would misstate the evidence boundary.

### Decision: keep production and admission unchanged

**Choice:** The native verifier remains an explicit diagnostic command. It cannot select MTP or change eager production.

**Rationale:** Target-only validity is necessary but insufficient. MTP still requires drafter timing, local acceptance, rejection rollback, end-to-end correctness, and admission evidence.

## Risks / Trade-offs

- Width-dependent matmul programs can produce small numerical changes. Exact target-token parity and bounded full-logit metrics remain mandatory.
- Full-logit host transfer is diagnostic and can have scheduling outliers. Replay stability is authoritative because a speculative loop keeps logits on device.
- Serialized K/V writes add fixed work, but prevent concurrent read-modify-write corruption in one physical block.
- The adapter uses package-owned private model seams. Device-free source checks must reject missing or changed seams before physical execution.
- A final-state comparison does not prove rejection rollback. The receipt must retain that blocker.
- If the packed path fails parity or cost, the result rejects this pinned-runtime route only. It does not reject MTP or a future immutable runtime uplift.

## Safety and evidence boundaries

The probe must release the parked decode trace before verifier allocations. It must restore eager production after physical work. It must bind the exact package, runtime, source, checkpoint, tokenizer, mesh, physical devices, CPU affinity, widths, and sample count. A simulated or device-free result cannot establish performance or physical correctness.
