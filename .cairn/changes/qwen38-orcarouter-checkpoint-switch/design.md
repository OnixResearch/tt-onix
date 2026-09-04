# OrcaRouter checkpoint-switch design

## Decision

The target is the dense Qwen3.8-27B-Uncensored family on the existing two Blackhole P150 cards. It is not Qwen3.8 Flash Next.

The recommended first implementation uses the publisher's BF16 safetensors source through the current TT-NN loader. It does not load the linked GGUF files. Source selection must record owner approval for this representation choice before retrieval and staging.

| Role | Repository | Observed immutable revision |
| --- | --- | --- |
| Requested artifact | `orcarouter/Qwen3.8-27B-Uncensored-GGUF` | `a855f377abf5cbda99a278414466743f427e97c8` |
| Recommended native source | `orcarouter/Qwen3.8-27B-Uncensored` | `404ea47aaa5d8a8b00049c9e9750089aca011ab2` |
| Existing official release authority | `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |

The publisher describes the BF16 release as the source for its quantizations. That statement does not prove tensor-by-tensor equivalence to the requested GGUF release. The authenticated manifest and conversion checks remain implementation gates.

## Current boundaries

`Qwen36ModelArgs.load_state_dict` uses the HF weight-mapping path. It supports the existing native checkpoint flow, not a GGUF file path. Its cache path distinguishes mesh and dtype, but a directory name alone does not prove source identity.

`nixos/module.nix` points both `HF_MODEL` and `TT_CACHE_PATH` at `qwen38.modelPath`. Replacing files inside that root can mix old caches with new weights. Staging must use a new immutable source root and a separate model-bound cache root.

`server.py` loads an HF tokenizer and applies its chat template. The upstream card advertises vision, tool calling, reasoning control, and a larger context. The current service does not acquire those capabilities merely because weights change.

The MTP adapter reads `model.safetensors.index.json`. The publisher states that its MTP head also received the weight edit. The old official BF16 head is therefore not an acceptable candidate companion.

## Component ownership

- The existing pure staging and admission cores own candidate validity, identity comparison, gate decisions, and promotion state.
- Typed Nickel records own human-authored candidate identities, precision policy, test limits, and allowed routes.
- The shell owns authenticated downloads, byte hashing, bounded conversion, filesystem publication, device ownership, and service changes.
- HF, safetensors, TT-NN, and systemd adapters translate those external operations. Their types do not define admission policy.
- The owning Onix host configuration consumes a published, pinned package after approval. It does not depend on this research worktree.

No new generic framework or cross-project component is necessary. Reuse the existing staging, recovery, service, and evidence components where their contracts fit.

## Stages

### Source and format admission

The first stage records authorized HF access without storing credentials in source or receipts. It inventories the exact candidate shard list, file hashes, tensor names, shapes, dtypes, and required metadata.

The comparison includes full-attention indices, GDN dimensions, convolution state, vocabulary, special tokens, RoPE parameters, normalization, and MTP tensors. Missing data produces a blocker, not a shape-compatibility claim.

The BF16 source remains immutable. The existing TT-NN precision policy creates device caches from that source. A BF16 source file does not imply BF16 execution throughout the model. Any optional FP8 staging derivative needs its own reproducible conversion manifest and measured quality gate.

### Cache and request isolation

The candidate uses distinct source, conversion, tensor-cache, prefix-cache, and receipt roots. Content identity includes the selected source, tokenizer, template, runtime closure, mesh layout, and precision recipe. Cross-model cache reuse fails before device acquisition.

The candidate reports a distinct model alias during evaluation. Production aliases change only through reviewed client compatibility. The host keeps its current supported context and generation limits.

The workload declares thinking settings explicitly. An upstream default such as `xhigh` cannot silently change the measured output budget. Token accounting includes reasoning tokens. Unsupported request fields and modalities remain unsupported.

### MTP separation

The initial candidate uses plain decode with MTP off. It preserves and identifies the candidate MTP tensors but does not load or capture that path.

Later MTP work depends on `qwen38-mtp-speculative-serving` and requires the matching OrcaRouter BF16 head. Acceptance rate, timing, and state-carry evidence must be new. No official-model speedup receipt transfers to this cohort.

### Validation and promotion

The first comparison holds the runtime, mesh, precision recipe, sampling policy, and workload fixed. The TT-Metal 0.77 experiment remains separate.

Numerical checks compare the converted candidate with the same OrcaRouter source on the reference path. Behavioral checks compare the official and OrcaRouter models without requiring identical outputs. Quality tasks cover representative coding, reasoning, multilingual text, structured output, and benign over-refusal cases.

A typed profile fixes acceptable quality, latency, throughput, memory, and stability bounds before measurement. Disk and host-memory preflight covers both retained and candidate sources, conversion peaks, caches, and rollback retention.

Device-free positive and negative fixtures precede a short simulator smoke. The smoke uses the same host and device binaries on a supported single-chip representative path. It covers weight loading, GDN, full attention, and recurrent-state carry. An unsupported simulator path retains its exact blocker before one authorized bounded hardware attempt.

Physical admission uses the exact `1x2` mesh with device IDs `0` and `1`. The screen includes repeated requests, state reset, supported greedy and sampled modes, and any currently admitted batching profile. It records cold staging separately from warm service timings.

The operator approves a service window and rollback procedure before activation. A failed trial cannot restart the old worker until ownership is safe. A timed-out live device thread requires explicit recovery, not concurrent cleanup.

## Dependencies and optimization order

The change depends on `qwen38-optimization-evidence-contract`. It does not block official-model research and does not require the 0.77 runtime candidate.

After adoption, the new cohort becomes the reference for optimizations intended for this deployment. Each precision, cache, batching, MTP, or collective result must name that cohort. Old results remain historical evidence only.

## Alternatives

| Family | Evidence | Decision / next discriminating check |
| --- | --- | --- |
| Native BF16 through TT-NN | Public HF metadata names the existing architecture and eighteen safetensors shards | Preferred candidate. Authenticated tensor and loader checks remain open. |
| GGUF through current TT-NN service | The loader expects HF configuration and native weights | Rejected as a direct path. Renaming a file or changing `modelPath` is insufficient. |
| GGUF through `llama-cpp-metalium` | The pinned fork contains `qwen35` and MTP frontend code | Separate backend project. Verify operation coverage, actual placement, state carry, and TP2 performance before selection. |
| Quantized GGUF converted back to BF16 | Conversion cannot recover precision already lost | Rejected while the publisher's native source is available. |

The pinned Metalium dispatch lacks `SSM_CONV` and `SSM_SCAN` cases. Frontend architecture recognition is therefore not proof of complete device execution. CPU fallback cannot count as the requested P150x2 migration.

## Risks and unresolved facts

- HF marks both repositories as gated. Anonymous access to the pinned BF16 config failed. This plan does not accept account terms.
- Public metadata and model-card counts are not a complete authenticated tensor manifest.
- The publisher's capability and refusal results are upstream claims, not local acceptance evidence.
- GGUF file size is not TT-NN device memory usage. Repacking can change layout and footprint.
- Initial scope remains text-only. No vision projector or new tool authority is necessary.

## Completion boundary

All implementation tasks remain open. The plan is complete when its artifacts and dependency links pass structural gates. Deployment completion requires the separate implementation, evidence, and owner-approval steps.
