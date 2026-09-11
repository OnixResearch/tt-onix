# OrcaRouter Qwen3.8-27B migration candidate

Date: 2026-09-04. Scope: checkpoint-switch planning for the existing P150x2 service.

## Result

The OrcaRouter model family is a reasonable migration candidate. The linked GGUF files are not direct inputs for the current TT-NN service.

The recommended first route uses the publisher's BF16 safetensors source. It preserves the existing runtime and TT-NN optimization path. This is a representation choice, not a claim of byte-equivalence to the requested GGUF quantization.

The new [Cairn package](../../.cairn/changes/qwen38-orcarouter-checkpoint-switch/proposal.md) contains seven requirements, fourteen scenarios, and ten open implementation tasks. It depends on the existing optimization-evidence contract.

## Source identities

| Role | Repository | Revision |
| --- | --- | --- |
| Requested release | [OrcaRouter GGUF](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-GGUF) | `a855f377abf5cbda99a278414466743f427e97c8` |
| Recommended native source | [OrcaRouter BF16](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored) | `404ea47aaa5d8a8b00049c9e9750089aca011ab2` |
| Existing release authority | `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |

Public API snapshots are retained in [sources/](qwen38-orcarouter-switch-2026-09-04/sources/). Both candidate repositories report `gated=auto`.

Anonymous access to the pinned BF16 `config.json` returned:

> Access to model orcarouter/Qwen3.8-27B-Uncensored is restricted. You must have access to it and be authenticated to access it. Please log in.

No account terms were accepted. No credentials were read. No model weights were downloaded. Complete file and tensor verification remains blocked on authorized access.

## What the available evidence establishes

The GGUF API identifies the architecture as `qwen35`. The model card describes standard and IQ quantizations, a separate vision projector, and an embedded MTP head.

The BF16 API identifies `Qwen3_5ForConditionalGeneration` and lists eighteen safetensors shards. The publisher describes a dense hybrid model with sixty-four layers, hidden size 5120, forty-eight GDN layers, and sixteen full-attention layers. These descriptions match the broad local model shape. They do not replace file-level compatibility checks.

The BF16 card identifies that release as the source for the publisher's quantizations. The GGUF card instead links the FP8 release in its introductory conversion description. Exact derivation lineage remains a verification task, not an established equality.

The publisher states that its MTP head received the same weight edit as the backbone. The plan therefore rejects an OrcaRouter backbone paired with the old official-model head. Initial adoption keeps MTP off.

The cards advertise capability retention and lower refusal rates. Those are upstream measurements, not local results. The migration needs same-source numerical checks and separate behavioral evaluation. It does not require identical outputs from the official and modified models.

## Local implementation evidence

- [`pkgs/qwen36/authority.ncl`](../../pkgs/qwen36/authority.ncl) binds the current official release authority and existing structural expectations.
- The packaged `Qwen36ModelArgs.load_state_dict` supports an FP8 dequantization path and an HF `Qwen3_5ForCausalLM` path. It has no direct GGUF input path.
- The same module qualifies cache directories by dtype and mesh. Its comments warn that cached tensors can bypass a new mesh mapping.
- [`nixos/module.nix`](../../nixos/module.nix) points `HF_MODEL` and `TT_CACHE_PATH` at `qwen38.modelPath`. The new source and cache roots must remain separate from the old deployment.
- [`server.py`](../../pkgs/qwen36/service/qwen36_service/server.py) loads an HF tokenizer and applies its template. A checkpoint change does not add new API capabilities.
- [`mtp_drafter_adapter.py`](../../pkgs/qwen36/service/qwen36_service/mtp_drafter_adapter.py) reads a safetensors index. The previous [MTP repair plan](../../.cairn/changes/qwen38-mtp-speculative-serving/proposal.md) remains necessary before speculative serving.

The loader review used `models/demos/blackhole/qwen36/tt/model_config.py` from the packaged source at:

```text
/nix/store/m28mhmkj3xsc5azbz10mad3wdhp1r8gl-source
```

The live process and deployed weight identity were not queried. The future control must record their actual identities rather than infer them from historical receipts.

## Route comparison

| Family | State | Exact next gate |
| --- | --- | --- |
| BF16 source through current TT-NN | Preferred candidate, compatibility unproved | Authorized manifest, exact tensor/config comparison, isolated conversion and cache fixtures |
| GGUF through current TT-NN | Rejected as a direct substitution | A new loader is necessary. Changing the path or file extension is insufficient. |
| GGUF through `llama-cpp-metalium` | Separate backend path, not admitted | Complete operation coverage, physical placement, recurrent-state correctness, and TP2 measurements |

The pinned Metalium fork at `04602ca95c1a6adb6c1533e024ad9c79ee6c428d` already contains `qwen35` and MTP frontend code. The Qwen graph calls `ggml_ssm_conv`, but the Metalium dispatch has no `GGML_OP_SSM_CONV` case. Architecture recognition therefore does not establish complete Tenstorrent execution.

The upstream card names CPU, CUDA, Metal, and ROCm. Apple Metal is not Tenstorrent Metalium. CPU fallback cannot count as a successful migration onto the two P150 cards.

## Planned sequence

1. Record the approved format and authenticated source manifests.
2. Verify compatibility without device access.
3. Build separate candidate caches with matching source identities.
4. Verify text requests, templates, token accounting, and disabled MTP.
5. Fix the evaluation profile before any measurements.
6. Run device-free fixtures and a representative simulator smoke.
7. Run one authorized, bounded P150x2 comparison.
8. Promote only after owner approval and a verified rollback path.

The first comparison retains the current TT-Metal runtime. The [0.77 candidate](../../.cairn/changes/qwen38-runtime-077-prefill-candidate/proposal.md) remains an independent experiment.

After adoption, optimizations intended for this deployment need an OrcaRouter-bound baseline. Old MTP, batching, precision, and cache results remain historical evidence. They do not automatically transfer.

## Validation and limits

The artifact checks and raw receipts are in [validation.md](qwen38-orcarouter-switch-2026-09-04/validation.md). The plan does not alter the running service, package pins, hardware state, or current model files.

The review used three serial lenses: checkpoint identity, native loader compatibility, and the alternative backend. The passes came from the same assistant and are correlated.

The retrieval budget was twelve public requests. Nine requests were used, including two retained API snapshots. The terminal result is a complete plan with explicit implementation blockers, not a completed switch.
