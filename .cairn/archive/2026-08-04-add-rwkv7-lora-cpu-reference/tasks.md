# Tasks: RWKV-7 LoRA CPU reference

## Phase 1: Model and data authority

- [x] [serial] R1 Record baseline RWKV inference parity and focused package checks before training changes. r[tenstorrent.rwkv7_lora_reference.model] r[tenstorrent.rwkv7_lora_reference.verification]
- [x] [serial] R2 Pin the model, tokenizer, HF source, paper equation, upstream training source, dimensions, and arithmetic class in one authority manifest. r[tenstorrent.rwkv7_lora_reference.model]
- [x] [parallel] R3 Add typed Nickel contracts for model, tokenizer, data, sequence, LoRA, optimizer, seed, tolerance, output, and resume configuration. r[tenstorrent.rwkv7_lora_reference.configuration]
- [x] [depends:tenstorrent.rwkv7_lora_reference.configuration] R4 Add pure JSONL admission, token-label planning, prompt masking, padding, truncation rejection, and data-manifest identities. r[tenstorrent.rwkv7_lora_reference.data]

## Phase 2: LoRA and equation reference

- [x] [depends:tenstorrent.rwkv7_lora_reference.model] R5 Add pure role-based LoRA target selection and parameter-freeze plans with an explicit default allowlist. r[tenstorrent.rwkv7_lora_reference.lora]
- [x] [depends:tenstorrent.rwkv7_lora_reference.model] R6 Add the FP32-from-BF16 RWKV-7 sequence reference and paper-matched naive WKV7 forward equation. r[tenstorrent.rwkv7_lora_reference.equations]
- [x] [depends:tenstorrent.rwkv7_lora_reference.equations] R7 Add the paper-matched WKV7 backward oracle, autograd comparison, and bounded finite-difference comparison. r[tenstorrent.rwkv7_lora_reference.gradients]
- [x] [depends:tenstorrent.rwkv7_lora_reference.lora] R8 Add the thin PyTorch shell for model construction, adapter insertion, loss, optimizer, clipping, step, and explicit seeded randomness. r[tenstorrent.rwkv7_lora_reference.training_step]

## Phase 3: Artifacts and validation

- [x] [depends:tenstorrent.rwkv7_lora_reference.training_step] R9 Add adapter-only Safetensors save, load, BLAKE3 manifests, receipts, and fail-closed resume admission. r[tenstorrent.rwkv7_lora_reference.checkpoint]
- [x] [parallel] V1 Add positive fixtures for data, labels, targets, initial equivalence, equations, gradients, one-step mutation, deterministic reruns, save, load, and resume. r[tenstorrent.rwkv7_lora_reference.verification]
- [x] [parallel] V2 Add negative fixtures for malformed data, empty labels, stale authorities, invalid targets, bad ranks, nonfinite values, shape mismatch, base mutation, corrupt artifacts, and incompatible resume. r[tenstorrent.rwkv7_lora_reference.verification]
- [x] [depends:tenstorrent.rwkv7_lora_reference.checkpoint] V3 Produce the deterministic CPU reference receipt and adapter fixture from the fixed tiny corpus. r[tenstorrent.rwkv7_lora_reference.evidence]
- [x] [serial] R10 Document data preparation, LoRA targets, CPU reference commands, artifact interpretation, resume rules, and non-claims. r[tenstorrent.rwkv7_lora_reference.documentation]
- [x] [depends:tenstorrent.rwkv7_lora_reference.verification] V4 Run formatting, focused package tests, Nix checks, existing inference regressions, Cairn validation, and all lifecycle gates. r[tenstorrent.rwkv7_lora_reference.verification]

## Verification Coverage

- `Scenario: Pinned model authority is admitted` -> R2, V1
- `Scenario: Model authority drift is rejected` -> R2, V2
- `Scenario: Valid SFT record produces deterministic labels` -> R4, V1
- `Scenario: Invalid SFT record is rejected` -> R4, V2
- `Scenario: Explicit LoRA plan freezes the base model` -> R5, R8, V1
- `Scenario: Unsafe LoRA target plan is rejected` -> R5, V2
- `Scenario: WKV7 gradient mechanisms agree` -> R6, R7, V1
- `Scenario: Invalid gradient evidence fails` -> R7, V2
- `Scenario: CPU optimizer step changes only adapters` -> R8, V1
- `Scenario: Adapter checkpoint resumes exactly` -> R9, V1
- `Scenario: Incompatible checkpoint cannot resume` -> R9, V2
- `Scenario: CPU evidence remains bounded` -> V3, R10
