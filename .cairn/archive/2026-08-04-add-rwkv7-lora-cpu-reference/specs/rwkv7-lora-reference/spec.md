# RWKV-7 LoRA CPU Reference Specification Delta

## ADDED Requirements

### Requirement: The CPU reference binds one exact RWKV-7 model authority

r[tenstorrent.rwkv7_lora_reference.model] The reference MUST bind the pinned World2.8 0.1B checkpoint, tokenizer, model source, equation sources, dimensions, precision class, and immutable content identities.

#### Scenario: Pinned model authority is admitted

r[tenstorrent.rwkv7_lora_reference.model.valid]
- GIVEN every model artifact and source matches its pinned identity
- AND the model has the admitted layer, hidden, head, and vocabulary dimensions
- WHEN model admission runs
- THEN the reference MUST produce one deterministic BLAKE3 authority identity.

#### Scenario: Model authority drift is rejected

r[tenstorrent.rwkv7_lora_reference.model.invalid]
- GIVEN any checkpoint, tokenizer, source, dimension, or precision identity differs
- WHEN model admission runs
- THEN admission MUST fail before tensor execution.

### Requirement: Fine-tuning configuration is typed and portable

r[tenstorrent.rwkv7_lora_reference.configuration] The reference MUST author model, tokenizer, SFT data, static sequence, LoRA, optimizer, seed, tolerance, output, and resume configuration in typed Nickel.

#### Scenario: Valid reference configuration exports deterministically

r[tenstorrent.rwkv7_lora_reference.configuration.valid]
- GIVEN a complete configuration satisfies all model, LoRA, optimizer, path, and resume constraints
- WHEN configuration export runs twice
- THEN both generated runner inputs MUST match byte for byte.

#### Scenario: Invalid reference configuration fails admission

r[tenstorrent.rwkv7_lora_reference.configuration.invalid]
- GIVEN a configuration has unknown fields, invalid bounds, empty targets, incompatible paths, or inconsistent resume identities
- WHEN Nickel validation runs
- THEN export MUST fail before model or data loading.

### Requirement: SFT records produce deterministic active labels

r[tenstorrent.rwkv7_lora_reference.data] The reference MUST admit local JSONL records with nonempty `prompt` and `response` strings and MUST create deterministic tokens, prompt masks, response labels, padding masks, and a BLAKE3 data manifest.

#### Scenario: Valid SFT record produces deterministic labels

r[tenstorrent.rwkv7_lora_reference.data.valid]
- GIVEN a valid local record and the pinned tokenizer
- WHEN preprocessing runs twice with the same static sequence profile
- THEN token IDs, active labels, masks, lengths, and manifest identities MUST match
- AND only response target positions MUST contribute to supervised loss.

#### Scenario: Invalid SFT record is rejected

r[tenstorrent.rwkv7_lora_reference.data.invalid]
- GIVEN a record is malformed, has unknown fields, has an empty prompt or response, exceeds the admitted policy, or produces no active labels
- WHEN preprocessing runs
- THEN preprocessing MUST fail before model execution.

### Requirement: LoRA targets are explicit and base weights stay frozen

r[tenstorrent.rwkv7_lora_reference.lora] The reference MUST add LoRA only to an explicit allowlist of admitted RWKV-7 projection roles and MUST freeze every base checkpoint tensor.

#### Scenario: Explicit LoRA plan freezes the base model

r[tenstorrent.rwkv7_lora_reference.lora.valid]
- GIVEN the default attention and feed-forward projection allowlist
- WHEN the LoRA plan is applied with zero-effect initialization
- THEN initial logits MUST match the frozen base reference within the named tolerance
- AND only adapter tensors MUST require gradients.

#### Scenario: Unsafe LoRA target plan is rejected

r[tenstorrent.rwkv7_lora_reference.lora.invalid]
- GIVEN a plan uses `all-linear`, an unknown role, a duplicate role, an empty role set, or a frozen-only tensor
- WHEN target admission runs
- THEN the plan MUST fail with exact target diagnostics.

### Requirement: The CPU sequence equations match RWKV-7 training authority

r[tenstorrent.rwkv7_lora_reference.equations] The reference MUST implement the full bounded RWKV-7 sequence path and the Appendix H naive WKV7 recurrence in FP32 after loading pinned BF16 values.

#### Scenario: Sequence forward matches the admitted equation

r[tenstorrent.rwkv7_lora_reference.equations.valid]
- GIVEN admitted fixed tensors, initial state, and sequence shape
- WHEN the sequence reference runs
- THEN logits, loss inputs, WKV outputs, and final state MUST be finite
- AND they MUST match stored authority fixtures within named tolerances.

#### Scenario: Invalid sequence input fails

r[tenstorrent.rwkv7_lora_reference.equations.invalid]
- GIVEN a tensor has a wrong shape, wrong dtype, nonfinite value, invalid state, or unsupported sequence profile
- WHEN the sequence reference runs
- THEN it MUST fail before it returns partial authority output.

### Requirement: Independent gradient mechanisms agree

r[tenstorrent.rwkv7_lora_reference.gradients] The reference MUST compare PyTorch autograd gradients with the paper-matched manual WKV7 backward equation and bounded finite differences under named tolerance profiles.

#### Scenario: WKV7 gradient mechanisms agree

r[tenstorrent.rwkv7_lora_reference.gradients.valid]
- GIVEN admitted bounded tensors and an explicit output gradient
- WHEN all gradient mechanisms run
- THEN each required input gradient MUST be finite
- AND the compared values MUST agree within their named tolerances.

#### Scenario: Invalid gradient evidence fails

r[tenstorrent.rwkv7_lora_reference.gradients.invalid]
- GIVEN a gradient is missing, nonfinite, stale, wrongly shaped, outside tolerance, or disconnected from the selected loss
- WHEN gradient validation runs
- THEN validation MUST fail with the exact tensor role and comparison class.

### Requirement: One CPU optimizer step changes only adapters

r[tenstorrent.rwkv7_lora_reference.training_step] The reference MUST run deterministic causal-language-model loss, gradient accumulation, named clipping policy, and an admitted optimizer over adapter tensors only.

#### Scenario: CPU optimizer step changes only adapters

r[tenstorrent.rwkv7_lora_reference.training_step.valid]
- GIVEN admitted data, configuration, base model, and zero-effect adapters
- WHEN one optimizer step completes
- THEN loss and required gradients MUST be finite
- AND at least one adapter tensor MUST change
- AND every base tensor MUST remain byte-identical.

#### Scenario: Invalid training step fails without an artifact

r[tenstorrent.rwkv7_lora_reference.training_step.invalid]
- GIVEN loss or gradients are nonfinite, no adapter receives a gradient, a base tensor changes, or clipping violates policy
- WHEN the step runs
- THEN the run MUST fail
- AND it MUST not emit a passing adapter artifact.

### Requirement: Adapter checkpoints are portable and resume safely

r[tenstorrent.rwkv7_lora_reference.checkpoint] The reference MUST save adapter tensors as Safetensors and MUST bind model, tokenizer, configuration, data, optimizer, step, and artifact identities in a BLAKE3 manifest.

#### Scenario: Adapter checkpoint resumes exactly

r[tenstorrent.rwkv7_lora_reference.checkpoint.valid]
- GIVEN a valid adapter and matching resume inputs
- WHEN save, load, and resume validation run
- THEN loaded adapter tensors MUST match the saved tensors byte for byte
- AND the next deterministic step MUST match the uninterrupted reference within named tolerances.

#### Scenario: Incompatible checkpoint cannot resume

r[tenstorrent.rwkv7_lora_reference.checkpoint.invalid]
- GIVEN an adapter is corrupt or any bound model, tokenizer, target, shape, optimizer, data, or configuration identity differs
- WHEN resume admission runs
- THEN resume MUST fail before model mutation.

### Requirement: CPU reference evidence is bounded

r[tenstorrent.rwkv7_lora_reference.evidence] The reference MUST emit a BLAKE3-bound receipt for authority, data, LoRA plan, equations, gradients, optimizer results, adapter artifact, and non-claims.

#### Scenario: CPU evidence remains bounded

r[tenstorrent.rwkv7_lora_reference.evidence.valid]
- GIVEN all reference validation passes
- WHEN receipt generation runs
- THEN the receipt MUST identify all portable inputs, compared outputs, tolerances, and artifacts
- AND it MUST not claim Tenstorrent execution, convergence, generalization, or production quality.

### Requirement: CPU reference has complete validation and documentation

r[tenstorrent.rwkv7_lora_reference.verification] The change MUST include positive and negative data, LoRA, equation, gradient, optimizer, artifact, determinism, and resume fixtures plus existing inference regressions.

#### Scenario: CPU reference validation closes the change

r[tenstorrent.rwkv7_lora_reference.verification.complete]
- GIVEN maintainers intend to sync and archive the CPU reference change
- WHEN all required validation runs
- THEN focused tests, positive fixtures, negative fixtures, inference regressions, package checks, Cairn validation, and lifecycle gates MUST pass.

### Requirement: CPU reference documentation gives exact procedures and limits

r[tenstorrent.rwkv7_lora_reference.documentation] The documentation MUST give exact data, configuration, reference, training-step, save, load, resume, and evidence procedures with explicit claim limits.

#### Scenario: Operator can reproduce the CPU fixture

r[tenstorrent.rwkv7_lora_reference.documentation.complete]
- GIVEN the pinned inputs and local fixture data
- WHEN an operator follows the documented commands
- THEN the operator MUST produce the documented reference receipt and adapter identities
- AND the documentation MUST keep accelerator and convergence claims out of scope.
