## ADDED Requirements

### Requirement: Explicit OrcaRouter source selection

The candidate manifest MUST distinguish the requested GGUF repository from any selected safetensors repository. It MUST bind immutable revisions, source lineage, file identities, license records, and tokenizer artifacts. The shell MUST require authorized Hugging Face access before restricted file retrieval. It MUST NOT accept account-sharing terms or substitute another model without owner approval. Metadata inspection MUST NOT execute remote model code.

r[tenstorrent.qwen38_p150x2.orca_source]

#### Scenario: Reviewed native source
- GIVEN the requested GGUF release and the approved OrcaRouter BF16 alternative with complete file manifests
- WHEN the core evaluates source selection
- THEN it records the distinct artifacts and permits only the selected source.

#### Scenario: Restricted, substituted, or executable metadata
- GIVEN missing access, an unapproved repository substitution, or metadata that requests remote code
- WHEN the shell prepares the source
- THEN it stops before restricted retrieval or remote-code execution and preserves the current service.

### Requirement: Proven checkpoint format and shape

The staging core MUST compare tensor names, shapes, dtypes, layer types, tokenizer IDs, and configuration against the pinned loader contract. The TT-NN path MUST reject GGUF files as direct checkpoint input. An exact GGUF route MUST require a separate backend capability decision and physical placement evidence. A loader success or CPU fallback MUST NOT establish P150x2 support.

r[tenstorrent.qwen38_p150x2.orca_format]

#### Scenario: Compatible safetensors inventory
- GIVEN a complete OrcaRouter tensor inventory compatible with the pinned native loader
- WHEN the evaluator verifies all required tensors and metadata
- THEN it admits staging without claiming model correctness or performance.

#### Scenario: Unsupported representation or shape
- GIVEN a GGUF file, missing shard, incompatible head dimension, or unsupported tensor encoding
- WHEN the evaluator verifies the candidate
- THEN it rejects the candidate before device acquisition.

### Requirement: Checkpoint-specific cache isolation

The cache identity MUST bind source bytes, conversion recipe, runtime closure, mesh, precision policy, tokenizer, template, and MTP cohort. The shell MUST use separate candidate cache roots and verify those identities before reuse. It MUST NOT overwrite or relabel the current checkpoint, weight caches, prefix states, or rollback artifacts. Requantized weights MUST retain their OrcaRouter source lineage.

r[tenstorrent.qwen38_p150x2.orca_cache]

#### Scenario: Fresh OrcaRouter cache
- GIVEN the selected source and an empty candidate cache root
- WHEN the authorized shell builds TT-NN tensors
- THEN it records the actual conversion recipe and leaves the old root unchanged.

#### Scenario: Same shape with stale bytes
- GIVEN an official-model cache or recurrent snapshot with the same tensor shapes
- WHEN the candidate attempts reuse
- THEN the identity mismatch rejects reuse despite matching names and shapes.

### Requirement: Bounded text-service compatibility

The service MUST report the selected model identity and use its reviewed tokenizer and chat template. It MUST preserve existing request limits, text-only support, sampling semantics, and tool-authority boundaries. The evaluator MUST compare explicit thinking settings and account for all generated tokens. The checkpoint switch MUST NOT imply new vision, tool-calling, reasoning-field, context-length, or streaming support.

r[tenstorrent.qwen38_p150x2.orca_service]

#### Scenario: Supported text request
- GIVEN the approved candidate template and a supported request with recorded thinking settings
- WHEN the candidate serves that request
- THEN the receipt reports the correct model, token accounting, and supported response contract.

#### Scenario: Unsupported advertised capability
- GIVEN an image request or an unadmitted feature from the upstream model card
- WHEN a client requests that feature
- THEN the service rejects it under the current API contract without broadening authority.

### Requirement: Matching OrcaRouter MTP cohort

Initial checkpoint adoption MUST keep MTP disabled. The candidate manifest MUST identify the OrcaRouter MTP tensors even while that route remains disabled. Any later enabled route MUST use a content-bound BF16 head from the same approved cohort. It MUST require the existing MTP repair and admission gates. Official-model heads or old MTP performance receipts MUST NOT authorize the OrcaRouter route.

r[tenstorrent.qwen38_p150x2.orca_mtp]

#### Scenario: Plain candidate adoption
- GIVEN the candidate backbone and a disabled MTP route
- WHEN the service starts
- THEN it does not load MTP weights or capture MTP traces.

#### Scenario: Mixed checkpoint cohort
- GIVEN an OrcaRouter backbone with the old official-model MTP head
- WHEN a caller requests MTP admission
- THEN the cohort check rejects the route even if every tensor shape matches.

### Requirement: Independent checkpoint admission evidence

The evaluator MUST separate same-source numerical correctness from intended behavioral differences against the official model. The shell MUST retain the current runtime for the first checkpoint comparison. It MUST predeclare quality, memory, latency, throughput, token, and timeout gates before measurement. It MUST require device-free tests, a representative simulator result or exact blocker, and authorized physical evidence on the exact P150x2 mesh.

r[tenstorrent.qwen38_p150x2.orca_admission]

#### Scenario: Matched physical comparison
- GIVEN an authorized window, explicit gates, and matched runtime profiles for control and candidate
- WHEN the complete P150x2 screen passes
- THEN the evaluator records checkpoint-scoped evidence without transferring old optimization claims.

#### Scenario: Confounded or simulated result
- GIVEN an unbound runtime change, incomplete timing, failed quality gate, or simulator-only result
- WHEN the evaluator considers production admission
- THEN it rejects promotion and retains the exact failed gate or evidence gap.

### Requirement: Authorized reversible checkpoint promotion

The shell MUST require owner approval before a service switch. It MUST retain the old closure, checkpoint, caches, service identity, and recovery procedure. Promotion MUST verify candidate health and a new model-bound optimization baseline. A failed trial MUST restore the old service only after safe device ownership is established. A live timed-out worker MUST block reuse and concurrent cleanup.

r[tenstorrent.qwen38_p150x2.orca_promotion]

#### Scenario: Approved switch with rollback
- GIVEN passing candidate evidence, owner approval, and a verified rollback procedure
- WHEN the shell promotes the candidate
- THEN it verifies the new service identity and preserves the old deployment artifacts.

#### Scenario: Failed trial with unresolved worker
- GIVEN a failed trial and a device worker that did not terminate
- WHEN the shell evaluates rollback
- THEN it blocks device reuse and reports the required recovery authority instead of starting another worker.
