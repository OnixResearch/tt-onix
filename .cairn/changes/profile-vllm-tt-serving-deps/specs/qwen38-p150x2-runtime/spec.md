## ADDED Requirements

### Requirement: The vLLM serving stack uses an isolated, ABI-checked dependency profile

r[tenstorrent.qwen38_p150x2.vllm_serving_dependency_profile]

The repository MAY provide an isolated dependency profile for a vLLM-based serving stack. The profile MUST NOT change any dependency of the deployed Qwen serialized environment, and no new requirement MAY be suppressed to force an install. The profile MUST build vLLM from source with the empty target device against a pinned tag, and it MUST install the common requirements explicitly so that no CUDA wheel metadata resolves. The profile MUST require Starlette at or above its security floor and MUST reject any resolution that would lower it. The profile MUST prove a coherent Python, Torch, and TT-NN ABI before any model trial, and a successful plugin import alone MUST NOT satisfy that check. The profile MUST own the load-bearing serving settings, including disabled prefix caching, the L1 small size, the maximum sequence count, the block size, the trace region size, host sampling, the mesh descriptor path, and a readiness budget above the measured startup time. The profile MUST record why host sampling is required. The repository MUST NOT claim a throughput, latency, concurrency, or accuracy result from this change.

#### Scenario: The profile resolves and the stack serves one bounded trial

r[tenstorrent.qwen38_p150x2.vllm_serving_dependency_profile.valid]

- **GIVEN** the pinned vLLM tag, the plugin revision, the isolated profile, and a coherent ABI check
- **WHEN** one bounded non-MTP model trial runs through the stack
- **THEN** the stack MUST load the model and answer a bounded request
- **AND** the receipt MUST record the resolved dependency versions, the ABI check result, the startup time, and the profile-owned serving settings
- **AND** the serialized P150x2 service MUST remain the served route.

#### Scenario: The profile cannot resolve or the ABI is incoherent

r[tenstorrent.qwen38_p150x2.vllm_serving_dependency_profile.invalid]

- **GIVEN** an unsatisfiable constraint pair, a second Torch in the closure, a lowered Starlette floor, a suppressed requirement, or a changed deployed dependency
- **WHEN** profile resolution or the ABI check runs
- **THEN** it MUST fail closed and name the exact constraint or the duplicated package
- **AND** no serving run MAY proceed
- **AND** the deployed Qwen environment MUST be unchanged.

#### Scenario: A load-bearing serving setting is missing

r[tenstorrent.qwen38_p150x2.vllm_serving_dependency_profile.settings]

- **GIVEN** an enabled prefix cache, a missing or lowered L1 small size, an unbounded sequence count, a missing trace region, on-device sampling below four devices, or a readiness budget below the measured startup time
- **WHEN** the pure profile policy validates the serving plan
- **THEN** it MUST fail before the engine starts
- **AND** the receipt MUST name the setting and the failure it prevents.
