## ADDED Requirements

### Requirement: Decode component attribution uses matched trace replay

r[tenstorrent.qwen38_p150x2.decode_component_attribution] The repository MAY provide an operator-enabled component attribution probe for the pinned Qwen3.8 P150x2 runtime. The probe MUST remain diagnostic-only. It MUST measure repeated trace replay for embedding, representative GDN mixers, representative MLPs, representative full-attention mixers, and LM-head reduction. It MUST also measure synchronized host input update, bounded winner readback, and the complete traced decode step. Each retained group MUST contain at least five complete samples. The deterministic evaluator MUST reject missing identities, wrong layer types, invalid durations, unstable samples, or an additive component model outside its configured closure bounds. A rejected result MUST select no optimization and MUST preserve eager production.

#### Scenario: Component attribution is valid

- **GIVEN** the exact pinned package, runtime, checkpoint, topology, profile, prompt state, and representative layer set
- **WHEN** every component and the full decode path complete repeated trace replay within the stability and closure bounds
- **THEN** the receipt MUST report direct medians, weighted component shares, host costs, and the selected bottleneck
- **AND** the result MUST remain diagnostic and ineligible for production admission.

#### Scenario: Component attribution is invalid

- **GIVEN** a missing group, wrong layer type, identity mismatch, malformed duration, unstable sample set, or additive model outside its closure bounds
- **WHEN** the deterministic evaluator processes the evidence
- **THEN** it MUST reject attribution and name each bounded reason
- **AND** it MUST select no kernel, precision, collective, or production change.
