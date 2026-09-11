## ADDED Requirements

### Requirement: Checkpoint layer types own hybrid dispatch

r[tenstorrent.qwen38_p150x2.layer_type_authority] The Qwen3.8 runtime MUST derive every layer's mixer type from the pinned checkpoint `layer_types` list. The list MUST contain exactly one recognized type for every declared text layer. A static mapper list MUST NOT override checkpoint dispatch. A legacy fallback MAY apply only when its length equals the declared layer count. Invalid or incomplete layer-type authority MUST fail before layer construction or weight loading.

#### Scenario: Pinned Qwen3.8 layer types are complete

- **GIVEN** the pinned 64-layer Qwen3.8 checkpoint with its complete `layer_types` list
- **WHEN** the pure resolver validates hybrid dispatch
- **THEN** it returns 48 linear-attention layers and 16 full-attention layers
- **AND** full-attention layers occur at every fourth index, starting at index 3.

#### Scenario: Layer-type authority is invalid

- **GIVEN** a missing non-legacy list, a list with the wrong length, or an unknown layer type
- **WHEN** the pure resolver validates hybrid dispatch
- **THEN** it rejects the configuration before layer construction or weight loading
- **AND** it does not substitute the obsolete eight-entry mapper set.
