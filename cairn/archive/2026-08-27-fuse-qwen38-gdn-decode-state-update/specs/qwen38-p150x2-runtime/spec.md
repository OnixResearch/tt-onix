## ADDED Requirements

### Requirement: GDN decode fusion requires matched exact-output evidence

r[tenstorrent.qwen38_p150x2.gdn_decode_fusion]

The repository MAY provide a default-off fusion candidate for the pinned Qwen3.8 P150x2 GDN decode path. The candidate MUST use a closed mode selected by a pure device-free policy. Missing or unknown modes MUST select `off`. The candidate MUST keep fp32 recurrent state, unchanged model equations, per-request state isolation, and the existing fallback path. Selection MUST use matched physical roots in exact `control,candidate,candidate,control` order. Selection MUST require exact token and text parity, clean reset, valid state evidence, stable duplicate roots, a GDN median ratio at most `0.95`, and a full decode-step median ratio at most `0.97`. A selected execution identity MUST pass fresh B=4 admission before deployment.

#### Scenario: A fusion mode passes matched physical gates

r[tenstorrent.qwen38_p150x2.gdn_decode_fusion.valid]

- **GIVEN** the pinned runtime, checkpoint, tokenizer, P150x2 devices, fixed profile, and matched control and candidate roots
- **WHEN** exact outputs, state isolation, reset, duplicate stability, GDN timing, and full-step timing pass every fixed bound
- **THEN** the evaluator MAY select that exact fusion mode for fresh B=4 admission
- **AND** deployment MUST wait for admission of the changed execution identity.

#### Scenario: Fusion evidence is invalid or incomplete

r[tenstorrent.qwen38_p150x2.gdn_decode_fusion.invalid]

- **GIVEN** an unknown mode, missing root, identity drift, output mismatch, state mismatch, reset failure, unstable duplicate, or missed timing bound
- **WHEN** fusion selection evaluates the evidence
- **THEN** it MUST reject the candidate and select `off`
- **AND** production MUST keep the previously admitted execution identity.
