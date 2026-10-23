## ADDED Requirements

### Requirement: Decode weight-traffic changes declare their evidence before hardware

r[tenstorrent.qwen38_p150x2.mlp_projection_streaming]

A decode-path weight-traffic change MUST declare its traffic model, its
numerical-retention screen, and its device-free evidence before any hardware
probe. The numerical-retention screen MUST name the NLL/perplexity floor, corpus,
and comparison method before implementation. The change MUST NOT be admitted
on a bandwidth figure above the packaged ceiling.

#### Scenario: A candidate is proposed for hardware

- GIVEN a candidate MLP gate/up/down traffic reduction
- WHEN a hardware probe is proposed
- THEN its traffic model, predefined numerical-retention screen, and device-free
  evidence are declared before the probe executes.

#### Scenario: An unchanged-bytes speedup exceeds the ceiling

- GIVEN a candidate whose bandwidth figure exceeds the packaged ceiling
- WHEN admission is considered
- THEN that bandwidth figure cannot justify admission.
