## ADDED Requirements

### Requirement: The fused GDN decode step serves greedy decode under a numerical quality gate

r[tenstorrent.qwen38_p150x2.gdn_decode_step]

The repository MAY serve deterministic P150x2 decode through the fused GDN decode step. On the owner decision of 2026-09-24, this route is exempt from bit-exact token parity with the composite route. The step MUST replace the decode chain between the fused `[qkv|z|a|b]` projection and the out projection with one conv-prep program and one recurrence program per GDN layer. The step MUST update the conv state and the fp32 recurrent state in place for batch rows `[0, B)`. It MUST leave the other rows unchanged. The decayed recurrent state MUST stay fp32 in DEST and MUST NOT pass through the 19-bit source registers. The step MUST reject unsupported geometry and keep the composite route. A service started in the step mode MUST keep greedy requests on it, and sampled requests MUST keep their separately admitted route. The step MUST NOT combine with MTP speculation. The quality gate MUST compare the step against a float64 reference of one layer. It MUST also compare teacher-forced logits against the composite route over a fixed prompt corpus and measure passage negative log-likelihood. The step MUST NOT raise passage NLL above the composite route, and its logit distance from the composite route MUST NOT exceed that of the fused recurrence already admitted for sampled serving.

#### Scenario: The fused step passes the quality gate and serves greedy decode

r[tenstorrent.qwen38_p150x2.gdn_decode_step.valid]

- **GIVEN** the Qwen3.8-27B TP=2 geometry, a warmed model, and the GDN layers started in the step mode
- **WHEN** a greedy request decodes
- **THEN** every GDN layer MUST run the conv-prep and recurrence programs in place of the composite chain
- **AND** the router MUST keep the request on the step route without a trace switch
- **AND** the recorded evidence MUST show no NLL increase and teacher-forced logit distances within the admitted sampled route's distances.

#### Scenario: Unsupported geometry, mode, or serving combination

r[tenstorrent.qwen38_p150x2.gdn_decode_step.invalid]

- **GIVEN** a head width other than 128, value heads that are not a multiple of key heads, a batch outside 1..32, a mode other than `step`, a sampled route named as deterministic, or MTP speculation with the step enabled
- **WHEN** the policy or the service configuration is evaluated
- **THEN** the geometry MUST be rejected and decode MUST keep the composite route
- **AND** the router MUST reject the invalid deterministic route
- **AND** the service configuration MUST fail its assertion.
