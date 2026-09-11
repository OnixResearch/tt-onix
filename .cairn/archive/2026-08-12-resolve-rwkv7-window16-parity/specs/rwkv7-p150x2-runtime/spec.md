# RWKV-7 P150x2 runtime delta

## ADDED Requirements

### Requirement: Bounded window-16 parity diagnosis

r[tenstorrent.rwkv7_p150x2.window16_parity_diagnosis]

The runtime and evidence package MUST provide a bounded diagnostic path that compares the known window-16 request through the first rejected generated-token index. The diagnostic MUST retain enough per-step and per-layer evidence to locate the first CPU/device divergence. It MUST NOT expand production admission.

#### Scenario: Diagnostic localizes the first divergence

- **GIVEN** the pinned checkpoint, known prompt, and pinned CPU authority
- **WHEN** the bounded device diagnostic reaches generated token index `10`
- **THEN** the evidence identifies the first differing layer and value boundary or proves all retained boundaries equal

#### Scenario: Diagnostic input is outside its bound

- **GIVEN** a request that exceeds the named step, layer, device, or artifact bound
- **WHEN** the diagnostic validates the request
- **THEN** it rejects the request before physical execution

### Requirement: Window-16 correction remains fail-closed

r[tenstorrent.rwkv7_p150x2.window16_parity_correction]

A window-16 source correction MUST preserve exact CPU generated-token parity and FP32 recurrent-state parity. It MUST include positive and negative regression tests. A new admission evaluation MUST use a new immutable candidate identity and repeat every required simulator and physical gate.

#### Scenario: Corrected bounded request has exact parity

- **GIVEN** a source correction and a new immutable candidate
- **WHEN** the bounded correctness request completes on physical hardware
- **THEN** generated tokens and FP32 recurrent state exactly match the pinned CPU authority before later gates run

#### Scenario: Corrected request still differs

- **GIVEN** a source correction whose bounded request differs from the CPU authority
- **WHEN** the correctness validator reads the result
- **THEN** it rejects the candidate, stops later gates, and retains production windows `[2, 4, 8]`
