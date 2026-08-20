# Qwen38 P150x4 Runtime Specification

## Purpose

Defines the `qwen38-p150x4-runtime` capability.

## Requirements

### Requirement: P150x4 remains unsupported without an exact physical topology
r[tenstorrent.qwen38_p150x4.availability_non_support] The repository MUST treat Qwen3.8-27B P150x4 as unsupported when four compatible, owned physical devices are unavailable. It MUST NOT add a P150x4 package, service, profile, runtime path, or local performance claim from a smaller topology. The admitted P150x2 deployment MUST remain unchanged. Future P150x4 support MUST use a new validated Cairn change with exact physical evidence.

#### Scenario: Required physical topology is absent
r[tenstorrent.qwen38_p150x4.availability_non_support.valid]
- GIVEN an available host with fewer than four compatible, owned physical devices
- WHEN P150x4 support is evaluated
- THEN the deployment MUST be returned as unsupported
- AND no P150x4 implementation or local performance claim MUST be added.

#### Scenario: Future hardware becomes available
r[tenstorrent.qwen38_p150x4.availability_non_support.invalid]
- GIVEN four compatible physical devices become available after this decision
- WHEN an operator requests P150x4 support
- THEN this archived decision MUST NOT imply support
- AND a new validated change with physical acceptance evidence MUST be required.
