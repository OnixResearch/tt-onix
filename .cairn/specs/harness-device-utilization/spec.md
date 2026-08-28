# Harness Device Utilization Specification

## Purpose

Defines the `harness-device-utilization` capability.

## Requirements

### Requirement: Device utilization metrics are measured
r[tenstorrent.harness.utilization] A measurement run MUST record a math or ALU busy fraction and a NoC idle fraction for the device. Each fraction SHALL be in the closed interval zero to one. The source of each metric MUST be named in the receipt.

#### Scenario: Utilization fractions are recorded and bounded
r[tenstorrent.harness.utilization.valid]
- GIVEN a completed measurement run with device profiling available
- WHEN the harness writes the receipt
- THEN the receipt SHALL contain a math busy fraction and a NoC idle fraction in the closed interval zero to one.

#### Scenario: Out-of-range utilization is rejected
r[tenstorrent.harness.utilization.invalid]
- GIVEN a utilization fraction outside the closed interval zero to one
- WHEN the admission core validates the receipt
- THEN admission MUST fail and name the out-of-range fraction.

### Requirement: Memory pressure is measured per device
r[tenstorrent.harness.memory_pressure] A measurement run MUST record peak L1 or SRAM and peak DRAM or sysmem usage per device. Each peak SHALL be non-negative and finite. Each peak SHALL bind the total available capacity for the same store.

#### Scenario: Memory peaks are recorded against capacity
r[tenstorrent.harness.memory_pressure.valid]
- GIVEN a completed measurement run that reads memory telemetry
- WHEN the harness writes the receipt
- THEN the receipt SHALL contain each peak and its total available capacity.

#### Scenario: Negative or non-finite memory peak is rejected
r[tenstorrent.harness.memory_pressure.invalid]
- GIVEN a memory peak that is negative or non-finite
- WHEN admission runs
- THEN admission MUST fail and name the invalid peak.

### Requirement: Per-operation latency table is recorded
r[tenstorrent.harness.op_latency] A measurement run MUST record a per-operation latency table with mean and p99 latency for each TT-NN operation in the step. Every operation executed in the step SHALL appear in the table.

#### Scenario: Op latency table covers every operation
r[tenstorrent.harness.op_latency.valid]
- GIVEN a completed measurement run that collects per-operation timing
- WHEN the harness writes the receipt
- THEN every operation in the step SHALL appear with mean and p99 latency.

#### Scenario: A step operation missing from the table is rejected
r[tenstorrent.harness.op_latency.invalid]
- GIVEN an operation in the executed step that is absent from the table
- WHEN admission runs
- THEN admission MUST fail and name the missing operation.

### Requirement: Scaling efficiency is measured across meshes
r[tenstorrent.harness.scaling_efficiency] A measurement run MUST record time per step on one device and on two devices under the same profile, when the profile names both meshes. The receipt SHALL report the ratio against the ideal data-parallel line.

#### Scenario: Scaling ratio is computed against the ideal line
r[tenstorrent.harness.scaling_efficiency.valid]
- GIVEN measured time per step on one device and on two devices under the same profile
- WHEN the harness computes the scaling ratio
- THEN the receipt SHALL bind the ratio and the ideal data-parallel line.

#### Scenario: Scaling ratio without a two-device run is rejected
r[tenstorrent.harness.scaling_efficiency.invalid]
- GIVEN a run with no two-device measurement but a scaling claim
- WHEN admission runs
- THEN admission MUST fail and MUST NOT report a scaling ratio.
