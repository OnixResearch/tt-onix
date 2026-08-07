# TTsim Runtime Bundle Specification Delta

## ADDED Requirements

### Requirement: Simulator assets match the host system

r[tenstorrent.ttsim.asset_selection] The package MUST select each simulator asset by exact host system, chip architecture, topology, upstream release name, and content hash.

#### Scenario: Supported host asset is selected

r[tenstorrent.ttsim.asset_selection.valid]
- GIVEN a supported host system and a pinned simulator release asset
- WHEN the package evaluates the asset table
- THEN it MUST select the exact host-specific asset and hash
- AND the resulting binary MUST match the host architecture.

#### Scenario: Unsupported host asset is rejected

r[tenstorrent.ttsim.asset_selection.invalid]
- GIVEN a host system, chip, or topology without a verified release asset
- WHEN the package evaluates the selection
- THEN evaluation MUST fail with the unsupported combination
- AND it MUST NOT substitute an asset for another host system.

### Requirement: Simulator runtime includes its TT-Metal descriptor

r[tenstorrent.ttsim.runtime_bundle] Each supported simulator runtime MUST place the selected simulator library and matching TT-Metal `soc_descriptor.yaml` in one immutable directory. A named wrapper MUST set the tested simulator environment.

#### Scenario: Complete simulator runtime evaluates

r[tenstorrent.ttsim.runtime_bundle.valid]
- GIVEN a supported simulator asset and package-matched TT-Metal descriptor
- WHEN the runtime package builds
- THEN the selected `libttsim.so` and `soc_descriptor.yaml` MUST be adjacent
- AND the wrapper MUST select slow dispatch and the exact library path.

#### Scenario: Incomplete simulator runtime fails

r[tenstorrent.ttsim.runtime_bundle.invalid]
- GIVEN a runtime with a missing descriptor, wrong chip descriptor, missing required ABI symbol, or mismatched host asset
- WHEN package validation runs
- THEN validation MUST fail before a TT-Metal workload starts.

### Requirement: Blackhole single-chip smoke is bounded

r[tenstorrent.ttsim.single_chip_smoke] The repository MUST run a named Blackhole `1x1` simulator smoke before a new covered hardware attempt. The smoke MUST use the same TT-Metal binaries as silicon.

#### Scenario: Blackhole single-chip smoke passes

r[tenstorrent.ttsim.single_chip_smoke.valid]
- GIVEN the complete Blackhole runtime and package-matched TT-Metal
- WHEN the bounded smoke runs under its named timeout
- THEN it MUST open one device, compile firmware, execute representative operations, and close cleanly
- AND it MUST emit a simulated receipt.

#### Scenario: Simulator smoke failure stays bounded

r[tenstorrent.ttsim.single_chip_smoke.invalid]
- GIVEN the smoke times out, reaches an unsupported operation, exits nonzero, or fails to close
- WHEN the bounded runner classifies the result
- THEN it MUST stop without a hardware claim
- AND it MUST record the exact blocker.

### Requirement: Multi-chip simulation requires matched authority

r[tenstorrent.ttsim.multi_chip_boundary] A multi-chip simulator path MUST require package-matched TT-Metal support, the exact simulator topology, and a reviewed cluster descriptor before mesh execution.

#### Scenario: Matched multi-chip path runs

r[tenstorrent.ttsim.multi_chip_boundary.valid]
- GIVEN package-matched support for `bh_x2`, its exact library, and its exact cluster descriptor
- WHEN the bounded multi-chip smoke runs
- THEN it MUST validate mesh open, replication, representative execution, and close
- AND it MUST classify the result as experimental simulated evidence.

#### Scenario: Unmatched multi-chip path is blocked

r[tenstorrent.ttsim.multi_chip_boundary.invalid]
- GIVEN an unmatched TT-Metal revision, missing cluster descriptor, or unsupported topology
- WHEN multi-chip launch is requested
- THEN launch MUST fail with an exact blocker before simulation.

### Requirement: Simulator evidence preserves its claim boundary

r[tenstorrent.ttsim.evidence] Simulator receipts MUST bind the simulator, descriptor, host system, TT-Metal, dispatch mode, workload, timeout, and outcome with BLAKE3. They MUST state that execution was simulated.

#### Scenario: Simulated receipt is admitted

r[tenstorrent.ttsim.evidence.valid]
- GIVEN a completed bounded simulator run with complete identities and non-claims
- WHEN receipt admission runs
- THEN it MUST classify only the observed simulated behavior
- AND it MUST NOT claim physical placement, hardware profiling, performance, thermals, health, or silicon success.

#### Scenario: Hardware claim appears in simulator evidence

r[tenstorrent.ttsim.evidence.invalid]
- GIVEN a simulator receipt that omits its simulated status or includes a hardware claim
- WHEN receipt admission runs
- THEN admission MUST fail with a claim-boundary diagnostic.
