# zyx Tenstorrent Evaluation Delta

## ADDED Requirements

### Requirement: Pinned zyx source authority
r[tenstorrent.rust_ml.zyx_authority] The repository MUST evaluate one exact `zyx` source revision. The authority pin SHALL bind source URL, version, commit, Nix source hash, BLAKE3 tree identity, Cargo lock identity, and license.

#### Scenario: Pinned zyx source is admitted
r[tenstorrent.rust_ml.zyx_authority.valid]
- GIVEN `zyx` version `0.16.0` at commit `7606188adac64b362c257159af9949d860759cc0`
- AND source, lock, and LGPL-3.0-only license identities that match the pin
- WHEN source admission runs
- THEN it produces one deterministic admitted authority

#### Scenario: Source or lock drift is rejected
r[tenstorrent.rust_ml.zyx_authority.drift]
- GIVEN a source tree or Cargo lock identity that differs from the pin
- WHEN source admission runs
- THEN admission fails and names the mismatched identity
- AND no package output is exposed

### Requirement: Hermetic and isolated zyx runtime
r[tenstorrent.rust_ml.zyx_hermetic_runtime] The `zyx-tenstorrent` package MUST use Nix-owned Rust, C++, TT-Metal, and runtime paths. The build MUST install its child runtime inside the package output. The build and runtime MUST NOT write user configuration paths or terminate processes by name.

#### Scenario: Hermetic package installs its runtime
r[tenstorrent.rust_ml.zyx_hermetic_runtime.valid]
- GIVEN the pinned source and packaged TT-Metal inputs
- WHEN Nix builds `zyx-tenstorrent`
- THEN the child runtime is installed under the package output
- AND runtime discovery uses only declared package paths

#### Scenario: Build tries to write a user directory
r[tenstorrent.rust_ml.zyx_hermetic_runtime.user_write]
- GIVEN a build that resolves output through `HOME` or `XDG_CONFIG_HOME`
- WHEN the hermetic package check runs
- THEN the check fails and names the forbidden path source
- AND no user directory is modified

#### Scenario: Runtime closes its owned child
r[tenstorrent.rust_ml.zyx_hermetic_runtime.owned_child]
- GIVEN one backend session with one owned child
- WHEN the session closes or times out
- THEN the backend terminates and reaps that exact child
- AND the lifecycle reaches one terminal state

#### Scenario: Runtime tries global termination
r[tenstorrent.rust_ml.zyx_hermetic_runtime.global_termination]
- GIVEN an unrelated process with the same executable name
- WHEN backend cleanup runs
- THEN the unrelated process remains unchanged
- AND any `pkill`, name search, or global signal path fails validation

### Requirement: Truthful zyx capability matrix
r[tenstorrent.rust_ml.zyx_capability_matrix] The evaluation MUST classify each reviewed dtype, operation family, and architecture as supported, unsupported, or unknown. Each supported entry SHALL bind a positive code-generation fixture. Unsupported and unknown requests MUST fail before runtime launch.

#### Scenario: Supported operation has matching fixtures
r[tenstorrent.rust_ml.zyx_capability_matrix.supported]
- GIVEN an operation marked supported for Blackhole with one declared dtype
- WHEN device-free graph lowering and Tenstorrent code generation run
- THEN both fixtures pass with deterministic output
- AND the matrix binds their identities

#### Scenario: Unsupported capability fails before launch
r[tenstorrent.rust_ml.zyx_capability_matrix.unsupported]
- GIVEN an unsupported dtype, operation, or architecture request
- WHEN capability admission runs
- THEN admission fails with the exact unsupported field
- AND no child runtime, simulator, or device opens

### Requirement: Bounded zyx evaluation verdict
r[tenstorrent.rust_ml.zyx_evaluation_verdict] The evaluation MUST produce one canonical BLAKE3-bound verdict. The verdict SHALL be `blocked` or `eligible_for_runtime_adapter_spike`. It MUST bind package, capability, process-isolation, device-free, and simulator results with exact blockers. It MUST NOT make model, physical execution, placement, or performance claims.

#### Scenario: Bounded simulator smoke is classified as simulated
r[tenstorrent.rust_ml.zyx_evaluation_verdict.simulated]
- GIVEN a passing Blackhole `1x1` ttsim smoke with the named timeout
- WHEN verdict construction runs
- THEN the receipt binds runtime startup, transfer, operation, readback, and cleanup
- AND it classifies every execution result as simulated

#### Scenario: Simulator evidence is presented as physical
r[tenstorrent.rust_ml.zyx_evaluation_verdict.false_physical]
- GIVEN only device-free and simulator evidence
- WHEN verdict construction requests a physical classification
- THEN construction fails with an evidence-class diagnostic
- AND no physical receipt is emitted

#### Scenario: Passing gates permit only an adapter spike
r[tenstorrent.rust_ml.zyx_evaluation_verdict.eligible]
- GIVEN passing package, capability, process-isolation, device-free, and simulator gates
- WHEN verdict construction runs
- THEN the verdict is `eligible_for_runtime_adapter_spike`
- AND it exposes no runnable model target

#### Scenario: Missing gate produces an exact blocker
r[tenstorrent.rust_ml.zyx_evaluation_verdict.blocked]
- GIVEN one missing or failed evaluation gate
- WHEN verdict construction runs
- THEN the verdict is `blocked`
- AND it names the failed gate and available evidence
