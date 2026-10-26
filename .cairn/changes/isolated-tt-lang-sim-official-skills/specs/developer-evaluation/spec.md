## ADDED Requirements

### Requirement: Isolated device-free developer tools preserve runtime boundaries

r[tenstorrent.tooling.isolated_simulator_references]

The developer tool packages MUST execute the pinned upstream TT-Lang simulator without production TT-NN or ambient Python imports, and MUST expose official review references from a reproducibly pinned source. Reference selection MUST load review-core first with at most two distinct additional domains. Shared integration MUST distinguish modeled, CPU, physical, historical unresolved and completed package-check evidence without changing serving defaults.

#### Scenario: Installed simulator arithmetic produces traffic evidence

- GIVEN the explicit simulator Python closure and a one-tile add and matmul kernel
- WHEN the installed runner executes with hostile ambient Python paths and trace collection enabled
- THEN both outputs match the independent float32 references, simulator TT-NN is used, and upstream statistics report four DRAM tile reads and two writes as modeled transfers only.

#### Scenario: Official reference selection is bounded and complete

- GIVEN the pinned review-core, precision, trace, L1 and kernel reference directories
- WHEN two domains are resolved or their local references are read
- THEN review-core appears first, duplicates are removed, local relative links remain available, and excess domains or file escapes are rejected without marketplace installation, hooks or home mutation.

#### Scenario: A full native package check does not finish

- GIVEN a bounded normal package check with all tests enabled
- WHEN the command reaches its declared timeout during the test phase
- THEN integration preserves that attempt as unresolved and MUST NOT replace it with a disabled-test build or retroactively label it as passed.

#### Scenario: A later full native package check resolves acceptance

- GIVEN optimized host test oracles with debug assertions, overflow checks, all workspace cases and numerical bounds retained
- WHEN a later normal package build completes its full workspace check, install and fixup
- THEN integration records its exact command, output, counts and elapsed time separately from historical timeouts, without claiming new hardware or performance qualification.

#### Scenario: Delivery records follow the exact aggregate acceptance

- GIVEN a completed exact quality launcher run and its original command output
- WHEN a portable acceptance receipt is added during later documentation closure
- THEN the receipt binds the observed aggregate output and keeps its historical manifest boundary explicit; later documentation checks MUST NOT be represented as part of that earlier run or as renewed runtime qualification.
