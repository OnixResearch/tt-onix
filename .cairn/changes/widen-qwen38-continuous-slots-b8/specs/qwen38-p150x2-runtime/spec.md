## MODIFIED Requirements

### Requirement: Deterministic bounded slot lifecycle

r[tenstorrent.qwen38_p150x2.continuous_slot_policy]

The pure scheduler MUST own slot generation, request identity, position, deadline, completion, and refill decisions. It MUST support unequal generation bounds without reassigning another user's state. The slot capacity MUST be declared by the profile and MUST be a member of the closed set {4, 8}; values outside that set MUST be refused. Queue and memory limits MUST come from typed policy.

#### Scenario: Staggered requests
- GIVEN one completed slot and another active request
- WHEN a compatible queued request is eligible
- THEN the scheduler plans refill without changing the active request's identity or position.

#### Scenario: Stale event or full queue
- GIVEN an old slot-generation event or exhausted queue capacity
- WHEN the scheduler evaluates the event
- THEN it rejects the event or request without corrupting active state.

#### Scenario: Unsupported capacity
- GIVEN a policy whose `slot_capacity` is 5, 6, or 16
- WHEN the policy is validated
- THEN it is refused before any device work.

## ADDED Requirements

### Requirement: B=8 scheduler candidate profile

r[tenstorrent.qwen38_p150x2.continuous_slot_capacity]

A B=8 continuous-slot policy MUST declare `slot_capacity = 8` and MUST reuse the B4 queue, decode-window, refill, bypass, deadline, and stall bounds. The core MUST accept it, the coordinator MUST open a width-8 slot group and a width-8 resident decode trace, and the package MUST export it beside the B4 policy without changing the B4 default. The profile MUST remain an unserved candidate until a separate measured admission receipt passes the existing queue-inclusive gates at eight slots.

#### Scenario: Candidate expressed and validated
- GIVEN the B=8 policy file
- WHEN the package exports it and the device-free probes run
- THEN the policy exports deterministically, the core accepts capacity 8, and unsupported capacities remain refused.

#### Scenario: Served default unchanged
- GIVEN a package that carries both policies
- WHEN the service starts without an explicit policy argument
- THEN the B4 profile is the one it serves.

#### Scenario: Widened admission is not assumed
- GIVEN a B=8 policy with no measured receipt
- WHEN an admission verdict is requested
- THEN the profile is treated as a candidate, not as an admitted serving default.
