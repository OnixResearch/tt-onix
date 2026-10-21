## ADDED Requirements

### Requirement: The hybrid prefix cache is admitted for the scheduler profile it was measured with

r[tenstorrent.qwen38_p150x2.hybrid_prefix_admission]

An admission of the hybrid prefix cache MUST name the continuous-slot scheduler
profile that every measurement root served, and a root that omits that profile or
names another one MUST be rejected. The gate set of the earlier schema MUST remain
bound: the same ABBA root order, the same workloads, the same hit, miss, parity, and
startup-DRAM bounds. A schema that binds the scheduler MUST NOT invalidate the roots
and receipts of the earlier schemas.

#### Scenario: Roots served the named profile
- GIVEN a policy that names a continuous-slot profile
- WHEN four roots that all reported that profile are compared
- THEN the admission is decided by the hit, miss, parity, and memory gates alone.

#### Scenario: A root served another profile
- GIVEN a policy that names a continuous-slot profile
- WHEN any root of the four reports a different profile or reports none
- THEN the admission is rejected with the scheduler mismatch as its reason.

#### Scenario: Earlier receipts stay verifiable
- GIVEN the roots and the policy of an earlier schema
- WHEN they are read by the admission core
- THEN they validate exactly as they did before, and the older schema's receipt is
  still reproducible from them.

### Requirement: The served package carries the receipts of the features it serves

r[tenstorrent.qwen38_p150x2.served_feature_receipts]

A package that serves a feature MUST carry that feature's admission receipt for its
own package and execution identity. When a package change adds an admission policy
for a feature that another package's receipt already covers, the scheduler admission
of that package MUST be measured again, and the served profile MUST move to the
package that carries both receipts.

#### Scenario: A new package changes the served evidence
- GIVEN a package that adds an admission policy to the one the service serves
- WHEN the served profile is moved to it
- THEN the scheduler admission of that package has been measured, and both receipts
  name it.
