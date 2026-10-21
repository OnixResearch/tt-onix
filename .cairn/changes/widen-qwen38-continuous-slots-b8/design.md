## Decisions

### Decision: a closed set, not an arbitrary capacity

**Choice:** Accept `slot_capacity` in {4, 8}; reject every other value in the pure core
and in the contracts.

**Rationale:** the resident decode trace and the KV pool are sized per profile. An
arbitrary capacity would let a policy request a trace the adapter cannot capture or a
pool the model cannot allocate. Two named capacities keep every accepted profile
testable and memory-checkable.

### Decision: duplicate the contracts rather than parameterize the frozen B4 file

**Choice:** add `continuous-slot-contracts-b8.ncl` and `continuous-slot-policy-b8.ncl`;
leave the B4 files byte-identical.

**Rationale:** the B4 profile and its admission receipt are historical artifacts. Editing
the B4 contract to accept another capacity would make the exported B4 JSON's provenance
ambiguous. A sibling file with one changed predicate is easier to audit.

### Decision: keep B4 as the served default

**Choice:** do not change the NixOS default or the served profile in this change.

**Rationale:** a wider trace changes memory and the inactive-row contract. Admission of
the B=8 profile is a separate measured decision; this change only makes the candidate
expressible, bound, and device-free validated.

### Decision: reuse the existing admission gates

**Choice:** the B=8 comparison uses the existing `continuous-slot-admission` gates and
workload shape, scaled to eight slots, with a new predeclared profile.

**Rationale:** the questions (short-request latency, tail, fairness, delivered-token
throughput, exact candidate parity) do not change with capacity. A new gate set would
make the two profiles incomparable.

## Risks / Trade-offs

- The resident width-8 trace needs more L1/DRAM than width 4; the smoke and the
  admission run must show it allocates and binds cleanly.
- A wider group has more inactive rows and a larger KV pool; the inactive-row and
  scratch-block contracts are the same, but the first smoke must prove them at 8.
- The B=8 policy is an execution identity input; a policy-only change still moves the
  package identity once it is bound.
