# Closed profile composed from immutable execution policy and mutable receipt binding.
let contracts = import "trace-reuse-contracts.ncl" in
let policy = import "trace-reuse-policy.ncl" in
let binding = import "trace-reuse-binding.ncl" in
let profile = policy & {
  physically_admitted = binding.physically_admitted,
  identities = policy.identities & { receipt_blake3 = binding.receipt_blake3 },
} in
profile
| contracts.ProfileClosed
| contracts.ProfileSemantic
| contracts.Profile
