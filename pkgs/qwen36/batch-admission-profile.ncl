# Closed profile composed from immutable execution policy and mutable receipt binding.
let contracts = import "batch-admission-contracts.ncl" in
let policy = import "batch-admission-policy.ncl" in
let binding = import "batch-admission-binding.ncl" in
let profile = policy & {
  physically_admitted = binding.physically_admitted,
  identities = policy.identities & { receipt_blake3 = binding.receipt_blake3 },
} in
profile
| contracts.ProfileClosed
| contracts.ProfileSemantic
| contracts.Profile
