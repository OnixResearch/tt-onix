# Closed B=16 profile composed from immutable execution policy and mutable receipt binding.
let contracts = import "batch-admission-b16-contracts.ncl" in
let policy = import "batch-admission-b16-policy.ncl" in
let binding = import "batch-admission-b16-binding.ncl" in
let profile = policy & {
  physically_admitted = binding.physically_admitted,
  identities = policy.identities & { receipt_blake3 = binding.receipt_blake3 },
} in
profile
| contracts.ProfileClosed
| contracts.ProfileSemantic
| contracts.Profile
