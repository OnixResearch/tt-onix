# Closed B=8 profile composed from the immutable interleaved execution policy and
# the mutable receipt binding.
let contracts = import "batch-admission-b8-contracts-v3.ncl" in
let policy = import "batch-admission-b8-policy-v3.ncl" in
let binding = import "batch-admission-b8-binding-v3.ncl" in
let profile = policy & {
  physically_admitted = binding.physically_admitted,
  identities = policy.identities & { receipt_blake3 = binding.receipt_blake3 },
} in
profile
| contracts.ProfileClosed
| contracts.ProfileSemantic
| contracts.Profile
