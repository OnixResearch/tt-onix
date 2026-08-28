let contracts = import "kernel-distribution-admission-contracts.ncl" in
(import "kernel-distribution-admission-policy.ncl")
| contracts.PolicyClosed
| contracts.PolicySemantic
| contracts.Policy
