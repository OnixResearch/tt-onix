let contracts = import "batch-admission-contracts.ncl" in
(import "batch-admission-policy.ncl")
| contracts.PolicyClosed
| contracts.PolicySemantic
| contracts.Policy
