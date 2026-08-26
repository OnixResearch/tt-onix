let contracts = import "batch-admission-b8-contracts.ncl" in
(import "batch-admission-b8-policy.ncl")
| contracts.PolicyClosed
| contracts.PolicySemantic
| contracts.Policy
