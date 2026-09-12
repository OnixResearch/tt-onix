let contracts = import "batch-admission-b16-contracts.ncl" in
(import "batch-admission-b16-policy.ncl")
| contracts.PolicyClosed
| contracts.PolicySemantic
| contracts.Policy
