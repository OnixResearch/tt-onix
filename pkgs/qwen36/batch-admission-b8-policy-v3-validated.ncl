let contracts = import "batch-admission-b8-contracts-v3.ncl" in
(import "batch-admission-b8-policy-v3.ncl")
| contracts.PolicyClosed
| contracts.PolicySemantic
| contracts.Policy
