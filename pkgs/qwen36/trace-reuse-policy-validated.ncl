let contracts = import "trace-reuse-contracts.ncl" in
(import "trace-reuse-policy.ncl")
| contracts.PolicyClosed
| contracts.PolicySemantic
| contracts.Policy
