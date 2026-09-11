let contracts = import "batch-admission-contracts.ncl" in
(import "batch-admission-binding.ncl")
| contracts.BindingClosed
| contracts.BindingSemantic
| contracts.Binding
