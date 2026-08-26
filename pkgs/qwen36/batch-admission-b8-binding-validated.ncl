let contracts = import "batch-admission-b8-contracts.ncl" in
(import "batch-admission-b8-binding.ncl")
| contracts.BindingClosed
| contracts.BindingSemantic
| contracts.Binding
