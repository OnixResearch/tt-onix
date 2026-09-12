let contracts = import "batch-admission-b16-contracts.ncl" in
(import "batch-admission-b16-binding.ncl")
| contracts.BindingClosed
| contracts.BindingSemantic
| contracts.Binding
