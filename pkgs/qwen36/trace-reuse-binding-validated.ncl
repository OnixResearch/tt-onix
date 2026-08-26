let contracts = import "trace-reuse-contracts.ncl" in
(import "trace-reuse-binding.ncl")
| contracts.BindingClosed
| contracts.BindingSemantic
| contracts.Binding
