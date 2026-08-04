# Negative fixture: an unknown precision class has no named comparison
# policy. Admission must fail with a contract violation before
# compilation.
let contracts = import "../contracts.ncl" in
let base = import "base-profile.ncl" in
let stripped = std.record.remove "precision" base in
(
  stripped
  & {
    precision = {
      precision_class = "fp8_mixed",
      model_dtype = "bf16",
      optimizer_dtype = "bf16",
      comparison_dtype = "fp32",
    },
  }
) | contracts.ProfileClosed
  | contracts.Profile
