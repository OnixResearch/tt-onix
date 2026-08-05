# Negative fixture: adapter gradients must reduce with the mean operation.
let contracts = import "../contracts.ncl" in
let base = import "base-profile.ncl" in
(std.record.remove "reduction_policy" base) & {
  reduction_policy = (std.record.remove "gradient_op" base.reduction_policy) & {
    gradient_op = "sum",
  },
}
| contracts.ProfileClosed | contracts.Profile
