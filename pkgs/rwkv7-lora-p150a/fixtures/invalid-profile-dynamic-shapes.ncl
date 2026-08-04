# Negative fixture: a non-positive static shape is a dynamic-shape proxy.
# Admission must fail with a contract violation before compilation.
let contracts = import "../contracts.ncl" in
let base = import "base-profile.ncl" in
let stripped = std.record.remove "static_shape" base in
(
  stripped
  & {
    static_shape = {
      micro_batch = 1,
      sequence_length = 0,
      gradient_accumulation_steps = 1,
    },
  }
) | contracts.ProfileClosed
  | contracts.Profile
