# Negative fixture: every tolerance role must be present. The tolerances
# field is replaced whole so the recursive merge cannot restore the field.
let contracts = import "../contracts.ncl" in
let base = import "base-profile.ncl" in
(std.record.remove "tolerances" base) & {
  tolerances = std.record.remove "adapter_delta" base.tolerances,
}
| contracts.ProfileClosed | contracts.Profile
