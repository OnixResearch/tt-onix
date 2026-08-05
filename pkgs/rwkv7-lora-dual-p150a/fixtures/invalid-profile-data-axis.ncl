# Negative fixture: the logical data axis must have exactly two replicas.
let contracts = import "../contracts.ncl" in
let base = import "base-profile.ncl" in
(std.record.remove "data_axis" base) & {
  data_axis = (std.record.remove "replicas" base.data_axis) & { replicas = 4 },
}
| contracts.ProfileClosed | contracts.Profile
