# Negative fixture: host-placed collectives are forbidden fallback.
let contracts = import "../contracts.ncl" in
let base = import "base-profile.ncl" in
(std.record.remove "reduction_policy" base) & {
  reduction_policy = (std.record.remove "placement" base.reduction_policy) & {
    placement = "host",
  },
}
| contracts.ProfileClosed | contracts.Profile
