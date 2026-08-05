# Negative fixture: replica equality must be exact for this runtime.
let contracts = import "../contracts.ncl" in
let base = import "base-profile.ncl" in
(std.record.remove "replica_policy" base) & {
  replica_policy = (std.record.remove "equality" base.replica_policy) & {
    equality = "tolerance",
  },
}
| contracts.ProfileClosed | contracts.Profile
