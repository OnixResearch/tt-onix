# Negative fixture: uneven shards are not admitted for this profile.
let contracts = import "../contracts.ncl" in
let base = import "base-profile.ncl" in
(std.record.remove "shard_policy" base) & {
  shard_policy = (std.record.remove "allow_uneven" base.shard_policy) & {
    allow_uneven = true,
  },
}
| contracts.ProfileClosed | contracts.Profile
