let contracts = import "../contract.ncl" in
let base = import "../profiles.ncl" in
let first = std.array.first base.profiles in
let changed = std.record.remove "consumer" first in
let changed_profiles = std.array.map
  (fun profile => if profile.target_name == first.target_name then changed else profile)
  base.profiles
in
(base & { profiles | force = changed_profiles }) | contracts.Profiles
