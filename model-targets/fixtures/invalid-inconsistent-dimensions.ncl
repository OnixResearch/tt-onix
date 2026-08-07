let base = import "../profiles.ncl" in
let first = std.array.first base.profiles in
let changed = first & {
  consumer.dimensions.head_count | force = 11,
} in
let changed_profiles = std.array.map
  (fun profile => if profile.target_name == first.target_name then changed else profile)
  base.profiles
in
base & { profiles | force = changed_profiles }
