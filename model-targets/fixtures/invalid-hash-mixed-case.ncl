let contracts = import "../contract.ncl" in
let base = import "../profiles.ncl" in
let first = std.array.first base.profiles in
let changed_artifacts = std.array.map
  (fun artifact =>
    if artifact.role == "checkpoint" then artifact & {
      blake3 | force = "905f82048a64b881f9267117A398feb8a8a92bcc5233666bf67904e0d899d0e5"
    }
    else artifact)
  first.artifacts
in
let changed = first & { artifacts | force = changed_artifacts } in
let changed_profiles = std.array.map
  (fun profile => if profile.target_name == first.target_name then changed else profile)
  base.profiles
in
(base & { profiles | force = changed_profiles }) | contracts.Profiles
