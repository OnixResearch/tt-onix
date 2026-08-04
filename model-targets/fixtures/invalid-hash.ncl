let contracts = import "../contract.ncl" in
let base = import "../profiles.ncl" in
let first = std.array.first base.profiles in
let changed_artifacts = std.array.map
  (fun artifact =>
    if artifact.role == "checkpoint" then artifact & { blake3 | force = "bad" }
    else artifact)
  first.artifacts
in
let changed = first & { artifacts | force = changed_artifacts } in
let changed_profiles = std.array.map
  (fun profile => if profile.target_name == first.target_name then changed else profile)
  base.profiles
in
(base & { profiles | force = changed_profiles }) | contracts.Profiles
