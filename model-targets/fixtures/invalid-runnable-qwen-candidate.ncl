# r[verify tenstorrent.model_targets.qwen3_6_candidate_profile.runnable_rejected]
let base = import "../profiles.ncl" in
let candidate_name = "qwen3-6-35b-a3b-p150x1-candidate" in
let changed_profiles =
  std.array.map
    (fun profile =>
      if profile.target_name == candidate_name then
        profile & { runnable | force = true }
      else
        profile
    )
    base.profiles
in
base & { profiles | force = changed_profiles }
