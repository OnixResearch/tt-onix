# r[verify tenstorrent.model_targets.hybrid_profile_contract.invalid]
let base = import "../profiles.ncl" in
let candidate_name = "qwen3-6-35b-a3b-p150x1-candidate" in
let changed_profiles =
  std.array.map
    (fun profile =>
      if profile.target_name == candidate_name then
        profile
        & {
          state.components | force = [
            (std.array.first profile.state.components) & { layer_count | force = 40 },
          ],
        }
      else
        profile
    )
    base.profiles
in
base & { profiles | force = changed_profiles }
