# SPDX-License-Identifier: LGPL-2.1-only
# r[verify tenstorrent.runtime_profile.contract.mode]
# `mode` is a closed set of serve and job.
let contracts = import "../runtime-contract.ncl" in
let base = import "../runtime-profiles.ncl" in
let qwen = std.array.first base.profiles in
let changed = qwen & { mode | force = "daemon" } in
let changed_profiles = std.array.map
  (fun profile => if profile.name == qwen.name then changed else profile)
  base.profiles
in
(base & { profiles | force = changed_profiles })
| contracts.Profiles
| contracts.Semantics
| contracts.Unique
