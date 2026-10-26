# SPDX-License-Identifier: LGPL-2.1-only
# r[verify tenstorrent.runtime_profile.contract.closed]
# A runtime profile must not accept a field the contract does not declare.
let contracts = import "../runtime-contract.ncl" in
let base = import "../runtime-profiles.ncl" in
let qwen = std.array.first base.profiles in
let changed = qwen & { undocumented_launch_flag | force = "--fast" } in
let changed_profiles = std.array.map
  (fun profile => if profile.name == qwen.name then changed else profile)
  base.profiles
in
(base & { profiles | force = changed_profiles })
| contracts.Profiles
| contracts.Semantics
| contracts.Unique
