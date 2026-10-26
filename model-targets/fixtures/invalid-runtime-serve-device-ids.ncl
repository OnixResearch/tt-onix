# SPDX-License-Identifier: LGPL-2.1-only
# r[verify tenstorrent.runtime_profile.contract.serve_mesh]
# A serve profile owns both P150 cards in the admitted order, exactly [0, 1].
# Reversing the order keeps the same device set but breaks the recorded order.
let contracts = import "../runtime-contract.ncl" in
let base = import "../runtime-profiles.ncl" in
let qwen = std.array.first base.profiles in
let changed = qwen & {
  mesh | force = qwen.mesh & { physical_device_ids | force = [ 1, 0 ] }
} in
let changed_profiles = std.array.map
  (fun profile => if profile.name == qwen.name then changed else profile)
  base.profiles
in
(base & { profiles | force = changed_profiles })
| contracts.Profiles
| contracts.Semantics
| contracts.Unique
