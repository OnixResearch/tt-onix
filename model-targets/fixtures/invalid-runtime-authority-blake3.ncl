# SPDX-License-Identifier: LGPL-2.1-only
# r[verify tenstorrent.runtime_profile.contract.hash]
# Authority digests are lowercase hexadecimal BLAKE3 text; uppercase must fail.
let contracts = import "../runtime-contract.ncl" in
let base = import "../runtime-profiles.ncl" in
let qwen = std.array.first base.profiles in
let changed = qwen & {
  checkpoint | force = qwen.checkpoint & {
    authority_blake3 | force = "C319F7EEF1B58911457C128CA5397710F4C9AF3B98D1CCC83914005F9D3726CD"
  }
} in
let changed_profiles = std.array.map
  (fun profile => if profile.name == qwen.name then changed else profile)
  base.profiles
in
(base & { profiles | force = changed_profiles })
| contracts.Profiles
| contracts.Semantics
| contracts.Unique
