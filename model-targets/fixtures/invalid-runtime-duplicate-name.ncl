# SPDX-License-Identifier: LGPL-2.1-only
# r[verify tenstorrent.runtime_profile.registry.unique]
# Two profiles with the same name make a lookup ambiguous; the registry refuses it.
let contracts = import "../runtime-contract.ncl" in
let base = import "../runtime-profiles.ncl" in
let qwen = std.array.first base.profiles in
(base & { profiles | force = base.profiles @ [ qwen ] })
| contracts.Profiles
| contracts.Semantics
| contracts.Unique
