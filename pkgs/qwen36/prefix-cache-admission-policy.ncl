# Predeclared gates for admitting the hybrid prefix-cache profile (tasks p6 and p7).
#
# r[impl tenstorrent.qwen38_p150x2.hybrid_prefix_admission]
#
# Cold roots (continuous slots, no cache) and cached roots run the same agent-shaped
# conversations on one package, each root in a fresh process. The first workload has
# fewer conversations than state buffers, so every turn after the first resumes. The
# second has more, so turns miss after eviction and must cost no more than cold prefill
# plus the save.
#
# The host that serves Qwen also runs unrelated work, and its load drifts. Version 1
# compared one cold root with one cached root measured ten minutes apart, so drift and
# cache cost were confounded. Version 2 keeps every gate bound and measures four roots
# per workload in ABBA order: cold, cached, cached, cold. Each gate compares, turn by
# turn, the mean of the two cached roots with the mean of the two cold roots, which
# cancels load that drifts linearly across the four roots. Fixed before the physical
# measurement.
let contracts = import "prefix-cache-admission-contracts.ncl" in
{
  schema_version = 2,
  profile_id = "qwen38-p150x2-prefix-cache-admission-v2",
  prefix_cache_profile_id = "qwen38-p150x2-prefix-cache-v1",
  root_order = ["cold", "cached", "cached", "cold"],
  workloads = [
    { conversations = 3, turns = 4 },
    { conversations = 5, turns = 2 },
  ],
  gates = {
    # One package, runtime and device pair; roots measured in root_order, and both
    # cached roots resuming the same prefix on every turn.
    require_matched_identity = true,
    # Every turn decodes exactly the same tokens in all four roots.
    require_exact_parity = true,
    require_resumed_hits = true,
    # A resumed turn's first token, and its whole request, must be clearly faster.
    maximum_hit_median_ttft_ratio = 0.7,
    maximum_hit_median_wall_ratio = 0.85,
    # A miss (a first turn, or after eviction) pays only for saving state.
    maximum_miss_median_wall_ratio = 1.10,
    maximum_miss_wall_ratio = 1.25,
    # Four saved states, about 72 MiB of recurrent state each per chip, plus the carry.
    maximum_startup_dram_drop_bytes = 536870912,
  },
}
| contracts.Closed
| contracts.Fields
| { gates | contracts.GatesClosed, .. }
