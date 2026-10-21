# Predeclared gates for admitting the hybrid prefix-cache profile in the served
# continuous-slot configuration (schema 5).
#
# r[impl tenstorrent.qwen38_p150x2.hybrid_prefix_admission]
#
# Every gate of schema 2 stays bound: the same ABBA root order, the same two
# workloads, the same hit, miss, parity, and startup-DRAM bounds. Schema 5 adds
# one thing: the policy names the continuous-slot scheduler profile that every
# root must have served, so a receipt cannot be read as covering a scheduler the
# roots did not run. The prefix cache and the scheduler profile are separate
# features, and the deployed configuration pairs this cache with the eight-slot
# scheduler, which the earlier schema-2 receipt did not cover.
#
# Fixed before the physical measurement.
let contracts = import "prefix-cache-admission-contracts.ncl" in
# The scheduler profile is taken from the packaged scheduler policy rather than
# written out here, so the bound identifier cannot drift from the profile the
# service actually reports.
let slot = import "continuous-slot-policy-b8.ncl" in
{
  schema_version = 5,
  profile_id = "qwen38-p150x2-prefix-cache-admission-v5",
  prefix_cache_profile_id = "qwen38-p150x2-prefix-cache-v1",
  # The continuous-slot profile every root must report, from the service's own
  # decode evidence.
  scheduler_profile_id = slot.profile_id,
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
| contracts.V5Closed
| contracts.V5Fields
| { gates | contracts.GatesClosed, .. }
