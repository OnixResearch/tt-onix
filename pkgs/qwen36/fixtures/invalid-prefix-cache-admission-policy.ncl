# Negative fixture: an admission that does not require exact parity.
#
#
# A cold root (continuous slots, no cache) and a cached root run the same agent-shaped
# conversations on one package. The first workload has fewer conversations than state
# buffers, so every turn after the first resumes. The second has more, so turns miss
# after eviction and must cost no more than cold prefill plus the save. Fixed before the
# physical measurement.
let contracts = import "prefix-cache-admission-contracts.ncl" in
{
  schema_version = 1,
  profile_id = "qwen38-p150x2-prefix-cache-admission-v1",
  prefix_cache_profile_id = "qwen38-p150x2-prefix-cache-v1",
  workloads = [
    { conversations = 3, turns = 4 },
    { conversations = 5, turns = 2 },
  ],
  gates = {
    require_matched_identity = true,
    # Every turn decodes exactly the tokens it decodes with cold prefill.
    require_exact_parity = false,
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
