# Opt-in policy for the Qwen3.8 P150x2 hybrid generation prefix cache.
#
# r[impl tenstorrent.qwen38_p150x2.hybrid_prefix_admission]
#
# An agent turn resends the whole conversation, so its prompt extends the previous
# turn's. An eligible long prompt saves the carried GDN state at the longest
# chunk boundary below its end that fits the KV budget, keeping the preceding
# KV blocks for the next turn. Entries are in process only and lost on restart.
let contracts = import "prefix-cache-contracts.ncl" in
{
  schema_version = 1,
  profile_id = "qwen38-p150x2-prefix-cache-v1",
  # Four saved states: about 4 x 72 MiB of recurrent state per chip.
  state_slots = 4,
  # Shorter prefixes are prefilled cold.
  minimum_prefix_tokens = 2048,
  # KV the cache may hold: 1,024 blocks of 64 tokens, a quarter of the pool. Live
  # requests evict unused entries whenever they need the blocks.
  maximum_cached_blocks = 1024,
  # The service has no client authentication: every request is this one tenant.
  tenant = "local",
}
| contracts.Closed
| contracts.Fields
