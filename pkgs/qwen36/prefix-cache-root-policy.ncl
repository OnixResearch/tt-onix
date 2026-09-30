# Opt-in Qwen3.8 P150x2 shared system/tool root; the deployed v1 profile is unchanged.
# One root is soft-retained. The first eligible prompt saves its system/tool prefix;
# later requests resume there and can save a conversation prefix in another state slot.
let contracts = import "prefix-cache-contracts.ncl" in
{
  schema_version = 2,
  profile_id = "qwen38-p150x2-prefix-cache-root-v2",
  state_slots = 4,
  minimum_prefix_tokens = 2048,
  maximum_cached_blocks = 1024,
  tenant = "local",
  shared_root_minimum_tokens = 2048,
}
| contracts.RootClosed
| contracts.RootFields
