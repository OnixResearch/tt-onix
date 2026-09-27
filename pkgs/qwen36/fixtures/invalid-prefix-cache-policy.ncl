# Negative fixture: a minimum prefix that does not end on a prefill chunk boundary.
let contracts = import "prefix-cache-contracts.ncl" in
{
  schema_version = 1,
  profile_id = "qwen38-p150x2-prefix-cache-invalid",
  state_slots = 4,
  minimum_prefix_tokens = 1500,
  maximum_cached_blocks = 1024,
  tenant = "local",
}
| contracts.Closed
| contracts.Fields
