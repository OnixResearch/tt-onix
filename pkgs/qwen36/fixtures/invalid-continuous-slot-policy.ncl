# Negative fixture: more refills per boundary than the policy has slots.
let contracts = import "continuous-slot-contracts.ncl" in
{
  schema_version = 1,
  profile_id = "qwen38-p150x2-continuous-slot-invalid",
  slot_capacity = 4,
  queue_capacity = 32,
  decode_window_steps = 16,
  refills_per_boundary = 5,
  refill_prompt_token_budget = 8192,
  maximum_head_bypasses = 2,
  maximum_queue_seconds = 1800,
  worker_stall_seconds = 900,
}
| contracts.Closed
| contracts.Coherent
| contracts.Fields
