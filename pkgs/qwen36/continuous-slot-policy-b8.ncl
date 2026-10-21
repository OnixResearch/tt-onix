# Typed policy for the Qwen3.8 P150x2 continuous-slot B=8 scheduler candidate.
#
# r[impl tenstorrent.qwen38_p150x2.continuous_slot_policy]
#
# The pure core (qwen36_service/continuous_slot_core.py) reads these bounds; the
# KV memory limit is the decode profile's kv_pool_blocks, which the adapter turns
# into the block range the core allocates from. Nothing here is a performance
# claim: the profile is a device-free candidate until the queue-inclusive
# comparison (task s6) admits it.
let contracts = import "continuous-slot-contracts-b8.ncl" in
{
  schema_version = 1,
  profile_id = "qwen38-p150x2-continuous-slot-b8-v1",
  # Rows of the one resident width-8 decode trace. Candidate profile only: the B4
  # policy remains the served default until a B=8 admission receipt passes.
  slot_capacity = 8,
  # Requests waiting for a slot; one more is refused instead of queued.
  queue_capacity = 32,
  # Decode steps between two refill boundaries. A slot freed by an end-of-sequence
  # token waits at most this many steps for its refill.
  decode_window_steps = 16,
  # Prefills beside active users at one boundary.
  refills_per_boundary = 1,
  # Prompt tokens those prefills may add at one boundary; the first refill is always
  # allowed, so a long prompt is delayed by the budget but never starved by it.
  refill_prompt_token_budget = 8192,
  # How often later requests may overtake a queue head that does not fit yet.
  maximum_head_bypasses = 2,
  # A request still queued after this long is refused.
  maximum_queue_seconds = 1800,
  # One device effect running longer than this quarantines the engine.
  worker_stall_seconds = 900,
}
| contracts.Closed
| contracts.Coherent
| contracts.Fields
