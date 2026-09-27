# Typed, fail-closed contracts for the Qwen3.8 P150x2 continuous-slot policy.
let PositiveIntC = fun maximum => std.contract.from_predicate (fun value =>
  std.is_number value
  && std.number.is_integer value
  && value > 0
  && value <= maximum
) in
let NonNegativeIntC = fun maximum => std.contract.from_predicate (fun value =>
  std.is_number value
  && std.number.is_integer value
  && value >= 0
  && value <= maximum
) in
let PositiveSecondsC = std.contract.from_predicate (fun value =>
  std.is_number value && value > 0 && value <= 86400
) in
let ClosedRecordC = fun allowed => std.contract.from_predicate (fun value =>
  std.is_record value
  && std.array.length (std.record.fields value) == std.array.length allowed
  && std.array.all (fun field => std.array.elem field allowed) (std.record.fields value)
) in
let RefillWithinCapacityC = std.contract.from_predicate (fun value =>
  value.refills_per_boundary <= value.slot_capacity
) in
{
  Closed = ClosedRecordC [
    "schema_version",
    "profile_id",
    "slot_capacity",
    "queue_capacity",
    "decode_window_steps",
    "refills_per_boundary",
    "refill_prompt_token_budget",
    "maximum_head_bypasses",
    "maximum_queue_seconds",
    "worker_stall_seconds",
  ],
  Coherent = RefillWithinCapacityC,
  Fields = {
    schema_version | std.contract.from_predicate (fun value => value == 1),
    profile_id | std.contract.from_predicate (fun value =>
      std.is_string value && std.string.is_match "^[a-z0-9][a-z0-9-]*$" value
    ),
    # B4 only: the rows of one resident decode trace. Wider traces are out of scope.
    slot_capacity | std.contract.from_predicate (fun value => value == 4),
    queue_capacity | PositiveIntC 256,
    decode_window_steps | PositiveIntC 256,
    refills_per_boundary | PositiveIntC 4,
    refill_prompt_token_budget | PositiveIntC 65536,
    maximum_head_bypasses | NonNegativeIntC 16,
    maximum_queue_seconds | PositiveSecondsC,
    worker_stall_seconds | PositiveSecondsC,
  },
}
