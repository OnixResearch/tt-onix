# Typed, fail-closed contracts for the Qwen3.8 P150x2 hybrid prefix-cache policy.
let PositiveIntC = fun maximum => std.contract.from_predicate (fun value =>
  std.is_number value
  && std.number.is_integer value
  && value > 0
  && value <= maximum
) in
let IdC = std.contract.from_predicate (fun value =>
  std.is_string value && std.string.is_match "^[a-z0-9][a-z0-9-]*$" value
) in
let ChunkMultipleC = std.contract.from_predicate (fun value =>
  std.is_number value
  && std.number.is_integer value
  && value >= 1024
  && value % 1024 == 0
) in
let ClosedRecordC = fun allowed => std.contract.from_predicate (fun value =>
  std.is_record value
  && std.array.length (std.record.fields value) == std.array.length allowed
  && std.array.all (fun field => std.array.elem field allowed) (std.record.fields value)
) in
{
  Closed = ClosedRecordC [
    "schema_version",
    "profile_id",
    "state_slots",
    "minimum_prefix_tokens",
    "maximum_cached_blocks",
    "tenant",
  ],
  Fields = {
    schema_version | std.contract.from_predicate (fun value => value == 1),
    profile_id | IdC,
    # Saved carried-state buffers, allocated with the masked prefill traces.
    state_slots | PositiveIntC 16,
    # Prefill chunks are 1,024 tokens; a prefix ends on a chunk boundary.
    minimum_prefix_tokens | ChunkMultipleC,
    maximum_cached_blocks | PositiveIntC 4096,
    tenant | IdC,
  },
}
