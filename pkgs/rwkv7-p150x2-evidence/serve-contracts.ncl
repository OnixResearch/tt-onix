# Typed, fail-closed contracts for the bounded RWKV-7 inference service.
let PositiveIntC = std.contract.from_predicate (fun value =>
  std.is_number value && value > 0 && std.number.is_integer value
) in
let NonNegativeIntC = std.contract.from_predicate (fun value =>
  std.is_number value && value >= 0 && std.number.is_integer value
) in
let NonEmptyStringC = std.contract.from_predicate (fun value =>
  std.is_string value && std.string.length value > 0
) in
let ExactC = fun expected => std.contract.from_predicate (fun value =>
  value == expected
) in
let ClosedRecordC = fun allowed => std.contract.from_predicate (fun value =>
  std.is_record value
  && std.array.length (std.record.fields value) == std.array.length allowed
  && std.array.all (fun field => std.array.elem field allowed) (std.record.fields value)
) in
let CapacityC = {
  id | NonEmptyStringC,
  per_card_slot_count | PositiveIntC,
  temporary_tensor_bytes_per_slot | PositiveIntC,
  device_budget_bytes | PositiveIntC,
  safety_reserve_bytes | PositiveIntC,
} in
let CapacityClosedC = ClosedRecordC [
  "id",
  "per_card_slot_count",
  "temporary_tensor_bytes_per_slot",
  "device_budget_bytes",
  "safety_reserve_bytes",
] in
let LimitsC = {
  maximum_prompt_tokens | PositiveIntC,
  strict_decode_windows | Array PositiveIntC,
  experimental_maximum_generated_tokens | PositiveIntC,
  maximum_message_bytes | PositiveIntC,
  maximum_http_header_bytes | PositiveIntC,
  maximum_http_body_bytes | PositiveIntC,
  maximum_active_connections | PositiveIntC,
  maximum_lifetime_connections | PositiveIntC,
  connection_io_timeout_seconds | PositiveIntC,
  worker_timeout_seconds | PositiveIntC,
  barrier_poll_milliseconds | PositiveIntC,
} in
let LimitsClosedC = ClosedRecordC [
  "maximum_prompt_tokens",
  "strict_decode_windows",
  "experimental_maximum_generated_tokens",
  "maximum_message_bytes",
  "maximum_http_header_bytes",
  "maximum_http_body_bytes",
  "maximum_active_connections",
  "maximum_lifetime_connections",
  "connection_io_timeout_seconds",
  "worker_timeout_seconds",
  "barrier_poll_milliseconds",
] in
let ServerC = {
  listen_address | ExactC "127.0.0.1:8080",
  openai_completions_path | ExactC "/v1/completions",
  health_path | ExactC "/health",
} in
let ServerClosedC = ClosedRecordC ["listen_address", "openai_completions_path", "health_path"] in
let ProfileC = {
  schema_version | ExactC 1,
  profile_id | NonEmptyStringC,
  model | ExactC "RWKV/RWKV7-Goose-World2.8-0.1B-HF",
  server | ServerClosedC | ServerC,
  limits | LimitsClosedC | LimitsC,
  devices | ExactC [
    { physical_device_id = 0, primary_thread_cpu = 2 },
    { physical_device_id = 1, primary_thread_cpu = 6 },
  ],
  strict_capacity | CapacityClosedC | CapacityC,
  experimental_capacity | CapacityClosedC | CapacityC,
  primary_thread_fifo_priority | PositiveIntC,
  default_mode | ExactC "strict",
  retain_request_artifacts | ExactC false,
} in
let ProfileClosedC = ClosedRecordC [
  "schema_version",
  "profile_id",
  "model",
  "server",
  "limits",
  "devices",
  "strict_capacity",
  "experimental_capacity",
  "primary_thread_fifo_priority",
  "default_mode",
  "retain_request_artifacts",
] in
let ProfileSemanticC = std.contract.from_predicate (fun value =>
  value.limits.strict_decode_windows == [2, 4, 8]
  && value.limits.experimental_maximum_generated_tokens > 8
  && value.strict_capacity.per_card_slot_count == 4
  && value.experimental_capacity.per_card_slot_count == 16
  && value.devices == [
    { physical_device_id = 0, primary_thread_cpu = 2 },
    { physical_device_id = 1, primary_thread_cpu = 6 },
  ]
) in
{
  Profile = ProfileC,
  ProfileClosed = ProfileClosedC,
  ProfileSemantic = ProfileSemanticC,
}
