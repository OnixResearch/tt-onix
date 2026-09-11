# Production policy for the bounded RWKV-7 P150x2 text service.
let contracts = import "serve-contracts.ncl" in
let configured_schema_version = 1 in
let configured_maximum_prompt_tokens = 32 in
let configured_strict_decode_windows = [2, 4, 8] in
let configured_experimental_maximum_generated_tokens = 64 in
let configured_maximum_message_bytes = 256 in
let configured_maximum_http_header_bytes = 16384 in
let configured_maximum_http_body_bytes = 262144 in
let configured_maximum_active_connections = 32 in
let configured_maximum_lifetime_connections = 1000000 in
let configured_connection_io_timeout_seconds = 30 in
let configured_worker_timeout_seconds = 900 in
let configured_barrier_poll_milliseconds = 10 in
let configured_temporary_tensor_bytes_per_slot = 5242880 in
let configured_device_budget_bytes = 3221225472 in
let configured_safety_reserve_bytes = 268435456 in
{
  schema_version = configured_schema_version,
  profile_id = "rwkv7-p150x2-serve-v1",
  model = "RWKV/RWKV7-Goose-World2.8-0.1B-HF",
  server = {
    listen_address = "127.0.0.1:8080",
    openai_completions_path = "/v1/completions",
    health_path = "/health",
  },
  limits = {
    maximum_prompt_tokens = configured_maximum_prompt_tokens,
    strict_decode_windows = configured_strict_decode_windows,
    experimental_maximum_generated_tokens = configured_experimental_maximum_generated_tokens,
    maximum_message_bytes = configured_maximum_message_bytes,
    maximum_http_header_bytes = configured_maximum_http_header_bytes,
    maximum_http_body_bytes = configured_maximum_http_body_bytes,
    maximum_active_connections = configured_maximum_active_connections,
    maximum_lifetime_connections = configured_maximum_lifetime_connections,
    connection_io_timeout_seconds = configured_connection_io_timeout_seconds,
    worker_timeout_seconds = configured_worker_timeout_seconds,
    barrier_poll_milliseconds = configured_barrier_poll_milliseconds,
  },
  devices = [
    { physical_device_id = 0, primary_thread_cpu = 2 },
    { physical_device_id = 1, primary_thread_cpu = 6 },
  ],
  strict_capacity = {
    id = "B4",
    per_card_slot_count = 4,
    temporary_tensor_bytes_per_slot = configured_temporary_tensor_bytes_per_slot,
    device_budget_bytes = configured_device_budget_bytes,
    safety_reserve_bytes = configured_safety_reserve_bytes,
  },
  experimental_capacity = {
    id = "B16",
    per_card_slot_count = 16,
    temporary_tensor_bytes_per_slot = configured_temporary_tensor_bytes_per_slot,
    device_budget_bytes = configured_device_budget_bytes,
    safety_reserve_bytes = configured_safety_reserve_bytes,
  },
  primary_thread_fifo_priority = 1,
  default_mode = "strict",
  retain_request_artifacts = false,
} | contracts.ProfileClosed | contracts.Profile | contracts.ProfileSemantic
