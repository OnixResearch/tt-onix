# Diagnostic-only same-process TTFT control for Qwen3.8-27B on P150x2.
let contracts = import "ttft-control-contracts.ncl" in
let profile_schema_version = 1 in
let configured_mesh_rows = 1 in
let configured_mesh_columns = 2 in
let first_pcie_device_id = 0 in
let second_pcie_device_id = 1 in
let configured_minimum_cpu = 8 in
let configured_maximum_cpu = 31 in
let configured_trace_region_size_bytes = 1073741824 in
let configured_maximum_sequence_length = 131072 in
let configured_kv_block_size = 64 in
let configured_request_cache_blocks = 2 in
let configured_trace_capacity_blocks = 2048 in
let configured_page_table_blocks = 2 in
let configured_generated_tokens = 64 in
let configured_warmup_samples = 1 in
let configured_retained_samples = 5 in
let configured_maximum_within_root_spread_ratio = 1.25 in
let configured_maximum_duplicate_root_ratio = 1.25 in
let configured_material_regression_ratio = 1.05 in
let configured_phase_sum_tolerance_seconds = 0.000001 in
{
  schema_version = profile_schema_version,
  profile_id = "qwen38-p150x2-traced-ttft-control-v1",
  diagnostic_only = true,
  identities = {
    authority_blake3 = "c319f7eef1b58911457c128ca5397710f4c9af3b98d1ccc83914005f9d3726cd",
    checkpoint_revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    model_source_revision = "d102e7f4512060e09e580e40cf26a870bfa88c55",
    tokenizer_revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
  },
  topology = {
    expected_pcie_device_ids = [first_pcie_device_id, second_pcie_device_id],
    mesh_rows = configured_mesh_rows,
    mesh_columns = configured_mesh_columns,
    minimum_cpu = configured_minimum_cpu,
    maximum_cpu = configured_maximum_cpu,
    trace_region_size_bytes = configured_trace_region_size_bytes,
    maximum_sequence_length = configured_maximum_sequence_length,
    kv_block_size = configured_kv_block_size,
    request_cache_blocks = configured_request_cache_blocks,
    trace_capacity_blocks = configured_trace_capacity_blocks,
    page_table_blocks = configured_page_table_blocks,
  },
  workload = {
    prompt = "The capital of France is",
    generated_tokens = configured_generated_tokens,
  },
  roots = {
    route_schedule = [
      "request-cache-untraced",
      "capacity-cache-untraced",
      "capacity-cache-traced",
      "capacity-cache-traced",
      "capacity-cache-untraced",
      "request-cache-untraced",
    ],
    warmup_samples = configured_warmup_samples,
    retained_samples = configured_retained_samples,
    maximum_within_root_spread_ratio = configured_maximum_within_root_spread_ratio,
    maximum_duplicate_root_ratio = configured_maximum_duplicate_root_ratio,
    material_regression_ratio = configured_material_regression_ratio,
    phase_sum_tolerance_seconds = configured_phase_sum_tolerance_seconds,
  },
} | contracts.Profile
