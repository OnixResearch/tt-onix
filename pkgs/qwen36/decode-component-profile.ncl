let contracts = import "decode-component-contracts.ncl" in

{
  profile_id = "qwen38-p150x2-decode-component-replay-v1",
  diagnostic_only = true,
  execution_path = "matched-component-trace-replay",
  samples_per_group = 5,
  warmup_replays = 1,
  maximum_spread_ratio = 1.25,
  trace_capture_timeout_seconds = 300,
  closure = {
    minimum_ratio = 0.8,
    maximum_ratio = 1.25,
  },
  selection = {
    minimum_weighted_share = 0.35,
    minimum_dominance_ratio = 1.25,
  },
  identities = {
    authority_blake3 = "c319f7eef1b58911457c128ca5397710f4c9af3b98d1ccc83914005f9d3726cd",
    checkpoint_revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    model_source_revision = "d102e7f4512060e09e580e40cf26a870bfa88c55",
    tokenizer_revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
  },
  topology = {
    expected_pcie_device_ids = [ 0, 1 ],
    mesh_rows = 1,
    mesh_columns = 2,
    minimum_cpu = 8,
    maximum_cpu = 31,
    trace_region_size_bytes = 1073741824,
    maximum_sequence_length = 2048,
    kv_block_size = 64,
  },
  architecture = {
    layer_count = 64,
    gdn_layer_count = 48,
    full_attention_layer_count = 16,
    mlp_layer_count = 64,
  },
  representatives = {
    gdn = [ 0, 32, 60 ],
    full_attention = [ 3, 31, 63 ],
    mlp = [ 0, 31, 63 ],
  },
  prompt = "The capital of France is",
} | contracts.Profile
