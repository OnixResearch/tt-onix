let contracts = import "../throughput-contracts.ncl" in
let invalid
  | contracts.ThroughputProfile
  | contracts.WindowSixteenV2Binding
  = {
    schema_version = 1,
    profile_id = "rwkv7-p150x2-window16-candidate-v2-throughput-v1",
    model_profile_blake3 = "d33ed1051c7595327368cce7b42e16f025e8fc78d96addb0353fba2e86314a88",
    persistent_profile_id = "rwkv7-p150x2-persistent-decode-window16-candidate-v2",
    persistent_profile_blake3 = "0fd161f7172276d1efbe51623b1d110947e68de40024109618547cf7a42bb01a",
    repeat_count = 5,
    workload = {
      request_count = 4,
      short_prompt_token_count = 8,
      long_prompt_token_count = 8,
      generated_token_count = 16,
      homogeneous_prompts = true,
    },
    scheduler = {
      mode = "deterministic_continuous_batching",
      tie_break = "least_loaded_card_then_lowest_card_then_lowest_slot",
      per_card_slot_count = 4,
    },
    runtime = {
      replica_mode = "independent_card_replicas",
      recurrent_state_residency = "device_fp32",
    },
    timing = {
      timed_region = "decode_only",
      warmup_token_count = 4,
      exclude_state_readback = true,
      exclude_profiler_flush = true,
      exclude_control_replay = true,
      exclude_artifact_writes = true,
    },
    admission = {
      minimum_dual_card_speedup = 1.0,
      maximum_p99_latency_regression_fraction = 0.10,
      require_physical_hardware = true,
      require_profiler_attribution = true,
      machine_fact_source = "tt-smi -s --snapshot_no_tty",
      one_card_physical_device_id = 0,
      two_card_physical_device_ids = [0, 1],
    },
  }
in
invalid
