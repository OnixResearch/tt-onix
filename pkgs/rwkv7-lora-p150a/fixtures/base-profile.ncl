# Raw profile record without contract application. profile.ncl applies the
# admitted contracts; negative fixtures mutate this record before applying
# the same contracts.
let contracts = import "../contracts.ncl" in
{
  schema_version = 1,
  profile_id = "rwkv7-lora-p150a-single",
  bundle_id = contracts.AdmittedBundleId,
  model = {
    model_id = "RWKV/RWKV7-Goose-World2.8-0.1B-HF",
    revision = "d81965cb4e1a9f96696b4f70b84212b8f2e43216",
    checkpoint_blake3 = "905f82048a64b881f9267117a398feb8a8a92bcc5233666bf67904e0d899d0e5",
    layer_count = 12,
    hidden_size = 768,
    head_count = 12,
    head_size = 64,
    vocabulary_size = 65536,
    precision = "fp32_from_bf16",
  },
  tokenizer = {
    vocabulary_blake3 = "3997a74891dd68ced8daadae0d7475274b08988c9263ca042896c8106967aef2",
    vocabulary_entry_count = 65529,
  },
  device = {
    physical_count = 2,
    architecture = "blackhole",
    board_class = "p150a",
    device_root = "/dev/tenstorrent",
    logical_mesh = {
      rows = 1,
      columns = 1,
    },
    selection_ordinal = 0,
    container_device_isolation = "unsupported",
  },
  static_shape = {
    micro_batch = 1,
    sequence_length = 64,
    gradient_accumulation_steps = 1,
  },
  probe = {
    focused = {
      batch = 1,
      time = 4,
      heads = 1,
      width = 16,
    },
    representative = {
      batch = 1,
      time = 64,
      heads = 12,
      width = 64,
    },
  },
  precision = {
    precision_class = contracts.AdmittedPrecisionClass,
    model_dtype = "bf16",
    optimizer_dtype = "bf16",
    comparison_dtype = "fp32",
  },
  compile_limits = {
    probe_compile_timeout_seconds = 1200,
    full_compile_timeout_seconds = 3600,
    max_stablehlo_operations = 200000,
  },
  memory_limits = {
    max_device_bytes = 6000000000,
    max_host_bytes = 8000000000,
  },
  tolerances = {
    logits = {
      rtol = 0.02,
      atol = 0.1,
    },
    loss = {
      rtol = 0.01,
      atol = 0.05,
    },
    wkv7_output = {
      rtol = 0.02,
      atol = 0.1,
    },
    wkv7_state = {
      rtol = 0.02,
      atol = 0.1,
    },
    adapter_gradient = {
      rtol = 0.05,
      atol = 0.2,
    },
    gradient_norm = {
      rtol = 0.02,
      atol = 0.01,
    },
    clip_coefficient = {
      rtol = 0.02,
      atol = 0.01,
    },
    adapter_delta = {
      rtol = 0.05,
      atol = 0.001,
    },
  },
  run_criteria = {
    min_steps = 8,
    max_steps = 32,
    loss_improvement_ratio = 0.9,
    min_active_labels_per_sample = 1,
    require_finite = true,
  },
  health_limits = {
    max_board_temperature_c = 90,
    max_power_watts = 180,
    require_dram_ok = true,
  },
  timeouts = {
    hardware_attempt_seconds = 3600,
    multi_step_run_seconds = 3600,
  },
  lora = {
    rank = 2,
    scale = 1.0,
    target_roles = [
      "attention.receptance",
      "attention.key",
      "attention.value",
      "attention.output",
      "feed_forward.key",
      "feed_forward.value",
    ],
  },
  optimizer = {
    kind = "adamw",
    learning_rate = 0.01,
    gradient_accumulation_steps = 1,
    clip_norm = 1.0,
  },
  seed = 0,
  artifact_policy = {
    format = contracts.AdmittedAdapterFormat,
    artifact_root = "/var/lib/tenstorrent-training/artifacts/rwkv7-lora-p150a",
  },
  evidence_policy = {
    hash_algorithm = "blake3",
    exclude = ["credentials", "raw_logs", "serial_numbers", "absolute_private_paths"],
    required_receipts = [
      "prerequisite",
      "compile_probe",
      "placement",
      "parity",
      "optimizer",
      "run",
      "artifact",
      "health",
      "restoration",
    ],
    non_claims = [
      "no dual-card scaling claim",
      "no useful generalization claim",
      "no speedup claim",
      "no support claim for other RWKV models",
    ],
  },
}
