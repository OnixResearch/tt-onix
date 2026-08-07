# Raw profile record without contract application. profile.ncl applies the
# admitted contracts; negative fixtures mutate this record before applying
# the same contracts.
let contracts = import "../contracts.ncl" in
let target_profiles = import "../../../model-targets/profiles.ncl" in
let target = std.array.first (std.array.filter
  (fun profile => profile.target_name == "rwkv7-goose-world2.8-0.1b-p150x2")
  target_profiles.profiles) in
let artifact = fun role => std.array.first (std.array.filter
  (fun value => value.role == role)
  target.artifacts) in
{
  schema_version = 1,
  profile_id = "rwkv7-lora-dual-p150a",
  bundle_id = contracts.AdmittedBundleId,
  model = {
    model_id = target.model.id,
    revision = target.model.revision,
    checkpoint_blake3 = (artifact "checkpoint").blake3,
    layer_count = target.resources.layer_count,
    hidden_size = target.resources.hidden_size,
    head_count = target.consumer.dimensions.head_count,
    head_size = target.consumer.dimensions.head_size,
    vocabulary_size = target.resources.vocabulary_size,
    precision = target.consumer.precisions.cpu_reference,
  },
  tokenizer = {
    vocabulary_blake3 = (artifact "tokenizer_vocabulary").blake3,
    vocabulary_entry_count = 65529,
  },
  device = {
    physical_count = 2,
    architecture = "blackhole",
    board_class = "p150a",
    device_root = "/dev/tenstorrent",
    physical_device_ids = [0, 1],
    mesh_descriptor = "p150_x2",
    logical_mesh = {
      rows = 1,
      columns = 2,
    },
    container_device_isolation = "unsupported",
  },
  data_axis = {
    name = "data",
    replicas = 2,
    orientation = "columns",
  },
  shard_policy = {
    kind = "even_contiguous",
    allow_uneven = false,
  },
  reduction_policy = {
    gradient_op = "mean",
    gradient_scope = "adapter_gradients",
    loss_reduction = "weighted_active_labels",
    placement = "tt",
  },
  replica_policy = {
    equality = "exact",
    mechanism = "spmd_replicated",
    boundaries = ["optimizer_update", "canonical_save", "resume"],
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
      # bf16 gradient-norm reduction precision on the chaotic full model.
      rtol = 0.05,
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
    # Fraction of gradient/adapter-delta elements that must satisfy their
    # named tolerance. The chaotic recurrence makes a few worst-case
    # elements flip sign, so the acceptance is a pass fraction, not a max.
    parity_pass_fraction = 0.9,
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
    collective_timeout_seconds = 600,
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
    canonical = "single_adapter_after_replica_equality",
    artifact_root = "/var/lib/tenstorrent-training/artifacts/rwkv7-lora-dual-p150a",
  },
  evidence_policy = {
    hash_algorithm = "blake3",
    exclude = ["credentials", "raw_logs", "serial_numbers", "absolute_private_paths"],
    required_receipts = [
      "prerequisite",
      "collective",
      "shard",
      "placement",
      "parity",
      "replica",
      "optimizer",
      "run",
      "artifact",
      "health",
      "cleanup",
      "restoration",
    ],
    non_claims = [
      "no speedup claim",
      "no scaling efficiency claim",
      "no long-run stability claim",
      "no useful generalization claim",
      "no support claim for other RWKV models",
    ],
  },
}
