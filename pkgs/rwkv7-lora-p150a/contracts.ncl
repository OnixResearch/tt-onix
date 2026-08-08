# r[impl tenstorrent.rwkv7_p150a_finetuning.configuration]
# Typed contracts for the single-P150A RWKV-7 LoRA acceptance profile.
# The admission core in core/src/ re-checks every rule in Rust before
# compilation, device access, or evidence admission.

let target_profiles = import "../../model-targets/profiles.ncl" in
let target = std.array.first (std.array.filter
  (fun profile => profile.target_name == "rwkv7-goose-world2.8-0.1b-p150x2")
  target_profiles.profiles) in

let NonEmptyC = std.contract.from_predicate (fun value =>
  std.is_string value && std.string.length value > 0
) in

let PositiveC = std.contract.from_predicate (fun value =>
  std.is_number value && value > 0
) in

let NonNegativeC = std.contract.from_predicate (fun value =>
  std.is_number value && value >= 0
) in

let PositiveIntC = std.contract.from_predicate (fun value =>
  std.is_number value && value > 0 && std.number.is_integer value
) in

let HashHexC = std.contract.from_predicate (fun value =>
  std.is_string value && std.string.is_match "^[0-9a-f]{64}$" value
) in

let GitRevisionC = std.contract.from_predicate (fun value =>
  std.is_string value && std.string.is_match "^[0-9a-f]{40}$" value
) in

let AbsolutePathC = std.contract.from_predicate (fun value =>
  std.is_string value && std.string.is_match "^/" value && !(std.string.is_match "\\.\\." value)
) in

let admitted_bundle_id = "tt-xla-training-2026-08-04" in

let BundleIdC = std.contract.from_predicate (fun value =>
  value == admitted_bundle_id
) in

let admitted_model_id = target.model.id in

let admitted_precision_class = "bf16_from_fp32_cpu" in

let admitted_adapter_format = target.consumer.adapter_format in

let admitted_hash_algorithm = "blake3" in

let admitted_roles = [
  "attention.receptance",
  "attention.key",
  "attention.value",
  "attention.output",
  "feed_forward.key",
  "feed_forward.value",
] in

let LoraRolesC = std.contract.from_predicate (fun value =>
  std.is_array value
  && std.array.length value > 0
  && std.array.all (fun role => std.array.elem role admitted_roles) value
) in

let DtypeC = std.contract.from_predicate (fun value =>
  std.array.elem value ["bf16", "fp32"]
) in

let ModelC = {
  model_id | std.contract.from_predicate (fun value => value == admitted_model_id),
  revision | GitRevisionC,
  checkpoint_blake3 | HashHexC,
  layer_count | PositiveIntC,
  hidden_size | PositiveIntC,
  head_count | PositiveIntC,
  head_size | PositiveIntC,
  vocabulary_size | PositiveIntC,
  precision | NonEmptyC,
} in

let TokenizerC = {
  vocabulary_blake3 | HashHexC,
  vocabulary_entry_count | PositiveIntC,
} in

let DeviceC = {
  physical_count | std.contract.from_predicate (fun value => value == 2),
  architecture | std.contract.from_predicate (fun value => value == "blackhole"),
  board_class | std.contract.from_predicate (fun value => value == "p150a"),
  device_root | AbsolutePathC,
  logical_mesh = {
    rows | std.contract.from_predicate (fun value => value == 1),
    columns | std.contract.from_predicate (fun value => value == 1),
  },
  selection_ordinal | NonNegativeC,
  container_device_isolation | std.contract.from_predicate (fun value => value == "unsupported"),
} in

# Static shapes are positive integers. A zero, negative, or non-integer
# shape fails admission before compilation.
let StaticShapeC = {
  micro_batch | PositiveIntC,
  sequence_length | PositiveIntC,
  gradient_accumulation_steps | PositiveIntC,
} in

let ProbeShapeC = {
  batch | PositiveIntC,
  time | PositiveIntC,
  heads | PositiveIntC,
  width | PositiveIntC,
} in

let ProbeC = {
  focused | ProbeShapeC,
  representative | ProbeShapeC,
} in

let PrecisionC = {
  precision_class | std.contract.from_predicate (fun value =>
    value == admitted_precision_class
  ),
  model_dtype | DtypeC,
  optimizer_dtype | DtypeC,
  comparison_dtype | DtypeC,
} in

let CompileLimitsC = {
  probe_compile_timeout_seconds | PositiveC,
  full_compile_timeout_seconds | PositiveC,
  max_stablehlo_operations | PositiveIntC,
} in

let MemoryLimitsC = {
  max_device_bytes | PositiveC,
  max_host_bytes | PositiveC,
} in

let ToleranceC = {
  rtol | PositiveC,
  atol | PositiveC,
} in

let tolerance_roles = [
  "logits",
  "loss",
  "wkv7_output",
  "wkv7_state",
  "adapter_gradient",
  "gradient_norm",
  "clip_coefficient",
  "adapter_delta",
] in

let TolerancesC = std.contract.from_predicate (fun value =>
  std.is_record value
  && std.array.all (fun role => std.array.elem role (std.record.fields value)) tolerance_roles
) in

let RunCriteriaC = {
  min_steps | PositiveIntC,
  max_steps | PositiveIntC,
  loss_improvement_ratio | PositiveC,
  min_active_labels_per_sample | PositiveIntC,
  parity_pass_fraction | PositiveC,
  require_finite | Bool,
} in

let HealthLimitsC = {
  max_board_temperature_c | PositiveC,
  max_power_watts | PositiveC,
  require_dram_ok | Bool,
} in

let TimeoutsC = {
  hardware_attempt_seconds | PositiveC,
  multi_step_run_seconds | PositiveC,
} in

let LoraC = {
  rank | PositiveIntC,
  scale | PositiveC,
  target_roles | LoraRolesC,
} in

let OptimizerC = {
  kind | std.contract.from_predicate (fun value => value == "adamw"),
  learning_rate | PositiveC,
  gradient_accumulation_steps | PositiveIntC,
  clip_norm | PositiveC,
} in

let minimum_measurement_repeats = 3 in
let measurement_metrics = ["stage_time_seconds", "loss"] in
let measurement_machine_facts = [
  "board_identity",
  "firmware_version",
  "mesh_descriptor_identity",
  "pcie_link",
  "power_state",
] in
let measurement_profile_fields = [
  "profile_id",
  "parity_profile_id",
  "repeat_count",
  "metric_set",
  "machine_fact_set",
] in
let measurement_profile_field_count = std.array.length measurement_profile_fields in
let RegressionMetricC = std.contract.from_predicate (fun value =>
  std.array.elem value ["time_seconds.mean", "time_seconds.p99", "loss.mean"]
) in

let RegressionMetricBoundC = {
  metric | RegressionMetricC,
  direction | std.contract.from_predicate (fun value => value == "lower_is_better"),
  maximum_regression_fraction | NonNegativeC,
} in

let DeterminismTolerancesC = {
  time_seconds | NonNegativeC,
  loss | NonNegativeC,
} in

let RegressionPolicyC = {
  schema_version | std.contract.from_predicate (fun value => value == 1),
  profile_id | NonEmptyC,
  metric_bounds | Array RegressionMetricBoundC,
  determinism_tolerances | DeterminismTolerancesC,
  hash_algorithm | std.contract.from_predicate (fun value => value == "blake3"),
} in

let StabilityPolicyC = {
  schema_version | std.contract.from_predicate (fun value => value == 1),
  profile_id | NonEmptyC,
  parity_profile_id | NonEmptyC,
  step_budget | PositiveIntC,
  trajectory_cadence_steps | PositiveIntC,
  thermal_cadence_steps | PositiveIntC,
  maximum_loss | PositiveC,
  maximum_gradient_norm | PositiveC,
  maximum_temperature_celsius | PositiveC,
  maximum_power_watts | PositiveC,
  maximum_eval_loss_ratio | std.contract.from_predicate (fun value =>
    std.is_number value && value > 0 && value <= 1
  ),
  thermal_source | NonEmptyC,
  generalization_non_claim | NonEmptyC,
  hash_algorithm | std.contract.from_predicate (fun value => value == "blake3"),
} in

let MeasurementProfileC = std.contract.from_predicate (fun value =>
  std.is_record value
  && std.array.length (std.record.fields value) == measurement_profile_field_count
  && std.array.all
    (fun field => std.array.elem field measurement_profile_fields)
    (std.record.fields value)
  && std.is_string value.profile_id
  && std.string.length value.profile_id > 0
  && std.is_string value.parity_profile_id
  && std.string.length value.parity_profile_id > 0
  && std.is_number value.repeat_count
  && std.number.is_integer value.repeat_count
  && value.repeat_count >= minimum_measurement_repeats
  && value.metric_set == measurement_metrics
  && value.machine_fact_set == measurement_machine_facts
) in

let ArtifactPolicyC = {
  format | std.contract.from_predicate (fun value => value == admitted_adapter_format),
  artifact_root | AbsolutePathC,
} in

let EvidenceExcludedC = std.contract.from_predicate (fun value =>
  std.is_array value
  && std.array.all
    (fun item =>
      std.array.elem
        item
        ["credentials", "raw_logs", "serial_numbers", "absolute_private_paths"])
    value
) in

let required_receipts = [
  "prerequisite",
  "compile_probe",
  "placement",
  "parity",
  "optimizer",
  "run",
  "artifact",
  "health",
  "restoration",
] in

let RequiredReceiptsC = std.contract.from_predicate (fun value =>
  std.is_array value
  && std.array.all (fun kind => std.array.elem kind value) required_receipts
) in

let boundary_non_claims = [
  "no dual-card scaling claim",
  "no useful generalization claim",
  "no speedup claim",
  "no support claim for other RWKV models",
] in

let BoundaryNonClaimsC = std.contract.from_predicate (fun value =>
  std.is_array value
  && std.array.all (fun claim => std.array.elem claim value) boundary_non_claims
) in

let EvidencePolicyC = {
  hash_algorithm | std.contract.from_predicate (fun value =>
    value == admitted_hash_algorithm
  ),
  exclude | EvidenceExcludedC,
  required_receipts | RequiredReceiptsC,
  non_claims | BoundaryNonClaimsC,
} in

let HttpsUrlC = std.contract.from_predicate (fun value =>
  std.is_string value && std.string.is_match "^https://" value
) in

# A dependency wheel whose identity is pinned by the metadata of an
# admitted runtime wheel. The sha256 binds the exact URL.
let WheelDependencyC = {
  package | NonEmptyC,
  version | NonEmptyC,
  url | HttpsUrlC,
  wheel_sha256 | HashHexC,
  pinned_by | NonEmptyC,
} in

let WheelsC = {
  schema_version | std.contract.from_predicate (fun value => value == 1),
  tt_lang | WheelDependencyC,
} in

let wheels_fields = [
  "schema_version",
  "tt_lang",
] in

let ProfileC = {
  schema_version | std.contract.from_predicate (fun value => value == 1),
  profile_id | NonEmptyC,
  bundle_id | BundleIdC,
  model | ModelC,
  tokenizer | TokenizerC,
  device | DeviceC,
  static_shape | StaticShapeC,
  probe | ProbeC,
  precision | PrecisionC,
  compile_limits | CompileLimitsC,
  memory_limits | MemoryLimitsC,
  tolerances | TolerancesC,
  run_criteria | RunCriteriaC,
  health_limits | HealthLimitsC,
  timeouts | TimeoutsC,
  lora | LoraC,
  optimizer | OptimizerC,
  seed | NonNegativeC,
  artifact_policy | ArtifactPolicyC,
  evidence_policy | EvidencePolicyC,
} in

# Closed-record contract: rejects any field outside the admitted set.
let ClosedRecordC = fun allowed =>
  std.contract.from_predicate (fun value =>
    std.is_record value
    && std.array.all (fun field => std.array.elem field allowed) (std.record.fields value)
  )
in

let profile_fields = [
  "schema_version",
  "profile_id",
  "bundle_id",
  "model",
  "tokenizer",
  "device",
  "static_shape",
  "probe",
  "precision",
  "compile_limits",
  "memory_limits",
  "tolerances",
  "run_criteria",
  "health_limits",
  "timeouts",
  "lora",
  "optimizer",
  "seed",
  "artifact_policy",
  "evidence_policy",
] in

{
  Profile = ProfileC,
  ProfileClosed = ClosedRecordC profile_fields,
  MeasurementProfile = MeasurementProfileC,
  RegressionPolicy = RegressionPolicyC,
  StabilityPolicy = StabilityPolicyC,
  Wheels = WheelsC,
  WheelsClosed = ClosedRecordC wheels_fields,
  HashHex = HashHexC,
  GitRevision = GitRevisionC,
  AdmittedBundleId = admitted_bundle_id,
  AdmittedModelId = admitted_model_id,
  AdmittedPrecisionClass = admitted_precision_class,
  AdmittedAdapterFormat = admitted_adapter_format,
  ToleranceRoles = tolerance_roles,
  RequiredReceipts = required_receipts,
  BoundaryNonClaims = boundary_non_claims,
}
