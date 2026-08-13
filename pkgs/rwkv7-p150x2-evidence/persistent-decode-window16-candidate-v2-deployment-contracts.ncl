# Typed deployment policy contract for the corrected window-16 candidate.
let ExactC = fun expected => std.contract.from_predicate (fun value => value == expected) in
let ClosedRecordC = fun allowed =>
  std.contract.from_predicate (fun value =>
    std.is_record value
    && std.array.length (std.record.fields value) == std.array.length allowed
    && std.array.all (fun field => std.array.elem field allowed) (std.record.fields value)
  )
in
let DeploymentPolicyC = {
  schema_version | ExactC 1,
  policy_id | ExactC "rwkv7-p150x2-persistent-decode-window16-v2-deployment",
  source_candidate_profile_id | ExactC "rwkv7-p150x2-persistent-decode-window16-candidate-v2",
  source_candidate_profile_blake3 | ExactC "acd6d609397beee32b6dbe83bd00d042661ce1961c2353dbfd5794870839fee6",
  source_candidate_physically_admitted | ExactC false,
  expected_production_profile_id | ExactC "rwkv7-p150x2-persistent-decode-v4",
  expected_admitted_windows | ExactC [2, 4, 8, 16],
  target_machine | ExactC "britton-desktop",
  activation_method | ExactC "exact_store_switch_to_configuration",
  required_programs
    | ExactC [
      "rwkv7-p150x2-runtime",
      "rwkv7-p150x2-persistent-decode-monitor",
    ],
  physical_device_ids | ExactC [0, 1],
  require_built_closure | ExactC true,
  require_active_link_matches_target | ExactC true,
  require_rollback_target_recorded | ExactC true,
  require_service_state_restoration | ExactC true,
  require_devices_unowned_after | ExactC true,
}
in
let DeploymentPolicyClosedC =
  ClosedRecordC [
    "schema_version",
    "policy_id",
    "source_candidate_profile_id",
    "source_candidate_profile_blake3",
    "source_candidate_physically_admitted",
    "expected_production_profile_id",
    "expected_admitted_windows",
    "target_machine",
    "activation_method",
    "required_programs",
    "physical_device_ids",
    "require_built_closure",
    "require_active_link_matches_target",
    "require_rollback_target_recorded",
    "require_service_state_restoration",
    "require_devices_unowned_after",
  ]
in
{
  DeploymentPolicy = DeploymentPolicyC,
  DeploymentPolicyClosed = DeploymentPolicyClosedC,
}
