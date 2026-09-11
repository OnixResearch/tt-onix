let Contracts = import "../persistent-decode-window16-candidate-v2-deployment-contracts.ncl" in
let invalid | Contracts.DeploymentPolicy = {
  schema_version = 1,
  policy_id = "rwkv7-p150x2-persistent-decode-window16-v2-deployment",
  source_candidate_profile_id = "rwkv7-p150x2-persistent-decode-window16-candidate-v2",
  source_candidate_profile_blake3 = "0fd161f7172276d1efbe51623b1d110947e68de40024109618547cf7a42bb01a",
  source_candidate_physically_admitted = false,
  expected_production_profile_id = "rwkv7-p150x2-persistent-decode-v4",
  expected_admitted_windows = [2, 4, 8, 16],
  target_machine = "britton-desktop",
  activation_method = "exact_store_switch_to_configuration",
  required_programs = [
    "rwkv7-p150x2-runtime",
    "rwkv7-p150x2-persistent-decode-monitor",
  ],
  physical_device_ids = [0, 1],
  require_built_closure = true,
  require_active_link_matches_target = true,
  require_rollback_target_recorded = true,
  require_service_state_restoration = true,
  require_devices_unowned_after = true,
}
in
invalid
