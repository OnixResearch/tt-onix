# Typed, fail-closed contracts for sampled-mode kernel distributional admission.
let PositiveIntC = std.contract.from_predicate (fun value =>
  std.is_number value && value > 0 && std.number.is_integer value
) in
let PositiveNumberC = std.contract.from_predicate (fun value =>
  std.is_number value && value > 0
) in
let NonEmptyStringC = std.contract.from_predicate (fun value =>
  std.is_string value && std.string.length value > 0
) in
let Blake3C = std.contract.from_predicate (fun value =>
  std.is_string value && std.string.is_match "^[0-9a-f]{64}$" value
) in
let RevisionC = std.contract.from_predicate (fun value =>
  std.is_string value && std.string.is_match "^[0-9a-f]{40}$" value
) in
let ReceiptC = std.contract.from_predicate (fun value =>
  std.is_string value
  && (value == "not-measured" || std.string.is_match "^[0-9a-f]{64}$" value)
) in
let UnitProbabilityC = std.contract.from_predicate (fun value =>
  std.is_number value && value > 0 && value <= 1
) in
let ExactC = fun expected => std.contract.from_predicate (fun value => value == expected) in
let ClosedRecordC = fun allowed => std.contract.from_predicate (fun value =>
  std.is_record value
  && std.array.length (std.record.fields value) == std.array.length allowed
  && std.array.all (fun field => std.array.elem field allowed) (std.record.fields value)
) in
let StringArrayC = std.contract.from_predicate (fun value =>
  std.is_array value
  && std.array.length value > 0
  && std.array.all std.is_string value
  && std.array.all (fun item => std.string.length item > 0) value
) in
let ProbeRootOrderC = std.contract.from_predicate (fun value =>
  value == ["control", "candidate", "candidate", "control"]
) in

let IdentityC = {
  authority_blake3 | Blake3C,
  candidate_patch | NonEmptyStringC,
  checkpoint_revision | RevisionC,
  model_source_revision | RevisionC,
  receipt_blake3 | ReceiptC,
  tokenizer_revision | RevisionC,
} in
let IdentityClosedC = ClosedRecordC [
  "authority_blake3",
  "candidate_patch",
  "checkpoint_revision",
  "model_source_revision",
  "receipt_blake3",
  "tokenizer_revision",
] in

let ProbeC = {
  minimum_position_count | PositiveIntC,
  minimum_prompt_count | PositiveIntC,
  probe_mode | ExactC "teacher_forced",
  prompts | StringArrayC,
  root_order | ProbeRootOrderC,
  top_k_candidates | PositiveIntC,
} in
let ProbeClosedC = ClosedRecordC [
  "minimum_position_count",
  "minimum_prompt_count",
  "probe_mode",
  "prompts",
  "root_order",
  "top_k_candidates",
] in
let ProbeSemanticC = std.contract.from_predicate (fun value =>
  std.array.length value.prompts >= value.minimum_prompt_count
) in

let AdmissionC = {
  eligible_serving_modes | ExactC ["sampled"],
  maximum_full_step_ratio | PositiveNumberC,
  maximum_gdn_ratio | PositiveNumberC,
  maximum_max_topk_tv | PositiveNumberC,
  maximum_mean_topk_residual_mass | PositiveNumberC,
  maximum_mean_topk_tv | PositiveNumberC,
  minimum_mean_logit_pcc | UnitProbabilityC,
  minimum_min_logit_pcc | UnitProbabilityC,
  minimum_paired_logit_samples | PositiveIntC,
  minimum_stable_agreement | UnitProbabilityC,
  minimum_stable_position_count | PositiveIntC,
  near_tie_margin_logits | PositiveNumberC,
  require_clean_reset | ExactC true,
  require_physical_evidence | ExactC true,
  require_state_isolation | ExactC true,
} in
let AdmissionClosedC = ClosedRecordC [
  "eligible_serving_modes",
  "maximum_full_step_ratio",
  "maximum_gdn_ratio",
  "maximum_max_topk_tv",
  "maximum_mean_topk_residual_mass",
  "maximum_mean_topk_tv",
  "minimum_mean_logit_pcc",
  "minimum_min_logit_pcc",
  "minimum_paired_logit_samples",
  "minimum_stable_agreement",
  "minimum_stable_position_count",
  "near_tie_margin_logits",
  "require_clean_reset",
  "require_physical_evidence",
  "require_state_isolation",
] in
let AdmissionSemanticC = std.contract.from_predicate (fun value =>
  value.minimum_min_logit_pcc <= value.minimum_mean_logit_pcc
  && value.maximum_mean_topk_tv <= value.maximum_max_topk_tv
) in

let PolicyC = {
  schema_version | ExactC 1,
  profile_id | NonEmptyStringC,
  identities | IdentityClosedC | IdentityC,
  probe | ProbeClosedC | ProbeC,
  admission | AdmissionClosedC | AdmissionC,
} in
let PolicyClosedC = ClosedRecordC [
  "schema_version",
  "profile_id",
  "identities",
  "probe",
  "admission",
] in
let PolicySemanticC = std.contract.from_predicate (fun value =>
  std.array.length value.probe.prompts >= value.probe.minimum_prompt_count
  && value.admission.minimum_min_logit_pcc <= value.admission.minimum_mean_logit_pcc
  && value.admission.maximum_mean_topk_tv <= value.admission.maximum_max_topk_tv
  && value.admission.minimum_stable_position_count
    <= value.probe.minimum_prompt_count * value.probe.minimum_position_count
) in

let BindingC = {
  physically_admitted | Bool,
  receipt_blake3 | ReceiptC,
} in
let BindingClosedC = ClosedRecordC [
  "physically_admitted",
  "receipt_blake3",
] in
let BindingSemanticC = std.contract.from_predicate (fun value =>
  !value.physically_admitted || value.receipt_blake3 != "not-measured"
) in

let ProfileC = {
  schema_version | ExactC 1,
  profile_id | NonEmptyStringC,
  physically_admitted | Bool,
  identities | IdentityClosedC | IdentityC,
  probe | ProbeClosedC | ProbeC,
  admission | AdmissionClosedC | AdmissionC,
} in
let ProfileClosedC = ClosedRecordC [
  "schema_version",
  "profile_id",
  "physically_admitted",
  "identities",
  "probe",
  "admission",
] in
let ProfileSemanticC = std.contract.from_predicate (fun value =>
  (!value.physically_admitted || value.identities.receipt_blake3 != "not-measured")
  && std.array.length value.probe.prompts >= value.probe.minimum_prompt_count
  && value.admission.minimum_min_logit_pcc <= value.admission.minimum_mean_logit_pcc
  && value.admission.maximum_mean_topk_tv <= value.admission.maximum_max_topk_tv
  && value.admission.minimum_stable_position_count
    <= value.probe.minimum_prompt_count * value.probe.minimum_position_count
) in
{
  Admission = AdmissionC,
  AdmissionClosed = AdmissionClosedC,
  AdmissionSemantic = AdmissionSemanticC,
  Binding = BindingC,
  BindingClosed = BindingClosedC,
  BindingSemantic = BindingSemanticC,
  Identity = IdentityC,
  IdentityClosed = IdentityClosedC,
  Policy = PolicyC,
  PolicyClosed = PolicyClosedC,
  PolicySemantic = PolicySemanticC,
  Profile = ProfileC,
  ProfileClosed = ProfileClosedC,
  ProfileSemantic = ProfileSemanticC,
  Probe = ProbeC,
  ProbeClosed = ProbeClosedC,
  ProbeSemantic = ProbeSemanticC,
}
