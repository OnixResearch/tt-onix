# Immutable sampled-mode kernel distributional admission policy.
# Every bound mirrors a named constant in
# qwen36_service/kernel_distribution_admission_core.py. A change to any bound
# here or there is an execution-identity change and needs a new receipt.
let configured_top_k_candidates = 20 in
let configured_minimum_prompt_count = 5 in
let configured_minimum_position_count = 48 in
let configured_minimum_mean_logit_pcc = 0.995 in
let configured_minimum_min_logit_pcc = 0.99 in
let configured_minimum_paired_logit_samples = 8 in
let configured_maximum_mean_topk_tv = 0.10 in
let configured_maximum_max_topk_tv = 0.25 in
let configured_maximum_mean_topk_residual_mass = 0.50 in
let configured_near_tie_margin_logits = 0.5 in
let configured_minimum_stable_agreement = 0.95 in
let configured_minimum_stable_position_count = 120 in
let configured_maximum_gdn_ratio = 0.95 in
let configured_maximum_full_step_ratio = 0.97 in
{
  schema_version = 1,
  profile_id = "qwen38-p150x2-fused-gdn-sampled-mode-v1",
  identities = {
    authority_blake3 = "c319f7eef1b58911457c128ca5397710f4c9af3b98d1ccc83914005f9d3726cd",
    candidate_patch = "gdn-decode-fusion-candidate.patch",
    checkpoint_revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    model_source_revision = "d102e7f4512060e09e580e40cf26a870bfa88c55",
    receipt_blake3 = "not-measured",
    tokenizer_revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
  },
  probe = {
    minimum_position_count = configured_minimum_position_count,
    minimum_prompt_count = configured_minimum_prompt_count,
    probe_mode = "teacher_forced",
    prompts = [
      "The capital of France is",
      "Explain why deterministic cleanup matters for recurrent state.",
      "Repeat the token boundary exactly:",
      "A train travels at 60 mph for 2 hours and 30 minutes.",
      "Write a function to compute the nth Fibonacci number.",
    ],
    root_order = ["control", "candidate", "candidate", "control"],
    top_k_candidates = configured_top_k_candidates,
  },
  admission = {
    eligible_serving_modes = ["sampled"],
    maximum_full_step_ratio = configured_maximum_full_step_ratio,
    maximum_gdn_ratio = configured_maximum_gdn_ratio,
    maximum_max_topk_tv = configured_maximum_max_topk_tv,
    maximum_mean_topk_residual_mass = configured_maximum_mean_topk_residual_mass,
    maximum_mean_topk_tv = configured_maximum_mean_topk_tv,
    minimum_mean_logit_pcc = configured_minimum_mean_logit_pcc,
    minimum_min_logit_pcc = configured_minimum_min_logit_pcc,
    minimum_paired_logit_samples = configured_minimum_paired_logit_samples,
    minimum_stable_agreement = configured_minimum_stable_agreement,
    minimum_stable_position_count = configured_minimum_stable_position_count,
    near_tie_margin_logits = configured_near_tie_margin_logits,
    require_clean_reset = true,
    require_physical_evidence = true,
    require_state_isolation = true,
  },
}
