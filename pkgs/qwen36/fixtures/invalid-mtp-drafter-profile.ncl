# Negative fixture: drafter selection cannot weaken prompt or draft bounds.
let profile = import "../mtp-drafter-profile.ncl" in
profile & {
  diagnostic_only = false,
  draft_token_count = 2,
  prompts = ["only one prompt"],
}
