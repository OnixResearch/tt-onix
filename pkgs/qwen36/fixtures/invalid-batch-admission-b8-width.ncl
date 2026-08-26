# Negative fixture: the closed contract rejects a candidate width outside {1, 8}.
let profile = import "../batch-admission-b8-policy.ncl" in
profile & {
  workload = profile.workload & { candidate_width = 9 },
}
