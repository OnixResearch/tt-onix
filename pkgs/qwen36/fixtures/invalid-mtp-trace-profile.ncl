# Negative fixture: five retained route samples violate the six-sample trace policy.
let profile = import "../mtp-trace-profile.ncl" in
profile & {
  minimum_samples_per_route = 5,
}
