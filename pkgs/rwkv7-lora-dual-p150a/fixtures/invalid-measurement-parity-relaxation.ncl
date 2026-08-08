# Negative fixture: a measurement profile cannot add or relax parity tolerances.
let contracts = import "../contracts.ncl" in
let valid = (import "../measurement-profile.ncl").measurement_profile in
valid & {
  tolerances = { loss = { rtol = 1.0, atol = 1.0 } },
}
| contracts.MeasurementProfile
