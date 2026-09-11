# Negative fixture: the measurement profile must request enough repeats.
let contracts = import "../contracts.ncl" in
let valid = (import "../measurement-profile.ncl").measurement_profile in
(std.record.remove "repeat_count" valid) & { repeat_count = 2 }
| contracts.MeasurementProfile
