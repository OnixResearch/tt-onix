# Negative fixture: an unmeasured candidate cannot be physically admitted.
let profile = import "../decode-profile.ncl" in
profile & {
  physically_admitted = true,
  trace = profile.trace & { slot_count = 0 },
}
