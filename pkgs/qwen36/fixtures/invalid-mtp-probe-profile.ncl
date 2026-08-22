# Negative fixture: the diagnostic width set is closed and ordered.
let profile = import "../mtp-probe-profile.ncl" in
profile & {
  diagnostic_only = false,
  verification_widths = [1, 4],
}
