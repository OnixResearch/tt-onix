let profile = import "../gdn-decode-fusion-profile.ncl" in

profile & {
  diagnostic_only | force = false,
}
