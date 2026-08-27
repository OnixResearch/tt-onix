let profile = import "../gdn-decode-fusion-profile.ncl" in

profile with {
  diagnostic_only | force = false,
}
