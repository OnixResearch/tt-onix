let profile = import "../decode-component-profile.ncl" in

profile with {
  diagnostic_only | force = false,
}
