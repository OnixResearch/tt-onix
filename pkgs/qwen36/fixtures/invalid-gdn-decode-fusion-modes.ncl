let profile = import "../gdn-decode-fusion-profile.ncl" in
profile & {
  workload = {
    modes | force = [ "off", "norm", "state", "norm_state", "full" ],
  },
}
