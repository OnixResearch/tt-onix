let profile = import "../gdn-decode-fusion-profile.ncl" in
profile & {
  gates = {
    require_exact_token_parity | force = false,
  },
}
