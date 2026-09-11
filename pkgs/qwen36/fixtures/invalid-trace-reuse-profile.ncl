# Negative fixture: reuse admission cannot bind an absent receipt.
let profile = import "../trace-reuse-profile.ncl" in
profile & {
  physically_admitted = true,
  identities = profile.identities & { receipt_blake3 = "not-measured" },
}
