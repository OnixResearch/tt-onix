# Negative fixture: B=8 admission cannot bind an absent receipt.
let profile = import "../batch-admission-b8-profile.ncl" in
profile & {
  physically_admitted = true,
  identities = profile.identities & { receipt_blake3 = "not-measured" },
}
