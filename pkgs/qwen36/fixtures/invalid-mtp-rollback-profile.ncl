# Negative fixture: a full-cache rollback profile weakens the bounded row contract.
let profile = import "../mtp-rollback-profile.ncl" in
profile & { snapshot_row_count = 2048 }
