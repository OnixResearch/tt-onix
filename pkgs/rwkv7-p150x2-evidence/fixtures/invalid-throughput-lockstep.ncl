# The throughput profile must reject lockstep mesh replication.
let contracts = import "../throughput-contracts.ncl" in
let profile = import "../throughput-profile.ncl" in
(profile & {
  runtime.replica_mode | force = "full-lockstep",
}) | contracts.ThroughputProfile
