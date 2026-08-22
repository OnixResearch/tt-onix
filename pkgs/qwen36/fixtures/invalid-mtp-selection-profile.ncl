let profile = import "../mtp-selection-profile.ncl" in
profile & {
  route_schedule = ["shard-local-top1", "full-logit-readback"],
}
