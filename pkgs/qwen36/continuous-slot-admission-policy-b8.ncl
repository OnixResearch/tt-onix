# Predeclared gates for admitting the continuous-slot B=8 scheduler profile.
#
# r[impl tenstorrent.qwen38_p150x2.continuous_slot_admission]
#
# The comparison runs one staggered, mixed-length workload against the group
# coordinator (baseline) and the continuous-slot coordinator (candidate), on the same
# package, runtime, and devices. Every time is client-observed and queue-inclusive:
# a request's wall runs from its send to the end of its stream. These bounds were fixed
# before the physical measurement; changing one after it requires a new measurement.
let contracts = import "continuous-slot-admission-contracts.ncl" in
{
  schema_version = 1,
  profile_id = "qwen38-p150x2-continuous-slot-admission-b8-v1",
  # The only profile a passing receipt may admit.
  scheduler_profile_id = "qwen38-p150x2-continuous-slot-b8-v1",
  workload = {
    workload_id = "qwen38-staggered-mixed-length-v1",
    # Measured passes of the concurrent workload per root; every gate must hold on
    # each matched pair, so one lucky pass cannot admit.
    repetitions = 2,
    short_class = "short",
    long_class = "long",
  },
  gates = {
    # Same package, runtime, devices, and workload bytes; no simulated root.
    require_matched_identity = true,
    # Every request in both roots ends with a stop or length finish.
    require_all_completed = true,
    # Every candidate request decodes exactly the tokens it decodes alone on the
    # candidate path: other slots' refills, retirements, and idle rows cannot reach it.
    require_exact_candidate_parity = true,
    # The purpose of the change: short requests stop waiting behind long ones.
    maximum_short_median_wall_ratio = 0.5,
    # No short request's tail may get worse.
    maximum_short_p95_wall_ratio = 1.0,
    # Long requests may pay for refill stalls and the wider trace, within a bound.
    maximum_long_p95_wall_ratio = 1.3,
    # Starvation guard: the worst queue-inclusive slowdown against a request's own solo
    # run may not exceed the baseline's worst.
    maximum_slowdown_ratio = 1.0,
    # Declared tail latency: the longest pause any active stream sees, including a
    # refill's prefill beside it. The workload's largest refill prompt is about
    # 6.5K tokens.
    maximum_inter_token_gap_seconds = 6.0,
    # Steady decode retention: the median pause between streamed tokens.
    maximum_median_inter_token_gap_ratio = 1.25,
    # Delivered tokens over the makespan.
    minimum_delivered_throughput_ratio = 0.95,
    # Device DRAM free per chip after startup.
    minimum_startup_dram_free_ratio = 0.99,
  },
}
| contracts.Closed
| contracts.Fields
| { workload | contracts.WorkloadClosed | contracts.WorkloadCoherent, gates | contracts.GatesClosed, .. }
