# Policy for the GDN decode occupancy planner.
#
# The planner chooses a state split to raise occupancy when the grid is idle, and
# it refuses to split once a core is already looping. Its decision depends on one
# measurement that does not exist yet: the share of a single instance's cost that
# does NOT divide with the value width (the `K x K` transpose plus reader, CB, and
# launch overhead).
#
# That value is therefore recorded as `not-measured` rather than guessed. The
# planner rejects a non-numeric share, so a split cannot be selected from this
# policy until the share is measured and recorded here. That is deliberate: an
# invented share would write an unmeasured performance claim into a policy and
# could enable a split that costs more than it saves.
{
  schema_version = 1,
  profile_id = "qwen38-p150x2-gdn-occupancy-v1",
  # Measured by the component probe at the width that uses the split. Until then
  # the string below is what the planner refuses.
  fixed_cost_share = "not-measured",
  # A split must reach at least this occupancy to be considered at all.
  minimum_occupancy = 0.5,
  # The value tile count the split divides (128 / 32 at TP=2).
  value_tiles = 4,
} | {
  schema_version | std.contract.from_predicate (fun value => value == 1),
  profile_id | std.contract.from_predicate (
    fun value => std.is_string value && std.string.length value > 0
  ),
  fixed_cost_share | std.contract.from_predicate (fun value =>
    value == "not-measured"
    || (std.is_number value && value >= 0 && value < 1)
  ),
  minimum_occupancy | std.contract.from_predicate (fun value =>
    std.is_number value && value > 0 && value <= 1
  ),
  value_tiles | std.contract.from_predicate (fun value =>
    std.is_number value && value > 0 && std.number.is_integer value
  ),
}
