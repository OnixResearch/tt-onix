# Explicit host budget for the measured diagnostic route, not a service default.
{
  host_threads
    | Number
    | std.contract.from_predicate (fun value => value == 1)
    = 1,
}
