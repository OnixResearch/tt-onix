# Fixture: the v3 receipt binding before promotion. installCheckPhase swaps this
# for the exported binding and proves the exported profile becomes unadmitted
# while the execution identity, whose input manifest never names a binding,
# stays the same.
{
  physically_admitted = false,
  receipt_blake3 = "not-measured",
}
