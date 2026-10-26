# Design

Qualify on simulator then isolated card 0, with real 768-wide projections and 65536-way LM head, model/stateful continuation gates, independent numerical oracles and warmed median timings. Throughput/overlap is the hypothesis; fewer bytes is not required. Keep baseline defaults until numerical and performance qualification.

Evidence distinguishes source contract, numerical execution, physical performance, and unestablished model/serving admission.

## Native F32 recurrent-state view consumer

The selected real materialization is the `[768,64]` state slice of the fused WKV7
`[768,96]` result, reshaped to `[12,64,64]`, before the next batched recurrence.
The existing `Strided` metadata identifies this exact origin-zero layout. An
opt-in `TT_WKV7_STATE_VIEW=1` route passes its existing parent allocation directly
to the WKV7 consumer, which reads only the first two whole tiles of each
three-tile row. There is no new tensor-view type, intermediate GDDR state,
arithmetic change, BF16 assumption, or affine face-row gather. Other views retain
the established materialization path; defaults stay unchanged.

Qualification compares direct consumption against compact-state consumption
bit-for-bit, poisons excluded output columns with NaNs, checks immutable parent
bits and standalone-transfer counters, and measures warmed host medians with
output validation outside timing. Silicon qualification is owned by the main
agent; simulator timing is not physical performance evidence.
