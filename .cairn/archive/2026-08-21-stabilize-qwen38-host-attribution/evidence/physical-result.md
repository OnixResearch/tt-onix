# Physical result

The final P150x2 sweep kept exact `eager,candidate,candidate,eager` order. Each root contains five independent service and capture samples. Each sample warmed before its measured request. The service used CPUs `8-31`. Production was restored eager and healthy.

A single long capture stalled on its third request. Tracy also refused a second capture connection to the same client. The final runner therefore used one package-matched service and capture process per sample. The comparator selected the final measured signpost pair from each sample and used the five-sample median as the root value.

Candidate duplicates passed the `1.25` bound:

- wall `1.1673619615353883`
- enqueue `1.0199312943580248`
- gaps `1.1955534484861683`

Eager duplicates failed:

- wall `1.3928346091490837`
- enqueue `1.2300527123313059`
- gaps `1.4276727516605083`

Verdict: `rejected`, reason: `duplicate-root-instability`. Repeated sampling stabilized the candidate roots but left persistent eager cross-process drift. No host attribution or optimization is selected.

Receipt: `pkgs/qwen36/evidence/2026-08-21/qwen38-27b-host-stability-attribution-receipt.json`

BLAKE3: `41fa1da2caf6aef20a0d1dfd8e9d5e65480a91d39c7b8e984c31e14199527497`
