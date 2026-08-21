## Context

One request per root could not separate transient host scheduling noise from mode effects.

## Decision

Keep two immutable roots per mode and exact `eager,candidate,candidate,eager` order. Capture five sequential measured requests per root after warm-up. Pin the service to CPUs `8-31`, leaving CPUs `0-7` for capture, HTTP, and system work. Parse every ordered signpost pair, calculate interval metrics independently, and use the odd-sample median as each root value. Compare duplicate root medians against `1.25`.

## Alternatives

More roots increase outage time without first reducing within-root noise. Real-time scheduling requires broader authority and can perturb system services. A median rejects isolated scheduling outliers without hiding persistent root drift.

## Non-Goals

No production admission, decode-mode change, kernel selection, or claim from unstable data.
