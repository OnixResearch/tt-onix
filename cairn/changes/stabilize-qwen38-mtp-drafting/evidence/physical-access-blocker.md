## Physical access blocker

The diagnostic package is ready for a matched P150x2 run, but `qwen38-p150x2.service` currently owns devices `0` and `1`. The service is active and healthy on port `8000`.

No new production outage has been authorized for this change. The prior authorized outage budget was exhausted. The trace probe has therefore not opened either device.

The blocked command is `qwen36-p150x2-mtp-trace-probe` from the device-free verified package. A valid run must use CPUs `8-31`, one loaded process, all five fixed prompts, and the 12-root schedule in `mtp-trace-profile.json`.

This blocker makes no trace-correctness, cache-idempotence, silicon-timing, or performance claim. Production remains eager.
