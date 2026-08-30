# Tasks

1. [x] Pure serving round policy core (`mtp_serving_core.py`): round
   schedule, exact greedy commit rule, closed validation.
2. [x] Device-free check probe (`probe_mtp_serving_core`) with positive and
   negative cases, bound to the spec delta.
3. [x] Bind the module into the package execution identity and pass the
   qwen36 device-free check.
4. [ ] Serving adapter: request-scoped trace capture in hygiene order
   (verify, draft, decode), round replay, snapshot rollback per round,
   plain-step fallback, disengage accounting.
5. [ ] Server wiring: `QWEN36_MTP_SERVING` gate (default off), module
   option, readiness payload field, metrics for round outcomes.
6. [ ] Physical exactness screen: greedy requests through the gated path
   versus the plain path, 64 tokens, exact token parity required; draft
   and verify trace capture costs excluded from round economics.
7. [ ] Owner review: k operating point, disengage thresholds, enablement.
