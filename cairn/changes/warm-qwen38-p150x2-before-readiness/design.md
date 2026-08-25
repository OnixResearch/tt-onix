## Context

The service binds its HTTP socket after model load. It does not execute production prefill or trace capture before it emits `ready`. The first request therefore performs cold TTNN compilation after health has passed.

## Decisions

### Decision: Warm synchronously before binding and notifying

**Choice:** Run startup warm-up before the HTTP server is created. Notify systemd only after all warm-up widths pass.

**Rationale:** No request can enter during compilation. A failed warm-up exits before traffic is accepted. This is simpler than an asynchronous server that must reject and coordinate requests during initialization.

### Decision: Warm B=1 and B=4

**Choice:** Execute prefill, exact-width trace capture, one trace replay, and clean reset for widths 1 and 4.

**Rationale:** B=1 removes the measured first-user cold path. B=4 covers the proven concurrent production path. Wider batches remain bounded soak checks, not startup requirements.

### Decision: Keep policy pure and effects at the edge

**Choice:** A pure readiness core validates widths and controls ordered lifecycle transitions. The batch coordinator executes TTNN effects. A small systemd adapter sends readiness notifications.

**Rationale:** Warm-up policy stays device-free. TTNN, sockets, clocks, and process environment remain in the imperative shell.

### Decision: Fail closed

**Choice:** A prefill, trace, replay, reset, or systemd-notification failure aborts startup. The service does not downgrade to cold request-time compilation.

**Rationale:** A healthy endpoint must not hide an unprepared production path.

## Rejected Alternatives

- **Serve immediately and return HTTP 503 while warming:** This adds request concurrency during device initialization and does not improve the current local-only deployment.
- **Warm on the first client request:** This preserves the false-ready condition.
- **Warm every width through B=16:** This increases startup time without evidence that all widths are normal production traffic.

## Risks / Trade-offs

- Startup takes longer because cold compilation moves before readiness.
- One exact-width trace remains resident. A later width change still recaptures the trace, but compiled programs should remain cached.
- Warm-up uses one bounded prompt shape. It does not prove that every prompt length is warm.
- The profile remains not admitted until the separate admission workflow succeeds.
