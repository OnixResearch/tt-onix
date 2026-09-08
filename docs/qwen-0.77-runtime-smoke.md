# Qwen runtime comparison: 0.74 and 0.77

## Result

Both runtime packages passed the bounded Qwen3.8-27B smoke on physical P150x2 hardware. They returned identical token IDs for one prompt, repeated twice per runtime. This is a narrow regression check, not broad model-accuracy or performance acceptance.

The production service remained stopped. No package deployment, firmware update, or production cache change occurred.

## Runtime identities

| Role | TT-Metal | Qwen package |
| --- | --- | --- |
| Baseline | `/nix/store/2n6r40bpg5a38aib9i8inxv5hrkyrf24-tt-metal-0.74.0` | `/nix/store/yx01j2cxvx8ma3hnpmljznfqb9j2ixs3-qwen36-0.1.0` |
| Candidate | `/nix/store/dr9bj8qwflylq39zl5ajcx42cdrb1bv9-tt-metal-0.77.0` | `/nix/store/qw9llks4k65np226xygfrmzx1qcbs6nz-qwen36-0.1.0` |

Both used checkpoint `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, the same Python environment, and the existing tensor cache. The candidate includes the HiFi4 diagnostic kernel fix. The model smoke kept the existing default-off fusion controls unchanged. It did not enable full fused GDN or MTP serving.

## Safety bounds

Each temporary server ran with:

- Physical PCIe devices 0 and 1 on the required 1x2 mesh.
- A 900-second timeout and 30-second termination grace.
- A 96 GiB memory limit and no swap allowance for the process group.
- A read-only filesystem view, except private scratch space, temporary files, and device access.
- Read-only production weights and tensor caches.
- Loopback-only HTTP access on a separate port.
- A maximum sequence length of 2048 and maximum generation length of 16.
- Fast dispatch, SFPLOADMACRO disabled, and no simulator override.

The tests used the packaged `qwen36-p150x2-trace-candidate` wrapper. The writable scratch directory held the compiler cache and profiler output. The service configuration did not change.

## Positive and negative results

Both runs loaded all 64 layers and passed startup warm-up at widths 1 and 4. Their `/readyz` responses reported `ready: true`.

An oversized request with `max_tokens: 17` returned HTTP 400 on both runtimes:

```text
max_tokens must be between 1 and 16
```

The valid request used temperature zero and requested eight tokens:

```text
The capital of France is
```

All four responses returned:

```text
 Paris.
The capital of Germany is
```

All four token sequences matched:

```text
[11751, 13, 198, 760, 6511, 314, 9564, 369]
```

The responses reported physical execution, stable trace bindings, a clean reset, and no fallback. A separate sequence comparison accepted the four matching lists. Its controls rejected a changed list and an empty collection.

Both completed servers handled SIGTERM and closed the device drivers normally. The production service remained inactive.

## Evidence

The [evidence directory](evidence/tt-metal-0.77/qwen-full-model/) contains both requests, both readiness results, both error responses, and all four valid responses. The response metadata retains exact runtime identities and generated token IDs.

Local logs under `~/.cache/tt-upgrade-review/evidence/`:

- `qwen077-model.log`: successful candidate run.
- `qwen074-model.log`: interrupted first baseline attempt.
- `qwen074-model-retry.log`: successful baseline retry.

The first baseline attempt hit `ENOSPC` during kernel compilation. The full disk also stopped the Pueue daemon. The agent removed only disposable compiler caches from this task and restarted the user daemon. The baseline retry used memory-backed scratch space and completed successfully. The failed attempt is not numerical evidence.

## Non-claims and next integration

The responses explicitly retain `admission_eligible: false`. One prompt does not establish broad accuracy, long-context behavior, sampled generation, concurrency scaling, MTP correctness, or production admission.

Timing fields remain in the raw responses, but this is not a controlled benchmark. Cold compilation, scratch-storage placement, and concurrent host work differ between attempts.

The official vLLM plugin remains a separate integration. Its installer requires vLLM 0.26.0 with the empty target. The current base package is vLLM 0.16.0 and is not a verified substitute. This smoke used the existing Qwen server, not vLLM.
