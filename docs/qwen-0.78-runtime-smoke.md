# Qwen runtime smoke: TT-Metal 0.78

## Result

The packaged Qwen3.8-27B server loaded all 64 layers and served two identical
greedy requests on physical P150x2 hardware with TT-Metal 0.78.0. The generated
token IDs and text match the earlier 0.74/0.77 comparison exactly. This is a
bounded regression smoke, not broad model-accuracy, long-context, MTP, or
production-admission evidence.

The production `qwen38-p150x2` service remained stopped and unchanged.

## Runtime identities

| Role | Identity |
| --- | --- |
| TT-Metal | `/nix/store/lz9hma19v7hfhc53fldq6qhdkyyyjmrh-tt-metal-0.78.0` |
| Qwen package | `/nix/store/mpnqg0ans7kzpssxambmr4g7mkgx48sp-qwen38-0.1.0` |
| Execution identity | `/nix/store/qwq8fic901rksw3xs9yivypni299h33k-qwen36-execution-identity-0.1.0` |
| Checkpoint | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Decode profile | `qwen38-p150x2-traced-decode-v1` |
| Physical devices | 0 and 1 (1x2 mesh) |

The package build installs the upstream `infra/data_collection` module, which
the tt-metal Python output omits. Without it, the server imported
`models.perf.benchmarking_utils` and failed at
`from infra.data_collection.pydantic_models import ...` before opening a
device.

## Bounded run

The temporary server used:

- loopback port 8012 and a private HOME/cache under
  `~/.cache/tt-upgrade-review/evidence/qwen078-model-run2`;
- `MESH_DEVICE=P150x2`, `HF_HUB_OFFLINE=1`, `QWEN35_TEMP=0`;
- a 2048-token sequence limit and 16-token generation limit;
- `qwen36-p150x2-trace-candidate`, which selects candidate decode mode;
- no production model or tensor-cache writes.

Startup opened both devices, compiled firmware, loaded the tensor cache, ran
width-1 and width-4 warmups, and reported ready in 127.5 s. The candidate
reported 22,919,363,584 bytes allocated, 10,185,625,600 bytes free, and one
resident trace.

## Requests and evidence

Valid request:

```json
{"model":"Qwen3.8-27B","prompt":"The capital of France is","max_tokens":8,"temperature":0,"stream":false}
```

Both valid responses returned:

```text
 Paris.
The capital of Germany is
```

Token IDs were identical in both responses:

```text
[11751, 13, 198, 760, 6511, 314, 9564, 369]
```

Each response reported `admission_eligible: false`, `decode_path:
batched-traced`, `fallback_reason: null`, `clean_reset: true`, a stable trace
binding, and no fusion switch. Decode rates were 18.40 and 18.39 tokens/s;
first-token times were 0.619 and 0.568 s. These are two short requests, not a
benchmark.

An oversized request with `max_tokens: 17` returned HTTP 400 with
`max_tokens must be between 1 and 16`.

Portable evidence is under
[`docs/evidence/tt-metal-0.78/qwen-full-model/`](evidence/tt-metal-0.78/qwen-full-model/):
readiness, model list, both requests, both responses, and the invalid request
and response. The raw server log is not committed.

SIGTERM closed the device drivers, but the server logged a `ServiceStop`
traceback from `batch_service.close()` while joining the worker. The requests
had already completed with clean resets; this shutdown exception is recorded,
not converted into a clean-shutdown claim.

## Non-claims

- No production service start, deployment, or admission.
- No MTP, fused GDN, sampled generation, concurrency, long-context, or
  performance claim.
- No model-accuracy claim beyond the single matched regression prompt.
- The official vLLM plugin remains separate; this run used the packaged Qwen
  server.
