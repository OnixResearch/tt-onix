## Context

The repository runs a serialized P150x2 service at `127.0.0.1:8000`. That
service admits B=4 with a matched receipt and reaches 57.33 aggregate
tokens/second. A third party running the same checkpoint on the same two cards
reaches 100.5 aggregate tokens/second at B=8 through a vLLM plugin.

The gap is a serving stack, not a kernel. Continuous batching, arrival-driven
scheduling, and a standard OpenAI endpoint all live in the serving layer.

The preflight recorded that the reviewed plugin targets vLLM 0.26.0 built with
the empty target device, and that the local base is vLLM 0.16.0. Several
packages differ in a direction a version override cannot fix. NumPy must go
below 2 while the base is 2.4.4, and Torch must go to 2.11.0 while the base is
2.12.0. Both are ABI-relevant for the packaged TT-NN runtime.

The reference two-card serving recipe also carries load-bearing flags whose
absence produces misleading failures. `--no-enable-prefix-caching` is mandatory,
because prefix caching breaks engine construction for every Qwen3.5, 3.6, and
3.8 GDN model. `l1_small_size: 24576` is required or `ttnn.conv1d` cannot
allocate, and the error misleadingly reads as an out-of-memory condition.
Readiness takes about 510 seconds against a default 30-second probe, which
restart-loops forever and reads as a broken image.

## Decisions

### Decision: Build an isolated profile and never mutate the deployed environment

**Choice:** Add a new package profile for the serving stack. Do not change a
dependency in the deployed Qwen environment, and do not suppress a new
requirement to force an install.

**Rationale:** The deployed environment carries admitted physical receipts. A
dependency change there would invalidate them silently. The preflight already
stated this constraint and this change keeps it.

### Decision: Keep the Starlette floor and record it as non-relaxable

**Choice:** Require Starlette at or above 1.0.1. A build that needs a lower
floor is rejected rather than adjusted.

**Rationale:** The floor is a security bound for CVE-2026-48710. A serving
profile is not a reason to move a security floor.

### Decision: Build vLLM from source with the empty target device

**Choice:** Build against a pinned vLLM tag with `VLLM_TARGET_DEVICE=empty`,
and install common requirements explicitly so no CUDA wheel metadata resolves.

**Rationale:** vLLM's PyPI metadata is generated on a CUDA machine, so a plain
install resolves `requirements/cuda.txt` regardless of the target device. That
pulls Torch and CUDA packages that fight the TT-Metal Torch. Building from
source with an explicit common requirement set keeps the packaged Torch.

### Decision: Prove the ABI before any serving run

**Choice:** A coherent Python, Torch, and TT-NN ABI check must pass before any
model trial. A plugin import alone is explicitly insufficient, because the
plugin does not declare vLLM as a resolvable dependency.

**Rationale:** An import that resolves proves only that a module was found. It
does not prove that two Torch builds agree or that the TT-NN extension loads.

### Decision: One bounded non-MTP trial, and no throughput claim

**Choice:** Run one bounded non-MTP model trial through the finished stack. Do
not enable MTP, do not tune a serving parameter, and do not claim a throughput
result in this change.

**Rationale:** MTP serving is a separate open change with its own gate. The
first useful fact is whether the stack serves this model correctly at all. A
serving-latency claim belongs to a later change with matched roots.

### Decision: Record the load-bearing serving flags as profile-owned policy

**Choice:** The profile owns `--no-enable-prefix-caching`, `l1_small_size`,
`--max-num-seqs`, the block size, the trace region size, host sampling, and the
P300 mesh descriptor path. It also owns a readiness budget above the measured
startup time.

**Rationale:** Every one of these flags prevents a specific misleading failure.
Recording them as policy keeps a later operator from removing one and reading
the resulting error as a broken image.

## Risks / Trade-offs

- The dependency resolution may not close. The published pinned set may simply
  be incompatible with this repository's Nix environment. The deliverable is
  then the exact unsatisfiable constraint pair, not a forced install.
- The profile may build while the model still fails to serve. The bounded trial
  exists to separate those two outcomes.
- A second Torch in the closure is a real ABI risk for the TT-NN extension. The
  ABI check must fail closed.
- On-device sampling is hard-refused below TP=4 because the vocabulary gives
  124160 logits per device against a 65536 ceiling. The profile must require
  host sampling and must record why.
