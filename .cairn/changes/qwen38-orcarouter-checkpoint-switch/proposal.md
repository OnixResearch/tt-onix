# Adopt the OrcaRouter Qwen3.8-27B checkpoint on P150x2

## Why

The owner requested the OrcaRouter Qwen3.8-27B-Uncensored model family. The linked release contains GGUF files, but the current TT-NN service loads safetensors.

OrcaRouter also publishes a BF16 source checkpoint. That source is the preferred candidate for the existing P150x2 path. It is not byte-equivalent to a GGUF quantization.

## What Changes

- Add a content-bound candidate manifest for the requested GGUF release and the recommended native BF16 source.
- Verify authenticated file access, tensor compatibility, tokenizer identity, and the complete source lineage before staging.
- Build separate TT-NN caches from the selected OrcaRouter source. Preserve the current checkpoint and its caches for rollback.
- Keep MTP disabled during initial adoption. Any later MTP path must use the matching OrcaRouter head and separate admission evidence.
- Compare quality and performance under the current runtime before an authorized service switch.

## Capabilities

### Modified Capabilities

- `qwen38-p150x2-runtime`: candidate selection, cache identity, text-service compatibility, and reversible checkpoint promotion.

## Impact

The future implementation touches candidate authority records, weight staging, cache validation, service identity, and the owning host configuration. The service keeps its exact P150x2 mesh and existing supported request limits.

The evidence-contract Cairn is a dependency. The runtime 0.77 candidate is not a dependency. Initial adoption must not combine runtime and checkpoint changes.

## Non-goals

This package does not implement a GGUF loader, port new Metalium operators, add vision, increase context limits, enable MTP, or change access controls. It does not deploy the Flash Next or MoE model.

This request authorizes a plan only. It does not authorize gated-model consent, large downloads, a service outage, device recovery, or a push.

## Evidence and verification

The [source review](../../../docs/research/qwen38-orcarouter-switch-2026-09-04.md) records immutable candidate revisions and the current compatibility blocker. Positive and negative fixtures precede a representative simulator smoke and any authorized physical run. Production promotion requires explicit owner approval and a verified rollback path.
