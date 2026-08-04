# RWKV-7 LoRA CPU Reference

Device-free authority for RWKV-7 supervised fine-tuning with LoRA. This
package establishes the bounded CPU reference that later Tenstorrent
bring-up Cairns compare against. It makes no accelerator claims.

## Model authority

The reference binds exactly one model:

- `RWKV/RWKV7-Goose-World2.8-0.1B-HF` at revision
  `d81965cb4e1a9f96696b4f70b84212b8f2e43216`.
- Checkpoint BLAKE3
  `905f82048a64b881f9267117a398feb8a8a92bcc5233666bf67904e0d899d0e5`.
- Tokenizer vocabulary BLAKE3
  `3997a74891dd68ced8daadae0d7475274b08988c9263ca042896c8106967aef2`.
- 12 layers, hidden 768, 12 heads, head width 64, vocabulary 65536.
- Precision class `fp32_from_bf16`.

Any drift in these identities fails admission before tensor execution.

## Data preparation

SFT input is local JSONL. Each record is an object with nonempty `prompt`
and `response` strings and no other fields. Records are tokenized and laid
out into a fixed sequence length. Prompt tokens and padding are masked from
the supervised loss; only response target positions contribute labels.

Malformed records, empty prompts or responses, unknown fields, overlong
samples, and records that produce no active labels fail preprocessing
before model execution.

## LoRA targets

LoRA is added only to an explicit role allowlist:

- `attention.receptance`, `attention.key`, `attention.value`,
  `attention.output`
- `feed_forward.key`, `feed_forward.value`

Every base checkpoint tensor stays frozen. The selectors `all-linear`,
`*`, unknown roles, duplicate roles, and empty role sets are rejected.

## Commands

Build and run the bounded validation:

```sh
nix build .#checks.x86_64-linux.rwkv7-lora-reference
```

The check exports the Nickel runner configuration twice and requires the
two exports to match byte for byte. It rejects a configuration with an
empty LoRA target set. It then runs `reference.py`, which validates:

- WKV7 forward output is finite on bounded tensors.
- Autograd gradients, the paper Appendix H manual backward, and bounded
  finite differences agree within named tolerances.
- One optimizer step changes at least one adapter tensor and leaves every
  base tensor byte-identical.
- Adapter tensors save to Safetensors, load byte-identically, and produce a
  BLAKE3 manifest. Resume fails closed when bound identities drift.
- The pinned checkpoint opens, stores `bf16`, and converts to finite FP32.

## Artifact interpretation

The receipt identifies the model authority, the loaded checkpoint dtype and
BLAKE3, the data plan, the LoRA plan, the gradient-agreement deviations,
the one-step result, and the adapter manifest BLAKE3. A passing receipt has
`outcome = bounded_cpu_reference_pass`.

## Resume rules

Resume is admitted only when the saved manifest and the current inputs agree
on model, tokenizer, LoRA targets, rank, optimizer policy, data identity,
and every adapter tensor BLAKE3. Any difference fails before model
mutation.

## Non-claims

A passing receipt does NOT establish:

- Tenstorrent execution, placement, or performance.
- Useful convergence or generalization.
- Dataset quality.
- Production fine-tuning quality.
