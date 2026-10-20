# metalium-rs Adoption Evidence

Date: 2026-10-19

## Fork identity

- Fork: `https://github.com/brittonr/metalium-rs`
- Commit: `dd5d316e3de31150fcee935ff321a6fc8ac863c2`
- Upstream base: `Knight-Ops/metalium-rs` at `b22a245db18252b131679aca992dac85c9f7d812`
- Fork tree BLAKE3: `8d82674236a37e5926f6ae30b11bfea7dca317eb4dcb2513ccb94928736ec188`
- `Cargo.lock` SHA-256: `ef26d17c2994cecba303c91753f265c1e623531ec42a0585d9b130330909e088`
- Nix source hash: `sha256-ODUQl+0E2MTNf0pvnHkspmxRByYQdo77N+xJWJGm8rc=`
- Fork delta: `LICENSE` (Apache-2.0 text), `crates/burn-tt/tests/essentials.rs` (F16 capability
  assertion now expects `Storage | Arithmetic`, without `Accelerated`), and `crates/rwkv7`
  (WKV reference/device gate, synthetic layer gate, real-checkpoint layer-zero loader, and the
  all-12-layer gate, logits/oracle comparison, the exact byte-trie tokenizer, the greedy next-token
  check, the stateful decode chain, and prompt generation with wrapper tokenization).
  Fourteen files, +2,646/−2.

## Validation

`cargo test --workspace --no-fail-fast` at the fork commit:

- 1,544 passed
- 0 failed
- 46 ignored
- 266 test binaries

The run used the pinned ISA specification, ttsim `v1.10.9`, and MNIST inputs. Simulator gates are
simulated evidence. No physical device was opened.

## Package

- Flake input: `rust-overlay` at `550547cefc67e01a6fd11a09e4096a3c844ca840`, following the
  repository nixpkgs.
- Toolchain: Rust 1.97.1 with `riscv32im-unknown-none-elf` and `llvm-tools`.
- Package: `.#metalium-rs`, revision `8600f14`, `Cargo.lock` from the source.
- MNIST: four CVDF files fetched with the upstream `PINS.toml` SHA-256 values and decompressed.
- Excluded dev-only crates: `tt-tests`, `tt-ttsim`, `tt-ttsim-sys`, `burn-tt-conformance`,
  `tt-models`, `xtask`.
- Output: `/nix/store/xqirzkrkidb62r414pwy31bk0g6l3pbg-metalium-rs-0.1.0-unstable-2026-10-19`
- Installed artifacts: `bin/tt-mnist` (26 MiB), `bin/wkv7-gate` (4.4 MiB),
  `bin/rwkv7-layer-gate` (6.3 MiB), `bin/rwkv7-model-gate` (6.3 MiB),
  `bin/rwkv7-tokenizer-gate` (896 KiB), and `bin/rwkv7-generate-gate` (6.4 MiB).
- Wrappers: `/nix/store/80ylgakhv481qb4zjwn1sdrpmkbz5cli-rwkv7-layer-checkpoint/bin/rwkv7-layer-checkpoint`,
  `/nix/store/4f58rghjjmjn17nf9alq30dvl5lplbfq-rwkv7-model-checkpoint/bin/rwkv7-model-checkpoint`,
  `/nix/store/9nydqg3zvb9aym8hgn4l5imnikhp4bl7-rwkv7-tokenizer/bin/rwkv7-tokenizer`, and
  `/nix/store/02lfdd67v5grrns504a6g8xv4fv19m5w-rwkv7-generate-checkpoint/bin/rwkv7-generate-checkpoint`,
  and `/nix/store/bc39r9djbyrkflyybjmby95370j50vq3-rwkv7-prompt-checkpoint/bin/rwkv7-prompt-checkpoint`,
  bound to the pinned checkpoint and vocabulary. The prompt wrapper sets `TT_BATCH=0`.
- The package check phase runs `cargo test --workspace --no-fail-fast` with the pinned specification,
  ttsim `v1.10.9`, MNIST, `cfg_defines.h`, and `ioctl.h`. It passes: 1,544 tests, 0 failures,
  46 ignored, 266 binaries, in about 21 minutes locally.
- App: `.#metalium-rs-mnist`.

## Binary smoke

- `tt-mnist --help` prints the usage text and exits 2.
- `tt-mnist --infer --host --batch 1 --passes 1` loads the embedded corpus
  (60,000 train / 10,000 test images), prints the model and storage plan, then fails at
  `could not open the card: Invalid argument (os error 22)`. The host flag does not bypass the
  device open in this mode.

The smoke proves the binary starts, parses arguments, and unpacks the embedded corpus. It was run
before isolation, when `qwen38-p150x2.service` held the cards, so it does not prove a successful
device path.

## Physical gate

`qwen38-p150x2.service` was stopped through the permitted `systemctl stop` rule before these runs.
The service was inactive and `lsof` showed no owner on `/dev/tenstorrent/1`.

Both runs exited 0:

| Command | Result |
|---|---|
| `nix run .#metalium-rs-mnist -- --steps 1 --card 0` | loss 2.3095; 127.34 ms/step; test accuracy 13.14%; 221.6 MB up, 0.04 MB down |
| `nix run .#metalium-rs-mnist -- --steps 1 --cards 0,1` | loss 2.3095; 119.18 ms/step; test accuracy 13.14%; 221.6 MB up, 0.04 MB down |

Both report a 784-512-256-10 network and one Tensix tile per invoked card. The mesh run reports
matmuls sharded over Ethernet. These are single-step smoke timings, not benchmarks: no warmup,
median, or repeated-run protocol was applied.

## WKV7 native gate

The fork now carries `crates/rwkv7`: the token-at-a-time Goose recurrence as a pure F32 host
reference and the same equation composed from Burn tensor operations on `TtBackend`. It uses
`accurate_matmul`, not the fused Tensix kernel.

`cargo run -p rwkv7 --bin wkv7-gate -- --card 0` and `nix run .#metalium-rs-wkv7 -- --card 0`
on 2026-10-19:

```text
host reference: step1 out 0.406937, state 0.247399; step2 out 0.236715; control gap 0.311922
card 0: step1 out gap 1.192e-7, state gap 2.980e-8; step2 out gap 5.960e-8, state gap 2.980e-8
device gate PASS
```

The second token consumes the first token's device state, so this is a state-carry check. It is one
card, two tokens, and a composed op graph. The packaged binary reproduced the same gaps. It does not establish a fused-kernel cost, a layer, a
model, or mesh behavior.

## Synthetic RWKV-7 layer gate

`crates/rwkv7/src/layer.rs` implements one layer from the pinned checkpoint equations: pre/attn/ffn
norms, token shift, the time-mixing LoRA projections, the WKV recurrence, group norm, value
correction, output projection, and the squared-ReLU channel mixer. The gate runs layer zero with
synthetic weights on `Flex` (CPU FP32) and `TtBackend`, for two tokens, with `accurate_matmul`.

`nix run .#metalium-rs-rwkv7-layer -- --card 0`:

```text
flex: token1 max 1.834199, token2 max 1.805208, state max 0.156451; control gap 0.000691
card 0: token1 hidden gap 5.960e-7
card 0: token2 hidden gap 3.576e-7
card 0: token1 state gap 1.788e-7
card 0: token2 state gap 1.490e-7
layer gate PASS
```

Two fresh runs produced the same gaps. A first attempt immediately after stopping the service
failed with a stopped backend queue (`element-wise 0x2 ... queued list 2 failed`); the cause is not
established and the later runs were clean.

The gate uses synthetic weights, layer zero only (no v-LoRA), and no real checkpoint. It is not a
real-weight layer, full-model, fused-kernel, or performance result.

## Real-checkpoint layer zero

`crates/rwkv7/src/checkpoint.rs` reads the pinned BF16 safetensors checkpoint and builds a
layer-zero `LayerSpec`: all norms, the token-shift factors, the projection matrices, the rank-64
w/a LoRA, the rank-128 g-LoRA, per-head `r_k`, and the FFN. Token inputs are embedding rows 1 and 2.

Checkpoint identity:

- Path: `/nix/store/x9czgfz6spvwg438pv6hfgsla6lnv1j4-rwkv7-goose-world2.8-0.1b-d81965cb4e1a9f96696b4f70b84212b8f2e43216.safetensors`
- BLAKE3: `905f82048a64b881f9267117a398feb8a8a92bcc5233666bf67904e0d899d0e5`
- 365 MiB, BF16

`cargo run -p rwkv7 --bin rwkv7-layer-gate -- --checkpoint <path> --card 0`, and the packaged
`nix run .#metalium-rs-rwkv7-checkpoint -- --card 0`, produce:

```text
checkpoint layer 0: flex token1 max 1.473932, token2 max 0.658408, state max 1.203975; control gap 1.700190
card 0: token1 hidden gap 1.621e-5
card 0: token2 hidden gap 4.113e-6
card 0: token1 state gap 1.989e-5
card 0: token2 state gap 1.974e-5
layer gate PASS
```

This is real layer-zero weights, not the full model, not nonzero-layer v-LoRA, and not a fused
kernel. The tolerance is absolute 1e-3 on hidden and state.

## All 12 real-weight layers

`crates/rwkv7/src/checkpoint.rs` now loads every layer, including the rank-32 v-LoRA on nonzero
layers. The model gate carries the layer-zero value anchor through the stack, runs embedding rows 1
and 2 through all 12 layers, applies `model.norm`, and compares the final hidden vector and all
recurrent matrices.

`cargo run -p rwkv7 --bin rwkv7-model-gate -- --checkpoint <path> --card 0`:

```text
flex: final hidden max 74.441063, recurrent max 54.999863; control gap 25.011158
card 0: final hidden gap 1.678e-3 (max 7.444e1, relative 2.255e-5)
card 0: recurrent state gap 4.005e-4 (max 5.500e1, relative 7.283e-6)
model gate PASS
```

The bound is relative `1e-4` of each output's max magnitude. This is Flex FP32 vs TtBackend on
real weights. It does not evaluate the 65536x768 logits, compare against the Python reference, or
establish generation or performance parity.

## Logits and Python-reference comparison

The model gate now evaluates `lm_head` (`[65536, 768]`) and accepts `--oracle DIR` with the
reference `final_hidden.f32`, `logits.f32`, and `states.f32`.

The Python reference was generated with `evidence/rwkv_make_reference.py`, the harness's own
equation source, and the pinned checkpoint under a
`python3.withPackages [torch safetensors blake3 numpy]` environment:

```text
final_hidden (768,) max 74.44098663330078
logits (65536,) max 19.674400329589844
states (589824,) max 54.99984359741211
```

Reference file identities:

| File | Bytes | SHA-256 |
|---|---:|---|
| `final_hidden.f32` | 3,072 | `f2531d339065bee2c518bc5f3e9ed32a67f32e161fef00c1ebf31a6aeda96f4e` |
| `logits.f32` | 262,144 | `c12eb0eb011dba3c04cc954a32f14dd6661e247229220a9d869d3eff141cc6b9` |
| `states.f32` | 2,359,296 | `1f1f17a2f464cdf32443e5a0eaba03d2575b90f17605938c17bb8d70de21f446` |

`cargo run -p rwkv7 --bin rwkv7-model-gate -- --checkpoint <path> --oracle /var/tmp/rwkv-ref --card 0`
and `nix run .#metalium-rs-rwkv7-model -- --oracle /var/tmp/rwkv-ref --card 0` produce:

```text
flex hidden vs oracle: gap 7.629e-5 (bound 5.0e-1)
flex logits vs oracle: gap 1.717e-5 (bound 1.0e0)
flex states vs oracle: gap 3.052e-5 (bound 1.0e0)
card 0: final hidden gap 1.678e-3 (max 7.444e1, relative 2.255e-5)
card 0: logits gap 2.680e-4 (max 1.967e1, relative 1.362e-5)
card 0: recurrent state gap 4.005e-4 (max 5.500e1, relative 7.283e-6)
tt hidden vs oracle: gap 1.614e-3 (bound 5.0e-1)
tt logits vs oracle: gap 2.728e-4 (bound 1.0e0)
tt states vs oracle: gap 3.939e-4 (bound 1.0e0)
model gate PASS
```

The oracle bounds are the harness's 0.5 final-hidden, 1.0 logit, and 1.0 recurrent-state
tolerances. This establishes the two-token logits and state against the Python equation reference.
It does not test tokenization, generation, serving, or performance.

## Tokenizer: toks refused, exact byte trie

`actual-computer/toks` 0.3.2 (Apache-2.0) was tried first. Its C `toks_load` accepts a HF
`tokenizer.json` or a tiktoken model; RWKV ships neither. The probe returned:

```text
/var/tmp/rwkv-model: load=-1 code=-1 what=model directory without tokenizer.json, tiktoken.model or qwen.tiktoken
/var/tmp/rwkv-model/rwkv_vocab_v20230424.txt: load=-1 code=-1 what=tiktoken.model needs tokenizer_config.json and tokenization_kimi.py beside it
/var/tmp/gpt2-tokenizer.json: load=0 code=0 what=
```

The RWKV tokenizer is leftmost-longest greedy matching over raw UTF-8 bytes. 486 of 65,529
vocabulary tokens are not valid UTF-8 individually, and valid text emits them: `😀` is
`[3319, 153, 129]` with bytes `f09f`, `98`, `80`. A JSON added token cannot carry those bytes.
See `docs/research/toks-rwkv-tokenizer.md`.

`crates/rwkv7/src/tokenizer.rs` implements the vocabulary's byte trie directly.
`cargo run -p rwkv7 --bin rwkv7-tokenizer-gate -- --vocab <path>` and
`nix run .#metalium-rs-rwkv7-tokenizer` pass the Python-derived vectors, including `😀` and
`U+10FFFF`, and round-trip the original bytes.

## Greedy next token

The model gate now ranks the top two logits. The harness token checkpoint's expected values are
`token 2, logit 2.8641083`, `runner-up 33, logit 0.89640886`, `margin 1.9676995`.

`cargo run -p rwkv7 --bin rwkv7-model-gate -- --checkpoint <path> --card 0` and
`nix run .#metalium-rs-rwkv7-model -- --card 0`:

```text
flex greedy: token 2 logit 2.8641133, runner 33 logit 0.8964117, margin 1.9677017
tt greedy: token 2 logit 2.8639894, runner 33 logit 0.8964123, margin 1.9675771
```

Flex is within 5e-6 of the expected logits; card 0 is within 1.3e-4. Both choose the same token
and runner-up. This is one greedy step, not a generation loop, sampler, or serving result.

## Stateful decode chain

`rwkv7-generate-gate` starts from seed token 1 with zero state and runs three greedy steps,
carrying every layer's recurrent matrix and token-shift vectors. The harness expects token 1 at each
step with logits `0.04493069`, `6.9543834`, `6.486726`.

`cargo run -p rwkv7 --bin rwkv7-generate-gate -- --checkpoint <path> --card 0` and
`nix run .#metalium-rs-rwkv7-generate -- --card 0` produce:

```text
flex step 0: token 1 logit 0.04495239
flex step 1: token 1 logit 6.95438051
flex step 2: token 1 logit 6.48672342
tt step 0: token 1 logit 0.04504108
tt step 1: token 1 logit 6.95432234
tt step 2: token 1 logit 6.48668861
decode gate PASS
```

Flex gaps are at most 2.2e-5; card 0 at most 1.1e-4. The gate compares generated token ids and
logits only; it is not a prompt-generation endpoint, sampler, or serving result.

## Prompt generation

`rwkv7-generate-gate` now accepts `--vocab PATH --prompt TEXT --generate N`. Wrapper
tokenization maps `<|rwkv_tokenizer_end_of_text|>` to id 0, `\n\n` to the wrapper EOS id 65530,
and every other segment through the byte trie.

Fixed harness prompt
`<|rwkv_tokenizer_end_of_text|>User: Hi\n\nAssistant:`. The packaged path is
`nix run .#metalium-rs-rwkv7-prompt -- --prompt '<fixed prompt>' --generate 3 --card 0`.

```text
prompt ids: [0, 24281, 59, 3880, 65530, 5585, 41693, 59]
flex generated ids: [3880, 45, 308]
flex generated text: " Hi, I"
tt generated ids: [3880, 45, 308]
tt generated text: " Hi, I"
prompt gate PASS
```

The bundled app sets `TT_BATCH=0` because the batched path exhausted contiguous GDDR slots after
the 8-token prompt. Disabling the pipeline alone is not required: the same run passes with
batching off and the default pipeline. Record this as an allocator blocker, not a numerical
difference.

This is a bounded greedy generation, not a sampler, stop-policy-complete endpoint, or serving
result.

## Card reset after a TT-Metal service

The first native run after stopping `qwen38-p150x2.service` failed at
`element-wise 0x2 ... queued list 2 failed (code 1)` on both the synthetic and real gates. A second
process passed. After `tt-smi -r` reset both cards, the first real-checkpoint run passed.

Treat a board reset as required between a TT-Metal service stop and native-stack execution. Record
it as an operational gate, not as a source defect.

## Evidence classes

- Device-free: fork build, package build, hermetic Nix test lane, toks probe, and the tokenizer
  gate.
- Simulated: the ttsim gates inside the fork validation tier.
- Physical: one training step on card 0, one on the `1x2` mesh, a two-token WKV7 device gate, the
  synthetic full-layer gate, the real-checkpoint layer-zero gate, the all-12-layer final-hidden,
  logits, and recurrent-state gate, one greedy next-token step, a three-step stateful decode chain,
  and a bounded prompt generation producing `" Hi, I"`. No sampler, serving, placement, long-run
  stability, or performance claim follows.

## Migration status

The native stack is the authority for new Rust device, firmware, kernel, and Burn-backend work.
Qwen3.6/3.8 and RWKV-7 remain on their TT-Metal/ttnn paths. Their native migration is blocked on
operator coverage, linear-attention/MoE kernels, recurrent-state carry, and physical evidence.
The per-target inventory is `docs/research/metalium-rs-parity-inventory.md`.
