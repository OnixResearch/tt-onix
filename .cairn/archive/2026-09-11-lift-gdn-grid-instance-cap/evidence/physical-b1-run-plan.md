# Physical B=1 argument-agreement run: exact plan

Recorded 2026-09-11 for change `lift-gdn-grid-instance-cap`, tasks V8 and V9.

Everything below was verified against the live machine. Nothing was executed:
the run needs several minutes of device time and result interpretation, and
leaving a half-run device is worse than not starting.

## Why B=1 first, not B=8

At B=1 the decode head count is 24 against about 110 cores, so the chunk length
is one and the instance loop runs exactly once. B=1 does **not** exercise
chunking. It exercises the thing that is actually unverified: the **new runtime
argument layout**. The reader and writer moved their addresses from arguments
1..6 to 2..7, and the compute kernel gained its first runtime argument.

If the factories and the kernels disagree about that layout, B=1 fails
immediately and cheaply. That is the right first test. B=8 then tests the
chunking itself, at two instances per core.

## Machine facts

| Item | Value |
| --- | --- |
| Devices | `/dev/tenstorrent/0`, `/dev/tenstorrent/1` |
| Service holding them | `qwen38-p150x2.service` (running) |
| Model snapshot | `/home/brittonr/models/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Variant package | `/nix/store/wpayxyj9xjwglqvv00jy4pasqpi37wd9-qwen38-0.1.0` |
| Probe | `$OUT/bin/qwen38-p150x2-decode-component-probe` |
| Probe args | `--model-path --decode-profile --component-profile --runtime-identity --package-identity --output [--sequence-steps]` |

The probe has no fusion flag. The fused path is selected by environment, so
`QWEN35_GDN_DECODE_FUSION=full` is what makes the chunked code run at all.
Without it the probe exercises `off` and proves nothing about this change.

The service environment supplies the rest: `MESH_DEVICE=P150x2`,
`HF_HUB_OFFLINE=1`, `QWEN35_TEMP=0`, `HOME=/var/lib/qwen38-p150x2`,
`TT_CACHE_PATH` beside the snapshot.

## Isolation

`AGENTS.md` grants standing authority for service stops and hardware probes.
Restoration is not required. Use the existing helper rather than a bare stop:

```sh
ttwkv7-owner-control stop      # passwordless isolation
ttwkv7-owner-control restore   # when finished
```

## Run

```sh
OUT=/nix/store/wpayxyj9xjwglqvv00jy4pasqpi37wd9-qwen38-0.1.0
MODEL=/home/brittonr/models/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
export HOME=/var/lib/qwen38-p150x2
export HF_HUB_OFFLINE=1 MESH_DEVICE=P150x2 QWEN35_TEMP=0
export TT_CACHE_PATH="$MODEL"
export QWEN35_GDN_DECODE_FUSION=full     # required: selects the chunked path
export TT_METAL_DISABLE_SFPLOADMACRO=1

"$OUT/bin/qwen38-p150x2-decode-component-probe" \
  --model-path "$MODEL" \
  --decode-profile "$OUT/share/qwen36/decode-profile.json" \
  --component-profile "$OUT/share/qwen36/decode-component-profile.json" \
  --runtime-identity "$(readlink -f /nix/store/6ccynlvv46p54pmp2lb7dbs65hkyj5w9-tt-metal-0.77.0)" \
  --package-identity "$OUT" \
  --output /tmp/gdn-grid-b1-probe.json \
  --sequence-steps 8
```

`--sequence-steps 8` keeps it bounded. Model load takes about 77 s and first-run
JIT compilation adds more, so budget at least 10 minutes and run it under
`pueue_run` with a timeout rather than a 30-second tool.

## Interpretation, fixed before the run

| Outcome | Meaning |
| --- | --- |
| Probe reports the fused path engaged and 8 tokens generated | Argument layout agrees. Proceed to B=8. |
| Immediate error naming a runtime argument or CB | Layout mismatch. Compare the factory's `emplace_runtime_args` order against each kernel's `get_arg_val` index and correct the smaller side. |
| Error inside the scan kernels | The prefill path, which `--sequence-steps` may not reach. Separate run needed. |
| Clean run but tokens differ from the admitted reference | The chunking changed numerics. The NV search was kept deliberately so this should not happen; treat it as a real finding. |

A pass here is a **mechanism** result. It is not an admission, and it does not
authorise B=8 serving.

## What was not verified

The live service state was read, not changed. `--sequence-steps` may not
exercise the prefill scan kernels at all, so V9 may need its own run.
