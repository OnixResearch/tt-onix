# Acceptance

## Result

The bounded diagnosis proved the first causal boundary. Package-default BF16 model matmuls use HiFi2 with BF16 destination accumulation.

At generated-token index `10`, layer `0` operation `r_projection` differs from the diagnostic HiFi4 FP32-accumulation control. The maximum BF16 output error is `0.00390625`.

The broad high-precision control restores all sixteen pinned CPU tokens exactly. The device FP32 recurrent-matrix BLAKE3 does not match the pinned CPU authority.

## Decision

Do not create candidate v3. Do not start admission, performance, canary, deployment, or observation gates.

Production remains profile `rwkv7-p150x2-persistent-decode-v3` with windows `[2, 4, 8]`.

## Evidence

- Aggregate receipt: `bounded-diagnosis.json`
- Device matrix BLAKE3: `9aca7260763e0a728b30175c40f65d43eee5743b68aea593675f94936d507c98`
- CPU matrix BLAKE3: `0d904c3766a7cb13aaaeb13266e65623d516056ab6cfa2d697ccb3610dc593d7`
- Exact generated-token parity: passed
- Exact FP32 recurrent-matrix parity: failed

Raw tensors, logs, caches, compiler output, and profiler output were not retained.
