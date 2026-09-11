# Physical result

## Scope

The bounded control used physical PCIe devices `0` and `1`, mesh `(1, 2)`, and CPUs `8-31`. Production stayed eager before and after each attempt.

## First warmup attempt

Package `/nix/store/yyxzjmgnn6bhfdrmnnbddbyz436rbw2c-qwen36-0.1.0` failed before measured roots. The persistent trace rejected this exact binding error:

`MTP trace binding changed before-controlled-replay: kv_buffer_identity`

Package-matched source inspection showed that `TPAttention.reset_state()` allocated new `k_caches` and `v_caches` objects. The trace remained bound to obsolete cache objects. The owner control restored eager service after the error.

## In-place reset attempt

Package `/nix/store/9hybv1i5zxc3sl811wsnnn854f1h7wvz-qwen36-0.1.0` allocated the MTP caches once and reset their data from fixed zero sources. This change removed the immediate cache-identity error.

The control started its authorized outage at `2026-08-22T20:52:04-04:00`. It produced no receipt before the named `7200 s` timeout. The watchdog terminated the probe and restored the service at `2026-08-22T22:52:11-04:00`.

The service reached eager-ready state at `2026-08-22T22:53:27-04:00`. The `/healthz` response was `{"model": "Qwen3.8-27B", "status": "ok"}`.

## Decision

The persistent recovery trace is rejected. No complete matched roots exist, so the control cannot establish state parity, timing stability, or recovery gain.

The timeout is not performance evidence. It does not identify whether trace coexistence, device execution, host execution, or another mechanism prevented completion.

MTP, traced target decode, and production behavior remain unadmitted. Production remains eager.
