## Why

Trace capture is a documented corruption risk in the reviewed Qwen3.6-A3B
bundle and in the local Qwen3.8 MTP work. A second resident trace corrupts the
captured one. Zeroed priming writes wrong state that later rounds condition
on. A reallocated buffer silently breaks a trace that bakes buffer addresses.

The local Qwen3.8 trace controls, draft trace, TTFT parked-trace control, and
recovery stability control, each capture and replay traces. Without explicit
capture-hygiene gates, a capture bug produces plausible but wrong trace
evidence instead of a failed gate.

## What Changes

- Add a pure, device-free trace-capture-hygiene validation core that rejects
  evidence from a process that kept a second trace resident during capture,
  captured from zeroed priming, or reallocated a trace-bound buffer.
- Require release-before-capture, a real warm seed round, and in-place
  stable-address mutation of trace-bound buffers.
- Keep the gates diagnostic and separate from token and speed evidence, with
  production eager behavior unchanged.

## Impact

- **Files**: `pkgs/qwen36/service/qwen36_service` trace-validation core and
  adapter, trace control wiring, and device-free tests.
- **Testing**: positive and negative device-free pytest, Python compilation,
  Ruff, Qwen package checks, Cairn validation, and repository-quality checks.
