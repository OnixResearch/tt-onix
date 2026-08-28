# TP2 tuning disposition

Date: 2026-08-20

## Verdict

`NO_PROFILED_CANDIDATE`

No TP2 tuning winner is selected.
Current Qwen3.8 TP2 configurations and safe fallbacks remain unchanged.

## Current behavior

- The service uses tensor parallel degree two on physical P150 device IDs `0` and `1`.
- The package keeps existing fixed decode program configurations.
- Unknown prefill tensor-parallel degrees inherit existing upstream fallback values.
- `tp2-runtime-compat.patch` avoids the TP4-tuned fused GDN prefill reduction on two devices.
- The production service reported `active` after the traced-decode maintenance window.

These are compatibility and execution facts.
They do not prove that the selected grids, matrix settings, or collective settings are TP2-optimal.

## Missing sweep evidence

No complete evidence set contains:

- attributed TP2 operation device time, host gaps, launch counts, fabric time, exact shapes, and memory;
- a typed legal candidate domain for every required operation family and prefill bucket;
- complete comparable baseline and candidate samples;
- deterministic operation winners;
- generated explicit TP2 configuration identities;
- matched end-to-end correctness, decode, prefill, TTFT, tail-latency, memory, and cleanup results.

## Decision

No configuration source, generated data, package setting, or service behavior changes.
A future tuning effort must use a new validated Cairn change and start with an attributed baseline profile.

## Non-claims

- Current configurations are not described as TP2-tuned.
- This disposition does not show that no faster TP2 configuration exists.
- TP4 or TP8 tuning results do not prove a TP2 winner.
- No microbenchmark or model speedup is claimed.
