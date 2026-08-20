# P150x4 hardware availability evidence

## Observation

- Date: 2026-08-20
- Host device directory: `/dev/tenstorrent`
- Visible device nodes: `0`, `1`
- Required topology: four compatible physical P150 devices
- Operator statement: no four-device system is available

## Verdict

`UNSUPPORTED_ON_AVAILABLE_HARDWARE`

The available host cannot run the required P150x4 physical comparison. The repository adds no P150x4 deployment path.

## Non-claims

- This verdict is not a measured P150x4 performance rejection.
- This verdict does not estimate capacity, latency, memory, fabric traffic, power, or load time.
- Simulator and upstream results do not replace the missing physical topology.
- Future hardware availability requires a new validated change and new physical evidence.
