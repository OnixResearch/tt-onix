# Blackhole hazard-audit source preflight

Recorded 2026-09-13. This is a reachability record, not a hazard verdict.
The full audit has not started.

## Source manifest

| Source | Role | Architecture and audit class | Reachable | Observation |
| --- | --- | --- | --- | --- |
| tt-isa-docs MCP | Primary ISA | WH/BH hardware semantics | No | The available MCP tools contain no ISA server. |
| DeepWiki MCP | Equivalent ISA corpus | WH/BH hardware semantics | Yes | `deepwiki_read_wiki_structure` returned the `tenstorrent/tt-isa-documentation` index. |
| Confluence / Atlassian MCP | Internal workaround authority | BH NoC and dataflow-CB | No | No corresponding tool is available. The required opaque pointer is `1001357404`. |
| Pinned sfpi-gcc | Compiler scheduling authority | BH instruction latency | Yes | GitHub API access timed out for a timing file. Direct public-source fetches reached all three required files. |
| Dataflow API source | Issued operations | BH NoC and dataflow-CB | Yes | The package-matched `tt_metal/hw/inc/api/dataflow/dataflow_api.h` is present. |
| LLK source | Issued operations and software contracts | BH, all applicable classes | Yes | The package-matched Blackhole LLK tree is present. |

Quasar and Wormhole kernels are outside this change's audit scope.
The missing ISA MCP therefore does not add a Quasar coverage claim.
DeepWiki supplies the supported Blackhole alternative.

The live exact path `BlackholeA0/NoC/Ordering.md` returned `404: Not Found` through the raw public source endpoint.
The API attempt timed out.
The missing page proves no positive or negative ordering property.
Without the internal source, the NoC/dataflow workaround review remains incomplete.

## Compiler pin

The unmodified TT-Metal 0.77 source names SFPI 7.69.0.
`pkgs/tt-metal/sfpi.nix` carries SFPI 7.70.0, release build 847.
`pkgs/tt-metal/default.nix` updates the installed compiler metadata to that version.
The audit must use the installed version, not the unmodified source metadata.

The source chain is:

- `tenstorrent/sfpi` at `6659aeb9e350dd8eef69edbdfa4f963759b04c44`.
- Its `gcc` submodule at `6ffe1f327045f30dadd241259463f4ce83eb0bd9`.
- Files under `gcc/config/riscv/tt/`: `rvtt.md`, `sfpu-ops-bh.h`, and `rtl-rvtt-schedule.cc`.

Reachability does not establish a latency verdict.
The later audit must read the exact source text, not Markdown-transformed compiler code.

## Required choice

`.pi/skills/ttllk-race-audit-all/SKILL.md` requires:

> Do not begin auditing until the user chooses

The user can authorize a bounded audit, supply access to the missing source, or add another source.
Until that choice, V4 stays open.
No full hazard pass, hardware test, integration, or physical admission follows from this preflight.

The separate capacity and symbolic-order reviews concern source/API contracts only.
Their runner produced a report but reached its process timeout before a clean exit.
The parent checked the reported arithmetic and regression results directly.
Those reports do not replace any of the nine required audits.
