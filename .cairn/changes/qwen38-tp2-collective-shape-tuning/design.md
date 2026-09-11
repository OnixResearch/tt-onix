## Context

The actual TP2 shapes include GDN output input width 3072 and MLP intermediate width 8704 per chip. The adopted source already uses wide 1D grids. Its line-mesh `tt_all_reduce` branch performs reduce-scatter.

## Decisions

Build an inventory from source plus matched device traces. Record semantic collective sites separately from kernel launches. Include norm/input gathers, output reductions, LM-head work, tile padding, and layout changes.

Reuse `decode_component_probe` for the full-step control. Add narrow attribution for the selected shape or collective. Synchronized phase timings rank device work but do not establish overlapped service latency.

The planner selects from a small predeclared candidate set. Each experiment changes one mechanism: a TP2 grid, a layout transition, or a small-message reduction implementation. Keep dtype, precision, links, and topology fixed unless they are the named candidate.

The 0.77 direct reduce-scatter is a lead, not an admitted substitute. Its signature, mesh support, and reduction order need source and capability review first.

## Validation

Positive tests cover legal TP2 shapes and exact output layout. Negative tests cover wrong head divisibility, unsupported grids, missing links, incomplete barriers, overflowed tile counts, and stale runtime capability records.

Kernel changes require applicable MMIO, config, semaphore, mailbox, CB, NoC, source-register, and latency audits. Test nonzero state and repeated calls, not only zero-state output.

The physical screen includes component traces and complete decode under matched roots. Report any full-step regression despite an isolated gain. Verify request latency after a surviving candidate composes with the service.

## Alternatives and rollback

The earlier DRAM-sharded projection experiment remains the negative control. Do not compare against AUTO when the selected path is already tuned. Do not assume all-reduce from a helper name.

Retain the existing grids and collectives for unsupported shapes and rejected candidates. This change does not buy more cards or change physical cabling.
