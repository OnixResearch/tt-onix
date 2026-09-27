## ADDED Requirements

### Requirement: Resident slot decode trace

r[tenstorrent.qwen38_p150x2.resident_slot_trace]

A new continuous-slot group MUST keep the resident decode trace only when it is the one resident trace, has the group's width and route, keeps a stable binding, and the group's refills can replay the masked prefill traces. Otherwise the adapter MUST release it before the group's first refill. A kept trace MUST NOT replay before every row is reset and every decode input is restaged. Admission MUST compare the candidate with the deployed package in ABBA order, with exact tokens, identical prefix-cache use, and no slower prefill.

#### Scenario: Isolated agent turn
- GIVEN a resident trace of the group's width and route, a stable binding, and intact masked prefill traces
- WHEN a turn that arrives alone opens a new group
- THEN the group resets its rows, restages every input, and decodes without capturing a trace.

#### Scenario: Unsafe residency
- GIVEN binding drift, a route change, or missing or stale masked prefill traces
- WHEN a new group opens
- THEN the adapter releases the trace, and the group's first decode window captures a new one.
