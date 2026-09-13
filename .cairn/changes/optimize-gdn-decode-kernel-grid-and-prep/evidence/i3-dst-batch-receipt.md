# Historical B=8 Dst receipts

Corrected 2026-09-13. All four receipts use the original, unbounded Dst candidate.
The bounded replacement does not inherit their results.
The B=8 binding remains false.

## Recorded results

| Round | Client | Group failures | Speedup | Retention | TTFT ratio | Tail ratio | Receipt verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | curl processes | 0 | 6.0338 | 0.7542 | 7.4895 | 1.2990 | admitted |
| 2 | curl processes | 0 | 6.1401 | 0.7675 | 6.0018 | 1.2846 | admitted |
| 3 | curl processes | 8 | 5.5761 | 0.6970 | 20.4056 | 1.6614 | rejected |
| 4 | preconnected barrier | 0 | 5.8395 | 0.7299 | 13.6542 | 1.4619 | rejected |

The bounds remain speedup >=6.0, retention >=0.75, TTFT ratio <=9.0, and tail ratio <=1.5.
Round 4 passed the tail bound and failed the other three bounds.
The earlier statement that it failed every metric was incorrect.

All four rounds used eight prompts, 64 generated tokens, and three measured runs per root.
The root order was `single,batch,batch,single`.
The receipts compare exact generated tokens and text between single and batch routes.
They also record distinct state-slice labels and clean reset flags.
They do not compare recurrent tensor contents against a control kernel.

## Identity and preservation

The original package was `/nix/store/yajnqp5cpbb65ky02nykj6m3yr0i8k3r-qwen38-0.1.0`.
Its execution identity was `/nix/store/9d3cs4kaj9ni80d5frhp9ayzpw6nax7d-qwen36-execution-identity-0.1.0`.
Its runtime was `/nix/store/s35xbznhgi5r3csls10lrjslarxaw278-tt-metal-0.77.0`.
The installed launcher overwrites `PYTHONPATH` with this runtime, despite the stale value in the outer harness.

The persistent private copy is `.pi/evidence/qwen38/gdn-reliability-20260913/original-rounds/` in the primary checkout.
It contains all four original receipts, raw response roots, server logs, and receipt harnesses.
The copy preserves the failed rounds rather than replacing them with new attempts.

Verified BLAKE3 digests:

| Receipt | Digest |
| --- | --- |
| `receipt-dst.json` | `9aefac99e6009f7b633a2207bbeb638bb887ed0345783fb28e1e98feb2224f2c` |
| `receipt-dst2.json` | `edeebc8f9ad5c58129d62e027524f64543a183b7f2bb08c9e07d81e87a0c5f54` |
| `receipt-dst3.json` | `04ab53399f7d8d7c57581ca699073e9a434e93569e179ee7772e3d6b9d129f1a` |
| `receipt-dst4.json` | `30530fd22069162d200e0ec46b63492ca354ee0fc816b841cfbe0c5c298c5c81` |

## Corrected interpretation

Zero logged group failures in round 4 do not establish that the client eliminated the defect.
The first two curl rounds also had zero failures.
A barrier does not guarantee request arrival within the collection window.

The old logs omit the exception message and do not identify the exact failing stage.
Therefore the earlier missing-warm-trace explanation was not proven.
A separate deterministic reproduction now shows that the service rejected unsupported measurement widths after device work.
The correction moves this same policy decision before all adapter effects.
It does not establish the cause of each historical HTTP 500.

The four receipts establish nonrepeatable admission under this protocol, not a proven noise mechanism.
Round 3 remains a rejected receipt with eight logged group failures.
No prior exclusion rule justifies removing it from the record.

The [capacity review](i3-dst-batch-candidate.md) rejects the original candidate on an independent source defect.
The [measurement review](protocol-measurement-power.md) records the remaining timing gap without changing the metric.
