# Design

Traffic follows the pinned MIT tensixfuse output-stationary tile model: A per N block, B per M block, optional corresponding grid multicast, BF8/BF4 exponent storage, explicit intermediate and L1 buffers. Results are analytical, never counters.

Retention freezes exact historical six texts, local tokenizer asset SHA-256s and IDs. Scoring consumes full-vocabulary logits aligned to target indices 13..end, casts float32 then float64 log-softmax, and averages passage means equally. Receipts bind corpus, model/checkpoint, runtime, package, cache, precision flags and baseline digest; comparison refuses unmatched provenance. The +0.010 nat ceiling is opt-in historical diagnostic only. A runtime collector uses the existing traced-decode adapters without changing shared runtime files.
