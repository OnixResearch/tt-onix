# Owner decision: keep speculative serving gated off, archive as not admitted

Date: 2026-09-11. The owner delegated the decision ("do what's best") after
receiving the measured record.

Decision:

- Keep `QWEN36_MTP_SERVING` default off. Do not enable speculation in
  production on this runtime.
- Close the change as a not-admitted candidate. The mechanism is implemented,
  exact for greedy output on the physical screen, and safely gated, but its
  measured economics lose: 0.992x at two drafts and 0.797x at three drafts
  against plain traced decode.
- Do not weaken the bitwise restoration oracle. The negative-zero blocker is
  root-caused as a platform property of tt-metal 0.77 (the bit does not
  survive first device residency on simulator or silicon), so the oracle is
  unattainable rather than unsatisfied. Zero-sign normalization remains a
  recorded future-work option should the economics or runtime change.
- No sampled, batch, or deterministic route changed; the default-off behavior
  of the serving loop is unchanged.

Task 4 closes by this decision: the rejected-KV restoration gap is a platform
boundary, not an implementable fix, and the path stays gated off. Task 7
closes as this recorded review.
