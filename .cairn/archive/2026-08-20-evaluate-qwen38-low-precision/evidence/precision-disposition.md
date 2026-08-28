# Low-precision candidate disposition

Date: 2026-08-20

## Verdict

`EXPERIMENTAL_UNADMITTED`

BFP8 remains the production precision for the affected Qwen3.8 weight families.
No BF4 candidate is admitted.

## Observed production state

The active `qwen38-p150x2.service` environment did not contain these variables:

```text
QWEN36_LM_HEAD_BF4
QWEN36_MLP_DOWN_BF4
QWEN36_GDN_BF4
```

The service reported `active`.
The package documentation states that these gates are opt-in and keep documented BFP8 defaults.
Distinct `.bfp4` cache names prevent an experiment from overwriting default BFP8 bins.

## Missing admission evidence

No current candidate has one complete evidence set containing:

- a canonical named precision profile;
- operator and logit results;
- recurrent-state error results at the accepted state precision;
- exact greedy token parity for the required corpus;
- matched physical decode, TTFT, tail-latency, memory, and load-time results;
- complete model, tokenizer, source, runtime, mesh, and machine identities.

## Decision

The existing flags remain experimental and opt-in.
No production profile, service setting, package default, or performance claim changes.

A future evaluation must use a new validated Cairn change and complete every numerical and physical gate.

## Non-claims

- This disposition does not show that BF4 is inaccurate.
- This disposition does not show that BF4 is slower or faster.
- Existing support for BF4 cache generation does not imply production admission.
- No physical low-precision run was performed for this disposition.
