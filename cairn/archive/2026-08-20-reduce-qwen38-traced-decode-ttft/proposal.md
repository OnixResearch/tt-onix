## Why

The physical traced-decode candidate preserved exact tokens and clean reset behavior.
It failed admission because its median TTFT ratio was `1.2164174383938555`, above the `1.05` limit.

Static review shows that trace capture starts after the first token.
The candidate instead gives short-prompt prefill a maximum-width page table while eager prefill uses only the required blocks.
The prior service-level samples also had high variance.

## What Changes

- Add pure request-bounded page-table sizing with strict capacity checks.
- Keep the maximum-width page table bound to stable decode trace inputs.
- Give prefill only the page-table prefix required by the prompt and generation bound.
- Add phase timing for reset, prefill device work, first-token readback, and trace preparation.
- Extend physical receipts with phase medians and page-table widths.
- Run order-balanced matched physical samples before any admission change.
- Keep eager production and every existing fail-closed gate until all gates pass.

## Impact

- **Core**: pure page-table sizing owns request and capacity rules.
- **Runtime shell**: prefill uses a request-bounded host table and records phase timing.
- **Trace adapter**: stable decode inputs and their device addresses remain unchanged.
- **Evidence**: receipts identify TTFT phases and both page-table widths.
- **Production**: eager decode stays selected until full physical admission.

## Out of Scope

- Moving trace capture outside its measured phase to hide cost.
- Changing model equations, precision, sampling, or prompt corpus.
- Continuous batching.
- Runtime uplift or TP2 kernel tuning.
- A speedup claim from static review, TTsim, or one execution order.
