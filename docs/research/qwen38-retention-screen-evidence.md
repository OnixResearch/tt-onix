# Qwen38 retention-screen evidence

Scope: only the four source groups requested; no new model execution. “Accepted”
below means a diagnostic numerical pass, not production admission. Missing means
not recorded in these sources. NLL is lower-is-better, so its “floor” is a ceiling.

## Evidence table

| Metric | Corpus / fixture | Control value | Worst accepted treatment value | Proposed floor | Citation (file + section) |
| --- | --- | --- | --- | --- | --- |
| Mean passage NLL | Six passages: prose, Python, history, recipe, calculus, SQL | BFP8 fused-step: 0.7451 | GDN BF4: Δ +0.0079 nats; LM-head BF4: Δ +0.0053 | Reuse declared ΔNLL ≤ +0.010 nats, on this same screen only | `.cairn/changes/enable-qwen38-bf4-weight-caches/evidence/screen-2026-09-24.md`, Setup / Results |
| Mean passage NLL, rejected MLP-down BF4 | Same six passages | 0.7451 | Missing: MLP-down was rejected, Δ +0.0192 | ΔNLL ≤ +0.010; +0.0192 exceeds it | Same file, Results / Verdict and non-claims |
| Teacher-forced top-1 agreement | Eight prompts × 96 reference-greedy positions; reference top-64 support | Missing: self-agreement not reported | GDN BF4 0.978; LM-head BF4 0.980; worst passing = 0.978 | Existing bound ≥ 0.97; not sufficient alone | Same file, Setup / Results |
| Teacher-forced top-1 agreement, rejected MLP-down BF4 | Same eight-prompt fixture | Missing | Missing: rejected family scored 0.980 despite failing NLL | Existing ≥ 0.97 did not reject MLP-down | Same file, Results |
| Mean top-64 KL | Same teacher-forced fixture | Missing | GDN BF4 0.021; LM-head BF4 0.0074; worst passing = 0.021; rejected MLP-down = 0.011 | Missing | Same file, Setup / Results |
| Per-passage NLL change, GDN BF4 | Prose / Python / history / recipe / calculus / SQL | Missing per-passage absolute controls | +0.050 / −0.017 / +0.028 / +0.015 / −0.007 / −0.022 | Missing per-passage bounds; mean gate only | Same file, Results |
| Passage NLL change, step versus composite | Same passage screen | Missing absolute composite values | Reported range −0.005 to +0.000 | Missing separate route-specific bound | Same file, Results |
| Greedy token and full-text parity | Fixed multi-prompt corpus; matched control,candidate,candidate,control roots | Composite reference; numeric score missing | Missing measured accepted candidate | Exact token and text equality, no relaxed parity bound | `.cairn/changes/stage-gdn-precision-preserving-kernel/specs/qwen38-p150x2-runtime/spec.md`, staged recurrence requirement / valid / exhausted scenarios |
| Layerwise precision quality | Prefill, teacher-forced decode, long carry, chunk boundaries; task quality; MTP when shared head changes | Missing | Missing | Missing numerical tolerances; preserve FP32 recurrent state and BF16 MTP head initially | `.cairn/changes/qwen38-layerwise-precision-frontier/specs/qwen38-p150x2-runtime/spec.md`, Explicit precision recipe / Decode and state quality coverage |
| Hidden relative error, decode path | Missing | Missing | Missing | Missing | `.cairn/changes/optimize-gdn-decode-kernel-grid-and-prep/evidence/`, no hidden-error bound found |
| Logit relative error, decode path | Missing | Missing | Missing | Missing | Same evidence directory; `docs/gdn-precision-candidate.md`, Separate evidence scopes / Hardware result; no logit-relative bound found |
| Recurrent-state comparison | NumPy recurrence; two heads, dimension 128, three steps; fresh/aliased storage and program-cache reuse | NumPy reference; absolute values missing | Maximum absolute error 7.927790284156799e-05 | Existing elementwise atol=0.0001, rtol=0.001; not a measured global relative-error bound | `docs/gdn-precision-candidate.md`, Hardware result |
| Recurrent-state rejected baseline | Same reference comparison; full fixture identity not specified in failure excerpt | NumPy reference | Missing accepted value in this baseline; 4890/32768 mismatches (14.9%), max absolute violation 0.00022145 | Same atol=0.0001, rtol=0.001 | Same file, Observed failure |
| Receipt perplexity / NLL / top-1 logit agreement | Missing | Missing | Missing | Missing | `pkgs/qwen36/evidence/**/*.json`, no explicit perplexity, NLL, or agreement field found |
| Receipt logits PCC / NMSE (not top-1 agreement) | Missing named text corpus | Missing standalone control measurement | Native MTP verifier summaries: minimum logits PCC 1.0, maximum logits NMSE 0.0 | Missing quality-retention threshold | `pkgs/qwen36/evidence/2026-08-22/qwen38-27b-native-mtp-verifier-receipt.json` and `qwen38-27b-native-mtp-verifier-untraced-receipt.json`, `minimum_logits_pcc` / `maximum_logits_nmse` summaries |

## What is missing

- The six passages' exact text, tokenization, scoring window, aggregation weights,
  and portable raw BF4 receipts are not supplied by the screen note: it points to
  `.pi/tmp/gdn-step/bf4_eval.py` and `.pi/tmp/gdn-step/bf4-*.json`, outside the
  requested source groups. The eight prompts are not fully enumerated there.
- Absolute treatment NLLs are not reported, only deltas. Per-passage controls and
  any per-passage ceiling are missing. The step/composite absolute controls and
  independent route threshold are missing.
- Top-1 and KL control measurements are missing. No accepted MLP-down treatment
  exists in this screen; its 0.980 agreement is a rejected-family observation.
  KL has no declared floor.
- “Passes” for GDN and LM-head BF4 is diagnostic only: no family was admitted;
  GDN changed greedy behavior on two of eight prompts. The screen has no matched
  order-balanced roots, long-context claim, or sampled-serving claim.
- Staging specifies exact parity but supplies neither a measured accepted value
  nor a named corpus in the requested change. Layerwise frontier supplies no
  numerical precision tolerance, control, treatment, or frozen corpus.
- Hidden/logit relative-error fixtures, controls, accepted treatments, and bounds
  are missing from the requested decode evidence. The state fixture supplies an
  absolute-error observation and elementwise tolerances, not control tensor values
  or a measured maximum state-relative error. These are not language-quality floors.
- The rejected state baseline has no accepted treatment value; its failure excerpt
  does not fully identify its fixture. The passing hardware fixture is identified
  separately above.
- JSON receipts supply no explicit perplexity, NLL, or token-agreement results,
  corpus names, control values, accepted treatments, or floors for those metrics.
  Native verifier PCC/NMSE figures are tensor equivalence evidence, not task NLL
  or top-1 retention; their named text corpus, standalone control, and quality floor
  are missing.

## Single metric for the MLP weight-layout experiment

Use mean teacher-forced passage NLL change against the matched BFP8 fused-step
control: it rejected MLP-down BF4 at +0.0192 nats while top-1 agreement still passed
at 0.980. The existing +0.010-nat ceiling is defensible only for the original six-
passage screen, not as a general retention floor. To apply it reproducibly, recover
and freeze those exact prose, Python, history, recipe, calculus, and SQL passages,
tokenization, scored positions, and averaging rule, then measure paired BFP8
control and MLP-layout-treatment NLL on that identical corpus. Without that corpus
identity and matched measurement, the in-repo evidence cannot establish a new
corpus-independent floor; do not invent one or substitute kernel relative error.
Corpus recovery result: [P1 retention corpus identity](../../.cairn/changes/optimize-qwen38-mlp-projection-streaming/evidence/p1-retention-corpus-identity-2026-10-22.md) records the recovered scratch passages, scoring recipe, NLLs, hashes, and remaining reproducibility limits.
