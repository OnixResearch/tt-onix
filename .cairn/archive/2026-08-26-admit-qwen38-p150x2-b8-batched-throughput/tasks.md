## Phase 1: Policy and evidence core

- [x] [serial] Add the closed Nickel B=8 admission profile set and positive and negative export checks for r[tenstorrent.qwen38_p150x2.batch_b8_admission].
- [x] [serial] Parameterize the pure admission core and the Rust receipt evaluator over candidate widths 4 and 8 with self-tests at both widths for r[tenstorrent.qwen38_p150x2.batch_b8_admission].

## Phase 2: Package and device-free gates

- [x] [serial] Package the B=8 profile and the `qwen36-p150x2-b8-admission` entry point with install checks for r[tenstorrent.qwen38_p150x2.batch_b8_admission].
- [x] [serial] Add positive and negative device-free tests for width acceptance, width rejection, admission eligibility, and missing receipts for r[tenstorrent.qwen38_p150x2.batch_b8_admission].

## Phase 3: Physical decision

- [x] [serial] Run matched B=1 versus B=8 roots in `single,batch,batch,single` order on devices `0` and `1` for r[tenstorrent.qwen38_p150x2.batch_b8_admission].
- [x] [serial] Bind a passing receipt or preserve the exact rejection without weakening any bound for r[tenstorrent.qwen38_p150x2.batch_b8_admission].

## Phase 4: Lifecycle completion

- [x] [serial] Restore production health and validate package, Cairn, formatting, and repository gates for r[tenstorrent.qwen38_p150x2.batch_b8_admission].
- [x] [serial] Sync, archive, and commit the validated rejection on the isolated branch for r[tenstorrent.qwen38_p150x2.batch_b8_admission].
