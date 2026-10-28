# Contributing

This repository is public (since 2026-10-27); the [publication review](docs/publication-review.md) records the risks that remain disclosed, including evidence that carries machine-specific paths and Git history that carries commit identities. For local changes, use the Nix development shell (`nix develop`) and run the device-free quality gate (`nix run .#quality`). Focused package checks are documented in [README.md](README.md). Hardware validation is separate and must not be inferred from a device-free build.

- Keep pull requests scoped; explain the package or behavior changed, the exact checks exercised, and the limits of any simulator or physical result. Do not turn a saved physical receipt into a claim about a new binary.
- Do not commit credentials, private SSH keys, customer/model conversations, downloaded checkpoints, host inventories, local machine paths, or raw operational logs. Keep private captures in the ignored `.pi/` directory; inspect *content* (including compressed and binary content) before proposing a portable evidence fixture. An ignored file can still exist in earlier Git history.
- Document the origin, version and applicable license of code, patches,
  datasets, models and generated weights that you add. New
  repository-authored code must be compatible with `LGPL-2.1-only`; keep
  upstream SPDX notices and attribution intact. The root license does not
  cover unrelated fetched packages or prove rights to publish saved
  model-derived artifacts. Do not infer redistribution rights from a Nix
  `meta.license`, a source URL or a checksum.
- Run the smallest relevant command or test for a behavior change, then the quality gate when feasible. The gate verifies source and saved artifact policy without accessing a Tenstorrent device; it is not a secret scanner or a publication approval.

For security issues, do not include exploit details or credentials in a public issue. No private reporting route is enabled on the hosting repository; seek a private maintainer contact rather than posting sensitive details publicly.
