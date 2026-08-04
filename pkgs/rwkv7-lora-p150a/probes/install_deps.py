# r[impl tenstorrent.rwkv7_p150a_finetuning.compiler]
# Collect the exact pinned dependency closure of the installed identity
# wheels. Reads each dist's METADATA, keeps non-extra requirements, drops
# direct-URL requirements (the four identity wheels are already installed
# from host-verified local files), and prints one requirement per line.
from __future__ import annotations

import sys
from importlib.metadata import distributions

# Distribution names whose requirements drive the closure. The identity
# wheels themselves are excluded from the emitted requirements because they
# are installed from host-verified local bytes.
ROOT_DISTS = {"pjrt-plugin-tt", "torch", "torch-xla", "tt-lang"}
IDENTITY_DISTS = {"pjrt-plugin-tt", "torch", "torch-xla", "tt-lang"}


def normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def main() -> int:
    seen = set()
    requirements = []
    by_name = {}
    for dist in distributions():
        by_name[normalize_name(dist.metadata["Name"])] = dist

    for root in sorted(ROOT_DISTS):
        dist = by_name.get(root)
        if dist is None:
            print(f"install-deps: missing root distribution {root}", file=sys.stderr)
            return 1
        for requirement in dist.requires or []:
            text = requirement.strip()
            if "extra ==" in text or "extra==" in text:
                continue
            if " @ " in text:
                # Direct-URL requirement; the target wheel is installed from
                # host-verified local bytes instead.
                continue
            # Reduce to the package specifier (name plus version constraint).
            spec = text.split(";")[0].strip()
            key = spec.split("<")[0].split(">")[0].split("=")[0].split("!")[0].split("[")[0]
            key = normalize_name(key)
            if key in IDENTITY_DISTS:
                continue
            if key in seen:
                continue
            seen.add(key)
            requirements.append(spec)

    for spec in sorted(requirements):
        print(spec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
