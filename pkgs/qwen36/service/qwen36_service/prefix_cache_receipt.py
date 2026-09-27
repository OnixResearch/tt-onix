"""Write the hybrid prefix-cache admission receipt from matched cold and cached roots.

r[impl tenstorrent.qwen38_p150x2.hybrid_prefix_admission]

Roots come in (cold, cached) pairs, one pair per declared workload, in policy order. The
shell binds every input with b3sum; prefix_cache_admission_core decides. An incomplete
root still yields a receipt, not admitted, with its specific evidence error.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
from pathlib import Path
from typing import Any

from .continuous_slot_receipt import blake3_of
from .prefix_cache_admission_core import (
    PrefixAdmissionEvidenceError,
    PrefixAdmissionPolicy,
    evaluate,
    root_from_mapping,
)

NON_CLAIMS = (
    "The receipt compares one declared agent-shaped workload on one package; it is not a "
    "general latency claim.",
    "Admission covers only the named prefix-cache profile on that package; the cache stays "
    "off until a deployment enables it.",
    "Entries live in one process and are lost on restart.",
)


def build_receipt(
    *,
    policy_mapping: dict[str, Any],
    pair_mappings: list[tuple[dict[str, Any], dict[str, Any]]],
    policy_blake3: str,
    root_blake3s: list[tuple[str, str]],
) -> dict[str, Any]:
    policy = PrefixAdmissionPolicy.from_mapping(policy_mapping)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "receipt_kind": "prefix-cache-admission",
        "policy": {"profile_id": policy.profile_id, "blake3": policy_blake3},
        "roots": [{"cold": cold, "cached": cached} for cold, cached in root_blake3s],
        "non_claims": list(NON_CLAIMS),
    }
    try:
        pairs = [
            (root_from_mapping(cold, expected_kind="cold"), root_from_mapping(cached, expected_kind="cached"))
            for cold, cached in pair_mappings
        ]
        result = evaluate(policy, pairs)
    except PrefixAdmissionEvidenceError as error:
        receipt["simulated"] = any(bool(cold.get("simulated") or cached.get("simulated")) for cold, cached in pair_mappings)
        receipt["admission"] = {"admitted": False, "admitted_profile": None, "evidence_error": str(error), "reasons": ["evidence-error"]}
        return receipt
    cached_identity = dict(pairs[0][1].identity)
    receipt["simulated"] = any(cold.simulated or cached.simulated for cold, cached in pairs)
    receipt["identity"] = cached_identity
    receipt["admission"] = {
        "admitted": result.admitted,
        "admitted_profile": (
            {
                "execution_identity": cached_identity["execution_identity"],
                "package": cached_identity["package"],
                "prefix_cache_profile_id": cached_identity.get("prefix_cache_profile_id"),
                "prefix_policy_blake3": cached_identity.get("prefix_policy_blake3"),
            }
            if result.admitted
            else None
        ),
        "evidence_error": None,
        "reasons": list(result.reasons),
    }
    receipt["gates"] = [dataclasses.asdict(gate) for gate in result.gates]
    receipt["turns"] = {"resumed": result.hit_turns, "missed": result.miss_turns}
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--pair", nargs=2, action="append", required=True, type=Path, metavar=("COLD", "CACHED"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--b3sum", default=shutil.which("b3sum"))
    args = parser.parse_args()
    if not args.b3sum:
        raise SystemExit("b3sum is required to bind the inputs")

    def load(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    receipt = build_receipt(
        policy_mapping=load(args.policy),
        pair_mappings=[(load(cold), load(cached)) for cold, cached in args.pair],
        policy_blake3=blake3_of(args.policy, args.b3sum),
        root_blake3s=[(blake3_of(cold, args.b3sum), blake3_of(cached, args.b3sum)) for cold, cached in args.pair],
    )
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    admission = receipt["admission"]
    print(json.dumps({"admitted": admission["admitted"], "reasons": admission["reasons"]}))


if __name__ == "__main__":
    main()
