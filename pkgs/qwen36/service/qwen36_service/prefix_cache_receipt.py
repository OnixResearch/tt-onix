"""Write the hybrid prefix-cache admission receipt from ABBA-ordered roots.

r[impl tenstorrent.qwen38_p150x2.hybrid_prefix_admission]

Roots come in groups of four, one group per declared workload in policy order, each group
in the policy's root order: cold, cached, cached, cold. The shell binds every input with
b3sum; prefix_cache_admission_core decides. An incomplete root still yields a receipt, not
admitted, with its specific evidence error.
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
    PAIR_NAMES,
    PrefixAdmissionEvidenceError,
    PrefixAdmissionPolicy,
    evaluate,
    root_from_mapping,
)

ROOT_LABELS = ("cold_first", "cached_first", "cached_second", "cold_second")
NON_CLAIMS = (
    "The receipt compares declared agent-shaped workloads on one package; it is not a "
    "general latency claim.",
    "Gates compare per-turn means of two cold and two cached roots measured in ABBA order. "
    "That cancels host load that drifts linearly, not bursts; per-pair ratios are "
    "diagnostics only.",
    "Admission covers only the named prefix-cache profile on that package; the cache stays "
    "off until a deployment enables it.",
    "Entries live in one process and are lost on restart.",
)


def build_receipt(
    *,
    policy_mapping: dict[str, Any],
    group_mappings: list[list[dict[str, Any]]],
    policy_blake3: str,
    root_blake3s: list[list[str]],
) -> dict[str, Any]:
    policy = PrefixAdmissionPolicy.from_mapping(policy_mapping)
    receipt: dict[str, Any] = {
        "schema_version": 2,
        "receipt_kind": "prefix-cache-admission",
        "policy": {"profile_id": policy.profile_id, "blake3": policy_blake3},
        "roots": [
            dict(zip(ROOT_LABELS, digests, strict=True)) for digests in root_blake3s
        ],
        "non_claims": list(NON_CLAIMS),
    }
    try:
        if any(len(group) != len(policy.root_order) for group in group_mappings):
            raise PrefixAdmissionEvidenceError(
                "every workload needs four roots: cold, cached, cached, cold"
            )
        groups = [
            [
                root_from_mapping(mapping, expected_kind=kind)
                for mapping, kind in zip(group, policy.root_order, strict=True)
            ]
            for group in group_mappings
        ]
        result = evaluate(policy, groups)
    except PrefixAdmissionEvidenceError as error:
        receipt["simulated"] = any(
            bool(mapping.get("simulated"))
            for group in group_mappings
            for mapping in group
        )
        receipt["admission"] = {
            "admitted": False,
            "admitted_profile": None,
            "evidence_error": str(error),
            "reasons": ["evidence-error"],
        }
        return receipt
    cached_identity = dict(groups[0][1].identity)
    receipt["simulated"] = any(root.simulated for group in groups for root in group)
    receipt["identity"] = cached_identity
    receipt["admission"] = {
        "admitted": result.admitted,
        "admitted_profile": (
            {
                "execution_identity": cached_identity["execution_identity"],
                "package": cached_identity["package"],
                "prefix_cache_profile_id": cached_identity.get(
                    "prefix_cache_profile_id"
                ),
                "prefix_policy_blake3": cached_identity.get("prefix_policy_blake3"),
            }
            if result.admitted
            else None
        ),
        "evidence_error": None,
        "reasons": list(result.reasons),
    }
    receipt["gates"] = [dataclasses.asdict(gate) for gate in result.gates]
    receipt["pair_diagnostics"] = [
        {"pair": name, **ratios}
        for name, ratios in zip(PAIR_NAMES, result.pairs, strict=True)
    ]
    receipt["turns"] = {"resumed": result.hit_turns, "missed": result.miss_turns}
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument(
        "--abba",
        nargs=4,
        action="append",
        required=True,
        type=Path,
        metavar=("COLD_FIRST", "CACHED_FIRST", "CACHED_SECOND", "COLD_SECOND"),
        help="one workload's roots in measurement order; repeat per declared workload",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--b3sum", default=shutil.which("b3sum"))
    args = parser.parse_args()
    if not args.b3sum:
        raise SystemExit("b3sum is required to bind the inputs")

    def load(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    receipt = build_receipt(
        policy_mapping=load(args.policy),
        group_mappings=[[load(path) for path in group] for group in args.abba],
        policy_blake3=blake3_of(args.policy, args.b3sum),
        root_blake3s=[
            [blake3_of(path, args.b3sum) for path in group] for group in args.abba
        ],
    )
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    admission = receipt["admission"]
    print(
        json.dumps({"admitted": admission["admitted"], "reasons": admission["reasons"]})
    )


if __name__ == "__main__":
    main()
