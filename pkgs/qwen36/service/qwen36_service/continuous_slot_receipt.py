"""Write the continuous-slot admission receipt from a policy and two measurement roots.

r[impl tenstorrent.qwen38_p150x2.continuous_slot_admission]

The shell reads files and computes content identities with b3sum; every decision is
made by continuous_slot_admission_core. A structurally incomplete root still yields a
receipt, marked not admitted with the specific evidence error, so a failed or missing
measurement cannot disappear from the record.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .continuous_slot_admission_core import (
    BASELINE_KIND,
    CANDIDATE_KIND,
    AdmissionEvidenceError,
    AdmissionPolicy,
    evaluate,
    root_from_mapping,
)

RECEIPT_SCHEMA_VERSION = 1
NON_CLAIMS = (
    "The receipt compares one declared workload on one package; it is not a general "
    "latency or throughput claim.",
    "Admission covers only the named scheduler profile and package; the group "
    "coordinator stays the served default until a deployment change selects the profile.",
    "Trace capture counts are reported and never gated.",
)


def blake3_of(path: Path, b3sum: str) -> str:
    output = subprocess.run(
        [b3sum, "--no-names", str(path)], check=True, capture_output=True, text=True
    ).stdout.strip()
    if len(output) != 64:
        raise RuntimeError(f"b3sum returned an unexpected digest for {path}")
    return output


def build_receipt(
    *,
    policy_mapping: dict[str, Any],
    baseline_mapping: dict[str, Any],
    candidate_mapping: dict[str, Any],
    policy_blake3: str,
    baseline_blake3: str,
    candidate_blake3: str,
) -> dict[str, Any]:
    policy = AdmissionPolicy.from_mapping(policy_mapping)
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "receipt_kind": "continuous-slot-admission",
        "policy": {"profile_id": policy.profile_id, "blake3": policy_blake3},
        "roots": {
            BASELINE_KIND: {"blake3": baseline_blake3},
            CANDIDATE_KIND: {"blake3": candidate_blake3},
        },
        "non_claims": list(NON_CLAIMS),
    }
    try:
        baseline = root_from_mapping(baseline_mapping, expected_kind=BASELINE_KIND)
        candidate = root_from_mapping(candidate_mapping, expected_kind=CANDIDATE_KIND)
        result = evaluate(policy, baseline, candidate)
    except AdmissionEvidenceError as error:
        receipt["simulated"] = bool(
            baseline_mapping.get("simulated") or candidate_mapping.get("simulated")
        )
        receipt["admission"] = {
            "admitted": False,
            "admitted_profile": None,
            "evidence_error": str(error),
            "reasons": ["evidence-error"],
        }
        return receipt
    receipt["simulated"] = baseline.simulated or candidate.simulated
    receipt["identity"] = dict(candidate.identity)
    receipt["admission"] = {
        "admitted": result.admitted,
        "admitted_profile": (
            {
                "execution_identity": candidate.identity["execution_identity"],
                "package": candidate.identity["package"],
                "scheduler_policy_blake3": candidate.identity[
                    "scheduler_policy_blake3"
                ],
                "scheduler_profile_id": candidate.identity["scheduler_profile_id"],
            }
            if result.admitted
            else None
        ),
        "evidence_error": None,
        "reasons": list(result.reasons),
    }
    receipt["gates"] = [dataclasses.asdict(gate) for gate in result.gates]
    receipt["metrics"] = {
        BASELINE_KIND: [dataclasses.asdict(metrics) for metrics in result.baseline],
        CANDIDATE_KIND: [dataclasses.asdict(metrics) for metrics in result.candidate],
        "cross_root_parity_mismatches": list(result.cross_root_parity_mismatches),
    }
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy", required=True, type=Path, help="exported admission policy JSON"
    )
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--b3sum", default=shutil.which("b3sum"))
    args = parser.parse_args()
    if not args.b3sum:
        raise SystemExit("b3sum is required to bind the inputs")
    paths = (args.policy, args.baseline_root, args.candidate_root)
    mappings = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    digests = [blake3_of(path, args.b3sum) for path in paths]
    receipt = build_receipt(
        policy_mapping=mappings[0],
        baseline_mapping=mappings[1],
        candidate_mapping=mappings[2],
        policy_blake3=digests[0],
        baseline_blake3=digests[1],
        candidate_blake3=digests[2],
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
