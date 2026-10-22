#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-only
"""Record a B=8 campaign's provenance before and after the run.

The batch receipt hashes response bytes, not launch history. This builder writes the
runbook's campaign record: the package store path and identity, the invoked profile
path and `profile_id`, the exact command line, the two recorded environment controls,
the checkpoint path and digest, the tree commit, and both containment-checker
verdicts. It fails closed when any field is absent or when a verdict names another
package, profile or identity.

    scripts/qwen38-campaign-provenance.py --package PATH --package-identity IDENTITY
        --profile PATH --runtime-identity IDENTITY --model-identity IDENTITY
        --command-line TEXT --gdn-decode-fusion MODE --disable-sfploadmacro VALUE
        --checkpoint PATH --checkpoint-blake3 DIGEST --source-revision COMMIT
        --containment-before REPORT --containment-after REPORT --output PATH

`--source-revision` is required; run it with the frozen commit explicitly so a
dirty or moved tree cannot be recorded silently. The output JSON is written only
after every field validates.

`--self-test` records a positive fixture and exercises every missing field and a
verdict mismatch in an isolated temporary directory.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

SCHEMA = "onix.qwen38.campaign-provenance.v1"
CONTAINMENT_SCHEMA = "onix.qwen38.admission-containment.v1"
DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
PASS_EXIT_CODE = 0
FAIL_EXIT_CODE = 1
REQUIRED_FIELDS = (
    "package_store_path",
    "package_identity",
    "profile_path",
    "profile_id",
    "runtime_identity",
    "model_identity",
    "command_line",
    "qwen35_gdn_decode_fusion",
    "tt_metal_disable_sfploadmacro",
    "checkpoint_path",
    "checkpoint_blake3",
    "source_revision",
    "containment_before",
    "containment_after",
)


class ProvenanceError(Exception):
    """A fail-closed provenance verdict."""


def require_string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ProvenanceError(f"provenance field {field} is absent or empty")
    return value


def read_json_object(path: str, label: str) -> dict[str, Any]:
    if not path:
        raise ProvenanceError(f"{label} is absent")
    location = Path(path)
    if not location.is_absolute():
        raise ProvenanceError(f"{label} must be absolute: {path}")
    try:
        payload = json.loads(location.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProvenanceError(f"{label} is not readable JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ProvenanceError(f"{label} is not a JSON object")
    return payload


def validate_verdict(
    verdict: dict[str, Any],
    label: str,
    package: Path,
    profile: Path,
    package_identity: str,
    profile_id: str,
) -> None:
    if verdict.get("schema") != CONTAINMENT_SCHEMA:
        raise ProvenanceError(f"{label} has the wrong schema")
    if verdict.get("passed") is not True:
        raise ProvenanceError(f"{label} did not pass")
    if verdict.get("package_store_path") != str(package):
        raise ProvenanceError(f"{label} names a different package")
    if verdict.get("profile_path") != str(profile):
        raise ProvenanceError(f"{label} names a different profile")
    if verdict.get("package_identity") != str(Path(package_identity)):
        raise ProvenanceError(f"{label} names a different package identity")
    if verdict.get("profile_id") != profile_id:
        raise ProvenanceError(f"{label} names a different profile_id")
    if verdict.get("mode") not in ("v3-admitted", "unadmitted-campaign", "non-v3"):
        raise ProvenanceError(f"{label} has an unknown containment mode")
    if verdict.get("mode") == "v3-admitted" and not verdict.get("receipt_path"):
        raise ProvenanceError(f"{label} admits v3 without naming a receipt file")


def build_record(payload: dict[str, Any]) -> dict[str, Any]:
    for field in REQUIRED_FIELDS:
        require_string(payload, field)

    package = Path(require_string(payload, "package_store_path"))
    if not package.is_absolute() or not package.is_dir():
        raise ProvenanceError(
            f"package_store_path is not an absolute directory: {package}"
        )
    if not package.name.endswith("qwen38-0.1.0"):
        raise ProvenanceError(
            f"package_store_path is not an exact qwen38 output: {package}"
        )

    profile = Path(require_string(payload, "profile_path"))
    if not profile.is_absolute() or not profile.is_file():
        raise ProvenanceError(f"profile_path is not an absolute file: {profile}")

    checkpoint = Path(require_string(payload, "checkpoint_path"))
    if not checkpoint.is_absolute() or not checkpoint.exists():
        raise ProvenanceError(
            f"checkpoint_path does not exist absolutely: {checkpoint}"
        )

    digest = require_string(payload, "checkpoint_blake3")
    if not DIGEST_PATTERN.fullmatch(digest):
        raise ProvenanceError("checkpoint_blake3 is not a BLAKE3 digest")
    commit = require_string(payload, "source_revision")
    if not COMMIT_PATTERN.fullmatch(commit):
        raise ProvenanceError("source_revision is not a 40-hex tree commit")

    package_identity = require_string(payload, "package_identity")
    if not Path(package_identity).is_absolute():
        raise ProvenanceError("package_identity is not an absolute path")
    runtime_identity = require_string(payload, "runtime_identity")
    if not Path(runtime_identity).is_absolute():
        raise ProvenanceError("runtime_identity is not an absolute path")
    model_identity = require_string(payload, "model_identity")
    if not Path(model_identity).is_absolute():
        raise ProvenanceError("model_identity is not an absolute path")

    profile_payload = read_json_object(str(profile), "profile")
    profile_id = profile_payload.get("profile_id")
    if not isinstance(profile_id, str) or not profile_id:
        raise ProvenanceError("profile carries no profile_id")
    if profile_id != payload["profile_id"]:
        raise ProvenanceError("recorded profile_id differs from the profile file")

    before = read_json_object(
        require_string(payload, "containment_before"), "containment_before"
    )
    after = read_json_object(
        require_string(payload, "containment_after"), "containment_after"
    )
    validate_verdict(
        before, "containment_before", package, profile, package_identity, profile_id
    )
    validate_verdict(
        after, "containment_after", package, profile, package_identity, profile_id
    )

    return {
        "schema": SCHEMA,
        "package_store_path": str(package),
        "package_identity": package_identity,
        "runtime_identity": runtime_identity,
        "model_identity": model_identity,
        "profile_path": str(profile),
        "profile_id": profile_id,
        "command_line": payload["command_line"],
        "environment": {
            "QWEN35_GDN_DECODE_FUSION": payload["qwen35_gdn_decode_fusion"],
            "TT_METAL_DISABLE_SFPLOADMACRO": payload["tt_metal_disable_sfploadmacro"],
        },
        "checkpoint_path": str(checkpoint),
        "checkpoint_blake3": digest,
        "source_revision": commit,
        "containment": {
            "before": before,
            "after": after,
        },
    }


def write_record(path: str, record: dict[str, Any]) -> None:
    destination = Path(path)
    if not destination.is_absolute():
        raise ProvenanceError("--output must be absolute")
    destination.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="qwen38-provenance-self-test-") as root:
        root_path = Path(root)
        package = root_path / "hash-qwen38-0.1.0"
        (package / "share/qwen36").mkdir(parents=True)
        identity = root_path / "hash-qwen36-execution-identity-0.1.0"
        identity.mkdir()
        runtime = root_path / "hash-tt-metal-0.77.0"
        runtime.mkdir()
        model = root_path / "hash-model-snapshot"
        model.mkdir()
        checkpoint = root_path / "checkpoint"
        checkpoint.mkdir()
        profile = package / "share/qwen36/batch-admission-b8-profile-v3.json"
        profile.write_text(
            json.dumps({"profile_id": "qwen38-p150x2-batched-throughput-b8-v3"}),
            encoding="utf-8",
        )
        digest = "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
        verdict = {
            "schema": CONTAINMENT_SCHEMA,
            "passed": True,
            "mode": "unadmitted-campaign",
            "package_store_path": str(package),
            "package_identity": str(identity),
            "profile_path": str(profile),
            "profile_id": "qwen38-p150x2-batched-throughput-b8-v3",
            "can_serve": False,
            "expected_receipt": str(root_path / "receipt.json"),
        }
        reports: list[Path] = []
        for name in ("before", "after"):
            path = root_path / f"containment-{name}.json"
            path.write_text(json.dumps(verdict), encoding="utf-8")
            reports.append(path)
        base: dict[str, Any] = {
            "package_store_path": str(package),
            "package_identity": str(identity),
            "runtime_identity": str(runtime),
            "model_identity": str(model),
            "profile_path": str(profile),
            "profile_id": "qwen38-p150x2-batched-throughput-b8-v3",
            "command_line": (
                f"{package}/bin/qwen38-p150x2-b8-admission --host 127.0.0.1 --port 8000 "
                f"--batch-admission-profile {profile}"
            ),
            "qwen35_gdn_decode_fusion": "full",
            "tt_metal_disable_sfploadmacro": "1",
            "checkpoint_path": str(checkpoint),
            "checkpoint_blake3": digest,
            "source_revision": "5e7fbc0" + "0" * 33,
            "containment_before": str(reports[0]),
            "containment_after": str(reports[1]),
        }
        record = build_record(base)
        if record["environment"]["QWEN35_GDN_DECODE_FUSION"] != "full":
            raise ProvenanceError("self-test lost the fusion control")
        output = root_path / "campaign-provenance.json"
        write_record(str(output), record)
        if json.loads(output.read_text(encoding="utf-8"))["schema"] != SCHEMA:
            raise ProvenanceError("self-test output did not round-trip")

        for field in REQUIRED_FIELDS:
            broken = dict(base)
            broken.pop(field)
            try:
                build_record(broken)
            except ProvenanceError as error:
                if field not in str(error):
                    raise ProvenanceError(
                        f"self-test missing {field} reported an unrelated error: {error}"
                    ) from error
            else:
                raise ProvenanceError(f"self-test accepted a record without {field}")

        mismatched = dict(base)
        other_verdict = dict(verdict)
        other_verdict["package_store_path"] = str(root_path / "other-package")
        mismatched_path = root_path / "containment-mismatched.json"
        mismatched_path.write_text(json.dumps(other_verdict), encoding="utf-8")
        mismatched["containment_after"] = str(mismatched_path)
        try:
            build_record(mismatched)
        except ProvenanceError as error:
            if "different package" not in str(error):
                raise ProvenanceError(
                    f"self-test mismatch reported: {error}"
                ) from error
        else:
            raise ProvenanceError("self-test accepted a mismatched containment verdict")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", dest="package_store_path", required=True)
    parser.add_argument("--package-identity", required=True)
    parser.add_argument("--profile", dest="profile_path", required=True)
    parser.add_argument("--runtime-identity", required=True)
    parser.add_argument("--model-identity", required=True)
    parser.add_argument("--command-line", required=True)
    parser.add_argument("--gdn-decode-fusion", required=True)
    parser.add_argument("--disable-sfploadmacro", required=True)
    parser.add_argument("--checkpoint", dest="checkpoint_path", required=True)
    parser.add_argument("--checkpoint-blake3", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--containment-before", required=True)
    parser.add_argument("--containment-after", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile-id", dest="profile_id", required=True)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        try:
            self_test()
        except ProvenanceError as error:
            print(
                f"qwen38-campaign-provenance self-test: FAIL: {error}", file=sys.stderr
            )
            return FAIL_EXIT_CODE
        print("qwen38-campaign-provenance self-test: PASS")
        return PASS_EXIT_CODE
    arguments = parse_arguments()
    payload = {
        "package_store_path": arguments.package_store_path,
        "package_identity": arguments.package_identity,
        "runtime_identity": arguments.runtime_identity,
        "model_identity": arguments.model_identity,
        "profile_path": arguments.profile_path,
        "profile_id": arguments.profile_id,
        "command_line": arguments.command_line,
        "qwen35_gdn_decode_fusion": arguments.gdn_decode_fusion,
        "tt_metal_disable_sfploadmacro": arguments.disable_sfploadmacro,
        "checkpoint_path": arguments.checkpoint_path,
        "checkpoint_blake3": arguments.checkpoint_blake3,
        "source_revision": arguments.source_revision,
        "containment_before": arguments.containment_before,
        "containment_after": arguments.containment_after,
    }
    try:
        record = build_record(payload)
        write_record(arguments.output, record)
    except ProvenanceError as error:
        print(f"qwen38-campaign-provenance: {error}", file=sys.stderr)
        return FAIL_EXIT_CODE
    print(json.dumps(record, indent=2, sort_keys=True))
    return PASS_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
