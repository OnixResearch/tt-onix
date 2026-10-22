#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-only
"""Check that an invoked Qwen3.8 admission profile is contained by its package.

The package exports its own v3 profile at `share/qwen36/batch-admission-b8-profile-v3.json`
and its execution identity as the `share/qwen36/execution-identity` symlink. A campaign
must select that exact file, and the wrapper that launches it must export the identity
the package derives. The service itself accepts any absolute profile path and any
later `--batch-admission-profile`, so a process can otherwise report v3 from an
external profile while carrying another identity.

    scripts/qwen38-admission-containment.py PACKAGE [--profile PATH]
        --package-identity IDENTITY [--wrapper PATH] [--receipt-root DIR]
        [--expect-unadmitted --receipt PATH] [--report PATH]

The checker fails closed on every missing or mismatched input. `--expect-unadmitted`
is the only way to pass an unadmitted v3 profile: it asserts the profile has
`physically_admitted: false` and `identities.receipt_blake3: "not-measured"`, names the
receipt the campaign will produce, and prints that the run cannot serve. There is no
flag that admits an unadmitted v3 profile.

`--self-test` builds isolated fixtures under a temporary directory and exercises the
containment, identity, admission, receipt-hash and unadmitted-campaign cases.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

SCHEMA = "onix.qwen38.admission-containment.v1"
PROFILE_RELATIVE_PATH = Path("share/qwen36/batch-admission-b8-profile-v3.json")
IDENTITY_RELATIVE_PATH = Path("share/qwen36/execution-identity")
WRAPPER_RELATIVE_PATH = Path("bin/qwen38-p150x2-b8-admission")
DEFAULT_RECEIPT_ROOT = Path(__file__).resolve().parent.parent / "pkgs/qwen36/evidence"
NOT_MEASURED = "not-measured"
V3_MODE = "v3-admitted"
UNADMITTED_MODE = "unadmitted-campaign"
NON_V3_MODE = "non-v3"
PASS_EXIT_CODE = 0
FAIL_EXIT_CODE = 1
PACKAGE_NAME = "qwen38-0.1.0"
WRAPPER_IDENTITY_PATTERN = re.compile(r"--package-identity\s+(\S+)")


class ContainmentError(Exception):
    """A fail-closed containment verdict."""


def blake3_bytes(data: bytes) -> str:
    """BLAKE3 of the supplied bytes, from the packaged module or the b3sum binary."""
    try:
        import blake3
    except ImportError:
        pass
    else:
        return blake3.blake3(data).hexdigest()
    try:
        result = subprocess.run(
            ["b3sum", "--no-names"],
            input=data,
            capture_output=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ContainmentError(
            f"no BLAKE3 implementation is available: {error}"
        ) from error
    digest = result.stdout.decode("ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ContainmentError(f"b3sum returned an invalid digest: {digest!r}")
    return digest


def strict_directory(value: str, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ContainmentError(f"{label} must be absolute: {value}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ContainmentError(f"{label} does not resolve: {value}: {error}") from error
    if not resolved.is_dir():
        raise ContainmentError(f"{label} is not a directory: {value}")
    return resolved


def strict_file(value: str, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ContainmentError(f"{label} must be absolute: {value}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ContainmentError(f"{label} does not resolve: {value}: {error}") from error
    if not resolved.is_file():
        raise ContainmentError(f"{label} is not a regular file: {value}")
    return resolved


def inside(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath((str(path), str(root))) == str(root)
    except ValueError:
        return False


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContainmentError(f"{label} is not readable JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ContainmentError(f"{label} is not a JSON object")
    return payload


def wrapper_identity(wrapper: Path) -> str:
    try:
        text = wrapper.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise ContainmentError(f"cannot read wrapper {wrapper}: {error}") from error
    match = WRAPPER_IDENTITY_PATTERN.search(text)
    if match is None:
        raise ContainmentError(f"wrapper exports no --package-identity: {wrapper}")
    exported = match.group(1).strip("\"'")
    if not exported.startswith("/"):
        raise ContainmentError(
            f"wrapper exports a non-absolute package identity: {exported}"
        )
    return exported


def receipt_digest_of(profile: dict[str, Any]) -> str:
    identities = profile.get("identities")
    if not isinstance(identities, dict):
        raise ContainmentError("profile carries no identities record")
    digest = identities.get("receipt_blake3")
    if not isinstance(digest, str):
        raise ContainmentError("profile carries no identities.receipt_blake3")
    if digest != NOT_MEASURED and not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ContainmentError(f"profile receipt digest is invalid: {digest!r}")
    return digest


def find_receipt(root: Path, digest: str) -> Path:
    if not root.is_dir():
        raise ContainmentError(f"receipt root is not a directory: {root}")
    matches: list[Path] = []
    for path in sorted(root.rglob("*.json")):
        if not path.is_file():
            continue
        try:
            actual = blake3_bytes(path.read_bytes())
        except OSError as error:
            raise ContainmentError(f"cannot hash receipt {path}: {error}") from error
        if actual == digest:
            matches.append(path)
    if not matches:
        raise ContainmentError(
            f"no receipt under {root} has BLAKE3 {digest}; the profile names no receipt"
        )
    return matches[0]


def check_containment(
    arguments: argparse.Namespace, *, require_store: bool = True
) -> dict[str, Any]:
    package = strict_directory(arguments.package, "package output")
    if not package.name.endswith(PACKAGE_NAME):
        raise ContainmentError(
            f"package output is not an exact qwen38 output: {package}"
        )
    if require_store and not str(package).startswith("/nix/store/"):
        raise ContainmentError(f"package output is not a store output: {package}")

    identity_link = package / IDENTITY_RELATIVE_PATH
    if not identity_link.is_symlink():
        raise ContainmentError(
            f"package carries no execution identity link: {identity_link}"
        )
    try:
        package_identity = identity_link.resolve(strict=True)
    except OSError as error:
        raise ContainmentError(
            f"package execution identity link does not resolve: {identity_link}: {error}"
        ) from error

    wrapper = (
        strict_file(arguments.wrapper, "wrapper")
        if arguments.wrapper
        else strict_file(str(package / WRAPPER_RELATIVE_PATH), "package wrapper")
    )
    exported_text = wrapper_identity(wrapper)
    if require_store and not exported_text.startswith("/nix/store/"):
        raise ContainmentError(
            f"wrapper exports a non-store package identity: {exported_text}"
        )
    try:
        exported_identity = Path(exported_text).resolve(strict=True)
    except OSError as error:
        raise ContainmentError(
            f"wrapper package identity does not resolve: {exported_text}: {error}"
        ) from error
    if exported_identity != package_identity:
        raise ContainmentError(
            "wrapper --package-identity differs from the package identity derivation: "
            f"{exported_identity} != {package_identity}"
        )
    try:
        invoked_identity = Path(arguments.package_identity).resolve(strict=True)
    except OSError as error:
        raise ContainmentError(
            f"invoked --package-identity does not resolve: {arguments.package_identity}: {error}"
        ) from error
    if invoked_identity != package_identity:
        raise ContainmentError(
            "invoked --package-identity differs from the package identity derivation: "
            f"{invoked_identity} != {package_identity}"
        )

    package_profile_path_candidate = package / PROFILE_RELATIVE_PATH
    package_profile_path: Path | None = None
    package_profile_id: str | None = None
    if package_profile_path_candidate.exists():
        package_profile_path = strict_file(
            str(package_profile_path_candidate), "package v3 profile"
        )
        package_profile = read_json_object(package_profile_path, "package v3 profile")
        candidate_id = package_profile.get("profile_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ContainmentError("package v3 profile carries no profile_id")
        package_profile_id = candidate_id

    if arguments.profile:
        profile = strict_file(arguments.profile, "invoked profile")
    elif package_profile_path is not None:
        profile = package_profile_path
    else:
        raise ContainmentError(
            "package ships no v3 profile; the invoked profile must be named explicitly"
        )
    if not inside(profile, package / "share/qwen36"):
        raise ContainmentError(
            f"invoked profile escapes the package's share/qwen36: {profile}"
        )
    profile_payload = read_json_object(profile, "invoked profile")
    profile_id = profile_payload.get("profile_id")
    if not isinstance(profile_id, str) or not profile_id:
        raise ContainmentError("invoked profile carries no profile_id")

    is_v3 = package_profile_id is not None and (
        profile == package_profile_path or profile_id == package_profile_id
    )
    if is_v3 and profile != package_profile_path:
        raise ContainmentError(
            "invoked profile claims the v3 profile_id but is not the package export: "
            f"{profile} != {package_profile_path}"
        )
    if not is_v3:
        if arguments.expect_unadmitted:
            raise ContainmentError(
                "--expect-unadmitted requires the package v3 profile"
            )
        return {
            "schema": SCHEMA,
            "package_store_path": str(package),
            "package_identity": str(package_identity),
            "wrapper": str(wrapper),
            "profile_path": str(profile),
            "profile_id": profile_id,
            "mode": NON_V3_MODE,
            "can_serve": False,
            "passed": True,
        }

    admitted = profile_payload.get("physically_admitted")
    if not isinstance(admitted, bool):
        raise ContainmentError("v3 profile carries no boolean physically_admitted")
    digest = receipt_digest_of(profile_payload)

    if arguments.expect_unadmitted:
        if not arguments.receipt:
            raise ContainmentError(
                "--expect-unadmitted requires --receipt naming its output"
            )
        expected_receipt = Path(arguments.receipt)
        if not expected_receipt.is_absolute():
            raise ContainmentError("--receipt must be absolute")
        if expected_receipt.exists():
            raise ContainmentError(
                f"expected receipt already exists, refusing a stale target: {expected_receipt}"
            )
        if admitted:
            raise ContainmentError("--expect-unadmitted rejects an admitted v3 profile")
        if digest != NOT_MEASURED:
            raise ContainmentError(
                f"--expect-unadmitted requires {NOT_MEASURED}, found {digest}"
            )
        print(
            "containment: unadmitted campaign profile; this run cannot serve "
            f"and must produce receipt {expected_receipt}"
        )
        return {
            "schema": SCHEMA,
            "package_store_path": str(package),
            "package_identity": str(package_identity),
            "wrapper": str(wrapper),
            "profile_path": str(profile),
            "profile_id": profile_id,
            "mode": UNADMITTED_MODE,
            "can_serve": False,
            "expected_receipt": str(expected_receipt),
            "passed": True,
        }

    if arguments.receipt:
        raise ContainmentError("--receipt is only valid with --expect-unadmitted")
    if not admitted:
        raise ContainmentError(
            "v3 profile is unadmitted; re-earn a receipt or pass --expect-unadmitted "
            "for a campaign that cannot serve"
        )
    if digest == NOT_MEASURED:
        raise ContainmentError("admitted v3 profile carries no receipt digest")
    receipt_root = (
        Path(arguments.receipt_root) if arguments.receipt_root else DEFAULT_RECEIPT_ROOT
    )
    receipt = find_receipt(receipt_root.resolve(strict=True), digest)
    return {
        "schema": SCHEMA,
        "package_store_path": str(package),
        "package_identity": str(package_identity),
        "wrapper": str(wrapper),
        "profile_path": str(profile),
        "profile_id": profile_id,
        "mode": V3_MODE,
        "can_serve": True,
        "receipt_blake3": digest,
        "receipt_path": str(receipt),
        "passed": True,
    }


def write_report(path: str | None, report: dict[str, Any]) -> None:
    if path is None:
        return
    destination = Path(path)
    if not destination.is_absolute():
        raise ContainmentError("--report must be absolute")
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="qwen38-containment-self-test-") as root:
        root_path = Path(root)
        package = root_path / "store" / "hash-qwen38-0.1.0"
        identity = root_path / "store" / "hash-qwen36-execution-identity-0.1.0"
        (identity / "share/qwen36").mkdir(parents=True)
        (identity / "share/qwen36/execution-inputs").write_text(
            "schema=onix.qwen36.execution-identity.v1\n", encoding="utf-8"
        )
        (package / "bin").mkdir(parents=True)
        (package / "share/qwen36").mkdir(parents=True)
        (package / "share/qwen36/execution-identity").symlink_to(identity)
        wrapper = package / "bin/qwen38-p150x2-b8-admission"
        wrapper.write_text(
            "#!/bin/sh\nexec python -m qwen36_service.server "
            f'--package-identity {identity} --runtime-identity /nix/store/runtime "$@"\n',
            encoding="utf-8",
        )
        wrapper.chmod(0o755)
        v3_profile_path = package / PROFILE_RELATIVE_PATH
        receipt_root = root_path / "evidence"
        receipt_root.mkdir()
        empty_receipt = receipt_root / "receipt.json"
        empty_receipt.write_bytes(b"")
        empty_digest = blake3_bytes(b"")
        if (
            empty_digest
            != "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
        ):
            raise ContainmentError("self-test BLAKE3 vector does not match")
        v3_profile = {
            "profile_id": "qwen38-p150x2-batched-throughput-b8-v3",
            "physically_admitted": True,
            "identities": {"receipt_blake3": empty_digest},
        }
        v3_profile_path.write_text(json.dumps(v3_profile), encoding="utf-8")
        generic_profile_path = package / "share/qwen36/batch-admission-b8-profile.json"
        generic_profile_path.write_text(
            json.dumps({"profile_id": "qwen38-p150x2-batched-throughput-b8-v2"}),
            encoding="utf-8",
        )

        def run(*extra: str) -> tuple[int, str]:
            argv = [
                "ignored",
                str(package),
                "--package-identity",
                str(identity),
                "--receipt-root",
                str(receipt_root),
                *extra,
            ]
            saved = sys.argv
            sys.argv = argv
            try:
                try:
                    arguments = parse_arguments()
                    check_containment(arguments, require_store=False)
                except ContainmentError as error:
                    return FAIL_EXIT_CODE, str(error)
                return PASS_EXIT_CODE, ""
            finally:
                sys.argv = saved

        cases: list[tuple[str, tuple[str, ...], int, str]] = [
            (
                "profile outside the package",
                ("--profile", str(root_path / "external.json")),
                FAIL_EXIT_CODE,
                "invoked profile",
            ),
            (
                "identity mismatch",
                ("--package-identity", str(root_path / "other-identity")),
                FAIL_EXIT_CODE,
                "invoked --package-identity",
            ),
            (
                "unadmitted v3 without the campaign mode",
                (),
                FAIL_EXIT_CODE,
                "unadmitted",
            ),
            (
                "receipt digest mismatch",
                (),
                FAIL_EXIT_CODE,
                "no receipt under",
            ),
        ]
        (root_path / "external.json").write_text(
            json.dumps({"profile_id": "qwen38-p150x2-batched-throughput-b8-v3"}),
            encoding="utf-8",
        )
        (root_path / "other-identity").mkdir()
        v3_profile["physically_admitted"] = False
        v3_profile["identities"]["receipt_blake3"] = NOT_MEASURED
        v3_profile_path.write_text(json.dumps(v3_profile), encoding="utf-8")
        for name, extra, expected, fragment in cases:
            if name == "profile outside the package":
                # An external profile is rejected before its bytes are read.
                code, message = run(*extra)
            elif name == "identity mismatch":
                code, message = run(*extra)
            elif name == "unadmitted v3 without the campaign mode":
                code, message = run(*extra)
            else:
                # An admitted profile that names a digest no receipt carries.
                v3_profile["physically_admitted"] = True
                v3_profile["identities"]["receipt_blake3"] = "0" * 64
                v3_profile_path.write_text(json.dumps(v3_profile), encoding="utf-8")
                code, message = run(*extra)
            if code != expected or fragment not in message:
                raise ContainmentError(
                    f"self-test case {name!r} expected {expected} containing "
                    f"{fragment!r}, got {code} {message!r}"
                )

        # Positive: admitted v3 with a receipt whose real BLAKE3 matches.
        v3_profile["physically_admitted"] = True
        v3_profile["identities"]["receipt_blake3"] = empty_digest
        v3_profile_path.write_text(json.dumps(v3_profile), encoding="utf-8")
        code, message = run()
        if code != PASS_EXIT_CODE:
            raise ContainmentError(f"self-test admitted profile failed: {message}")

        # Positive: unadmitted campaign mode names the receipt it will produce.
        v3_profile["physically_admitted"] = False
        v3_profile["identities"]["receipt_blake3"] = NOT_MEASURED
        v3_profile_path.write_text(json.dumps(v3_profile), encoding="utf-8")
        code, message = run(
            "--expect-unadmitted",
            "--receipt",
            str(root_path / "future-receipt.json"),
        )
        if code != PASS_EXIT_CODE:
            raise ContainmentError(f"self-test campaign mode failed: {message}")

        # Positive: a deliberately non-v3 package profile is accepted as non-servable.
        code, message = run("--profile", str(generic_profile_path))
        if code != PASS_EXIT_CODE:
            raise ContainmentError(f"self-test non-v3 profile failed: {message}")

        # Negative: campaign mode refuses an admitted profile.
        v3_profile["physically_admitted"] = True
        v3_profile["identities"]["receipt_blake3"] = empty_digest
        v3_profile_path.write_text(json.dumps(v3_profile), encoding="utf-8")
        code, message = run(
            "--expect-unadmitted", "--receipt", str(root_path / "future.json")
        )
        if code != FAIL_EXIT_CODE or "rejects an admitted" not in message:
            raise ContainmentError(
                f"self-test accepted an admitted campaign: {message}"
            )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", help="exact qwen38 package output directory")
    parser.add_argument("--profile", help="invoked batch admission profile JSON")
    parser.add_argument(
        "--package-identity",
        required=True,
        help="the --package-identity the process is invoked with",
    )
    parser.add_argument("--wrapper", help="package wrapper to inspect")
    parser.add_argument("--receipt-root", help="tree of receipt JSON files")
    parser.add_argument(
        "--expect-unadmitted",
        action="store_true",
        help="assert an unadmitted v3 campaign profile that cannot serve",
    )
    parser.add_argument(
        "--receipt", help="absolute receipt path the unadmitted campaign will produce"
    )
    parser.add_argument("--report", help="absolute path for the JSON verdict")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        try:
            self_test()
        except ContainmentError as error:
            print(
                f"qwen38-admission-containment self-test: FAIL: {error}",
                file=sys.stderr,
            )
            return FAIL_EXIT_CODE
        print("qwen38-admission-containment self-test: PASS")
        return PASS_EXIT_CODE
    arguments = parse_arguments()
    try:
        report = check_containment(arguments)
        write_report(arguments.report, report)
    except ContainmentError as error:
        print(f"qwen38-admission-containment: {error}", file=sys.stderr)
        return FAIL_EXIT_CODE
    print(json.dumps(report, indent=2, sort_keys=True))
    return PASS_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
