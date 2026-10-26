#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-only
# r[impl tenstorrent.runtime_profile.status]
# r[impl tenstorrent.runtime_profile.check]
# r[impl tenstorrent.runtime_profile.record]
"""Report, validate, and record the active Tenstorrent model runtime.

    tt-model status [--json]
    tt-model check <profile> [--package PATH] [--deep [--allow-large]] [--json]
    tt-model record <profile> --package PATH [--force] [--json]

`status` reads only: no device, no write, no unit mutation. `check` is pure
validation and also writes nothing. `record` writes the active record, an
append-only history entry, and one GC root per store path the record names, all
inside the state root and GC-root directory below.

No subcommand needs root or sudo, opens a device, stops or starts a unit, or
resets the cards. The privileged owner helper (stop/start, `lsof`, device reset)
is a later step.

Exit status:

  0  success
  1  a check failed, a reported source is dangling/unrooted, or a record was
     refused
  2  usage error, or the registry export is missing or does not match its contract

Portability and test hooks, all with the host defaults:

  TT_MODEL_REGISTRY      registry export path
  TT_MODEL_REPO_ROOT     default for --repo-root, which resolves the registry's
                         repository-relative receipt and authority paths
  TT_MODEL_STATE_ROOT    root-owned record directory
  TT_MODEL_GC_ROOTS      root-owned GC-root directory
  TT_MODEL_HOME          home directory that holds the legacy operator state
  TT_MODEL_PROFILE_ROOT  directory that holds Nix profile generations
  TT_MODEL_SYSTEMCTL     systemctl to use, or `none` to skip the unit probe
  TT_MODEL_NIX           nix to query package registration, or `none` to accept
                         store presence as the equivalent read-only query
  TT_MODEL_UNATTENDED    `1` refuses `record --force`
"""

from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

STATUS_SCHEMA = "onix.tt-model.status.v1"
CHECK_SCHEMA = "onix.tt-model.check.v1"
RECORD_SCHEMA = "onix.tt-model.record.v1"
PROFILE_SCHEMA = "onix.runtime-profile.v1"
REGISTRY_SCHEMA_VERSION = 1
TOOL_VERSION = "0.1.0"
DEFAULT_STATE_ROOT = Path("/var/lib/tenstorrent-models")
DEFAULT_GC_ROOTS = Path("/nix/var/nix/gcroots/tenstorrent-models")
DEFAULT_PROFILE_ROOT = Path("/nix/var/nix/profiles")
# The legacy owner, its Nix profile, and the operator's served-version symlink.
LEGACY_UNIT = "qwen38-p150x2.service"
LEGACY_PROFILE = "qwen38-p150x2"
LEGACY_CURRENT = Path(".local") / "state" / "qwen38-p150x2" / "current"
# A deep re-hash above this size needs an explicit second decision from the
# caller. A 62 GB checkpoint is never hashed implicitly.
MAX_DEEP_BYTES = 1 << 30
DEEP_PROGRESS_FILES = 64
OK = 0
FAILED = 1
USAGE = 2
STORE_PATH = re.compile(r"^/nix/store/[0-9a-z]{32}-[A-Za-z0-9+._?=-]+$")
# Items that only an admission record can satisfy. `--force` may bypass these and
# nothing else: a missing package or entry binary is a broken profile, not an
# unadmitted one.
ADMISSION_ITEMS = ("receipt-", "authority-", "gate-", "model-target")


class RegistryError(Exception):
    """The registry export is missing or does not match its contract."""


class JsonInputError(Exception):
    """A JSON file could not be read as the object this command expects."""


class Refused(Exception):
    """A refusal that names its item and the path that produced it."""

    def __init__(self, item: str, path: str, reason: str) -> None:
        super().__init__(f"{item}: {reason}")
        self.item = item
        self.path = path
        self.reason = reason

    def as_problem(self) -> dict[str, str]:
        return {"item": self.item, "path": self.path, "reason": self.reason}


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise JsonInputError(f"{path}: file is missing") from error
    except OSError as error:
        raise JsonInputError(f"{path}: {error}") from error
    except json.JSONDecodeError as error:
        raise JsonInputError(f"{path}: not valid JSON ({error})") from error


def env_path(name: str, default: str | Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else Path(default)


def repo_root(requested: str | None) -> Path:
    return Path(requested or os.environ.get("TT_MODEL_REPO_ROOT") or ".")


def registry_path(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    # The package installs the script and the export side by side.
    return Path(__file__).resolve().parent / "runtime-profiles.json"


def b3sum_binary() -> str | None:
    return shutil.which("b3sum")


def b3sum_bytes(data: bytes) -> str:
    binary = b3sum_binary()
    if binary is None:
        raise Refused(
            "digest", "", "b3sum is unavailable, so no digest can be verified"
        )
    result = subprocess.run([binary], input=data, capture_output=True, timeout=600)
    if result.returncode != 0:
        raise Refused(
            "digest",
            "",
            f"b3sum failed: {result.stderr.decode(errors='replace').strip()}",
        )
    return result.stdout.split()[0].decode()


def b3sum_file(path: Path) -> str:
    binary = b3sum_binary()
    if binary is None:
        raise Refused(
            "digest", str(path), "b3sum is unavailable, so no digest can be verified"
        )
    try:
        result = subprocess.run([binary, str(path)], capture_output=True, timeout=3600)
    except (OSError, subprocess.SubprocessError) as error:
        raise Refused(
            "digest", str(path), f"b3sum could not read the file: {error}"
        ) from error
    if result.returncode != 0:
        raise Refused(
            "digest",
            str(path),
            f"b3sum failed: {result.stderr.decode(errors='replace').strip()}",
        )
    return result.stdout.split()[0].decode()


def load_registry(path: Path) -> dict[str, Any]:
    """Load and validate the export. A mismatch is an error, never a guess."""
    document = read_json(path)
    if not isinstance(document, dict):
        raise RegistryError(f"{path}: registry export must be a JSON object")
    if document.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise RegistryError(
            f"{path}: schema_version must be {REGISTRY_SCHEMA_VERSION}, "
            f"got {document.get('schema_version')!r}"
        )
    profiles = document.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise RegistryError(f"{path}: profiles must be a non-empty array")
    seen: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        if not isinstance(profile, dict):
            raise RegistryError(f"{path}: every profile must be a JSON object")
        name = profile.get("name")
        if not isinstance(name, str) or not name:
            raise RegistryError(f"{path}: every profile needs a non-empty name")
        if name in seen:
            raise RegistryError(f"{path}: duplicate profile name {name!r}")
        if profile.get("schema") != PROFILE_SCHEMA:
            raise RegistryError(
                f"{path}: profile {name!r} schema must be {PROFILE_SCHEMA}, "
                f"got {profile.get('schema')!r}"
            )
        mode = profile.get("mode")
        if mode not in ("serve", "job"):
            raise RegistryError(f"{path}: profile {name!r} mode must be serve or job")
        if mode == "serve" and not isinstance(profile.get("port"), int):
            raise RegistryError(
                f"{path}: serve profile {name!r} must declare an integer port"
            )
        package = profile.get("package")
        if not isinstance(package, dict) or not isinstance(
            package.get("store_name"), str
        ):
            raise RegistryError(f"{path}: profile {name!r} needs package.store_name")
        seen[name] = profile
    return {"path": path, "document": document, "profiles": seen}


def registration(package: str) -> dict[str, Any]:
    """Prove the package is a registered store path, without writing anything."""
    override = os.environ.get("TT_MODEL_NIX")
    candidate = (
        override
        if override and override != "none"
        else (
            None
            if override == "none"
            else shutil.which("nix") or "/run/current-system/sw/bin/nix"
        )
    )
    if candidate is None or not Path(candidate).exists():
        # A materialized store path is a registered one; the store only keeps
        # valid paths. Report the weaker method instead of pretending to run nix.
        return {
            "method": "store-presence",
            "ok": Path(package).is_dir(),
            "detail": "nix is unavailable; accepted the store presence as the read-only query",
        }
    try:
        result = subprocess.run(
            [candidate, "path-info", "--offline", "--json", package],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"method": "nix-path-info", "ok": False, "detail": str(error)}
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        return {"method": "nix-path-info", "ok": False, "detail": detail}
    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        return {
            "method": "nix-path-info",
            "ok": False,
            "detail": f"unreadable output: {error}",
        }
    return {
        "method": "nix-path-info",
        "ok": package in info,
        "detail": "package is registered",
    }


def live_model_targets(root: Path) -> dict[str, str] | None:
    """The model-target identities from the committed consumer projection."""
    projection = root / "model-targets" / "generated" / "consumer-identities.nix"
    try:
        text = projection.read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"builtins\.fromJSON ''\n(.*)\n''", text, re.S)
    if match is None:
        return None
    try:
        entries = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    return {
        entry["target_name"]: entry["profile_blake3"]
        for entry in entries
        if isinstance(entry, dict)
        and "target_name" in entry
        and "profile_blake3" in entry
    }


def tree_summary(path: Path) -> tuple[int, int]:
    """Total byte size and file count, from metadata only."""
    total = 0
    files = 0
    if not path.is_dir():
        return (0, 0)
    for directory, _subdirectories, names in os.walk(path):
        for name in names:
            try:
                stat = os.stat(Path(directory) / name)
            except OSError:
                continue
            total += stat.st_size
            files += 1
    return (total, files)


def checkpoint_name_matches(path: Path, revision: str) -> bool:
    """A checkpoint directory carries either the full revision or its short form."""
    return path.name == revision or path.name.endswith(revision[:12])


def check_profile(
    registry: dict[str, Any],
    name: str,
    package: str | None,
    deep: bool,
    allow_large: bool,
    root: Path,
) -> tuple[dict[str, Any], int]:
    """Validate one profile against one candidate package. Writes nothing."""
    failures: list[dict[str, str]] = []

    def fail(item: str, path: str, reason: str) -> None:
        failures.append({"item": item, "path": path, "reason": reason})

    entry = registry["profiles"].get(name)
    report: dict[str, Any] = {
        "schema": CHECK_SCHEMA,
        "profile": name,
        "mode": None,
        "registry": {"path": str(registry["path"]), "blake3": None},
        "package": None,
        "runtime": None,
        "admission": None,
        "checkpoint": None,
        "model_target": None,
        "deep": {
            "enabled": deep,
            "bytes": None,
            "files": None,
            "max_deep_bytes": MAX_DEEP_BYTES,
            "allow_large": allow_large,
        },
        "failures": failures,
        "verdict": "fail",
    }
    try:
        report["registry"]["blake3"] = b3sum_file(registry["path"])
    except Refused as refusal:
        fail(refusal.item, refusal.path, refusal.reason)
    if entry is None:
        fail("profile", name, "no such runtime profile")
        return report, FAILED

    report["mode"] = entry["mode"]
    if package is None:
        fail("package-candidate", "", "pass --package or record one first")
        return report, FAILED

    package_path = Path(package)
    package_report: dict[str, Any] = {
        "path": package,
        "store_name": entry["package"]["store_name"],
        "registration": None,
        "store_name_ok": False,
        "entry_binary": None,
        "execution_identity": None,
    }
    report["package"] = package_report
    if not STORE_PATH.match(package):
        fail("package-path", package, "not a /nix/store path")
    elif not package_path.is_dir() or package_path.is_symlink():
        fail(
            "package-exists",
            package,
            "package is missing, not a directory, or a symlink",
        )
    else:
        package_report["registration"] = registration(package)
        if not package_report["registration"]["ok"]:
            fail(
                "package-registered",
                package,
                f"package is not registered: {package_report['registration']['detail']}",
            )
        if not package_path.name.endswith(entry["package"]["store_name"]):
            fail(
                "package-store-name",
                package,
                f"store name must end with {entry['package']['store_name']}",
            )
        else:
            package_report["store_name_ok"] = True

        binary = package_path / entry["entry"]["binary"]
        package_report["entry_binary"] = str(binary)
        if not binary.is_file() or not os.access(binary, os.X_OK):
            fail(
                "entry-binary", str(binary), "entry binary is missing or not executable"
            )

        runtime = entry.get("runtime")
        if runtime is None:
            report["runtime"] = {
                "kind": "package",
                "root": ".",
                "path": package,
                "execution_identity": None,
            }
        else:
            runtime_path = package_path / runtime["root"]
            execution_identity = runtime.get("execution_identity")
            identity_path = (
                package_path / execution_identity if execution_identity else None
            )
            report["runtime"] = {
                "kind": "package-root",
                "root": runtime["root"],
                "path": str(runtime_path),
                "execution_identity": (
                    {"relative": execution_identity, "path": str(identity_path)}
                    if identity_path is not None
                    else None
                ),
            }
            if not runtime_path.exists():
                fail(
                    "runtime-root",
                    str(runtime_path),
                    "runtime root is missing from the package",
                )
            if identity_path is not None:
                if not identity_path.is_symlink():
                    fail(
                        "execution-identity",
                        str(identity_path),
                        "execution identity must be a symlink",
                    )
                else:
                    target = identity_path.parent / os.readlink(identity_path)
                    if not target.exists():
                        fail(
                            "execution-identity",
                            str(identity_path),
                            f"execution identity target is missing: {target}",
                        )
                    else:
                        package_report["execution_identity"] = str(target)

        admission = entry["admission"]
        admission_report: dict[str, Any] = {
            "kind": admission["kind"],
            "receipts": [],
            "gates": [],
        }
        report["admission"] = admission_report
        for receipt in admission.get("receipts", []):
            receipt_path = root / receipt["path"]
            record = {
                "kind": receipt["kind"],
                "path": str(receipt_path),
                "blake3": receipt["blake3"],
                "admitted": None,
            }
            admission_report["receipts"].append(record)
            if not receipt_path.is_file():
                fail("receipt-missing", str(receipt_path), "receipt file is missing")
                continue
            try:
                actual = b3sum_file(receipt_path)
            except Refused as refusal:
                fail(refusal.item, refusal.path, refusal.reason)
                continue
            if actual != receipt["blake3"]:
                fail(
                    "receipt-hash",
                    str(receipt_path),
                    f"receipt bytes hash to {actual}, not the recorded {receipt['blake3']}",
                )
                continue
            try:
                document = read_json(receipt_path)
            except JsonInputError as error:
                fail("receipt-schema", str(receipt_path), str(error))
                continue
            verdict = (
                (document.get("admission") or {}) if isinstance(document, dict) else {}
            )
            record["admitted"] = verdict.get("admitted")
            if verdict.get("admitted") is not True:
                fail(
                    "receipt-admitted",
                    str(receipt_path),
                    "receipt no longer reports an admitted verdict",
                )
            named = (verdict.get("admitted_profile") or {}).get("package")
            if named != package:
                fail(
                    "receipt-package",
                    str(receipt_path),
                    f"no receipt names this package (receipt names {named!r})",
                )
        for gate in admission.get("gates", []):
            gate_record = {
                "name": gate["name"],
                "artifact": gate.get("artifact"),
                "blake3": gate.get("blake3"),
            }
            admission_report["gates"].append(gate_record)
            if "artifact" not in gate:
                continue
            artifact_path = package_path / gate["artifact"]
            if not artifact_path.is_file():
                fail(
                    "gate-artifact",
                    str(artifact_path),
                    "recorded gate artifact is missing",
                )
                continue
            try:
                actual = b3sum_file(artifact_path)
            except Refused as refusal:
                fail(refusal.item, refusal.path, refusal.reason)
                continue
            if actual != gate["blake3"]:
                fail(
                    "gate-hash",
                    str(artifact_path),
                    f"gate artifact hashes to {actual}, not the recorded {gate['blake3']}",
                )
        checkpoint = entry["checkpoint"]
        checkpoint_report = {
            "path": checkpoint.get("path"),
            "revision": checkpoint["revision"],
            "authority_kind": checkpoint["authority_kind"],
            "authority_path": checkpoint.get("authority_path"),
            "name_matches": None,
        }
        report["checkpoint"] = checkpoint_report
        if "path" in checkpoint:
            checkpoint_path = Path(checkpoint["path"])
            if not checkpoint_path.exists():
                fail(
                    "checkpoint-path",
                    str(checkpoint_path),
                    "checkpoint path is missing",
                )
            else:
                checkpoint_report["name_matches"] = checkpoint_name_matches(
                    checkpoint_path, checkpoint["revision"]
                )
                if not checkpoint_report["name_matches"]:
                    fail(
                        "checkpoint-name",
                        str(checkpoint_path),
                        f"checkpoint directory name does not carry revision {checkpoint['revision']}",
                    )
        if "authority_path" in checkpoint:
            authority_path = root / checkpoint["authority_path"]
            if not authority_path.is_file():
                fail(
                    "authority-path",
                    str(authority_path),
                    "authority document is missing",
                )
            else:
                try:
                    actual = b3sum_file(authority_path)
                except Refused as refusal:
                    fail(refusal.item, refusal.path, refusal.reason)
                else:
                    if actual != checkpoint["authority_blake3"]:
                        fail(
                            "authority-hash",
                            str(authority_path),
                            f"authority document hashes to {actual}, not the recorded {checkpoint['authority_blake3']}",
                        )
        if "model_target_name" in entry:
            live = live_model_targets(root)
            target_name = entry["model_target_name"]
            model_target_report = {
                "name": target_name,
                "expected": entry["model_target_blake3"],
                "live": None,
            }
            report["model_target"] = model_target_report
            if live is None:
                fail(
                    "model-target",
                    str(
                        root / "model-targets" / "generated" / "consumer-identities.nix"
                    ),
                    "the model-target identity projection is unreadable",
                )
            else:
                model_target_report["live"] = live.get(target_name)
                if live.get(target_name) != entry["model_target_blake3"]:
                    fail(
                        "model-target",
                        target_name,
                        f"live identity is {live.get(target_name)!r}, not {entry['model_target_blake3']}",
                    )

    if deep:
        deep_report = report["deep"]
        checkpoint = entry["checkpoint"]
        if "path" not in checkpoint:
            deep_report.update({"bytes": 0, "files": 0, "digest": None})
        else:
            checkpoint_path = Path(checkpoint["path"])
            total, files = tree_summary(checkpoint_path)
            deep_report.update({"bytes": total, "files": files})
            if total > MAX_DEEP_BYTES and not allow_large:
                fail(
                    "deep-size-bound",
                    str(checkpoint_path),
                    f"deep re-hash would read {total} bytes, above the {MAX_DEEP_BYTES}-byte bound; pass --allow-large",
                )
            else:
                digests: list[str] = []
                seen = 0
                walked = 0
                for directory, _subdirectories, names in os.walk(checkpoint_path):
                    for name in sorted(names):
                        item = Path(directory) / name
                        try:
                            digests.append(
                                f"{item.relative_to(checkpoint_path)}\0{b3sum_file(item)}"
                            )
                        except Refused as refusal:
                            fail(refusal.item, refusal.path, refusal.reason)
                        walked += 1
                        seen += 1
                        if seen >= DEEP_PROGRESS_FILES:
                            seen = 0
                            print(
                                f"tt-model: deep: {walked}/{files} files, {total} bytes",
                                file=sys.stderr,
                            )
                deep_report["digest"] = b3sum_bytes("\n".join(sorted(digests)).encode())
                print(f"tt-model: deep: {walked}/{files} files hashed", file=sys.stderr)

    report["verdict"] = "pass" if not failures else "fail"
    return report, (OK if not failures else FAILED)


def utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def write_atomic(path: Path, data: bytes) -> None:
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def ensure_directory(path: Path, item: str) -> None:
    if path.is_symlink():
        raise Refused(item, str(path), "directory must not be a symlink")
    if path.exists() and not path.is_dir():
        raise Refused(item, str(path), "path exists and is not a directory")
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise Refused(
            item, str(path), f"directory cannot be created: {error}"
        ) from error


def create_root(root: Path, target: str) -> None:
    if root.is_symlink():
        if os.readlink(root) == target:
            return
    elif root.exists():
        raise Refused("gc-root", str(root), "GC root exists and is not a symlink")
    temporary = root.with_name(f".{root.name}-{os.getpid()}")
    if temporary.is_symlink() or temporary.exists():
        temporary.unlink()
    try:
        os.symlink(target, temporary)
        os.replace(temporary, root)
    except OSError as error:
        raise Refused(
            "gc-root", str(root), f"GC root cannot be written: {error}"
        ) from error


def record_paths(
    entry: dict[str, Any], package: str, registry: dict[str, Any], root: Path
) -> dict[str, Any]:
    """The identity tuple a record carries, resolved from the registry."""
    package_path = Path(package)
    argv = [str(package_path / entry["entry"]["binary"])] + [
        token.replace("@package@", package) for token in entry["entry"]["argv_prefix"]
    ]
    runtime = entry.get("runtime")
    if runtime is None:
        runtime_record = {"kind": "package", "root": ".", "path": package}
    else:
        runtime_record = {
            "kind": "package-root",
            "root": runtime["root"],
            "path": str(package_path / runtime["root"]),
        }
    execution_identity = None
    if runtime is not None and "execution_identity" in runtime:
        identity_path = package_path / runtime["execution_identity"]
        execution_identity = str(identity_path.parent / os.readlink(identity_path))
    checkpoint = entry["checkpoint"]
    receipts = []
    for receipt in entry["admission"].get("receipts", []):
        receipt_path = root / receipt["path"]
        document = read_json(receipt_path)
        receipts.append({
            "kind": receipt["kind"],
            "path": str(receipt_path.resolve()),
            "blake3": receipt["blake3"],
            "admitted": (document.get("admission") or {}).get("admitted") is True,
        })
    gates = [
        {
            "name": gate["name"],
            "artifact": gate.get("artifact"),
            "blake3": gate.get("blake3"),
            "path": str(package_path / gate["artifact"])
            if "artifact" in gate
            else None,
        }
        for gate in entry["admission"].get("gates", [])
    ]
    return {
        "argv": argv,
        "runtime": runtime_record,
        "execution_identity": execution_identity,
        "receipts": receipts,
        "gates": gates,
        "checkpoint": {
            "path": checkpoint.get("path"),
            "revision": checkpoint["revision"],
            "authority_kind": checkpoint["authority_kind"],
            "authority_path": (
                str((root / checkpoint["authority_path"]).resolve())
                if "authority_path" in checkpoint
                else None
            ),
            "authority_blake3": checkpoint["authority_blake3"],
        },
    }


def previous_record(state_root: Path) -> dict[str, Any] | None:
    path = state_root / "active.json"
    if not path.exists():
        return None
    try:
        document = read_json(path)
        data = path.read_bytes()
    except (JsonInputError, OSError) as error:
        raise Refused(
            "previous", str(path), f"the active record is unreadable: {error}"
        ) from error
    if not isinstance(document, dict) or document.get("schema") != RECORD_SCHEMA:
        raise Refused(
            "previous", str(path), "the active record does not carry the record schema"
        )
    return {
        "profile": document.get("profile"),
        "package": (document.get("package") or {}).get("path"),
        "recorded_at": document.get("recorded_at"),
        "record_blake3": b3sum_bytes(data),
    }


def record_profile(
    registry: dict[str, Any],
    name: str,
    package: str,
    force: bool,
    root: Path,
) -> tuple[dict[str, Any], int]:
    entry = registry["profiles"].get(name)
    if entry is None:
        raise Refused("profile", name, "no such runtime profile")
    if entry["mode"] != "serve":
        raise Refused(
            "mode",
            name,
            f"the active owner must be a serve profile; {name} is {entry['mode']}",
        )
    state_root = env_path("TT_MODEL_STATE_ROOT", DEFAULT_STATE_ROOT)
    gc_roots = env_path("TT_MODEL_GC_ROOTS", DEFAULT_GC_ROOTS)
    ensure_directory(state_root, "state-root")
    lock_path = state_root / ".record.lock"
    with open(lock_path, "a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise Refused(
                "lock",
                str(lock_path),
                f"another record operation holds the lock: {error}",
            ) from error

        report, status = check_profile(registry, name, package, False, False, root)
        if status != OK:
            structural = [
                failure
                for failure in report["failures"]
                if not failure["item"].startswith(ADMISSION_ITEMS)
            ]
            if structural:
                raise Refused(
                    "check",
                    ",".join(sorted({failure["item"] for failure in structural})),
                    "the profile does not check out and this failure is not an admission failure",
                )
            if not force:
                raise Refused(
                    "check",
                    ",".join(
                        sorted({failure["item"] for failure in report["failures"]})
                    ),
                    "the profile does not check out; pass --force to record the bypass",
                )
            if os.environ.get("TT_MODEL_UNATTENDED") == "1":
                raise Refused(
                    "force-unattended",
                    name,
                    "--force is refused when TT_MODEL_UNATTENDED=1",
                )
        ensure_directory(gc_roots, "gc-root")

        try:
            resolved = record_paths(entry, package, registry, root)
        except JsonInputError as error:
            raise Refused("receipt-schema", "", str(error)) from error
        previous = previous_record(state_root)
        recorded_at = utc_now()
        canonical_argv = "\0".join(resolved["argv"]).encode()
        record = {
            "schema": RECORD_SCHEMA,
            "profile": name,
            "profile_schema": entry["schema"],
            "mode": entry["mode"],
            "package": {
                "path": package,
                "store_name": entry["package"]["store_name"],
                "execution_identity": resolved["execution_identity"],
            },
            "runtime": resolved["runtime"],
            "checkpoint": resolved["checkpoint"],
            "receipts": resolved["receipts"],
            "gates": resolved["gates"],
            "argv": resolved["argv"],
            "argv_blake3": b3sum_bytes(canonical_argv),
            "env": entry["env"],
            "port": entry.get("port"),
            "reset": entry["reset"],
            "admission_bypassed": bool(force and status != OK),
            "previous": previous,
            "recorded_at": recorded_at.isoformat(),
            "recorded_by": f"tt-model {TOOL_VERSION} uid={os.getuid()}",
            "registry_blake3": report["registry"]["blake3"],
        }
        data = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()

        roots = [
            ("package", package),
            ("execution-identity", resolved["execution_identity"]),
            ("runtime", resolved["runtime"]["path"]),
        ]
        for role, path in roots:
            # Only a whole store path can be a GC root; a subpath of one is not a
            # root target and the collector rejects it.
            if isinstance(path, str) and STORE_PATH.match(path):
                create_root(gc_roots / f"{name}-{role}", path)

        history_dir = state_root / "history"
        ensure_directory(history_dir, "state-root")
        stamp = recorded_at.strftime("%Y%m%dT%H%M%S.%fZ")
        history_path = history_dir / f"{stamp}-{name}.json"
        try:
            handle = os.open(history_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            with os.fdopen(handle, "wb") as output:
                output.write(data)
        except FileExistsError as error:
            raise Refused(
                "history", str(history_path), "history entry already exists"
            ) from error
        write_atomic(state_root / "active.json", data)
        return {
            "schema": RECORD_SCHEMA,
            "profile": name,
            "package": package,
            "active": str(state_root / "active.json"),
            "history": str(history_path),
            "gc_roots": {
                role: str(gc_roots / f"{name}-{role}")
                for role, path in roots
                if isinstance(path, str) and STORE_PATH.match(path)
            },
            "admission_bypassed": record["admission_bypassed"],
            "argv_blake3": record["argv_blake3"],
            "registry_blake3": record["registry_blake3"],
        }, OK


# ---------------------------------------------------------------------------
# Status: read what is active, never trust one source blindly.
# ---------------------------------------------------------------------------


def source(state: str, **fields: Any) -> dict[str, Any]:
    return {"state": state, **fields}


def readlink_target(path: Path) -> str:
    """The raw link target, without following it into a missing store path."""
    return os.readlink(path)


def absolute_target(path: Path) -> Path:
    """The link target, anchored at the link's directory when it is relative."""
    target = Path(readlink_target(path))
    return target if target.is_absolute() else path.parent / target


def probe_operator_symlink(home: Path) -> dict[str, Any]:
    path = home / LEGACY_CURRENT
    try:
        is_link = path.is_symlink()
    except OSError as error:
        return source("unavailable", path=str(path), reason=str(error))
    if not is_link:
        if path.exists():
            return source(
                "broken",
                path=str(path),
                reason="expected a symlink; the served version is not a plain directory",
            )
        return source("absent", path=str(path))
    target = str(absolute_target(path))
    if not Path(target).exists():
        return source("dangling", path=str(path), target=target)
    return source("ok", path=str(path), target=target)


def profile_packages(profile_path: Path) -> list[str] | None:
    """The store paths a Nix profile generation installs, from its manifest.

    `None` means the generation carries no readable manifest; the caller reports
    that instead of guessing a package.
    """
    manifest = profile_path / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        document = read_json(manifest)
    except JsonInputError:
        return None
    if not isinstance(document, dict):
        return None
    packages: list[str] = []
    elements = document.get("elements")
    if not isinstance(elements, dict):
        return packages
    for element in elements.values():
        if not isinstance(element, dict):
            continue
        for store_path in element.get("storePaths") or []:
            if isinstance(store_path, str):
                packages.append(store_path)
    return sorted(packages)


def probe_nix_profile(profile_root: Path, name: str) -> dict[str, Any]:
    path = profile_root / name
    try:
        is_link = path.is_symlink()
    except OSError as error:
        return source("unavailable", path=str(path), reason=str(error))
    if not is_link:
        if path.exists():
            return source("broken", path=str(path), reason="expected a profile symlink")
        return source("absent", path=str(path))
    generation = readlink_target(path)
    generation_path = profile_root / generation
    try:
        target = str(absolute_target(generation_path))
    except OSError as error:
        return source(
            "broken",
            path=str(path),
            generation=generation,
            reason=f"generation link is not a symlink: {error}",
        )
    if not Path(target).exists():
        return source(
            "dangling",
            path=str(path),
            generation=generation,
            target=target,
        )
    packages = profile_packages(Path(target))
    if packages is not None:
        missing = [package for package in packages if not Path(package).exists()]
        if missing:
            return source(
                "dangling",
                path=str(path),
                generation=generation,
                target=target,
                packages=packages,
                missing_package=missing[0],
            )
    return source(
        "ok", path=str(path), generation=generation, target=target, packages=packages
    )


def probe_unit(name: str, override: str | None) -> dict[str, Any]:
    if override == "none":
        return source(
            "unavailable", name=name, reason="unit probe disabled by TT_MODEL_SYSTEMCTL"
        )
    command = (
        override or shutil.which("systemctl") or "/run/current-system/sw/bin/systemctl"
    )
    if not Path(command).exists():
        return source("unavailable", name=name, reason=f"no systemctl at {command}")
    fields = ("LoadState", "UnitFileState", "ActiveState", "SubState", "FragmentPath")
    try:
        result = subprocess.run(
            [command, "show", name, *(f"-p{field}" for field in fields)],
            text=True,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return source("unavailable", name=name, reason=str(error))
    if result.returncode != 0:
        return source(
            "unavailable",
            name=name,
            reason=(
                result.stderr.strip() or f"systemctl show exited {result.returncode}"
            ),
        )
    values = dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )
    load_state = values.get("LoadState", "unknown")
    if load_state == "masked":
        state = "masked"
    elif load_state == "not-found":
        state = "not-found"
    elif load_state == "loaded":
        state = "loaded"
    else:
        state = "unknown"
    return source(
        state,
        name=name,
        unit_file_state=values.get("UnitFileState"),
        active_state=values.get("ActiveState"),
        sub_state=values.get("SubState"),
        fragment=values.get("FragmentPath"),
    )


def probe_record(
    state_root: Path, gc_roots: Path, profiles: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Read the root-owned record and prove it is rooted and consistent."""
    path = state_root / "active.json"
    if not path.exists():
        return source("absent", path=str(path))
    problems: list[dict[str, str]] = []
    try:
        record = read_json(path)
    except JsonInputError as error:
        return source(
            "broken",
            path=str(path),
            problems=[
                {"source": "active record", "path": str(path), "reason": str(error)}
            ],
        )
    if not isinstance(record, dict):
        return source(
            "broken",
            path=str(path),
            problems=[
                {
                    "source": "active record",
                    "path": str(path),
                    "reason": "record must be a JSON object",
                }
            ],
        )
    if record.get("schema") != RECORD_SCHEMA:
        problems.append({
            "source": "active record",
            "path": str(path),
            "reason": f"record schema must be {RECORD_SCHEMA}, got {record.get('schema')!r}",
        })
    name = record.get("profile")
    profile = profiles.get(name) if isinstance(name, str) else None
    if profile is None:
        problems.append({
            "source": "active record",
            "path": str(path),
            "reason": f"record names unknown profile {name!r}",
        })
    package = record.get("package")
    package_path = package.get("path") if isinstance(package, dict) else None
    if not isinstance(package_path, str):
        problems.append({
            "source": "active record",
            "path": str(path),
            "reason": "record needs package.path",
        })
    elif not Path(package_path).is_dir():
        problems.append({
            "source": "active record",
            "path": package_path,
            "reason": "record package is missing from the store",
        })
    elif profile is not None and not Path(package_path).name.endswith(
        profile["package"]["store_name"]
    ):
        problems.append({
            "source": "active record",
            "path": package_path,
            "reason": f"record package does not end with {profile['package']['store_name']}",
        })
    for field in ("runtime", "checkpoint"):
        value = record.get(field)
        nested = value.get("path") if isinstance(value, dict) else None
        if isinstance(nested, str) and not Path(nested).exists():
            problems.append({
                "source": "active record",
                "path": nested,
                "reason": f"record {field}.path is missing",
            })
    execution_identity = (
        package.get("execution_identity") if isinstance(package, dict) else None
    )
    if isinstance(execution_identity, str) and not Path(execution_identity).exists():
        problems.append({
            "source": "active record",
            "path": execution_identity,
            "reason": "record execution identity is missing from the store",
        })
    root = gc_roots / name if isinstance(name, str) else None
    if root is None:
        problems.append({
            "source": "active record",
            "path": str(path),
            "reason": "record needs a string profile name to locate its GC root",
        })
    else:
        roots = [(f"{name}-package", package_path)]
        if isinstance(execution_identity, str):
            roots.append((f"{name}-execution-identity", execution_identity))
        runtime = record.get("runtime")
        runtime_path = runtime.get("path") if isinstance(runtime, dict) else None
        if isinstance(runtime_path, str) and STORE_PATH.match(runtime_path):
            roots.append((f"{name}-runtime", runtime_path))
        for root_name, expected in roots:
            root_path = gc_roots / root_name
            if not root_path.is_symlink():
                problems.append({
                    "source": "active record",
                    "path": str(root_path),
                    "reason": "record GC root is missing",
                })
                continue
            rooted = str(absolute_target(root_path))
            if not Path(rooted).exists():
                problems.append({
                    "source": "active record",
                    "path": str(root_path),
                    "reason": "GC root target is missing from the store",
                })
            elif isinstance(expected, str) and rooted != expected:
                problems.append({
                    "source": "active record",
                    "path": str(root_path),
                    "reason": f"GC root resolves to {rooted}, not {expected}",
                })
    return source(
        "ok" if not problems else "broken",
        path=str(path),
        record=record,
        problems=problems,
    )


def collect(registry: dict[str, Any]) -> tuple[dict[str, Any], int]:
    profiles = registry["profiles"]
    home = Path(os.environ.get("TT_MODEL_HOME") or Path.home())
    state_root = env_path("TT_MODEL_STATE_ROOT", DEFAULT_STATE_ROOT)
    gc_roots = env_path("TT_MODEL_GC_ROOTS", DEFAULT_GC_ROOTS)
    profile_root = env_path("TT_MODEL_PROFILE_ROOT", DEFAULT_PROFILE_ROOT)

    active = probe_record(state_root, gc_roots, profiles)
    operator = probe_operator_symlink(home)
    nix_profile = probe_nix_profile(profile_root, LEGACY_PROFILE)
    unit = probe_unit(LEGACY_UNIT, os.environ.get("TT_MODEL_SYSTEMCTL"))

    problems: list[dict[str, str]] = list(active.get("problems", []))
    for label, probe in (("operator symlink", operator), ("nix profile", nix_profile)):
        if probe["state"] == "dangling":
            if probe.get("missing_package"):
                reason = (
                    "generation installs a package that is missing: "
                    f"{probe['missing_package']}"
                )
            else:
                reason = f"dangling symlink to {probe.get('target')}"
            problems.append({"source": label, "path": probe["path"], "reason": reason})
        elif probe["state"] == "broken":
            problems.append({
                "source": label,
                "path": probe["path"],
                "reason": probe.get("reason", "broken"),
            })

    record = active.get("record") if active["state"] == "ok" else None
    if isinstance(record, dict) and isinstance(record.get("package"), dict):
        recorded_package = record["package"].get("path")
        if operator["state"] == "ok" and operator.get("target") != recorded_package:
            problems.append({
                "source": "operator symlink",
                "path": operator["path"],
                "reason": f"resolves to {operator.get('target')}, but the record names {recorded_package}",
            })
        installed = nix_profile.get("packages")
        if (
            nix_profile["state"] == "ok"
            and installed is not None
            and recorded_package not in installed
        ):
            problems.append({
                "source": "nix profile",
                "path": nix_profile["path"],
                "reason": f"generation installs {', '.join(installed) or 'no package'}, but the record names {recorded_package}",
            })
        if unit["state"] in ("masked", "not-found"):
            problems.append({
                "source": "unit",
                "path": unit["name"],
                "reason": f"record names an owner unit whose state is {unit['state']}",
            })

    if problems:
        verdict = "broken"
        status = FAILED
    elif active["state"] == "ok":
        verdict = "managed"
        status = OK
    elif (
        operator["state"] == "ok"
        or nix_profile["state"] == "ok"
        or unit["state"] == "loaded"
    ):
        verdict = "legacy"
        status = OK
    elif "unavailable" in (operator["state"], nix_profile["state"], unit["state"]):
        verdict = "degraded"
        status = OK
    else:
        verdict = "none"
        status = OK

    report = {
        "schema": STATUS_SCHEMA,
        "verdict": verdict,
        "registry": {
            "path": str(registry["path"]),
            "schema_version": registry["document"]["schema_version"],
            "profiles": sorted(profiles),
        },
        "active": active,
        "legacy": {
            "operator_symlink": operator,
            "nix_profile": nix_profile,
            "unit": unit,
        },
        "unavailable": sorted(
            label
            for label, probe in (
                ("active record", active),
                ("operator symlink", operator),
                ("nix profile", nix_profile),
                ("unit", unit),
            )
            if probe["state"] == "unavailable"
        ),
        "problems": problems,
    }
    return report, status


def render_status_human(report: dict[str, Any]) -> str:
    lines = [
        f"verdict: {report['verdict']}",
        f"registry: {report['registry']['path']} "
        f"(schema {report['registry']['schema_version']}, "
        f"{len(report['registry']['profiles'])} profiles: "
        f"{', '.join(report['registry']['profiles'])})",
    ]
    active = report["active"]
    if active["state"] == "absent":
        lines.append(f"active record: absent ({active['path']})")
    elif active["state"] == "ok":
        record = active["record"]
        lines.append(f"active record: {active['path']}")
        lines.append(f"  profile: {record.get('profile')}")
        package = (
            record.get("package") if isinstance(record.get("package"), dict) else {}
        )
        lines.append(f"  package: {package.get('path')}")
        if package.get("execution_identity"):
            lines.append(f"  execution identity: {package['execution_identity']}")
        lines.append(f"  recorded at: {record.get('recorded_at')}")
    else:
        lines.append(f"active record: {active['state']} ({active['path']})")
    legacy = report["legacy"]
    for label, key in (
        ("operator symlink", "operator_symlink"),
        ("nix profile", "nix_profile"),
    ):
        probe = legacy[key]
        detail = probe.get("target") or probe.get("reason") or ""
        generation = (
            f", generation {probe['generation']}" if probe.get("generation") else ""
        )
        lines.append(
            f"{label}: {probe['state']} ({probe['path']}{generation}){(' -> ' + detail) if detail else ''}"
        )
        if probe.get("packages") is not None:
            lines.append(f"  packages: {', '.join(probe['packages']) or 'none'}")
    unit = legacy["unit"]
    if unit["state"] == "unavailable":
        lines.append(f"unit: unavailable ({unit['name']}: {unit.get('reason')})")
    else:
        lines.append(
            f"unit: {unit['state']} ({unit['name']}, active={unit.get('active_state')}, "
            f"file={unit.get('unit_file_state')})"
        )
    for problem in report["problems"]:
        source_name = problem.get("source", "active record")
        lines.append(f"problem: {source_name}: {problem['path']}: {problem['reason']}")
    if report["unavailable"]:
        lines.append(f"unavailable: {', '.join(report['unavailable'])}")
    return "\n".join(lines)


def render_check_human(report: dict[str, Any]) -> str:
    lines = [
        f"profile: {report['profile']} ({report['mode']})",
        f"verdict: {report['verdict']}",
    ]
    package = report.get("package") or {}
    if package:
        registration_info = package.get("registration") or {}
        lines.append(
            f"package: {package.get('path')} "
            f"(registered: {registration_info.get('method')}, "
            f"store name ok: {package.get('store_name_ok')})"
        )
        lines.append(f"  entry binary: {package.get('entry_binary')}")
    runtime = report.get("runtime") or {}
    if runtime:
        lines.append(f"runtime: {runtime.get('kind')} {runtime.get('path')}")
        if runtime.get("execution_identity"):
            lines.append(
                f"  execution identity: {runtime['execution_identity']['path']}"
            )
    admission = report.get("admission") or {}
    if admission:
        lines.append(f"admission: {admission.get('kind')}")
        for receipt in admission.get("receipts", []):
            lines.append(
                f"  receipt {receipt['kind']}: {receipt['path']} admitted={receipt['admitted']}"
            )
        for gate in admission.get("gates", []):
            lines.append(
                f"  gate {gate['name']}: {gate.get('artifact') or 'no recorded artifact'}"
            )
    checkpoint = report.get("checkpoint") or {}
    if checkpoint:
        lines.append(
            f"checkpoint: {checkpoint.get('path') or 'no host path'} "
            f"revision {checkpoint.get('revision')}"
        )
    model_target = report.get("model_target") or {}
    if model_target:
        lines.append(
            f"model target: {model_target.get('name')} "
            f"expected {model_target.get('expected')} live {model_target.get('live')}"
        )
    deep = report.get("deep") or {}
    if deep.get("enabled"):
        lines.append(
            f"deep: files={deep.get('files')} bytes={deep.get('bytes')} "
            f"bound={deep.get('max_deep_bytes')} allow-large={deep.get('allow_large')}"
        )
    for failure in report["failures"]:
        lines.append(
            f"failure: {failure['item']}: {failure['path']}: {failure['reason']}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tt-model",
        description="Report, validate, and record the active Tenstorrent model runtime.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    status = subcommands.add_parser(
        "status", help="report the active runtime and every source"
    )
    status.add_argument(
        "--json", action="store_true", help="emit the stable JSON schema"
    )
    check = subcommands.add_parser(
        "check", help="validate one profile against one package"
    )
    check.add_argument("profile")
    check.add_argument("--package", help="candidate package store path")
    check.add_argument(
        "--repo-root", help="directory that resolves repository-relative paths"
    )
    check.add_argument(
        "--deep", action="store_true", help="re-hash the checkpoint as well"
    )
    check.add_argument(
        "--allow-large",
        action="store_true",
        help=f"allow a deep re-hash above {MAX_DEEP_BYTES} bytes",
    )
    check.add_argument(
        "--json", action="store_true", help="emit the stable JSON schema"
    )
    record = subcommands.add_parser(
        "record", help="write the active record and its GC roots"
    )
    record.add_argument("profile")
    record.add_argument("--package", required=True, help="exact package store path")
    record.add_argument(
        "--repo-root", help="directory that resolves repository-relative paths"
    )
    record.add_argument(
        "--force",
        action="store_true",
        help="record an admission failure as admission_bypassed",
    )
    record.add_argument(
        "--json", action="store_true", help="emit the stable JSON schema"
    )
    args = parser.parse_args(argv)

    try:
        registry = load_registry(registry_path(os.environ.get("TT_MODEL_REGISTRY")))
    except (RegistryError, JsonInputError) as error:
        print(f"tt-model: {error}", file=sys.stderr)
        return USAGE

    if args.command == "status":
        report, status_code = collect(registry)
        print(
            json.dumps(report, indent=2, sort_keys=True)
            if args.json
            else render_status_human(report)
        )
        return status_code

    root = repo_root(args.repo_root)
    package = args.package
    if args.command == "check" and package is None:
        state_root = env_path("TT_MODEL_STATE_ROOT", DEFAULT_STATE_ROOT)
        try:
            previous = read_json(state_root / "active.json")
            if isinstance(previous, dict) and previous.get("profile") == args.profile:
                package = (previous.get("package") or {}).get("path")
        except JsonInputError:
            package = None

    if args.command == "check":
        report, status_code = check_profile(
            registry, args.profile, package, args.deep, args.allow_large, root
        )
        print(
            json.dumps(report, indent=2, sort_keys=True)
            if args.json
            else render_check_human(report)
        )
        return status_code

    try:
        report, status_code = record_profile(
            registry, args.profile, args.package, args.force, root
        )
    except Refused as refusal:
        print(
            f"tt-model: record refused: {refusal.item}: {refusal.path}: {refusal.reason}",
            file=sys.stderr,
        )
        return FAILED
    print(
        json.dumps(report, indent=2, sort_keys=True)
        if args.json
        else "\n".join(
            f"{key}: {value}" for key, value in report.items() if key != "schema"
        )
    )
    return status_code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
