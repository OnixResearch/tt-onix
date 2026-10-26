#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-only
# r[impl tenstorrent.runtime_profile.status]
"""Report which Tenstorrent model runtime is active, without privilege.

    tt-model status [--json]

This command reads only. It never opens a device, never writes a file, never
stops or starts a unit, and needs neither root nor sudo. It reports the active
model from the root-owned record when one exists, and otherwise reports the
legacy layout it can see: the operator symlink, the Nix profile generation, and
the owner unit's state. A source that cannot be read is reported as unavailable
instead of failing the run.

Exit status:

  0  every source resolves, or is explicitly absent, or is unavailable
  1  a reported source is dangling, unrooted, or contradicts another source
  2  usage error, or the registry export is missing or invalid

Portability and test hooks, all with the host defaults:

  TT_MODEL_REGISTRY          registry export path
  TT_MODEL_STATE_ROOT        root-owned record directory
  TT_MODEL_GC_ROOTS          root-owned GC-root directory
  TT_MODEL_HOME              home directory that holds the operator state
  TT_MODEL_PROFILE_ROOT      directory that holds Nix profile generations
  TT_MODEL_SYSTEMCTL         systemctl to use, or `none` to skip the unit probe
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA = "onix.tt-model.status.v1"
RECORD_SCHEMA = "onix.tt-model.record.v1"
PROFILE_SCHEMA = "onix.runtime-profile.v1"
REGISTRY_SCHEMA_VERSION = 1
DEFAULT_STATE_ROOT = Path("/var/lib/tenstorrent-models")
DEFAULT_GC_ROOTS = Path("/nix/var/nix/gcroots/tenstorrent-models")
DEFAULT_PROFILE_ROOT = Path("/nix/var/nix/profiles")
# The legacy owner, its Nix profile, and the operator's served-version symlink.
LEGACY_UNIT = "qwen38-p150x2.service"
LEGACY_PROFILE = "qwen38-p150x2"
LEGACY_CURRENT = Path(".local") / "state" / "qwen38-p150x2" / "current"
OK = 0
BROKEN = 1
USAGE = 2


class RegistryError(Exception):
    """The registry export is missing or does not match its contract."""


class JsonInputError(Exception):
    """A JSON file could not be read as the object this command expects."""


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise JsonInputError(f"{path}: file is missing") from error
    except OSError as error:
        raise JsonInputError(f"{path}: {error}") from error
    except json.JSONDecodeError as error:
        raise JsonInputError(f"{path}: not valid JSON ({error})") from error


def registry_path(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    # The package installs the script and the export side by side.
    return Path(__file__).resolve().parent / "runtime-profiles.json"


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


def readlink_target(path: Path) -> str:
    """The raw link target, without following it into a missing store path."""
    return os.readlink(path)


def absolute_target(path: Path) -> Path:
    """The link target, anchored at the link's directory when it is relative."""
    target = Path(readlink_target(path))
    return target if target.is_absolute() else path.parent / target


def source(state: str, **fields: Any) -> dict[str, Any]:
    return {"state": state, **fields}


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
    root = gc_roots / name if isinstance(name, str) else None
    if root is None:
        problems.append({
            "source": "active record",
            "path": str(path),
            "reason": "record needs a string profile name to locate its GC root",
        })
    elif not root.is_symlink():
        problems.append({
            "source": "active record",
            "path": str(root),
            "reason": "record GC root is missing",
        })
    else:
        rooted = str(absolute_target(root))
        if isinstance(package_path, str) and rooted != package_path:
            problems.append({
                "source": "active record",
                "path": str(root),
                "reason": f"GC root resolves to {rooted}, not {package_path}",
            })
        elif not Path(rooted).exists():
            problems.append({
                "source": "active record",
                "path": str(root),
                "reason": "GC root target is missing from the store",
            })
    return source(
        "ok" if not problems else "broken",
        path=str(path),
        record=record,
        problems=problems,
    )


def collect(
    args: argparse.Namespace, registry: dict[str, Any]
) -> tuple[dict[str, Any], int]:
    profiles = registry["profiles"]
    home = Path(os.environ.get("TT_MODEL_HOME") or Path.home())
    state_root = Path(os.environ.get("TT_MODEL_STATE_ROOT") or DEFAULT_STATE_ROOT)
    gc_roots = Path(os.environ.get("TT_MODEL_GC_ROOTS") or DEFAULT_GC_ROOTS)
    profile_root = Path(os.environ.get("TT_MODEL_PROFILE_ROOT") or DEFAULT_PROFILE_ROOT)

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
        status = BROKEN
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
        "schema": SCHEMA,
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


def render_human(report: dict[str, Any]) -> str:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tt-model",
        description="Report the active Tenstorrent model runtime. Read-only.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    status = subcommands.add_parser(
        "status", help="report the active runtime and every source"
    )
    status.add_argument(
        "--json", action="store_true", help="emit the stable JSON schema"
    )
    args = parser.parse_args(argv)

    try:
        registry = load_registry(registry_path(os.environ.get("TT_MODEL_REGISTRY")))
    except (RegistryError, JsonInputError) as error:
        print(f"tt-model: {error}", file=sys.stderr)
        return USAGE

    report, status_code = collect(args, registry)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_human(report))
    return status_code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
