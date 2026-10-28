#!/usr/bin/env python3
"""Compare the packaged Tenstorrent stack with tt-sw-manifest golden.json."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

SCHEMA = "onix.golden-drift.v1"
COMPONENTS = ("kmd", "smi", "flash", "sfpi", "metal")


def normalize(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return "unknown"
    return value[1:] if value.startswith("v") and value[1:2].isdigit() else value


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except OSError as error:
        raise SystemExit(f"golden-drift: cannot read {path}: {error}") from error
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"golden-drift: {path} is not JSON: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"golden-drift: {path} is not an object")
    return value


def compare(golden: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "kmd": normalize(golden.get("kmd")),
        "smi": normalize(golden.get("smi")),
        "flash": normalize(golden.get("flash")),
        "sfpi": normalize(golden.get("sfpi")),
        "metal": normalize(golden.get("metal-version")),
    }
    components = {}
    drift = []
    for name in COMPONENTS:
        present = normalize(actual.get(name))
        wanted = expected[name]
        status = "match" if present == wanted else "drift"
        if status == "drift":
            drift.append(name)
        components[name] = {"expected": wanted, "packaged": present, "status": status}
    firmware = normalize(golden.get("firmware"))
    components["firmware"] = {
        "expected": firmware,
        "packaged": normalize(actual.get("firmware")),
        "status": "unpinned",
    }
    return {
        "schema": SCHEMA,
        "golden": normalize(
            os.environ.get(
                "GOLDEN_DRIFT_VERSION", golden.get("golden-version", "unknown")
            )
        ),
        "components": components,
        "drift": drift,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--golden",
        default=os.environ.get("GOLDEN_DRIFT_GOLDEN"),
        help="golden.json path (default: packaged snapshot)",
    )
    parser.add_argument(
        "--actual",
        default=os.environ.get("GOLDEN_DRIFT_ACTUAL"),
        help="packaged-version JSON path (default: packaged snapshot)",
    )
    parser.add_argument(
        "--strict", action="store_true", help="fail when a component drifted"
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    arguments = parser.parse_args()
    if not arguments.golden or not arguments.actual:
        parser.error("no snapshot paths are packaged")
    report = compare(
        load_json(Path(arguments.golden)), load_json(Path(arguments.actual))
    )
    if arguments.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"golden: {report['golden']}")
        for name, item in report["components"].items():
            print(
                f"{name:9} expected={item['expected']:<10} "
                f"packaged={item['packaged']:<10} {item['status']}"
            )
    if arguments.strict and report["drift"]:
        print(
            "golden-drift: drift in " + ", ".join(report["drift"]),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
