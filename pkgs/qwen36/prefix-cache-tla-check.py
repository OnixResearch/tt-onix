"""Device-free TLC model-checking harness for the Qwen3.8 prefix-cache lifecycle."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

SPEC_FILE = "PrefixCacheLifecycle.tla"

POSITIVE_RUN = {
    "name": "PrefixCacheLifecycle",
    "config": "PrefixCacheLifecycle.cfg",
    "expect_pass": True,
    "expected_invariant": None,
}

MUTANT_RUNS = [
    {
        "name": "MutantNoUnpinnedEvictionGuard",
        "config": "MutantNoUnpinnedEvictionGuard.cfg",
        "expect_pass": False,
        "expected_invariant": "LiveBlocksDisjointFromOtherRuns",
    },
    {
        "name": "MutantNoProtectedBlocksGuard",
        "config": "MutantNoProtectedBlocksGuard.cfg",
        "expect_pass": False,
        "expected_invariant": "ReadyEntryBlocksImmutable",
    },
    {
        "name": "MutantPinnedDropOnRead",
        "config": "MutantPinnedDropOnRead.cfg",
        "expect_pass": False,
        "expected_invariant": "ReadyEntryBlocksImmutable",
    },
    {
        "name": "MutantCancelCompletedSlot",
        "config": "MutantCancelCompletedSlot.cfg",
        "expect_pass": False,
        "expected_invariant": "CancelPreservesFinishReason",
    },
    {
        "name": "MutantSkipBoundaryRetire",
        "config": "MutantSkipBoundaryRetire.cfg",
        "expect_pass": False,
        "expected_invariant": "CancelledReleasedAtBoundary",
    },
]

_STATES_RE = re.compile(r"(\d+)\s+states generated,\s+(\d+)\s+distinct states found")
_INVARIANT_RE = re.compile(r"Error:\s+Invariant\s+(\S+)\s+is violated\.")
_STATE_STEP_RE = re.compile(r"^State\s+(\d+):", re.MULTILINE)


def _find_tlc() -> str:
    tlc_env = os.environ.get("TLC_BIN")
    if tlc_env and Path(tlc_env).is_file():
        return tlc_env
    found = shutil.which("tlc")
    if found:
        return found
    raise FileNotFoundError(
        "tlc executable not found (set TLC_BIN or add tlaplus to PATH)"
    )


def _run_tlc(
    tlc_bin: str,
    formal_dir: Path,
    config_name: str,
    metadir: Path,
    workers: int,
) -> dict[str, Any]:
    spec_path = formal_dir / SPEC_FILE
    cfg_path = formal_dir / config_name
    if not spec_path.is_file():
        raise FileNotFoundError(f"missing TLA+ spec: {spec_path}")
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing TLC config: {cfg_path}")

    cmd = [
        tlc_bin,
        "-deadlock",
        "-cleanup",
        "-workers",
        str(workers),
        "-metadir",
        str(metadir),
        "-config",
        str(cfg_path),
        str(spec_path),
    ]
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    output = completed.stdout + "\n" + completed.stderr
    generated = None
    distinct = None
    for match in _STATES_RE.finditer(output):
        generated = int(match.group(1))
        distinct = int(match.group(2))
    inv_match = _INVARIANT_RE.search(output)
    violated_invariant = inv_match.group(1) if inv_match else None
    steps = [int(m.group(1)) for m in _STATE_STEP_RE.finditer(output)]
    counterexample_length = max(steps) if steps else 0
    return {
        "returncode": completed.returncode,
        "passed": completed.returncode == 0
        and "Model checking completed. No error has been found." in output,
        "states_generated": generated,
        "distinct_states": distinct,
        "violated_invariant": violated_invariant,
        "counterexample_length": counterexample_length,
        "output_tail": "\n".join(output.strip().splitlines()[-20:]),
    }


def main() -> int:
    if len(sys.argv) > 1:
        formal_dir = Path(sys.argv[1]).resolve()
    else:
        formal_dir = (Path(__file__).resolve().parent / "formal").resolve()

    tlc_bin = _find_tlc()
    workers = int(os.environ.get("TLC_WORKERS", "4"))

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="qwen38-prefix-tla-") as tmp:
        tmp_path = Path(tmp)
        for case in [POSITIVE_RUN, *MUTANT_RUNS]:
            meta = tmp_path / case["name"]
            meta.mkdir(parents=True, exist_ok=True)
            res = _run_tlc(tlc_bin, formal_dir, case["config"], meta, workers)
            record = {
                "name": case["name"],
                "config": case["config"],
                "expect_pass": case["expect_pass"],
                "expected_invariant": case["expected_invariant"],
                "passed": res["passed"],
                "returncode": res["returncode"],
                "states_generated": res["states_generated"],
                "distinct_states": res["distinct_states"],
                "violated_invariant": res["violated_invariant"],
                "counterexample_length": res["counterexample_length"],
            }
            results.append(record)
            if case["expect_pass"]:
                if not res["passed"]:
                    raise SystemExit(
                        f"positive TLA+ check {case['name']} failed:\n{res['output_tail']}"
                    )
            else:
                if res["passed"]:
                    raise SystemExit(
                        f"guard-removal mutant {case['name']} unexpectedly passed TLC"
                    )
                if res["violated_invariant"] != case["expected_invariant"]:
                    raise SystemExit(
                        f"mutant {case['name']} violated {res['violated_invariant']!r}, "
                        f"expected {case['expected_invariant']!r}:\n{res['output_tail']}"
                    )
                if res["counterexample_length"] < 2:
                    raise SystemExit(
                        f"mutant {case['name']} did not emit a counterexample trace"
                    )

    print(
        json.dumps(
            {
                "prefix_cache_tla_check": "passed",
                "checker": "tlc",
                "formal_dir": str(formal_dir),
                "runs": results,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
