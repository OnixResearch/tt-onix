"""Device-free validation, readiness checking, and reduction for qwen38-step-probe.

This module enforces strict screen-integrity rules for ``scripts/qwen38-step-probe.sh``:
- Every requested width must receive all W expected request records with valid HTTP/JSON
  success payloads and valid ``x_onix_decode_evidence`` before emitting a usable timing row.
- Numeric and container types are strictly validated (finite positive numbers, no booleans
  masquerading as numbers, sufficient samples after the declared trimming rule).
- Requested vs observed effective GDN decode fusion route and package/execution/runtime
  identities are recorded and verified; unconfirmed or mismatched routes block any
  route-specific performance conclusion.
- Failed attempts are retained rather than dropped or replaced by cherry-picking.
- Output is explicitly labeled as a screen (not admission evidence) and states whether
  timing samples are repeated views of shared batch steps or per-request observations.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import socket
import statistics
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "onix.qwen38-step-probe.v1"
VALID_GDN_ROUTES = ("step", "full", "off")
DEFAULT_TRIM_HEAD = 1
DEFAULT_TRIM_TAIL = 1
DEFAULT_MIN_INTERIOR_SAMPLES = 2
DEFAULT_USABLE_CORES = 110
GDN_HEADS_PER_LAYER = 24

SCREEN_NOTICE = (
    "SCREEN ONLY — NOT ADMISSION EVIDENCE. "
    "One fresh server session per width with service-reported per-token decode latencies. "
    "Pooled token observations across requests in a shared batch are repeated views of "
    "shared batch steps, not independent experimental replications."
)


def is_finite_positive_number(value: Any) -> bool:
    """Return True iff value is a finite positive int/float and not a bool."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    numeric = float(value)
    return math.isfinite(numeric) and numeric > 0.0


def is_non_bool_int(value: Any) -> bool:
    """Return True iff value is an int and not a bool."""
    return isinstance(value, int) and not isinstance(value, bool)


def validate_gdn_route(route: str) -> str:
    """Validate an explicit GDN decode fusion route name."""
    if not isinstance(route, str) or route not in VALID_GDN_ROUTES:
        allowed = ", ".join(VALID_GDN_ROUTES)
        raise ValueError(f"GDN route must be one of ({allowed}), got {route!r}")
    return route


def compute_file_sha256(path: Path) -> str | None:
    """Return lowercase SHA-256 hex digest of a file, or None if unreadable."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return hashlib.sha256(data).hexdigest()


def resolve_package_identities(
    package_path: Path | None,
    *,
    runtime_identity_override: str | None = None,
) -> dict[str, str | None]:
    """Read existing qwen36/qwen38 Nix execution and runtime identities."""
    if package_path is None:
        return {
            "package_path": None,
            "execution_identity": None,
            "runtime_identity": runtime_identity_override,
        }
    resolved_pkg = str(package_path.resolve())
    exec_link = package_path / "share" / "qwen36" / "execution-identity"
    execution_identity: str | None = None
    runtime_identity: str | None = runtime_identity_override
    if exec_link.exists():
        execution_identity = os.path.realpath(exec_link)
        inputs_file = Path(execution_identity) / "share" / "qwen36" / "execution-inputs"
        if runtime_identity is None and inputs_file.is_file():
            try:
                for line in inputs_file.read_text(encoding="utf-8").splitlines():
                    if line.startswith("input=") and "-tt-metal-" in line:
                        candidate = line.split("=", 1)[1].strip()
                        if "-tt-metal-tracy-tools-" not in candidate:
                            runtime_identity = candidate
                            break
            except OSError:
                pass
    return {
        "package_path": resolved_pkg,
        "execution_identity": execution_identity,
        "runtime_identity": runtime_identity,
    }


def _collect_process_tree_pids(root_pid: int) -> set[int]:
    """Return root_pid and all descendant PIDs readable in /proc."""
    pids: set[int] = set()
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        if pid in pids:
            continue
        if not Path(f"/proc/{pid}").exists():
            continue
        pids.add(pid)
        for children_path in glob.glob(f"/proc/{pid}/task/*/children"):
            try:
                text = Path(children_path).read_text(encoding="utf-8")
            except OSError:
                continue
            for token in text.split():
                if token.isdigit():
                    stack.append(int(token))
    return pids


def _socket_inodes_for_pids(pids: set[int]) -> set[str]:
    inodes: set[str] = set()
    for pid in pids:
        for fd_path in glob.glob(f"/proc/{pid}/fd/*"):
            try:
                target = os.readlink(fd_path)
            except OSError:
                continue
            if target.startswith("socket:[") and target.endswith("]"):
                inodes.add(target[8:-1])
    return inodes


def _listening_socket_inodes_for_port(port: int) -> set[str]:
    hex_port = f"{port:04X}"
    inodes: set[str] = set()
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(table).read_text(encoding="utf-8").splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 10:
                continue
            # State 0A is TCP_LISTEN.
            if parts[3] == "0A" and parts[1].endswith(f":{hex_port}"):
                inodes.add(parts[9])
    return inodes


def is_pid_alive(pid: int) -> bool:
    """Return True if process exists and is not a zombie."""
    if pid <= 0:
        return False
    stat_path = Path(f"/proc/{pid}/stat")
    if stat_path.exists():
        try:
            content = stat_path.read_text(encoding="utf-8")
            rparen = content.rfind(")")
            if rparen != -1:
                state = content[rparen + 2 : rparen + 3]
                if state == "Z":
                    return False
        except OSError:
            pass
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def check_port_free(host: str, port: int, timeout: float = 1.0) -> tuple[bool, str]:
    """Return (True, 'port_free') iff nothing is listening on (host, port)."""
    inodes = _listening_socket_inodes_for_port(port)
    if inodes:
        return False, f"unrelated_server_already_listening_on_port_{port}"
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return False, f"unrelated_server_accepted_connection_on_port_{port}"
    except OSError:
        return True, "port_free"


def check_server_readiness(
    *,
    pid: int,
    url: str,
    host: str,
    port: int,
    expected_model: str,
    expected_session_id: str | None = None,
    timeout: float = 2.0,
) -> tuple[int, str, dict[str, Any] | None]:
    """Check whether the spawned server process is alive, owns the port, and is ready.

    Exit code contract:
      0  -> ready and verified owned by pid
      2  -> spawned child exited before readiness
      3  -> unrelated server or identity mismatch on the listening port
      10 -> child still alive and starting (not ready yet)
    """
    if not is_pid_alive(pid):
        return 2, "server_exited_before_readiness", None

    listening_inodes = _listening_socket_inodes_for_port(port)
    tree_pids = _collect_process_tree_pids(pid)
    owned_inodes = _socket_inodes_for_pids(tree_pids)

    if listening_inodes and not (listening_inodes & owned_inodes):
        return 3, "unrelated_server_owns_listening_socket", None

    health_url = f"{url.rstrip('/')}/healthz"
    request = urllib.request.Request(health_url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status_code = response.status
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        if not is_pid_alive(pid):
            return 2, "server_exited_before_readiness", None
        if listening_inodes and not (listening_inodes & owned_inodes):
            return 3, "unrelated_server_owns_listening_socket", None
        if error.code == 503:
            return 10, "service_not_ready_yet_503", None
        return 3, f"unexpected_healthz_http_{error.code}", None
    except (urllib.error.URLError, TimeoutError, OSError):
        if not is_pid_alive(pid):
            return 2, "server_exited_before_readiness", None
        return 10, "healthz_not_reachable_yet", None

    if status_code != 200:
        return 10, f"healthz_http_{status_code}", None

    # Re-verify process ownership after receiving a response so a race with
    # an unrelated listener cannot slip past an empty initial proc scan.
    if not is_pid_alive(pid):
        return 2, "server_exited_before_readiness", None
    listening_inodes = _listening_socket_inodes_for_port(port)
    owned_inodes = _socket_inodes_for_pids(_collect_process_tree_pids(pid))
    if not listening_inodes or not (listening_inodes & owned_inodes):
        return 3, "unrelated_server_responded_on_healthz", None

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return 3, "healthz_malformed_json", None

    if not isinstance(payload, dict):
        return 3, "healthz_non_object_json", None

    if payload.get("ready") is not True or payload.get("status") != "ok":
        return 10, "healthz_reports_not_ready", payload

    if expected_model and payload.get("model") != expected_model:
        return 3, "healthz_model_mismatch", payload

    if (
        expected_session_id is not None
        and "probe_session_id" in payload
        and payload.get("probe_session_id") != expected_session_id
    ):
        return 3, "healthz_probe_session_mismatch", payload

    return 0, "ready", payload


def _read_http_status_sidecar(response_path: Path) -> int | None:
    meta_path = response_path.with_suffix(".http_status")
    if not meta_path.is_file():
        return None
    try:
        raw = meta_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw or not raw.isdigit():
        return -1
    return int(raw)


def validate_response_record(
    *,
    prompt_index: int,
    response_path: Path,
    expected_width: int,
    expected_route: str,
    http_status: int | None = None,
    expected_package_identity: str | None = None,
    expected_runtime_identity: str | None = None,
    expected_run_id: str | None = None,
    expected_request_id: str | None = None,
    corpus_request_sha256: str | None = None,
    trim_head: int = DEFAULT_TRIM_HEAD,
    trim_tail: int = DEFAULT_TRIM_TAIL,
    min_interior_samples: int = DEFAULT_MIN_INTERIOR_SAMPLES,
) -> dict[str, Any]:
    """Validate one prompt response record strictly without dropping failures."""
    reasons: list[str] = []
    if http_status is None:
        http_status = _read_http_status_sidecar(response_path)

    base: dict[str, Any] = {
        "prompt_index": prompt_index,
        "response_path": str(response_path),
        "http_status": http_status,
        "status": "invalid",
        "completed": False,
        "valid": False,
        "reasons": reasons,
        "observed_route": None,
        "observed_batch_width": None,
        "observed_package_identity": None,
        "observed_runtime_identity": None,
        "observed_request_id": None,
        "observed_run_id": None,
        "observed_state_slice": None,
        "raw_sample_count": 0,
        "trimmed_sample_count": 0,
        "interior_latencies": [],
        "aggregate_tokens_per_second": None,
        "corpus_request_sha256": corpus_request_sha256,
    }

    if not response_path.is_file():
        base["status"] = "missing"
        reasons.append("missing_response_file")
        return base

    if http_status is not None and http_status != 200:
        base["status"] = "http_error"
        reasons.append(f"http_status_{http_status}")

    try:
        raw_text = response_path.read_text(encoding="utf-8")
    except OSError:
        base["status"] = "read_error"
        reasons.append("response_read_error")
        return base

    if not raw_text.strip():
        base["status"] = "empty_response"
        reasons.append("empty_response_file")
        return base

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        base["status"] = "malformed_json"
        reasons.append("malformed_json")
        return base

    if not isinstance(payload, dict):
        base["status"] = "malformed_json"
        reasons.append("non_object_json")
        return base

    if "error" in payload and payload["error"] is not None:
        base["status"] = "error_response"
        reasons.append("api_error_payload")
        return base

    if http_status is not None and http_status != 200:
        return base

    base["completed"] = True

    observed_req_id = payload.get("request_id")
    if observed_req_id is None:
        observed_req_id = payload.get("id")
    if isinstance(observed_req_id, str):
        base["observed_request_id"] = observed_req_id

    observed_run_id = payload.get("probe_run_id") or payload.get("run_id")
    evidence = payload.get("x_onix_decode_evidence")
    if not isinstance(evidence, dict):
        base["status"] = "malformed_evidence"
        reasons.append("missing_or_malformed_evidence")
        return base

    if observed_run_id is None:
        observed_run_id = evidence.get("probe_run_id") or evidence.get("run_id")
    if isinstance(observed_run_id, str):
        base["observed_run_id"] = observed_run_id

    if expected_run_id is not None:
        if observed_run_id is None:
            reasons.append("missing_run_identity")
        elif observed_run_id != expected_run_id:
            reasons.append("run_identity_mismatch")

    if expected_request_id is not None:
        if observed_req_id is None:
            reasons.append("missing_request_identity")
        elif observed_req_id != expected_request_id:
            reasons.append("request_identity_mismatch")

    observed_pkg = evidence.get("package_identity")
    if isinstance(observed_pkg, str):
        base["observed_package_identity"] = observed_pkg
    if expected_package_identity is not None:
        if observed_pkg is not None and observed_pkg != expected_package_identity:
            reasons.append("package_identity_mismatch")

    observed_rt = evidence.get("runtime_identity")
    if isinstance(observed_rt, str):
        base["observed_runtime_identity"] = observed_rt
    if expected_runtime_identity is not None:
        if observed_rt is not None and observed_rt != expected_runtime_identity:
            reasons.append("runtime_identity_mismatch")

    if "batch_width" in evidence and evidence["batch_width"] is not None:
        bw = evidence["batch_width"]
        if not is_non_bool_int(bw) or bw != expected_width:
            reasons.append("actual_width_mismatch")
        else:
            base["observed_batch_width"] = bw

    if "state_slice" in evidence and evidence["state_slice"] is not None:
        ss = evidence["state_slice"]
        if is_non_bool_int(ss) and 0 <= ss < expected_width:
            base["observed_state_slice"] = ss
        else:
            reasons.append("invalid_state_slice")

    observed_route = evidence.get("serving_route")
    if observed_route is None:
        observed_route = evidence.get("gdn_decode_fusion")
    if not isinstance(observed_route, str) or observed_route not in VALID_GDN_ROUTES:
        reasons.append("unconfirmed_effective_route")
    else:
        base["observed_route"] = observed_route
        if observed_route != expected_route:
            reasons.append("route_mismatch")

    raw_latencies = evidence.get("token_latency_seconds")
    if not isinstance(raw_latencies, list):
        reasons.append("invalid_latency_container")
    elif not raw_latencies:
        reasons.append("empty_latency_array")
    else:
        base["raw_sample_count"] = len(raw_latencies)
        has_bool = any(isinstance(item, bool) for item in raw_latencies)
        has_bad = any(not is_finite_positive_number(item) for item in raw_latencies)
        if has_bool:
            reasons.append("boolean_latency_value")
        elif has_bad:
            reasons.append("nonfinite_or_nonpositive_latency_value")
        else:
            min_required_raw = trim_head + trim_tail + min_interior_samples
            if len(raw_latencies) < min_required_raw:
                reasons.append("insufficient_samples_after_trim")
            else:
                end_index = len(raw_latencies) - trim_tail if trim_tail > 0 else None
                interior = [float(x) for x in raw_latencies[trim_head:end_index]]
                base["interior_latencies"] = interior
                base["trimmed_sample_count"] = len(interior)

    agg = evidence.get("aggregate_decode_tokens_per_second")
    if isinstance(agg, bool):
        reasons.append("boolean_aggregate_tokens_per_second")
    elif not is_finite_positive_number(agg):
        reasons.append("invalid_aggregate_tokens_per_second")
    else:
        base["aggregate_tokens_per_second"] = float(agg)

    if not reasons:
        base["status"] = "valid"
        base["valid"] = True
    else:
        base["status"] = "invalid_evidence"
    return base


def _summarize_valid_rows(
    valid_rows: list[dict[str, Any]],
    *,
    width: int,
    usable: bool,
) -> dict[str, Any]:
    """Compute step latency summary with explicit sample-independence labeling."""
    per_request_lists = [row["interior_latencies"] for row in valid_rows]
    all_identical = len(per_request_lists) >= 1 and all(
        seq == per_request_lists[0] for seq in per_request_lists[1:]
    )
    pooled = [value for seq in per_request_lists for value in seq]
    aggregates = [
        row["aggregate_tokens_per_second"]
        for row in valid_rows
        if row["aggregate_tokens_per_second"] is not None
    ]

    if all_identical:
        canonical_steps = per_request_lists[0]
        sample_independence = "shared_batch_steps_repeated_per_request"
        unique_step_count: int | None = len(canonical_steps)
    else:
        canonical_steps = pooled
        sample_independence = "per_request_observations_shared_session"
        unique_step_count = None

    median_s = statistics.median(canonical_steps)
    mean_s = statistics.fmean(canonical_steps)
    k_instances = -(-GDN_HEADS_PER_LAYER * width // DEFAULT_USABLE_CORES)

    return {
        "usable": usable,
        "sample_independence": sample_independence,
        "independent_experimental_replications": False,
        "unique_step_count": unique_step_count,
        "pooled_observation_count": len(pooled),
        "per_request_trimmed_step_counts": [len(seq) for seq in per_request_lists],
        "median_step_ms": median_s * 1e3,
        "mean_step_ms": mean_s * 1e3,
        "per_user_tokens_per_second": 1.0 / median_s,
        "aggregate_tokens_per_second": (
            statistics.median(aggregates) if aggregates else None
        ),
        "k_instances_per_core": k_instances,
    }


def evaluate_attempt(
    *,
    attempt_index: int,
    attempt_dir: Path,
    width: int,
    expected_route: str,
    expected_package_identity: str | None = None,
    expected_runtime_identity: str | None = None,
    expected_run_id: str | None = None,
    expected_request_ids: dict[int, str] | None = None,
    corpus_request_hashes: dict[int, str | None] | None = None,
    trim_head: int = DEFAULT_TRIM_HEAD,
    trim_tail: int = DEFAULT_TRIM_TAIL,
    min_interior_samples: int = DEFAULT_MIN_INTERIOR_SAMPLES,
) -> dict[str, Any]:
    """Validate every requested prompt in one attempt directory."""
    expected_route = validate_gdn_route(expected_route)
    rows: list[dict[str, Any]] = []
    for prompt in range(width):
        req_id = expected_request_ids.get(prompt) if expected_request_ids else None
        req_hash = corpus_request_hashes.get(prompt) if corpus_request_hashes else None
        row = validate_response_record(
            prompt_index=prompt,
            response_path=attempt_dir / f"response-{prompt}.json",
            expected_width=width,
            expected_route=expected_route,
            expected_package_identity=expected_package_identity,
            expected_runtime_identity=expected_runtime_identity,
            expected_run_id=expected_run_id,
            expected_request_id=req_id,
            corpus_request_sha256=req_hash,
            trim_head=trim_head,
            trim_tail=trim_tail,
            min_interior_samples=min_interior_samples,
        )
        rows.append(row)

    completed_count = sum(1 for row in rows if row["completed"])
    valid_rows = [row for row in rows if row["valid"]]
    valid_count = len(valid_rows)
    group_reasons: list[str] = []

    observed_routes = sorted({
        row["observed_route"] for row in rows if row["observed_route"] is not None
    })
    if len(observed_routes) > 1:
        group_reasons.append("mixed_observed_routes")

    state_slices = [
        row["observed_state_slice"]
        for row in rows
        if row["observed_state_slice"] is not None
    ]
    if len(state_slices) != len(set(state_slices)):
        group_reasons.append("duplicate_state_slice")

    if valid_count < width:
        group_reasons.append(f"incomplete_valid_rows_{valid_count}_of_{width}")

    row_reason_summary = sorted({reason for row in rows for reason in row["reasons"]})
    all_reasons = sorted(set(group_reasons + row_reason_summary))

    usable = valid_count == width and not group_reasons and not row_reason_summary
    metrics = (
        _summarize_valid_rows(valid_rows, width=width, usable=True) if usable else None
    )
    partial_diagnostics = (
        _summarize_valid_rows(valid_rows, width=width, usable=False)
        if (not usable and valid_rows)
        else None
    )

    return {
        "attempt_index": attempt_index,
        "attempt_dir": str(attempt_dir),
        "requested": width,
        "completed": completed_count,
        "valid": valid_count,
        "usable": usable,
        "status": "valid" if usable else "failed",
        "observed_routes": observed_routes,
        "reasons": all_reasons,
        "metrics": metrics,
        "partial_diagnostics_unusable": partial_diagnostics,
        "rows": rows,
    }


def evaluate_width(
    *,
    width: int,
    width_dir: Path,
    expected_route: str,
    expected_package_identity: str | None = None,
    expected_runtime_identity: str | None = None,
    expected_run_id: str | None = None,
    expected_request_ids: dict[int, str] | None = None,
    corpus_request_hashes: dict[int, str | None] | None = None,
    trim_head: int = DEFAULT_TRIM_HEAD,
    trim_tail: int = DEFAULT_TRIM_TAIL,
    min_interior_samples: int = DEFAULT_MIN_INTERIOR_SAMPLES,
) -> dict[str, Any]:
    """Evaluate a requested width directory, retaining all attempts and failure reasons."""
    expected_route = validate_gdn_route(expected_route)
    meta_path = width_dir / "meta.json"
    meta: dict[str, Any] = {}
    if meta_path.is_file():
        try:
            loaded = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                meta = loaded
        except (OSError, json.JSONDecodeError):
            meta = {"meta_parse_error": True}

    pre_dispatch_failure = meta.get("pre_dispatch_failure")
    core_range_fatals = meta.get("core_range_fatals", 0)
    if not is_non_bool_int(core_range_fatals):
        core_range_fatals = 0

    attempts_root = width_dir / "attempts"
    attempt_dirs: list[tuple[int, Path]] = []
    if attempts_root.is_dir():
        for entry in sorted(attempts_root.iterdir()):
            if entry.is_dir() and entry.name.startswith("attempt-"):
                suffix = entry.name.removeprefix("attempt-")
                idx = int(suffix) if suffix.isdigit() else len(attempt_dirs) + 1
                attempt_dirs.append((idx, entry))
    elif width_dir.is_dir() and any(width_dir.glob("response-*.json")):
        # Direct flat directory layout (useful for unit fixtures).
        attempt_dirs.append((1, width_dir))

    attempts: list[dict[str, Any]] = []
    for idx, attempt_dir in attempt_dirs:
        attempts.append(
            evaluate_attempt(
                attempt_index=idx,
                attempt_dir=attempt_dir,
                width=width,
                expected_route=expected_route,
                expected_package_identity=expected_package_identity,
                expected_runtime_identity=expected_runtime_identity,
                expected_run_id=expected_run_id,
                expected_request_ids=expected_request_ids,
                corpus_request_hashes=corpus_request_hashes,
                trim_head=trim_head,
                trim_tail=trim_tail,
                min_interior_samples=min_interior_samples,
            )
        )

    width_reasons: list[str] = []
    if pre_dispatch_failure:
        width_reasons.append(str(pre_dispatch_failure))
    if core_range_fatals > 0:
        width_reasons.append(f"core_range_fatals_{core_range_fatals}")
    if not attempts and not pre_dispatch_failure:
        width_reasons.append("no_attempts_recorded")

    # Select the first usable attempt (never cherry-pick fastest among retries).
    selected_attempt = next((a for a in attempts if a["usable"]), None)
    last_attempt = attempts[-1] if attempts else None

    if selected_attempt is None and last_attempt is not None:
        width_reasons.extend(last_attempt["reasons"])
        if len(attempts) > 1:
            width_reasons.append("retries_exhausted")

    observed_routes = sorted({
        route for attempt in attempts for route in attempt["observed_routes"]
    })

    usable = selected_attempt is not None and not width_reasons
    if usable:
        assert selected_attempt is not None
        status = "passed"
        metrics = selected_attempt["metrics"]
        completed_count = selected_attempt["completed"]
        valid_count = selected_attempt["valid"]
        partial_diagnostics = None
    else:
        status = (
            "inconclusive"
            if any(
                reason in ("unconfirmed_effective_route", "route_mismatch")
                for reason in width_reasons
            )
            else "failed"
        )
        metrics = None
        completed_count = last_attempt["completed"] if last_attempt else 0
        valid_count = last_attempt["valid"] if last_attempt else 0
        partial_diagnostics = (
            last_attempt["partial_diagnostics_unusable"] if last_attempt else None
        )

    return {
        "width": width,
        "status": status,
        "usable": usable,
        "requested": width,
        "completed": completed_count,
        "valid": valid_count,
        "attempt_count": len(attempts),
        "failed_attempt_count": sum(1 for a in attempts if not a["usable"]),
        "selected_attempt_index": (
            selected_attempt["attempt_index"] if usable and selected_attempt else None
        ),
        "core_range_fatals": core_range_fatals,
        "pre_dispatch_failure": pre_dispatch_failure,
        "requested_gdn_route": expected_route,
        "observed_gdn_routes": observed_routes,
        "route_confirmed": usable and observed_routes == [expected_route],
        "reasons": sorted(set(width_reasons)),
        "metrics": metrics,
        "partial_diagnostics_unusable": partial_diagnostics,
        "attempts": attempts,
    }


def evaluate_probe_run(
    *,
    work_dir: Path,
    widths: list[int],
    expected_route: str,
    package_path: Path | None = None,
    runtime_identity: str | None = None,
    model_path: str | None = None,
    model_alias: str | None = None,
    corpus_path: Path | None = None,
    probe_session_id: str | None = None,
    expected_run_id: str | None = None,
    trim_head: int = DEFAULT_TRIM_HEAD,
    trim_tail: int = DEFAULT_TRIM_TAIL,
    min_interior_samples: int = DEFAULT_MIN_INTERIOR_SAMPLES,
) -> dict[str, Any]:
    """Evaluate all requested widths for a step-probe run."""
    expected_route = validate_gdn_route(expected_route)
    if not widths or any(not is_non_bool_int(w) or w <= 0 for w in widths):
        raise ValueError(f"widths must be positive integers, got {widths!r}")

    identities = resolve_package_identities(
        package_path, runtime_identity_override=runtime_identity
    )
    max_width = max(widths)
    corpus_hashes: dict[int, str | None] = {}
    if corpus_path is not None:
        for prompt in range(max_width):
            req_path = corpus_path / f"request-{prompt:02d}.json"
            corpus_hashes[prompt] = compute_file_sha256(req_path)

    width_results: list[dict[str, Any]] = []
    for width in widths:
        width_results.append(
            evaluate_width(
                width=width,
                width_dir=work_dir / f"w{width}",
                expected_route=expected_route,
                expected_package_identity=identities["execution_identity"],
                expected_runtime_identity=identities["runtime_identity"],
                expected_run_id=expected_run_id,
                corpus_request_hashes=corpus_hashes
                if corpus_path is not None
                else None,
                trim_head=trim_head,
                trim_tail=trim_tail,
                min_interior_samples=min_interior_samples,
            )
        )

    observed_routes = sorted({
        route for result in width_results for route in result["observed_gdn_routes"]
    })
    all_usable = all(result["usable"] for result in width_results)
    route_confirmed = all(result["route_confirmed"] for result in width_results)
    overall_passed = all_usable and route_confirmed

    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "screen_not_admission",
        "notice": SCREEN_NOTICE,
        "probe_session_id": probe_session_id,
        "requested_gdn_route": expected_route,
        "observed_gdn_routes": observed_routes,
        "route_confirmed": route_confirmed,
        "identities": {
            **identities,
            "model_path": model_path,
            "model_alias": model_alias,
            "corpus_path": str(corpus_path) if corpus_path is not None else None,
            "corpus_request_sha256": {
                f"request-{idx:02d}.json": digest
                for idx, digest in corpus_hashes.items()
                if digest is not None
            },
        },
        "trimming_rule": {
            "trim_head": trim_head,
            "trim_tail": trim_tail,
            "min_interior_samples": min_interior_samples,
        },
        "widths": width_results,
        "overall_status": "passed" if overall_passed else "failed",
        "exit_code": 0 if overall_passed else 1,
    }


def format_report_table(report: dict[str, Any]) -> str:
    """Format a human-readable screen summary table that never masks failures."""
    lines = [
        "",
        "QWEN38 STEP PROBE — SCREEN ONLY (NOT ADMISSION EVIDENCE)",
        f"notice: {report['notice']}",
        (
            f"gdn_route: requested={report['requested_gdn_route']} "
            f"observed={','.join(report['observed_gdn_routes']) or 'unconfirmed'} "
            f"route_confirmed={report['route_confirmed']}"
        ),
        (
            f"identities: execution={report['identities'].get('execution_identity') or 'unknown'} "
            f"runtime={report['identities'].get('runtime_identity') or 'unknown'}"
        ),
        (
            f"{'width':>5} {'status':>12} {'req/cmp/val':>11} {'steps(u/p)':>10} "
            f"{'median ms':>10} {'mean ms':>9} {'user t/s':>9} {'agg t/s':>9} "
            f"{'k':>3}  notes"
        ),
    ]
    for item in report["widths"]:
        counts = f"{item['requested']}/{item['completed']}/{item['valid']}"
        if item["usable"] and item["metrics"] is not None:
            m = item["metrics"]
            unique_str = (
                str(m["unique_step_count"])
                if m["unique_step_count"] is not None
                else "-"
            )
            steps_col = f"{unique_str}/{m['pooled_observation_count']}"
            agg_val = (
                f"{m['aggregate_tokens_per_second']:.3f}"
                if m["aggregate_tokens_per_second"] is not None
                else "-"
            )
            note = f"{m['sample_independence']} (attempts={item['attempt_count']})"
            lines.append(
                f"{item['width']:>5} {item['status']:>12} {counts:>11} {steps_col:>10} "
                f"{m['median_step_ms']:>10.3f} {m['mean_step_ms']:>9.3f} "
                f"{m['per_user_tokens_per_second']:>9.3f} {agg_val:>9} "
                f"{m['k_instances_per_core']:>3}  {note}"
            )
        else:
            reasons = ",".join(item["reasons"]) or "failed"
            partial = item.get("partial_diagnostics_unusable")
            partial_note = ""
            if partial is not None:
                partial_note = (
                    f" [UNUSABLE partial median={partial['median_step_ms']:.3f}ms]"
                )
            lines.append(
                f"{item['width']:>5} {item['status']:>12} {counts:>11} {'-':>10} "
                f"{'-':>10} {'-':>9} {'-':>9} {'-':>9} {'-':>3}  "
                f"reasons={reasons}{partial_note}"
            )
    lines.append(f"overall_status: {report['overall_status']} (screen_not_admission)")
    return "\n".join(lines)


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_route = sub.add_parser("validate-route")
    p_route.add_argument("route")

    p_free = sub.add_parser("check-port-free")
    p_free.add_argument("--host", default="127.0.0.1")
    p_free.add_argument("--port", type=int, required=True)
    p_free.add_argument("--timeout", type=float, default=1.0)

    p_ready = sub.add_parser("check-readiness")
    p_ready.add_argument("--pid", type=int, required=True)
    p_ready.add_argument("--url", required=True)
    p_ready.add_argument("--host", default="127.0.0.1")
    p_ready.add_argument("--port", type=int, required=True)
    p_ready.add_argument("--expected-model", required=True)
    p_ready.add_argument("--probe-session-id")
    p_ready.add_argument("--timeout", type=float, default=2.0)

    p_attempt = sub.add_parser("validate-attempt")
    p_attempt.add_argument("--attempt-dir", type=Path, required=True)
    p_attempt.add_argument("--attempt-index", type=int, required=True)
    p_attempt.add_argument("--width", type=int, required=True)
    p_attempt.add_argument("--gdn-route", required=True)
    p_attempt.add_argument("--package-path", type=Path)
    p_attempt.add_argument("--runtime-identity")
    p_attempt.add_argument("--expected-run-id")

    p_reduce = sub.add_parser("reduce")
    p_reduce.add_argument("--work-dir", type=Path, required=True)
    p_reduce.add_argument("--gdn-route", required=True)
    p_reduce.add_argument("--package-path", type=Path)
    p_reduce.add_argument("--runtime-identity")
    p_reduce.add_argument("--model-path")
    p_reduce.add_argument("--model-alias")
    p_reduce.add_argument("--corpus-path", type=Path)
    p_reduce.add_argument("--probe-session-id")
    p_reduce.add_argument("--expected-run-id")
    p_reduce.add_argument("--output-json", type=Path)
    p_reduce.add_argument("widths", type=int, nargs="+")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_cli()
    args = parser.parse_args(argv)

    if args.command == "validate-route":
        try:
            validate_gdn_route(args.route)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2
        return 0

    if args.command == "check-port-free":
        ok, reason = check_port_free(args.host, args.port, timeout=args.timeout)
        if not ok:
            print(reason, file=sys.stderr)
            return 3
        return 0

    if args.command == "check-readiness":
        code, reason, _payload = check_server_readiness(
            pid=args.pid,
            url=args.url,
            host=args.host,
            port=args.port,
            expected_model=args.expected_model,
            expected_session_id=args.probe_session_id,
            timeout=args.timeout,
        )
        if code != 0:
            print(reason, file=sys.stderr)
        return code

    if args.command == "validate-attempt":
        identities = resolve_package_identities(
            args.package_path, runtime_identity_override=args.runtime_identity
        )
        result = evaluate_attempt(
            attempt_index=args.attempt_index,
            attempt_dir=args.attempt_dir,
            width=args.width,
            expected_route=args.gdn_route,
            expected_package_identity=identities["execution_identity"],
            expected_runtime_identity=identities["runtime_identity"],
            expected_run_id=args.expected_run_id,
        )
        summary_path = args.attempt_dir / "attempt-result.json"
        summary_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return 0 if result["usable"] else 1

    if args.command == "reduce":
        report = evaluate_probe_run(
            work_dir=args.work_dir,
            widths=args.widths,
            expected_route=args.gdn_route,
            package_path=args.package_path,
            runtime_identity=args.runtime_identity,
            model_path=args.model_path,
            model_alias=args.model_alias,
            corpus_path=args.corpus_path,
            probe_session_id=args.probe_session_id,
            expected_run_id=args.expected_run_id,
        )
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        print(format_report_table(report))
        return int(report["exit_code"])

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
