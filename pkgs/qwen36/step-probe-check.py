"""Device-free unit and shell-boundary tests for qwen38-step-probe."""

from __future__ import annotations

import http.server
import importlib.util
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import textwrap
import threading
from pathlib import Path
from typing import Any


def _load_core():
    if len(sys.argv) > 2:
        core_path = Path(sys.argv[2]).resolve()
    else:
        core_path = Path(__file__).with_name("step_probe_core.py")
    spec = importlib.util.spec_from_file_location("step_probe_core", core_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, core_path


def _find_script() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).resolve()
    candidate = Path(__file__).resolve().parents[2] / "scripts" / "qwen38-step-probe.sh"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError("pass path to scripts/qwen38-step-probe.sh as argv[1]")


def _valid_payload(
    *,
    width: int = 4,
    route: str = "step",
    latencies: list[Any] | None = None,
    agg: Any = 80.0,
    state_slice: int | None = None,
    package_identity: str | None = None,
    runtime_identity: str | None = None,
    request_id: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "token_latency_seconds": (
            [0.05, 0.05, 0.05, 0.05] if latencies is None else latencies
        ),
        "aggregate_decode_tokens_per_second": agg,
        "serving_route": route,
        "batch_width": width,
    }
    if state_slice is not None:
        evidence["state_slice"] = state_slice
    if package_identity is not None:
        evidence["package_identity"] = package_identity
    if runtime_identity is not None:
        evidence["runtime_identity"] = runtime_identity
    if run_id is not None:
        evidence["probe_run_id"] = run_id
    payload: dict[str, Any] = {
        "choices": [{"text": "ok", "finish_reason": "length"}],
        "x_onix_decode_evidence": evidence,
    }
    if request_id is not None:
        payload["id"] = request_id
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_reducer_controls(core) -> None:
    with tempfile.TemporaryDirectory(prefix="step-probe-unit-") as tmp:
        root = Path(tmp)

        # 1. Positive control: 4/4 valid shared batch steps.
        w_ok = root / "ok" / "w4" / "attempts" / "attempt-01"
        for i in range(4):
            _write_json(
                w_ok / f"response-{i}.json",
                _valid_payload(
                    width=4,
                    route="step",
                    state_slice=i,
                    package_identity="/nix/store/fake-exec-id",
                    runtime_identity="/nix/store/fake-tt-metal",
                ),
            )
        report = core.evaluate_probe_run(
            work_dir=root / "ok",
            widths=[4],
            expected_route="step",
            runtime_identity="/nix/store/fake-tt-metal",
        )
        assert report["exit_code"] == 0, report
        assert report["overall_status"] == "passed"
        assert report["classification"] == "screen_not_admission"
        assert report["route_confirmed"] is True
        w4 = report["widths"][0]
        assert w4["usable"] is True
        assert (w4["requested"], w4["completed"], w4["valid"]) == (4, 4, 4)
        assert w4["metrics"]["unique_step_count"] == 2
        assert w4["metrics"]["pooled_observation_count"] == 8
        assert (
            w4["metrics"]["sample_independence"]
            == "shared_batch_steps_repeated_per_request"
        )
        assert w4["metrics"]["independent_experimental_replications"] is False
        assert abs(w4["metrics"]["median_step_ms"] - 50.0) < 1e-6
        table = core.format_report_table(report)
        assert "SCREEN ONLY (NOT ADMISSION EVIDENCE)" in table

        # 2. Distinct per-request observations label.
        w_diff = root / "diff" / "w4" / "attempts" / "attempt-01"
        for i in range(4):
            _write_json(
                w_diff / f"response-{i}.json",
                _valid_payload(
                    width=4,
                    route="step",
                    latencies=[0.05, 0.05 + i * 0.001, 0.05 + i * 0.001, 0.06],
                    state_slice=i,
                ),
            )
        rep_diff = core.evaluate_probe_run(
            work_dir=root / "diff", widths=[4], expected_route="step"
        )
        assert rep_diff["exit_code"] == 0
        assert (
            rep_diff["widths"][0]["metrics"]["sample_independence"]
            == "per_request_observations_shared_session"
        )
        assert rep_diff["widths"][0]["metrics"]["unique_step_count"] is None

        # 3. 1/4 present: must fail, never emit usable row, retain partial diagnostics.
        w_one = root / "one" / "w4" / "attempts" / "attempt-01"
        _write_json(w_one / "response-0.json", _valid_payload(width=4, route="step"))
        rep_one = core.evaluate_probe_run(
            work_dir=root / "one", widths=[4], expected_route="step"
        )
        assert rep_one["exit_code"] == 1
        assert rep_one["overall_status"] == "failed"
        item = rep_one["widths"][0]
        assert item["usable"] is False
        assert item["metrics"] is None
        assert (item["requested"], item["completed"], item["valid"]) == (4, 1, 1)
        assert "missing_response_file" in item["reasons"]
        assert "incomplete_valid_rows_1_of_4" in item["reasons"]
        assert item["partial_diagnostics_unusable"] is not None
        assert item["partial_diagnostics_unusable"]["usable"] is False
        assert "UNUSABLE partial" in core.format_report_table(rep_one)

        # 4. One malformed JSON out of 4.
        w_mal = root / "mal" / "w4" / "attempts" / "attempt-01"
        for i in range(3):
            _write_json(w_mal / f"response-{i}.json", _valid_payload(width=4))
        (w_mal / "response-3.json").write_text("{not-json", encoding="utf-8")
        rep_mal = core.evaluate_probe_run(
            work_dir=root / "mal", widths=[4], expected_route="step"
        )
        assert rep_mal["exit_code"] == 1
        assert (
            rep_mal["widths"][0]["completed"],
            rep_mal["widths"][0]["valid"],
        ) == (3, 3)
        assert "malformed_json" in rep_mal["widths"][0]["reasons"]

        # 5. All error objects.
        w_err = root / "err" / "w4" / "attempts" / "attempt-01"
        for i in range(4):
            _write_json(w_err / f"response-{i}.json", {"error": {"message": "busy"}})
        rep_err = core.evaluate_probe_run(
            work_dir=root / "err", widths=[4], expected_route="step"
        )
        assert rep_err["exit_code"] == 1
        assert (
            rep_err["widths"][0]["completed"],
            rep_err["widths"][0]["valid"],
        ) == (0, 0)
        assert "api_error_payload" in rep_err["widths"][0]["reasons"]

        # 6. Missing x_onix_decode_evidence.
        w_noev = root / "noev" / "w4" / "attempts" / "attempt-01"
        for i in range(4):
            _write_json(w_noev / f"response-{i}.json", {"choices": [{"text": "hi"}]})
        rep_noev = core.evaluate_probe_run(
            work_dir=root / "noev", widths=[4], expected_route="step"
        )
        assert rep_noev["exit_code"] == 1
        assert (
            rep_noev["widths"][0]["completed"],
            rep_noev["widths"][0]["valid"],
        ) == (4, 0)
        assert "missing_or_malformed_evidence" in rep_noev["widths"][0]["reasons"]

        # 7. Empty and truncated latency arrays.
        for label, bad_lat, expected_reason in (
            ("empty", [], "empty_latency_array"),
            ("trunc1", [0.05], "insufficient_samples_after_trim"),
            ("trunc2", [0.05, 0.05], "insufficient_samples_after_trim"),
            ("trunc3", [0.05, 0.05, 0.05], "insufficient_samples_after_trim"),
            ("nonlist", {"a": 0.05}, "invalid_latency_container"),
        ):
            wd = root / label / "w2" / "attempts" / "attempt-01"
            for i in range(2):
                _write_json(
                    wd / f"response-{i}.json",
                    _valid_payload(width=2, latencies=bad_lat),
                )
            rep = core.evaluate_probe_run(
                work_dir=root / label, widths=[2], expected_route="step"
            )
            assert rep["exit_code"] == 1, label
            assert expected_reason in rep["widths"][0]["reasons"], (
                label,
                rep["widths"][0]["reasons"],
            )

        # 8. Nonfinite, negative, zero, and boolean numeric masquerading.
        bad_numeric_cases = [
            (
                "neg",
                [-0.01, 0.05, 0.05, 0.05],
                80.0,
                "nonfinite_or_nonpositive_latency_value",
            ),
            (
                "zero",
                [0.05, 0.0, 0.05, 0.05],
                80.0,
                "nonfinite_or_nonpositive_latency_value",
            ),
            ("bool_lat", [0.05, True, 0.05, 0.05], 80.0, "boolean_latency_value"),
            (
                "bool_agg",
                [0.05, 0.05, 0.05, 0.05],
                True,
                "boolean_aggregate_tokens_per_second",
            ),
            (
                "neg_agg",
                [0.05, 0.05, 0.05, 0.05],
                -10.0,
                "invalid_aggregate_tokens_per_second",
            ),
        ]
        for label, lat, agg, expected_reason in bad_numeric_cases:
            wd = root / label / "w2" / "attempts" / "attempt-01"
            for i in range(2):
                _write_json(
                    wd / f"response-{i}.json",
                    _valid_payload(width=2, latencies=lat, agg=agg),
                )
            rep = core.evaluate_probe_run(
                work_dir=root / label, widths=[2], expected_route="step"
            )
            assert rep["exit_code"] == 1, label
            assert expected_reason in rep["widths"][0]["reasons"], (
                label,
                rep["widths"][0]["reasons"],
            )

        # NaN / Infinity via Python json.loads-accepted non-standard literals.
        for label, literal in (
            ("nan", "NaN"),
            ("inf", "Infinity"),
            ("ninf", "-Infinity"),
        ):
            wd = root / label / "w1" / "attempts" / "attempt-01"
            wd.mkdir(parents=True, exist_ok=True)
            (wd / "response-0.json").write_text(
                '{"x_onix_decode_evidence":{"token_latency_seconds":[0.05,'
                + literal
                + ',0.05,0.05],"aggregate_decode_tokens_per_second":80.0,'
                '"serving_route":"step","batch_width":1}}',
                encoding="utf-8",
            )
            rep = core.evaluate_probe_run(
                work_dir=root / label, widths=[1], expected_route="step"
            )
            assert rep["exit_code"] == 1, label
            assert (
                "nonfinite_or_nonpositive_latency_value" in rep["widths"][0]["reasons"]
            )

        # 9. Wrong request/run/package/width identity.
        w_id = root / "idcheck" / "w2" / "attempts" / "attempt-01"
        _write_json(
            w_id / "response-0.json",
            _valid_payload(
                width=2,
                request_id="req-wrong",
                run_id="run-wrong",
                package_identity="/nix/store/other-exec",
            ),
        )
        _write_json(
            w_id / "response-1.json",
            _valid_payload(
                width=3,
                request_id="req-1",
                run_id="run-expected",
                package_identity="/nix/store/expected-exec",
            ),
        )
        attempt = core.evaluate_attempt(
            attempt_index=1,
            attempt_dir=w_id,
            width=2,
            expected_route="step",
            expected_package_identity="/nix/store/expected-exec",
            expected_run_id="run-expected",
            expected_request_ids={0: "req-0", 1: "req-1"},
        )
        assert attempt["usable"] is False
        assert "request_identity_mismatch" in attempt["reasons"]
        assert "run_identity_mismatch" in attempt["reasons"]
        assert "package_identity_mismatch" in attempt["reasons"]
        assert "actual_width_mismatch" in attempt["reasons"]

        # 10. Unconfirmed route, mismatched route, and mixed routes.
        w_unconf = root / "unconf" / "w2" / "attempts" / "attempt-01"
        for i in range(2):
            p = _valid_payload(width=2)
            del p["x_onix_decode_evidence"]["serving_route"]
            _write_json(w_unconf / f"response-{i}.json", p)
        rep_unconf = core.evaluate_probe_run(
            work_dir=root / "unconf", widths=[2], expected_route="step"
        )
        assert rep_unconf["exit_code"] == 1
        assert rep_unconf["route_confirmed"] is False
        assert rep_unconf["widths"][0]["status"] == "inconclusive"
        assert "unconfirmed_effective_route" in rep_unconf["widths"][0]["reasons"]

        w_mix = root / "mix" / "w2" / "attempts" / "attempt-01"
        _write_json(w_mix / "response-0.json", _valid_payload(width=2, route="step"))
        _write_json(w_mix / "response-1.json", _valid_payload(width=2, route="full"))
        rep_mix = core.evaluate_probe_run(
            work_dir=root / "mix", widths=[2], expected_route="step"
        )
        assert rep_mix["exit_code"] == 1
        assert "mixed_observed_routes" in rep_mix["widths"][0]["reasons"]
        assert "route_mismatch" in rep_mix["widths"][0]["reasons"]

        # 11. Exhausted retries preserve all failed attempts.
        for idx in (1, 2, 3):
            att = root / "exhausted" / "w2" / "attempts" / f"attempt-{idx:02d}"
            for i in range(2):
                _write_json(att / f"response-{i}.json", {"error": {"message": "fail"}})
        rep_ex = core.evaluate_probe_run(
            work_dir=root / "exhausted", widths=[2], expected_route="step"
        )
        assert rep_ex["exit_code"] == 1
        assert rep_ex["widths"][0]["attempt_count"] == 3
        assert rep_ex["widths"][0]["failed_attempt_count"] == 3
        assert "retries_exhausted" in rep_ex["widths"][0]["reasons"]


def _make_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _create_fake_fixture_env(base: Path) -> dict[str, Path]:
    pkg = base / "fake-pkg"
    exec_id = base / "fake-exec-id"
    (exec_id / "share" / "qwen36").mkdir(parents=True)
    (exec_id / "share" / "qwen36" / "execution-inputs").write_text(
        "schema=onix.qwen36.execution-identity.v1\n"
        "input=/nix/store/00000000000000000000000000000000-tt-metal-0.78.0\n",
        encoding="utf-8",
    )
    (pkg / "share" / "qwen36").mkdir(parents=True)
    (pkg / "share" / "qwen36" / "execution-identity").symlink_to(exec_id)
    (pkg / "bin").mkdir(parents=True)

    corpus = base / "corpus"
    corpus.mkdir(parents=True)
    for i in range(4):
        _write_json(
            corpus / f"request-{i:02d}.json",
            {"model": "Qwen3.8-27B", "prompt": f"hello {i}", "max_tokens": 4},
        )
    model = base / "model"
    model.mkdir(parents=True)
    site_pkgs = base / "tt-metal-site-packages"
    site_pkgs.mkdir(parents=True)
    return {
        "pkg": pkg,
        "exec_id": exec_id,
        "corpus": corpus,
        "model": model,
        "site_pkgs": site_pkgs,
    }


def _stub_http_server_script(
    *,
    mode: str,
    route: str = "step",
    exec_id: str = "",
    runtime_id: str = "/nix/store/00000000000000000000000000000000-tt-metal-0.78.0",
    model_alias: str = "Qwen3.8-27B",
    counter_file: Path | None = None,
) -> str:
    return textwrap.dedent(
        f"""\
        #!{sys.executable}
        import argparse
        import http.server
        import json
        import os
        import signal
        import sys
        import threading
        from pathlib import Path

        MODE = {mode!r}
        ROUTE = {route!r}
        EXEC_ID = {exec_id!r}
        RUNTIME_ID = {runtime_id!r}
        MODEL_ALIAS = {model_alias!r}
        COUNTER_FILE = {str(counter_file) if counter_file else ""!r}

        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--host", default="127.0.0.1")
            parser.add_argument("--port", type=int, required=True)
            parser.add_argument("--model-path")
            parser.add_argument("--model-alias")
            parser.add_argument("--max-sequence-length")
            parser.add_argument("--max-generation-tokens")
            args = parser.parse_args()

            if MODE == "die_early":
                sys.stderr.write("fatal startup crash\\n")
                sys.exit(17)
            if MODE == "hang_no_bind":
                signal.pause()
                return

            lock = threading.Lock()
            req_counter = [0]

            class Handler(http.server.BaseHTTPRequestHandler):
                def log_message(self, fmt, *a):
                    pass

                def do_GET(self):
                    if self.path == "/healthz":
                        body = json.dumps({{
                            "status": "ok",
                            "ready": True,
                            "model": MODEL_ALIAS,
                            "probe_session_id": os.environ.get("QWEN38_PROBE_SESSION_ID"),
                        }}).encode("utf-8")
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                        return
                    self.send_response(404)
                    self.end_headers()

                def do_POST(self):
                    length = int(self.headers.get("Content-Length", "0"))
                    _ = self.rfile.read(length)
                    with lock:
                        req_counter[0] += 1
                        count = req_counter[0]
                        if COUNTER_FILE:
                            Path(COUNTER_FILE).write_text(str(count), encoding="utf-8")
                    if MODE == "always_error":
                        status = 503
                        payload = {{"error": {{"message": "overloaded"}}}}
                    elif MODE == "fail_first_group" and count <= 2:
                        status = 503
                        payload = {{"error": {{"message": "warmup group rejected"}}}}
                    else:
                        status = 200
                        payload = {{
                            "id": self.headers.get("X-Onix-Probe-Request", f"req-{{count}}"),
                            "choices": [{{"text": "ok", "finish_reason": "length"}}],
                            "x_onix_decode_evidence": {{
                                "token_latency_seconds": [0.06, 0.05, 0.05, 0.07],
                                "aggregate_decode_tokens_per_second": 40.0,
                                "serving_route": ROUTE,
                                "batch_width": 2,
                                "package_identity": EXEC_ID,
                                "runtime_identity": RUNTIME_ID,
                            }},
                        }}
                    body = json.dumps(payload).encode("utf-8")
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

            server = http.server.ThreadingHTTPServer((args.host, args.port), Handler)
            def _stop(_sig, _frame):
                threading.Thread(target=server.shutdown, daemon=True).start()
            signal.signal(signal.SIGTERM, _stop)
            server.serve_forever()

        if __name__ == "__main__":
            main()
        """
    )


def test_shell_boundary(script_path: Path, core_path: Path) -> None:
    if shutil.which("curl") is None:
        raise RuntimeError("curl is required for shell boundary tests")

    with tempfile.TemporaryDirectory(prefix="step-probe-shell-") as tmp:
        base = Path(tmp)
        fx = _create_fake_fixture_env(base)
        bash_bin = shutil.which("bash") or "/bin/sh"
        owner_log = base / "owner.log"
        owner_ok = base / "owner-ok.sh"
        _make_executable(
            owner_ok,
            f'#!{bash_bin}\necho "$1" >> {owner_log}\nexit 0\n',
        )
        owner_fail = base / "owner-fail.sh"
        _make_executable(
            owner_fail,
            f'#!{bash_bin}\necho "$1" >> {owner_log}\nexit 7\n',
        )

        def base_env(work_dir: Path, port: int, server_bin: Path, owner_bin: Path):
            env = os.environ.copy()
            env.pop("QWEN35_GDN_DECODE_FUSION", None)
            env.update({
                "QWEN38_PROBE_MODEL": str(fx["model"]),
                "QWEN38_PROBE_CORPUS": str(fx["corpus"]),
                "QWEN38_PROBE_TT_METAL_SITE_PACKAGES": str(fx["site_pkgs"]),
                "QWEN38_PROBE_OWNER_CONTROL": str(owner_bin),
                "QWEN38_PROBE_SERVER_BIN": str(server_bin),
                "QWEN38_PROBE_PYTHON": sys.executable,
                "QWEN38_PROBE_CORE": str(core_path),
                "QWEN38_PROBE_PORT": str(port),
                "QWEN38_PROBE_WORK_DIR": str(work_dir),
                "QWEN38_PROBE_STARTUP_ATTEMPTS": "4",
                "QWEN38_PROBE_STARTUP_INTERVAL_SECONDS": "0.1",
                "QWEN38_PROBE_GROUP_RETRY_MAX": "2",
                "QWEN38_PROBE_GROUP_RETRY_INTERVAL_SECONDS": "0.1",
                "QWEN38_PROBE_STOP_DELAY_SECONDS": "0",
            })
            return env

        # A. Missing explicit GDN route must fail closed before isolation.
        srv_ok = base / "srv-ok.py"
        _make_executable(
            srv_ok,
            _stub_http_server_script(mode="ok", exec_id=str(fx["exec_id"].resolve())),
        )
        owner_log.write_text("", encoding="utf-8")
        res = subprocess.run(
            ["bash", str(script_path), str(fx["pkg"]), "2"],
            env=base_env(base / "w-noroute", _free_tcp_port(), srv_ok, owner_ok),
            capture_output=True,
            text=True,
            check=False,
        )
        assert res.returncode == 2, (res.returncode, res.stderr)
        assert owner_log.read_text(encoding="utf-8").strip() == ""

        # B. Ownership isolation failure must abort before dispatch and not call restore.
        owner_log.write_text("", encoding="utf-8")
        res = subprocess.run(
            ["bash", str(script_path), "--gdn-route", "step", str(fx["pkg"]), "2"],
            env=base_env(base / "w-isofail", _free_tcp_port(), srv_ok, owner_fail),
            capture_output=True,
            text=True,
            check=False,
        )
        assert res.returncode == 2, (res.returncode, res.stderr)
        assert owner_log.read_text(encoding="utf-8").splitlines() == ["isolate"]

        # C. Early child death must fail BEFORE dispatch and propagate non-zero through trap.
        srv_die = base / "srv-die.py"
        _make_executable(srv_die, _stub_http_server_script(mode="die_early"))
        owner_log.write_text("", encoding="utf-8")
        w_die = base / "w-die"
        res = subprocess.run(
            ["bash", str(script_path), "--gdn-route", "step", str(fx["pkg"]), "2"],
            env=base_env(w_die, _free_tcp_port(), srv_die, owner_ok),
            capture_output=True,
            text=True,
            check=False,
        )
        assert res.returncode == 1, (res.returncode, res.stdout, res.stderr)
        assert owner_log.read_text(encoding="utf-8").splitlines() == [
            "isolate",
            "restore",
        ]
        rep_die = json.loads((w_die / "report.json").read_text(encoding="utf-8"))
        assert rep_die["overall_status"] == "failed"
        assert (
            rep_die["widths"][0]["pre_dispatch_failure"]
            == "server_exited_before_readiness"
        )
        assert rep_die["widths"][0]["attempt_count"] == 0

        # D. Readiness timeout must fail BEFORE dispatch.
        srv_hang = base / "srv-hang.py"
        _make_executable(srv_hang, _stub_http_server_script(mode="hang_no_bind"))
        w_hang = base / "w-hang"
        res = subprocess.run(
            ["bash", str(script_path), "--gdn-route", "step", str(fx["pkg"]), "2"],
            env=base_env(w_hang, _free_tcp_port(), srv_hang, owner_ok),
            capture_output=True,
            text=True,
            check=False,
        )
        assert res.returncode == 1, (res.returncode, res.stdout, res.stderr)
        rep_hang = json.loads((w_hang / "report.json").read_text(encoding="utf-8"))
        assert rep_hang["widths"][0]["pre_dispatch_failure"] == "readiness_timeout"
        assert rep_hang["widths"][0]["attempt_count"] == 0

        # E. Unrelated server already listening on port must be rejected before launch/dispatch.
        unrelated_port = _free_tcp_port()
        unrelated_hits = [0]

        class UnrelatedHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, fmt, *a):
                pass

            def do_GET(self):
                body = json.dumps({
                    "status": "ok",
                    "ready": True,
                    "model": "Qwen3.8-27B",
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                unrelated_hits[0] += 1
                self.send_response(200)
                self.end_headers()

        unrelated_srv = http.server.ThreadingHTTPServer(
            ("127.0.0.1", unrelated_port), UnrelatedHandler
        )
        unrelated_thread = threading.Thread(
            target=unrelated_srv.serve_forever, daemon=True
        )
        unrelated_thread.start()
        try:
            w_unrel = base / "w-unrel"
            res = subprocess.run(
                ["bash", str(script_path), "--gdn-route", "step", str(fx["pkg"]), "2"],
                env=base_env(w_unrel, unrelated_port, srv_hang, owner_ok),
                capture_output=True,
                text=True,
                check=False,
            )
            assert res.returncode == 1, (res.returncode, res.stdout, res.stderr)
            assert unrelated_hits[0] == 0
            rep_unrel = json.loads(
                (w_unrel / "report.json").read_text(encoding="utf-8")
            )
            assert (
                rep_unrel["widths"][0]["pre_dispatch_failure"]
                == "unrelated_server_already_listening"
            )
        finally:
            unrelated_srv.shutdown()
            unrelated_srv.server_close()

        # F. Wrong model on spawned server healthz must fail as unrelated_server_detected.
        srv_wrong_model = base / "srv-wrong-model.py"
        _make_executable(
            srv_wrong_model,
            _stub_http_server_script(mode="ok", model_alias="OtherModel-7B"),
        )
        w_wm = base / "w-wm"
        res = subprocess.run(
            ["bash", str(script_path), "--gdn-route", "step", str(fx["pkg"]), "2"],
            env=base_env(w_wm, _free_tcp_port(), srv_wrong_model, owner_ok),
            capture_output=True,
            text=True,
            check=False,
        )
        assert res.returncode == 1
        rep_wm = json.loads((w_wm / "report.json").read_text(encoding="utf-8"))
        assert (
            rep_wm["widths"][0]["pre_dispatch_failure"] == "unrelated_server_detected"
        )

        # G. Exhausted retries: retains all failed attempts and exits non-zero.
        srv_err = base / "srv-err.py"
        _make_executable(srv_err, _stub_http_server_script(mode="always_error"))
        w_ex = base / "w-ex"
        res = subprocess.run(
            ["bash", str(script_path), "--gdn-route", "step", str(fx["pkg"]), "2"],
            env=base_env(w_ex, _free_tcp_port(), srv_err, owner_ok),
            capture_output=True,
            text=True,
            check=False,
        )
        assert res.returncode == 1, (res.returncode, res.stdout, res.stderr)
        rep_ex = json.loads((w_ex / "report.json").read_text(encoding="utf-8"))
        assert rep_ex["widths"][0]["attempt_count"] == 2
        assert rep_ex["widths"][0]["failed_attempt_count"] == 2
        assert (w_ex / "w2" / "attempts" / "attempt-01" / "response-0.json").is_file()
        assert (w_ex / "w2" / "attempts" / "attempt-02" / "response-0.json").is_file()

        # H. Retry recovery retains failed attempt-01 and succeeds on attempt-02.
        srv_retry = base / "srv-retry.py"
        counter = base / "retry-counter.txt"
        _make_executable(
            srv_retry,
            _stub_http_server_script(
                mode="fail_first_group",
                route="step",
                exec_id=str(fx["exec_id"].resolve()),
                counter_file=counter,
            ),
        )
        w_retry = base / "w-retry"
        res = subprocess.run(
            ["bash", str(script_path), "--gdn-route", "step", str(fx["pkg"]), "2"],
            env=base_env(w_retry, _free_tcp_port(), srv_retry, owner_ok),
            capture_output=True,
            text=True,
            check=False,
        )
        assert res.returncode == 0, (res.returncode, res.stdout, res.stderr)
        rep_retry = json.loads((w_retry / "report.json").read_text(encoding="utf-8"))
        assert rep_retry["overall_status"] == "passed"
        assert rep_retry["route_confirmed"] is True
        assert rep_retry["widths"][0]["attempt_count"] == 2
        assert rep_retry["widths"][0]["failed_attempt_count"] == 1
        assert rep_retry["widths"][0]["selected_attempt_index"] == 2
        assert rep_retry["identities"]["execution_identity"] == str(
            fx["exec_id"].resolve()
        )
        assert "SCREEN ONLY (NOT ADMISSION EVIDENCE)" in res.stdout

        # I. Route mismatch on live stub server (requested step, server reports full).
        srv_full = base / "srv-full.py"
        _make_executable(
            srv_full,
            _stub_http_server_script(
                mode="ok", route="full", exec_id=str(fx["exec_id"].resolve())
            ),
        )
        w_rm = base / "w-rm"
        res = subprocess.run(
            ["bash", str(script_path), "--gdn-route", "step", str(fx["pkg"]), "2"],
            env=base_env(w_rm, _free_tcp_port(), srv_full, owner_ok),
            capture_output=True,
            text=True,
            check=False,
        )
        assert res.returncode == 1
        rep_rm = json.loads((w_rm / "report.json").read_text(encoding="utf-8"))
        assert rep_rm["route_confirmed"] is False
        assert rep_rm["observed_gdn_routes"] == ["full"]
        assert "route_mismatch" in rep_rm["widths"][0]["reasons"]


def main() -> int:
    core, core_path = _load_core()
    script_path = _find_script()
    test_reducer_controls(core)
    test_shell_boundary(script_path, core_path)
    print(
        json.dumps(
            {
                "step_probe_check": "passed",
                "reducer_controls": 11,
                "shell_boundary_controls": 9,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
