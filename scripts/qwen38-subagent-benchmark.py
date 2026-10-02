#!/usr/bin/env python3
"""Run bounded serial/parallel OMP task-subagent roots against local Qwen3.8.

Example: python3 scripts/qwen38-subagent-benchmark.py --output /tmp/qwen38-task-receipt.json
The JSONL siblings of --output retain the complete OMP event streams. Physical route
receipts are separate from task quality and timing comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

MODEL = "mesh/Qwen3.8-27B:low"
GATEWAY_MODEL = "Qwen3.8-27B"
MAX_EVENT_BYTES = 64 * 1024 * 1024
MAX_STDERR_BYTES = 1024 * 1024
NAME_RE = re.compile(r"[A-Z][a-zA-Z0-9]*\Z")
ANCHOR_RE = re.compile(r"\bAnchor:\s*([A-Za-z][A-Za-z0-9_-]*)\b", re.I)


class BenchmarkError(Exception):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BenchmarkError(message)


def load_workload(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    require(isinstance(value, dict) and value.get("schema_version") == 1,
            "workload must have schema_version 1")
    require(isinstance(value.get("workload_id"), str) and bool(value["workload_id"].strip()),
            "workload_id must be a nonempty string")
    require(isinstance(value.get("shared_context"), str) and bool(value["shared_context"].strip()),
            "shared_context must be a nonempty string")
    require(isinstance(value.get("synthesis"), str) and bool(value["synthesis"].strip()),
            "synthesis must be a nonempty string")
    workers = value.get("workers")
    require(isinstance(workers, list) and len(workers) == 3,
            "workload must declare exactly three workers")
    names = set()
    anchors = set()
    for worker in workers:
        require(isinstance(worker, dict), "each worker must be an object")
        name, task = worker.get("name"), worker.get("task")
        require(isinstance(name, str) and NAME_RE.fullmatch(name) is not None,
                "worker name must be CamelCase")
        require(name not in names, "duplicate worker name")
        names.add(name)
        require(isinstance(task, str) and bool(task.strip()), f"{name}: missing task")
        match = ANCHOR_RE.search(task)
        require(match is not None, f"{name}: task must contain an explicit Anchor: token")
        anchor = match.group(1).casefold()
        require(anchor not in anchors, f"{name}: duplicate task anchor")
        anchors.add(anchor)
        terms = worker.get("required_terms")
        require(isinstance(terms, list) and bool(terms) and all(
            isinstance(term, str) and bool(term) and term == term.lower()
            for term in terms), f"{name}: required_terms must be lowercase strings")
    return value, hashlib.sha256(raw).hexdigest()


def http_json(base_url: str, method: str, path: str, payload: dict[str, Any] | None = None
              ) -> tuple[int, dict[str, Any]]:
    parts = urlsplit(base_url)
    require(parts.scheme == "http" and bool(parts.hostname) and parts.port is not None
            and parts.path in ("", "/") and not parts.query and not parts.fragment,
            f"invalid local HTTP URL: {base_url!r}")
    require(parts.hostname in ("127.0.0.1", "localhost", "::1"),
            f"only loopback HTTP endpoints are allowed: {base_url!r}")
    connection = http.client.HTTPConnection(parts.hostname, parts.port, timeout=90)
    try:
        body = json.dumps(payload).encode() if payload is not None else None
        connection.request(method, path, body, {"Content-Type": "application/json"} if body else {})
        response = connection.getresponse()
        require(int(response.headers.get("Content-Length", "0")) <= 1024 * 1024,
                f"oversized response from {base_url}{path}")
        content = response.read(1024 * 1024 + 1)
        require(len(content) <= 1024 * 1024, f"oversized response from {base_url}{path}")
        if payload and payload.get("stream") and response.status == 200:
            evidence = None
            for line in content.splitlines():
                if line.startswith(b"data: ") and line[6:] != b"[DONE]":
                    chunk = json.loads(line[6:])
                    if "error" in chunk:
                        raise BenchmarkError(f"gateway stream error: {chunk['error']}")
                    evidence = chunk.get("x_onix_decode_evidence", evidence)
            return response.status, {"x_onix_decode_evidence": evidence}
        parsed = json.loads(content)
        require(isinstance(parsed, dict), f"non-object JSON response from {base_url}{path}")
        return response.status, parsed
    finally:
        connection.close()


def probe(gateway_url: str, service_url: str) -> dict[str, Any]:
    status, health = http_json(service_url, "GET", "/healthz")
    require(status == 200 and health.get("status") == "ok"
            and health.get("model") == GATEWAY_MODEL,
            f"production service is not ready for {GATEWAY_MODEL}: {status} {health}")
    status, response = http_json(gateway_url, "POST", "/v1/chat/completions", {
        "model": GATEWAY_MODEL,
        "messages": [{"role": "user", "content": "Reply with the word ready."}],
        "max_tokens": 24, "temperature": 0, "enable_thinking": False,
        "stream": True, "stream_options": {"include_usage": True},
    })
    require(status == 200, f"gateway probe failed with HTTP {status}: {response}")
    evidence = response.get("x_onix_decode_evidence")
    require(isinstance(evidence, dict), "gateway provided no decode evidence")
    require(evidence.get("physical_pcie_device_ids") == [0, 1]
            and evidence.get("simulated") is False,
            "gateway did not attest physical PCIe devices [0, 1] with simulated=false")
    require(evidence.get("decode_path") == "continuous-slot",
            "gateway is not running continuous-slot decode")
    require(bool(evidence.get("package_identity")) and bool(evidence.get("runtime_identity")),
            "gateway physical execution identity is missing")
    return {"service_health": health, "gateway_decode_evidence": evidence,
            "gateway_url": gateway_url, "service_url": service_url}


def root_prompt(workload: dict[str, Any], mode: str) -> str:
    workers = workload["workers"]
    assignments = [{"name": w["name"], "task": w["task"]} for w in workers]
    synthesis = workload["synthesis"]
    call_rule = ("Make exactly THREE separate task tool calls, IN ORDER, with one "
                 "worker per call. Await each result before the next call."
                 if mode == "serial" else
                 "Make exactly ONE task tool call with all THREE workers in one tasks array.")
    return (f"Run a {mode} local task-subagent code-review benchmark. {call_rule}\n"
            "Use the task tool, not a description of hypothetical workers. The shared "
            "context and exact worker names must be passed to the tool. Keep every "
            "worker's explicit Anchor token in its assigned task even when paraphrasing. "
            "Do not perform any other tool calls. Each child must analyze only the "
            "supplied text: no tools, files, shell, tests, builds, lint, formatting or "
            "network access. The parent must not use other tools.\n"
            f"shared_context: {workload['shared_context']}\n"
            f"workers: {json.dumps(assignments, ensure_ascii=False)}\n"
            "After receiving the actual child outputs, write a substantive final "
            "synthesis mentioning all three worker names and following this instruction: "
            f"{synthesis}. Do not merely claim workers ran.")


def stop_group(process: subprocess.Popen[bytes]) -> None:
    """Terminate all descendants, even if their OMP parent has already exited."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # Parent exit does not imply a task child has exited or closed inherited pipes.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_omp(omp: str, config: Path, prompt: str, raw_path: Path,
            timeout: int) -> dict[str, Any]:
    omp = str(Path(omp).resolve()) if os.sep in omp else omp
    command = [omp, "--mode=json", "--no-session", "--no-rules", "--no-skills",
               "--no-extensions", "--no-title", "--tools=task", f"--config={config}",
               f"--model={MODEL}", "--thinking=low", "-p", prompt]
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="qwen38-task-") as cwd, raw_path.open("wb") as raw:
        process = subprocess.Popen(command, cwd=cwd, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   start_new_session=True)
        overflow: list[str] = []
        stderr = bytearray()

        def drain_stdout() -> None:
            count = 0
            assert process.stdout is not None
            while chunk := process.stdout.read(65536):
                count += len(chunk)
                if count > MAX_EVENT_BYTES:
                    overflow.append("OMP JSONL exceeds 64 MiB")
                    stop_group(process)
                    break
                raw.write(chunk)

        def drain_stderr() -> None:
            assert process.stderr is not None
            while chunk := process.stderr.read(65536):
                if len(stderr) < MAX_STDERR_BYTES:
                    stderr.extend(chunk[:MAX_STDERR_BYTES - len(stderr)])

        streams = [threading.Thread(target=drain_stdout), threading.Thread(target=drain_stderr)]
        for thread in streams:
            thread.start()
        timed_out = False
        try:
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                stop_group(process)
        except BaseException:
            stop_group(process)
            raise
        for thread in streams:
            thread.join(timeout=5)
        if any(thread.is_alive() for thread in streams):
            stop_group(process)
            overflow.append("OMP event streams did not close after process exit")
        raw.flush()
        return {"wall_seconds": time.monotonic() - started, "exit_code": process.returncode,
                "timed_out": timed_out, "stream_error": overflow,
                "stderr": stderr.decode(errors="replace"), "events_path": str(raw_path),
                "command": command[:-1] + ["<prompt: workload recorded by hash>"]}


def event_payload(event: dict[str, Any]) -> dict[str, Any]:
    data = event.get("data")
    return data if isinstance(data, dict) else event


def model_name(value: Any) -> bool:
    if isinstance(value, dict):
        return value.get("provider") == "mesh" and value.get("id") in (
            GATEWAY_MODEL, f"{GATEWAY_MODEL}:low")
    return value in (MODEL, f"mesh/{GATEWAY_MODEL}")


def parent_model(event: dict[str, Any]) -> list[Any]:
    data = event_payload(event)
    message = data.get("message")
    models: list[Any] = []
    for source in (data, message):
        if not isinstance(source, dict) or not ("model" in source or "provider" in source):
            continue
        model = source.get("model")
        if isinstance(model, dict):
            models.append(model)
            if "provider" in source:
                models.append({"provider": source["provider"], "id": model.get("id")})
        elif "provider" in source or not isinstance(model, str) or "/" not in model:
            models.append({"provider": source.get("provider"), "id": model})
        else:
            models.append(model)
    return models


def event_type(event: dict[str, Any]) -> Any:
    return event.get("type") or event.get("event")


def tool_id(data: dict[str, Any]) -> Any:
    return data.get("toolCallId") or data.get("tool_call_id") or data.get("id")


def text_output(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(item.get("text", "") for item in value if isinstance(item, dict)
                         and isinstance(item.get("text"), str))
    return ""


def validate_root(path: Path, workload: dict[str, Any], mode: str,
                  run: dict[str, Any]) -> dict[str, Any]:
    require(not run["timed_out"] and not run["stream_error"],
            f"{mode}: timeout or event stream cap exceeded")
    require(run["exit_code"] == 0, f"{mode}: OMP exited {run['exit_code']}: {run['stderr'][-1000:]}")
    starts: list[dict[str, Any]] = []
    ends: list[dict[str, Any]] = []
    parent_models: list[Any] = []
    parent_usage: list[dict[str, Any]] = []
    final_text = ""
    final_index = -1
    final_tool_end = -1
    call_sequence: list[str] = []
    saw_turn_end = False
    with path.open("rb") as lines:
        for index, raw in enumerate(lines, 1):
            require(len(raw) <= 1024 * 1024, f"{mode}: oversized JSON event {index}")
            try:
                event = json.loads(raw)
            except (ValueError, UnicodeError) as error:
                raise BenchmarkError(f"{mode}: invalid JSON event {index}: {error}") from error
            require(isinstance(event, dict), f"{mode}: non-object JSON event {index}")
            kind = event_type(event)
            data = event_payload(event)
            if kind in ("tool_execution_start", "tool_execution_end"):
                call_sequence.append(kind)
                if kind == "tool_execution_end":
                    final_tool_end = index
                require(data.get("toolName") == "task", f"{mode}: non-task tool executed")
                (starts if kind == "tool_execution_start" else ends).append(data)
            elif kind in ("message_end", "turn_end"):
                message = data.get("message")
                models = parent_model(event)
                if kind == "message_end" and (
                    not isinstance(message, dict)
                    or message.get("role") not in ("assistant", "user", "toolResult")
                ):
                    parent_models.append(None)
                elif not models and isinstance(message, dict) and message.get("role") == "assistant":
                    parent_models.append(None)
                parent_models.extend(models)
                usage = data.get("usage") or (
                    message.get("usage") if isinstance(message, dict) else None)
                if isinstance(usage, dict) and usage:
                    parent_usage.append(usage)
                if kind == "turn_end":
                    saw_turn_end = True
                content = (message.get("content") if isinstance(message, dict) else None)
                content = data.get("content", content)
                if isinstance(content, str) and content.strip():
                    final_text, final_index = content, index
                elif isinstance(content, list):
                    extracted = text_output(content)
                    if extracted.strip():
                        final_text, final_index = extracted, index
    expected = workload["workers"]
    require(len(starts) == len(ends) == (3 if mode == "serial" else 1),
            f"{mode}: expected {'three' if mode == 'serial' else 'one'} completed task call(s), "
            f"got {len(starts)} starts and {len(ends)} ends")
    require(call_sequence == ["tool_execution_start", "tool_execution_end"] *
            (3 if mode == "serial" else 1),
            f"{mode}: task calls did not complete in the required order")
    require(saw_turn_end and parent_models and all(model_name(model) for model in parent_models),
            f"{mode}: missing or alternate root model/turn completion: {parent_models}")
    require(parent_usage, f"{mode}: missing parent model usage")
    validated = []
    seen_ids = set()
    for call_index, (start, end) in enumerate(zip(starts, ends)):
        sid, eid = tool_id(start), tool_id(end)
        require(sid is not None and sid == eid, f"{mode}: task call identity mismatch")
        arguments = start.get("args")
        require(isinstance(arguments, dict) and isinstance(arguments.get("tasks"), list),
                f"{mode}: missing task call arguments")
        require(isinstance(arguments.get("context"), str)
                and bool(arguments["context"].strip()),
                f"{mode}: task call omitted shared context")
        tasks = arguments["tasks"]
        intended = expected[call_index:call_index + 1] if mode == "serial" else expected
        require(len(tasks) == len(intended), f"{mode}: incorrect task batch size")
        for assigned, wanted in zip(tasks, intended):
            require(isinstance(assigned, dict) and assigned.get("name") == wanted["name"],
                    f"{mode}: task worker identity/order changed")
            task = assigned.get("task")
            anchor = ANCHOR_RE.search(wanted["task"]).group(1)
            require(isinstance(task, str) and anchor.casefold() in task.casefold(),
                    f"{mode}: {wanted['name']} task lost its distinctive anchor")
        result = end.get("result")
        details = result.get("details") if isinstance(result, dict) else None
        children = details.get("results") if isinstance(details, dict) else None
        require(isinstance(children, list) and len(children) == len(intended),
                f"{mode}: task result missing child details")
        by_name = {}
        for child in children:
            require(isinstance(child, dict) and isinstance(child.get("id"), str)
                    and bool(child["id"]), f"{mode}: child result missing execution id")
            name = child["id"]
            require(name not in by_name, f"{mode}: duplicate child result {name}")
            by_name[name] = child
        require(set(by_name) == {w["name"] for w in intended},
                f"{mode}: child execution ids differ from assigned workers")
        for wanted in intended:
            child = by_name[wanted["name"]]
            child_id = child["id"]
            require(child_id not in seen_ids, f"{mode}: duplicate child execution id")
            seen_ids.add(child_id)
            require(model_name(child.get("resolvedModel")),
                    f"{mode}: {wanted['name']} did not resolve to {MODEL}: {child.get('resolvedModel')!r}")
            require(type(child.get("exitCode")) is int and child["exitCode"] == 0,
                    f"{mode}: {wanted['name']} child failed")
            require(child.get("resolvedModelIsFallback") is not True,
                    f"{mode}: {wanted['name']} used fallback model")
            duration = child.get("durationMs")
            require(isinstance(duration, (int, float)) and not isinstance(duration, bool)
                    and 0 <= duration <= run["wall_seconds"] * 1000 + 5000,
                    f"{mode}: {wanted['name']} missing/invalid child duration")
            output = text_output(child.get("output"))
            require(bool(output.strip()), f"{mode}: {wanted['name']} has no child output")
            absent = [term for term in wanted["required_terms"] if term not in output.lower()]
            require(not absent, f"{mode}: {wanted['name']} missing output terms {absent}")
            usage = child.get("usage")
            require(isinstance(usage, dict) and bool(usage),
                    f"{mode}: {wanted['name']} missing child token usage")
            tokens, requests = child.get("tokens"), child.get("requests")
            require(isinstance(tokens, (int, float)) and not isinstance(tokens, bool)
                    and tokens >= 0 and isinstance(requests, int) and not isinstance(requests, bool)
                    and requests > 0,
                    f"{mode}: {wanted['name']} missing child tokens/requests")
            validated.append({"name": wanted["name"], "id": child_id,
                              "resolved_model": child["resolvedModel"], "duration_ms": duration,
                              "tokens": tokens, "usage": usage, "requests": requests,
                              "output": output, "required_terms_passed": True})
    require(final_index > final_tool_end and len(final_text.strip()) >= 80,
            f"{mode}: missing substantive final synthesis after child results")
    for worker in expected:
        require(worker["name"].casefold() in final_text.casefold(),
                f"{mode}: final synthesis omitted {worker['name']}")
    return {"parent_model_evidence": parent_models, "parent_usage": parent_usage,
            "final_synthesis": final_text, "children": validated,
            "makespan_seconds": run["wall_seconds"]}


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workload", type=Path, default=here / "qwen38-subagent-workload.json")
    parser.add_argument("--config", type=Path, default=here / "qwen38-subagent-benchmark.yml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--omp", default="omp")
    parser.add_argument("--gateway-url", default="http://127.0.0.1:9337")
    parser.add_argument("--service-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout-seconds", type=int, default=300,
                        help="per-root OMP deadline, integer 1–600 seconds (default 300)")
    args = parser.parse_args()
    receipt: dict[str, Any] = {"schema_version": 1, "status": "failed", "roots": {},
        "interpretation": {
            "physical_route": "Independent gateway probe, not per-child hardware attestation",
            "timing": "Diagnostic wall/makespan and child timing on a shared host; "
                      "serial and parallel parent prompts differ",
            "quality": "Required-term checks are bounded workload diagnostics, not broad accuracy admission",
        }}
    output = args.output.resolve()
    try:
        require(1 <= args.timeout_seconds <= 600, "--timeout-seconds must be in [1, 600]")
        workload_path, config_path = args.workload.resolve(), args.config.resolve()
        workload, digest = load_workload(workload_path)
        require(config_path.is_file(), f"missing configuration: {config_path}")
        config_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
        require(output not in (workload_path, config_path, Path(__file__).resolve()),
                "--output must not overwrite an input or the runner")
        output.parent.mkdir(parents=True, exist_ok=True)
        receipt.update(workload_id=workload["workload_id"], workload_path=str(workload_path),
                       workload_sha256=digest, config_path=str(config_path),
                       config_sha256=config_digest, expected_model=MODEL)
        receipt["route"] = probe(args.gateway_url, args.service_url)
        for mode in ("serial", "parallel"):
            require(hashlib.sha256(workload_path.read_bytes()).hexdigest() == digest
                    and hashlib.sha256(config_path.read_bytes()).hexdigest() == config_digest,
                    "workload or config changed during the benchmark")
            raw_path = output.with_name(output.stem + f".{mode}.jsonl")
            require(raw_path not in (workload_path, config_path, Path(__file__).resolve(),
                                     output), "event output must not overwrite an input")
            receipt["roots"][mode] = {"events_path": str(raw_path)}
            run = run_omp(args.omp, config_path, root_prompt(workload, mode), raw_path,
                          args.timeout_seconds)
            receipt["roots"][mode] = run
            validated = validate_root(raw_path, workload, mode, run)
            receipt["roots"][mode].update(validated)
        require(hashlib.sha256(workload_path.read_bytes()).hexdigest() == digest
                and hashlib.sha256(config_path.read_bytes()).hexdigest() == config_digest,
                "workload or config changed during the benchmark")
        serial, parallel = receipt["roots"]["serial"], receipt["roots"]["parallel"]
        receipt["comparison"] = {
            "serial_wall_seconds": serial["wall_seconds"],
            "parallel_wall_seconds": parallel["wall_seconds"],
            "wall_ratio_serial_over_parallel": serial["wall_seconds"] / parallel["wall_seconds"],
            "serial_child_duration_sum_ms": sum(c["duration_ms"] for c in serial["children"]),
            "parallel_child_duration_sum_ms": sum(c["duration_ms"] for c in parallel["children"]),
        }
        receipt["status"] = "verified"
    except (BenchmarkError, OSError, ValueError, KeyError, http.client.HTTPException) as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "receipt": str(output),
                      "error": receipt.get("error")}), flush=True)
    return 0 if receipt["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
