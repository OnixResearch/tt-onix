#!/usr/bin/env python3
"""Measure one continuous-slot admission root against a running Qwen3.8 P150x2 server.

Evidence driver for tasks s5 and s6 of `.cairn/changes/qwen38-continuous-slot-scheduler`.
It talks HTTP only; the operator starts the server (group coordinator for the baseline,
`--continuous-slot-policy` for the candidate) on a diagnostic port and stops it after.

Per root it records:

- solo references: every workload request run alone, one after another;
- `--repetitions` concurrent passes of the staggered, mixed-length workload, each
  request streamed and timed at the client from its send to `[DONE]`;
- the listener identity reported by the server's own evidence, which must name the
  expected coordinator, devices 0 and 1, and a non-simulated run.

`--extras` (candidate only) adds the s5 checks: a structured read issued while a long
request decodes, which must wait for the group to drain and match an idle read, a short
request queued behind that read, and sampled requests whose seeded outputs must not
change when they share the slots with other requests.

The output is the root schema of `qwen36_service.continuous_slot_admission_core`.
"""

from __future__ import annotations

import argparse
import http.client
import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

MODEL_ALIAS = "Qwen3.8-27B"
WORKLOAD_ID = "qwen38-staggered-mixed-length-v1"
SMOKE_WORKLOAD_ID = "qwen38-driver-smoke-v1"
SHARED_MAX_TOKENS = 1024
EXPECTED_DEVICE_IDS = [0, 1]
DECODE_PATHS = {"group": "batched-traced", "continuous": "continuous-slot"}
REQUEST_TIMEOUT_SECONDS = 1800
PASS_SETTLE_SECONDS = 5.0
READ_DELAY_SECONDS = 3.0
QUEUED_SHORT_DELAY_SECONDS = 5.0
EXTRA_LONG_MAX_TOKENS = 256
SAMPLED_MAX_TOKENS = 48
SAMPLED_TEMPERATURE = 0.6

LONG_INSTRUCTION = (
    "Rewrite the passage above as a detailed technical explanation of at least 900 words."
)
READ_PAYLOAD = {
    "state": {"ticket": "Everything is down and we have a demo at noon."},
    "questions": {
        "urgent": {
            "type": "noul",
            "instructions": "Does the customer need a reply within the hour?",
        }
    },
}


def log(stage: str, **fields: Any) -> None:
    print(json.dumps({"stage": stage, **fields}, sort_keys=True), flush=True)


def workload(filler: str, smoke: bool) -> list[dict[str, Any]]:
    """Declared requests: name, class, arrival offset, and chat messages."""

    def passage(start: int, length: int) -> str:
        return filler[start : start + length]

    requests = [
        ("l1", "long", 0.0, passage(0, 6000) + "\n\n" + LONG_INSTRUCTION),
        ("l2", "long", 0.0, passage(8000, 12000) + "\n\n" + LONG_INSTRUCTION),
        ("s1", "short", 4.0, "What is the capital of France? Answer in one word."),
        (
            "s2",
            "short",
            8.0,
            passage(22000, 3200) + "\n\nIn one sentence, what is this passage about?",
        ),
        (
            "s3",
            "short",
            12.0,
            passage(27000, 24000)
            + "\n\nName one technical term from the passage. Answer with the term only.",
        ),
        ("m1", "short", 16.0, "List the first 40 prime numbers separated by commas."),
        (
            "s4",
            "short",
            20.0,
            "Translate 'good morning' into Spanish. Answer with the translation only.",
        ),
        (
            "s5",
            "short",
            24.0,
            passage(53000, 8000) + "\n\nGive this passage a title of at most eight words.",
        ),
    ]
    if smoke:
        requests = [requests[2], requests[5]]
    return [
        {"name": name, "class": request_class, "arrival_offset_seconds": offset, "content": content}
        for name, request_class, offset, content in requests
    ]


def chat_payload(content: str, *, max_tokens: int, temperature: float | None = None, seed: int | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": MODEL_ALIAS,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "enable_thinking": False,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if seed is not None:
        payload["seed"] = seed
    return payload


class Server:
    def __init__(self, base_url: str) -> None:
        parts = urlsplit(base_url)
        if parts.scheme != "http" or not parts.hostname or not parts.port:
            raise SystemExit("--base-url must be http://host:port")
        self.host = parts.hostname
        self.port = parts.port

    def _connection(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(self.host, self.port, timeout=REQUEST_TIMEOUT_SECONDS)

    def get_json(self, path: str) -> tuple[int, dict[str, Any]]:
        connection = self._connection()
        try:
            connection.request("GET", path)
            response = connection.getresponse()
            return response.status, json.loads(response.read() or b"{}")
        finally:
            connection.close()

    def post_json(self, path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any], float]:
        connection = self._connection()
        started = time.monotonic()
        try:
            connection.request(
                "POST", path, json.dumps(payload).encode(), {"Content-Type": "application/json"}
            )
            response = connection.getresponse()
            body = json.loads(response.read() or b"{}")
            return response.status, body, time.monotonic() - started
        finally:
            connection.close()

    def stream_chat(self, payload: dict[str, Any], origin: float) -> dict[str, Any]:
        """Stream one chat completion; every time is seconds since `origin`."""
        record: dict[str, Any] = {
            "arrival_seconds": time.monotonic() - origin,
            "first_event_seconds": None,
            "done_seconds": None,
            "content_event_seconds": [],
            "token_ids": [],
            "finish_reason": None,
            "evidence": None,
            "usage": None,
            "error": None,
        }
        connection = self._connection()
        try:
            connection.request(
                "POST",
                "/v1/chat/completions",
                json.dumps(payload).encode(),
                {"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            if response.status != 200:
                record["error"] = {"status": response.status, "body": response.read().decode(errors="replace")}
                return record
            while True:
                line = response.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith(b"data: "):
                    continue
                now = time.monotonic() - origin
                data = line[len(b"data: ") :]
                if record["first_event_seconds"] is None:
                    record["first_event_seconds"] = now
                if data == b"[DONE]":
                    record["done_seconds"] = now
                    break
                chunk = json.loads(data)
                if "error" in chunk:
                    record["error"] = chunk["error"]
                    continue
                if chunk.get("usage") is not None and not chunk.get("choices"):
                    record["usage"] = chunk["usage"]
                    continue
                for choice in chunk.get("choices", ()):
                    delta = choice.get("delta") or {}
                    if delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls"):
                        record["content_event_seconds"].append(now)
                    if choice.get("finish_reason") is not None:
                        record["finish_reason"] = choice["finish_reason"]
                if "x_onix_decode_evidence" in chunk:
                    record["evidence"] = chunk["x_onix_decode_evidence"]
                    record["token_ids"] = list(chunk["x_onix_decode_evidence"]["generated_token_ids"])
        except (OSError, http.client.HTTPException, json.JSONDecodeError) as error:
            record["error"] = {"type": type(error).__name__, "message": str(error)}
        finally:
            connection.close()
        return record


def blake3_bytes(b3sum: str, data: bytes) -> str:
    return subprocess.run(
        [b3sum, "--no-names"], input=data, check=True, capture_output=True
    ).stdout.decode().strip()


def check_identity(kind: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    """The listener must be the expected coordinator on devices 0 and 1, not simulated."""
    identities = set()
    for record in records:
        evidence = record["evidence"]
        if evidence is None:
            raise SystemExit(f"a response carried no decode evidence: {record['error']}")
        if evidence.get("decode_path") != DECODE_PATHS[kind]:
            raise SystemExit(
                f"the listener served {evidence.get('decode_path')!r}, not the {kind} coordinator"
            )
        if list(evidence.get("physical_pcie_device_ids") or ()) != EXPECTED_DEVICE_IDS:
            raise SystemExit("the listener did not report devices 0 and 1")
        if evidence.get("simulated") is not False:
            raise SystemExit("the listener reported a simulated run")
        identities.add((evidence["package_identity"], evidence["runtime_identity"]))
    if len(identities) != 1:
        raise SystemExit(f"the listener's identity changed during the run: {identities}")
    package_identity, runtime_identity = identities.pop()
    return {"execution_identity": package_identity, "runtime_identity": runtime_identity}


def run_pass(server: Server, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    origin = time.monotonic()
    results: dict[str, dict[str, Any]] = {}

    def worker(request: dict[str, Any]) -> None:
        delay = origin + request["arrival_offset_seconds"] - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        results[request["name"]] = server.stream_chat(
            chat_payload(request["content"], max_tokens=SHARED_MAX_TOKENS), origin
        )

    threads = [threading.Thread(target=worker, args=(request,)) for request in requests]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return [{"name": request["name"], "class": request["class"], **results[request["name"]]} for request in requests]


def root_request(record: dict[str, Any]) -> dict[str, Any]:
    evidence = record["evidence"] or {}
    captures = evidence.get("trace_captures", evidence.get("trace_capture_count", 0))
    return {
        "name": record["name"],
        "class": record["class"],
        "arrival_seconds": record["arrival_seconds"],
        "first_event_seconds": record["first_event_seconds"],
        "done_seconds": record["done_seconds"] if record["error"] is None else None,
        "content_event_seconds": record["content_event_seconds"],
        "token_ids": record["token_ids"],
        "finish_reason": record["finish_reason"] if record["error"] is None else None,
        "trace_captures": int(captures or 0),
    }


def extras(server: Server, requests: list[dict[str, Any]], solo: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """s5 checks on the candidate: reads wait for the group, seeds survive shared slots."""
    by_name = {request["name"]: request for request in requests}
    checks: dict[str, Any] = {}

    status, idle_read, idle_read_seconds = server.post_json("/v1/systemone", READ_PAYLOAD)
    if status != 200:
        raise SystemExit(f"the idle structured read failed with {status}: {idle_read}")
    origin = time.monotonic()
    outcome: dict[str, Any] = {}

    def long_request() -> None:
        outcome["long"] = server.stream_chat(
            chat_payload(by_name["l1"]["content"], max_tokens=EXTRA_LONG_MAX_TOKENS), origin
        )

    def read_request() -> None:
        time.sleep(READ_DELAY_SECONDS)
        sent = time.monotonic() - origin
        status, body, seconds = server.post_json("/v1/systemone", READ_PAYLOAD)
        outcome["read"] = {"status": status, "body": body, "sent": sent, "done": sent + seconds}

    def short_request() -> None:
        time.sleep(QUEUED_SHORT_DELAY_SECONDS)
        outcome["short"] = server.stream_chat(
            chat_payload(by_name["s1"]["content"], max_tokens=SHARED_MAX_TOKENS), origin
        )

    threads = [threading.Thread(target=target) for target in (long_request, read_request, short_request)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    read = outcome["read"]
    long_tokens = outcome["long"]["token_ids"]
    checks["read_status_ok"] = read["status"] == 200
    checks["read_matches_idle_read"] = read["body"].get("answers") == idle_read.get("answers")
    checks["read_waited_for_long_request"] = (
        outcome["long"]["done_seconds"] is not None and read["done"] >= outcome["long"]["done_seconds"]
    )
    checks["long_matches_solo_prefix"] = (
        bool(long_tokens) and long_tokens == solo["l1"]["token_ids"][: len(long_tokens)]
    )
    checks["short_after_read_matches_solo"] = outcome["short"]["token_ids"] == solo["s1"]["token_ids"]
    checks["short_served_after_read"] = (
        outcome["short"]["first_event_seconds"] is not None
        and outcome["short"]["first_event_seconds"] >= read["done"]
    )

    sampled_prompts = {
        "sampled_a": ("Write one sentence about the sea.", 1234),
        "sampled_b": ("Write one sentence about mountains.", 5678),
    }
    alone = {}
    for name, (content, seed) in sampled_prompts.items():
        alone[name] = server.stream_chat(
            chat_payload(content, max_tokens=SAMPLED_MAX_TOKENS, temperature=SAMPLED_TEMPERATURE, seed=seed),
            time.monotonic(),
        )
    origin = time.monotonic()
    shared: dict[str, dict[str, Any]] = {}

    def sampled(name: str) -> None:
        content, seed = sampled_prompts[name]
        shared[name] = server.stream_chat(
            chat_payload(content, max_tokens=SAMPLED_MAX_TOKENS, temperature=SAMPLED_TEMPERATURE, seed=seed),
            origin,
        )

    def greedy() -> None:
        shared["greedy"] = server.stream_chat(
            chat_payload(by_name["m1"]["content"], max_tokens=SHARED_MAX_TOKENS), origin
        )

    threads = [threading.Thread(target=sampled, args=(name,)) for name in sampled_prompts]
    threads.append(threading.Thread(target=greedy))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for name in sampled_prompts:
        checks[f"{name}_matches_alone"] = (
            bool(alone[name]["token_ids"]) and shared[name]["token_ids"] == alone[name]["token_ids"]
        )
        checks[f"{name}_route"] = (shared[name]["evidence"] or {}).get("serving_route")
    checks["greedy_beside_sampled_matches_solo"] = shared["greedy"]["token_ids"] == solo["m1"]["token_ids"]
    return {
        "checks": checks,
        "idle_read_seconds": idle_read_seconds,
        "structured_read": {"sent": read["sent"], "done": read["done"], "answers": read["body"].get("answers")},
        "long": {k: outcome["long"][k] for k in ("first_event_seconds", "done_seconds", "finish_reason")},
        "short": {k: outcome["short"][k] for k in ("arrival_seconds", "first_event_seconds", "done_seconds")},
        "sampled": {
            name: {
                "alone": alone[name]["token_ids"],
                "shared": shared[name]["token_ids"],
                "evidence": shared[name]["evidence"],
            }
            for name in sampled_prompts
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--root-kind", required=True, choices=sorted(DECODE_PATHS))
    parser.add_argument("--package", required=True, help="store path of the qwen38 package the server runs")
    parser.add_argument("--filler", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--device-dram-free-bytes", type=int, nargs="+", required=True)
    parser.add_argument("--scheduler-policy", type=Path, help="continuous-slot policy JSON (candidate)")
    parser.add_argument("--b3sum", required=True)
    parser.add_argument("--extras-output", type=Path)
    parser.add_argument("--smoke", action="store_true", help="two short requests; never admission evidence")
    args = parser.parse_args()
    if (args.root_kind == "continuous") != (args.scheduler_policy is not None):
        raise SystemExit("--scheduler-policy is required for, and only for, the continuous root")
    if args.extras_output is not None and args.root_kind != "continuous":
        raise SystemExit("--extras-output is a candidate check")

    server = Server(args.base_url)
    status, health = server.get_json("/healthz")
    if status != 200 or health.get("status") != "ok" or health.get("model") != MODEL_ALIAS:
        raise SystemExit(f"the listener is not a ready {MODEL_ALIAS} service: {status} {health}")
    requests = workload(args.filler.read_text(encoding="utf-8"), args.smoke)
    declared = [
        {"name": r["name"], "class": r["class"], "arrival_offset_seconds": r["arrival_offset_seconds"],
         "payload": chat_payload(r["content"], max_tokens=SHARED_MAX_TOKENS)}
        for r in requests
    ]
    workload_blake3 = blake3_bytes(args.b3sum, json.dumps(declared, sort_keys=True).encode())
    log("workload", blake3=workload_blake3, requests=len(requests))

    solo_records = []
    for request in requests:
        record = server.stream_chat(chat_payload(request["content"], max_tokens=SHARED_MAX_TOKENS), time.monotonic())
        record["name"] = request["name"]
        if record["error"] is not None or record["done_seconds"] is None:
            raise SystemExit(f"solo {request['name']} failed: {record['error']}")
        solo_records.append(record)
        log(
            "solo",
            name=request["name"],
            tokens=len(record["token_ids"]),
            wall_seconds=record["done_seconds"] - record["arrival_seconds"],
            finish_reason=record["finish_reason"],
        )
    identity = check_identity(args.root_kind, solo_records)

    passes = []
    for repetition in range(args.repetitions):
        time.sleep(PASS_SETTLE_SECONDS)
        records = run_pass(server, requests)
        failed = [r["name"] for r in records if r["error"] is not None]
        log(
            "pass",
            repetition=repetition,
            failed=failed,
            walls={r["name"]: None if r["done_seconds"] is None else round(r["done_seconds"] - r["arrival_seconds"], 3) for r in records},
        )
        check_identity(args.root_kind, [r for r in records if r["error"] is None])
        passes.append({"requests": [root_request(r) for r in records], "raw_evidence": {r["name"]: r["evidence"] for r in records}})

    scheduler_policy_blake3 = None
    scheduler_profile_id = None
    if args.scheduler_policy is not None:
        policy_bytes = args.scheduler_policy.read_bytes()
        scheduler_policy_blake3 = blake3_bytes(args.b3sum, policy_bytes)
        scheduler_profile_id = json.loads(policy_bytes)["profile_id"]
    root = {
        "schema_version": 1,
        "root_kind": args.root_kind,
        "simulated": False,
        "identity": {
            "package": args.package,
            **identity,
            "physical_pcie_device_ids": EXPECTED_DEVICE_IDS,
            "scheduler_profile_id": scheduler_profile_id,
            "scheduler_policy_blake3": scheduler_policy_blake3,
        },
        "workload": {
            "workload_id": SMOKE_WORKLOAD_ID if args.smoke else WORKLOAD_ID,
            "blake3": workload_blake3,
            "requests": [{"name": r["name"], "class": r["class"], "arrival_offset_seconds": r["arrival_offset_seconds"]} for r in requests],
        },
        "startup": {"device_dram_free_bytes": args.device_dram_free_bytes},
        "solo": [
            {
                "name": r["name"],
                "token_ids": r["token_ids"],
                "wall_seconds": r["done_seconds"] - r["arrival_seconds"],
                "evidence": r["evidence"],
            }
            for r in solo_records
        ],
        "repetitions": passes,
    }
    args.output.write_text(json.dumps(root, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    log("root-written", path=str(args.output))
    if args.extras_output is not None:
        solo = {r["name"]: r for r in solo_records}
        result = extras(server, requests, solo)
        args.extras_output.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        log("extras", checks=result["checks"])


if __name__ == "__main__":
    main()
