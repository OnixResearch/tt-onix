#!/usr/bin/env python3
"""Measure one hybrid prefix-cache root against a running Qwen3.8 P150x2 server.

Evidence driver for tasks p5 and p6 of `.cairn/changes/qwen38-hybrid-prefix-cache`. The
operator starts the continuous-slot server on a diagnostic port without the cache (the
cold root) or with `--prefix-cache-policy` (the cached root), and stops it after.

The workload is agent-shaped: several conversations, each turn resending the whole
conversation plus the previous reply and a new passage, as an agent resends its history
with a new tool result. Turns run one at a time in a fixed interleaved order, so each
request's time is its own. Every request is streamed and timed at the client; the server's
evidence reports the prefix it resumed from and the prefix it saved.

The output root lists every turn's tokens, client TTFT and wall, and evidence. The
evaluator compares a cold root and a cached root turn by turn.
"""

from __future__ import annotations

import argparse
import http.client
import json
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

MODEL_ALIAS = "Qwen3.8-27B"
WORKLOAD_ID = "qwen38-agent-conversations-v1"
MAX_TOKENS = 48
REQUEST_TIMEOUT_SECONDS = 1800
EXPECTED_DEVICE_IDS = [0, 1]
SYSTEM_PROMPT_CHARACTERS = 6000
TURN_PASSAGE_CHARACTERS = 5000
QUESTION = "Summarize the newest passage in one sentence."


def log(stage: str, **fields: Any) -> None:
    print(json.dumps({"stage": stage, **fields}, sort_keys=True), flush=True)


def conversations(filler: str, count: int) -> list[dict[str, Any]]:
    """Each conversation opens with its own instructions and passage."""
    result = []
    for index in range(count):
        start = index * 7000
        result.append({
            "name": f"c{index}",
            "system": "You are a careful research assistant.\n\n"
            + filler[start : start + SYSTEM_PROMPT_CHARACTERS],
        })
    return result


def turn_passage(filler: str, conversation: int, turn: int) -> str:
    start = 60000 + (conversation * 11 + turn * 3) * 1700
    return filler[start : start + TURN_PASSAGE_CHARACTERS]


class Server:
    def __init__(self, base_url: str) -> None:
        parts = urlsplit(base_url)
        if parts.scheme != "http" or not parts.hostname or not parts.port:
            raise SystemExit("--base-url must be http://host:port")
        self.host, self.port = parts.hostname, parts.port

    def health(self) -> dict[str, Any]:
        connection = http.client.HTTPConnection(self.host, self.port, timeout=30)
        try:
            connection.request("GET", "/healthz")
            response = connection.getresponse()
            return {
                **json.loads(response.read() or b"{}"),
                "http_status": response.status,
            }
        finally:
            connection.close()

    def stream_chat(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        payload = {
            "model": MODEL_ALIAS,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "stream": True,
            "stream_options": {"include_usage": True},
            "enable_thinking": False,
        }
        record: dict[str, Any] = {
            "ttft_seconds": None,
            "wall_seconds": None,
            "text": "",
            "evidence": None,
            "usage": None,
            "error": None,
        }
        connection = http.client.HTTPConnection(
            self.host, self.port, timeout=REQUEST_TIMEOUT_SECONDS
        )
        started = time.monotonic()
        try:
            connection.request(
                "POST",
                "/v1/chat/completions",
                json.dumps(payload).encode(),
                {"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            if response.status != 200:
                record["error"] = {
                    "status": response.status,
                    "body": response.read().decode(errors="replace"),
                }
                return record
            while True:
                line = response.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith(b"data: "):
                    continue
                now = time.monotonic() - started
                data = line[len(b"data: ") :]
                if record["ttft_seconds"] is None:
                    record["ttft_seconds"] = now
                if data == b"[DONE]":
                    record["wall_seconds"] = now
                    break
                chunk = json.loads(data)
                if "error" in chunk:
                    record["error"] = chunk["error"]
                    continue
                if chunk.get("usage") is not None and not chunk.get("choices"):
                    record["usage"] = chunk["usage"]
                    continue
                for choice in chunk.get("choices", ()):
                    record["text"] += (choice.get("delta") or {}).get("content") or ""
                if "x_onix_decode_evidence" in chunk:
                    record["evidence"] = chunk["x_onix_decode_evidence"]
        except (OSError, http.client.HTTPException, json.JSONDecodeError) as error:
            record["error"] = {"type": type(error).__name__, "message": str(error)}
        finally:
            connection.close()
        return record


def blake3_bytes(b3sum: str, data: bytes) -> str:
    return (
        subprocess
        .run([b3sum, "--no-names"], input=data, check=True, capture_output=True)
        .stdout.decode()
        .strip()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--root-kind", required=True, choices=("cold", "cached"))
    parser.add_argument("--package", required=True)
    parser.add_argument("--filler", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--conversations", type=int, default=3)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--device-dram-free-bytes", type=int, nargs="+", required=True)
    parser.add_argument(
        "--prefix-policy", type=Path, help="prefix-cache policy JSON (cached root)"
    )
    parser.add_argument("--b3sum", required=True)
    args = parser.parse_args()
    if (args.root_kind == "cached") != (args.prefix_policy is not None):
        raise SystemExit(
            "--prefix-policy is required for, and only for, the cached root"
        )
    server = Server(args.base_url)
    health = server.health()
    if (
        health.get("http_status") != 200
        or health.get("status") != "ok"
        or health.get("model") != MODEL_ALIAS
    ):
        raise SystemExit(f"the listener is not a ready {MODEL_ALIAS} service: {health}")
    filler = args.filler.read_text(encoding="utf-8")
    opened = conversations(filler, args.conversations)
    plan = [
        {
            "conversation": index,
            "turn": turn,
            "passage": turn_passage(filler, index, turn),
        }
        for turn in range(args.turns)
        for index in range(args.conversations)
    ]
    workload_blake3 = blake3_bytes(
        args.b3sum,
        json.dumps(
            {
                "conversations": opened,
                "plan": plan,
                "question": QUESTION,
                "max_tokens": MAX_TOKENS,
            },
            sort_keys=True,
        ).encode(),
    )
    log("workload", blake3=workload_blake3, requests=len(plan))
    history = {
        index: [{"role": "system", "content": conversation["system"]}]
        for index, conversation in enumerate(opened)
    }
    turns = []
    identity = None
    for step in plan:
        messages = history[step["conversation"]] + [
            {"role": "user", "content": step["passage"] + "\n\n" + QUESTION}
        ]
        record = server.stream_chat(messages)
        evidence = record["evidence"] or {}
        if record["error"] is not None or record["wall_seconds"] is None:
            raise SystemExit(f"turn {step} failed: {record['error']}")
        if (
            evidence.get("decode_path") != "continuous-slot"
            or evidence.get("simulated") is not False
        ):
            raise SystemExit(
                f"the listener is not the continuous-slot service: {evidence.get('decode_path')}"
            )
        if list(evidence.get("physical_pcie_device_ids") or ()) != EXPECTED_DEVICE_IDS:
            raise SystemExit("the listener did not report devices 0 and 1")
        seen = (
            evidence["package_identity"],
            evidence["runtime_identity"],
            evidence.get("prefix_cache_profile_id"),
        )
        if identity is None:
            identity = seen
        elif seen != identity:
            raise SystemExit(f"the listener's identity changed: {identity} -> {seen}")
        if (args.root_kind == "cached") != (
            evidence.get("prefix_cache_profile_id") is not None
        ):
            raise SystemExit("the listener's prefix cache does not match the root kind")
        history[step["conversation"]] = messages + [
            {"role": "assistant", "content": record["text"]}
        ]
        turns.append({
            "conversation": step["conversation"],
            "turn": step["turn"],
            "prompt_tokens": (record["usage"] or {}).get("prompt_tokens"),
            "token_ids": evidence["generated_token_ids"],
            "ttft_seconds": record["ttft_seconds"],
            "wall_seconds": record["wall_seconds"],
            "prefix_hit_tokens": evidence.get("prefix_hit_tokens", 0),
            "prefix_saved_tokens": evidence.get("prefix_saved_tokens", 0),
            "prefill_seconds": evidence.get("prefill_seconds"),
            "server_ttft_seconds": evidence.get("ttft_seconds"),
        })
        log(
            "turn",
            conversation=step["conversation"],
            turn=step["turn"],
            prompt_tokens=turns[-1]["prompt_tokens"],
            ttft=round(record["ttft_seconds"], 3),
            wall=round(record["wall_seconds"], 3),
            hit=turns[-1]["prefix_hit_tokens"],
            saved=turns[-1]["prefix_saved_tokens"],
        )
    policy_blake3 = (
        blake3_bytes(args.b3sum, args.prefix_policy.read_bytes())
        if args.prefix_policy
        else None
    )
    root = {
        "schema_version": 1,
        "root_kind": args.root_kind,
        "simulated": False,
        "identity": {
            "package": args.package,
            "execution_identity": identity[0],
            "runtime_identity": identity[1],
            "physical_pcie_device_ids": EXPECTED_DEVICE_IDS,
            "prefix_cache_profile_id": identity[2],
            "prefix_policy_blake3": policy_blake3,
        },
        "workload": {
            "workload_id": WORKLOAD_ID,
            "blake3": workload_blake3,
            "conversations": args.conversations,
            "turns": args.turns,
        },
        "startup": {"device_dram_free_bytes": args.device_dram_free_bytes},
        "requests": turns,
    }
    args.output.write_text(
        json.dumps(root, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    log("root-written", path=str(args.output))


if __name__ == "__main__":
    main()
