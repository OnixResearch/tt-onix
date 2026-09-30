#!/usr/bin/env python3
"""Measure a hybrid prefix-cache root against a running Qwen3.8 P150x2 server.

The operator runs fresh cold and cached processes in ABBA order. By default this is
the v1 agent-shaped workload. --shared-system reuses one system header across five
interleaved conversations; --shared-tools adds the same offered function tool and,
after those turns, probes a changed description with the same function name.
Every turn resends its conversation with the previous reply and a new passage.

Each root records exact tokens, client TTFT and wall, saved/resumed prefixes, and
host CPU pressure when the listener is on loopback. The admission evaluator
compares the four roots turn by turn.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

MODEL_ALIAS = "Qwen3.8-27B"
WORKLOAD_ID = "qwen38-agent-conversations-v1"
SHARED_WORKLOAD_ID = "qwen38-agent-shared-system-v1"
SHARED_TOOL_WORKLOAD_ID = "qwen38-agent-shared-tools-v1"
SAMPLED_WORKLOAD_ID = "qwen38-agent-conversations-sampled-v1"
MAX_TOKENS = 48
REQUEST_TIMEOUT_SECONDS = 1800
EXPECTED_DEVICE_IDS = [0, 1]
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
SYSTEM_PROMPT_CHARACTERS = 6000
SHARED_SYSTEM_PROMPT_CHARACTERS = 9000
TURN_PASSAGE_CHARACTERS = 5000
QUESTION = "Summarize the newest passage in one sentence."
BASE_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_observation",
        "description": "Retrieve archived station observations by station id.",
        "parameters": {
            "type": "object",
            "properties": {"station": {"type": "integer", "description": "Station number to retrieve."}},
            "required": ["station"],
        },
    },
}
CHANGED_TOOL = {
    "type": "function",
    "function": {
        **BASE_TOOL["function"],
        "description": "Return only revised station metadata by station id, not observations.",
    },
}


def log(stage: str, **fields: Any) -> None:
    print(json.dumps({"stage": stage, **fields}, sort_keys=True), flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cpu_pressure() -> tuple[float, int, float]:
    """(some avg10 percent, some total stall microseconds, 1-minute load average)."""
    some = next(
        line
        for line in Path("/proc/pressure/cpu").read_text().splitlines()
        if line.startswith("some ")
    )
    fields = dict(item.split("=", 1) for item in some.split()[1:])
    return float(fields["avg10"]), int(fields["total"]), os.getloadavg()[0]


def conversations(filler: str, count: int, *, shared_system: bool = False) -> list[dict[str, Any]]:
    """Distinct conversation histories; optionally share the system instructions."""
    result = []
    for index in range(count):
        start = 0 if shared_system else index * 7000
        length = SHARED_SYSTEM_PROMPT_CHARACTERS if shared_system else SYSTEM_PROMPT_CHARACTERS
        result.append({
            "name": f"c{index}",
            "system": "You are a careful research assistant.\n\n"
            + filler[start : start + length],
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
        self.on_this_host = parts.hostname in LOOPBACK_HOSTS

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

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        sampling: dict[str, Any] | None = None,
        *,
        tools: tuple[dict[str, Any], ...] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "model": MODEL_ALIAS,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "stream": True,
            "stream_options": {"include_usage": True},
            "enable_thinking": False,
            **({"tools": list(tools)} if tools is not None else {}),
            **(sampling or {}),
        }
        record: dict[str, Any] = {
            "ttft_seconds": None,
            "wall_seconds": None,
            "text": "",
            "evidence": None,
            "usage": None,
            "tool_calls_seen": False,
            "error": None,
            "host_cpu_pressure": None,
        }
        connection = http.client.HTTPConnection(
            self.host, self.port, timeout=REQUEST_TIMEOUT_SECONDS
        )
        before = cpu_pressure() if self.on_this_host else None
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
                    delta = choice.get("delta") or {}
                    record["text"] += delta.get("content") or ""
                    record["tool_calls_seen"] |= bool(delta.get("tool_calls") or choice.get("finish_reason") == "tool_calls")
                if "x_onix_decode_evidence" in chunk:
                    record["evidence"] = chunk["x_onix_decode_evidence"]
        except (OSError, http.client.HTTPException, json.JSONDecodeError) as error:
            record["error"] = {"type": type(error).__name__, "message": str(error)}
        finally:
            connection.close()
            if before is not None:
                elapsed = time.monotonic() - started
                after = cpu_pressure()
                record["host_cpu_pressure"] = {
                    "some_avg10_percent_at_start": before[0],
                    # Share of the request during which runnable tasks waited for a CPU.
                    "some_stall_fraction": (after[1] - before[1]) / (elapsed * 1e6),
                    "load_average_1m_at_start": before[2],
                }
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
    parser.add_argument("--shared-system", action="store_true", help="interleave five or more chats with one system header")
    parser.add_argument("--shared-tools", action="store_true", help="measure one shared system/tool header and a changed-tool miss")
    parser.add_argument("--device-dram-free-bytes", type=int, nargs="+", required=True)
    parser.add_argument(
        "--prefix-policy", type=Path, help="prefix-cache policy JSON (cached root)"
    )
    parser.add_argument("--b3sum", required=True)
    parser.add_argument(
        "--temperature",
        type=float,
        help="sample every turn at this temperature, seeded per turn (needs --seed)",
    )
    parser.add_argument("--seed", type=int, help="base seed of the sampled workload")
    args = parser.parse_args()
    if (args.root_kind == "cached") != (args.prefix_policy is not None):
        raise SystemExit(
            "--prefix-policy is required for, and only for, the cached root"
        )
    if (args.temperature is None) != (args.seed is None) or (
        args.temperature is not None and args.temperature <= 0
    ):
        raise SystemExit(
            "a sampled workload needs a positive --temperature and a --seed"
        )
    sampled = args.temperature is not None
    if args.shared_system and args.shared_tools:
        raise SystemExit("--shared-system and --shared-tools select distinct workloads")
    shared_header = args.shared_system or args.shared_tools
    if shared_header and (args.conversations < 5 or args.turns < 2 or sampled):
        raise SystemExit("a shared header needs at least five conversations, two turns, and greedy decoding")
    policy = json.loads(args.prefix_policy.read_text(encoding="utf-8")) if args.prefix_policy else None
    if shared_header and policy is not None and (
        policy.get("schema_version") != 2
        or policy.get("shared_root_minimum_tokens", 0) < 2048
        or policy.get("state_slots") != 4
    ):
        raise SystemExit("a shared header needs the four-state v2 root policy")
    server = Server(args.base_url)
    health = server.health()
    if (
        health.get("http_status") != 200
        or health.get("status") != "ok"
        or health.get("model") != MODEL_ALIAS
    ):
        raise SystemExit(f"the listener is not a ready {MODEL_ALIAS} service: {health}")
    filler = args.filler.read_text(encoding="utf-8")
    opened = conversations(filler, args.conversations, shared_system=shared_header)
    if args.shared_tools:
        for conversation in opened:
            conversation["system"] = (
                "Answer solely from the supplied passage. Do not call tools.\n\n"
                + conversation["system"]
            )
    plan = [
        {
            "conversation": index,
            "turn": turn,
            "passage": turn_passage(filler, index, turn),
        }
        for turn in range(args.turns)
        for index in range(args.conversations)
    ]
    workload = {
        "conversations": opened,
        "plan": plan,
        "question": QUESTION,
        "max_tokens": MAX_TOKENS,
    }
    if shared_header:
        workload["shared_system"] = True
    if args.shared_tools:
        workload["shared_tools"] = True
        workload["tools"] = [BASE_TOOL]
        workload["changed_tools"] = [CHANGED_TOOL]
        workload["changed_probe"] = turn_passage(filler, 0, args.turns) + "\n\n" + QUESTION
    if sampled:
        workload["sampling"] = {"temperature": args.temperature, "seed": args.seed}
    workload_blake3 = blake3_bytes(
        args.b3sum, json.dumps(workload, sort_keys=True).encode()
    )
    log("workload", blake3=workload_blake3, requests=len(plan))
    history = {
        index: [{"role": "system", "content": conversation["system"]}]
        for index, conversation in enumerate(opened)
    }
    turns = []
    identity = None
    started_utc = utc_now()
    for step in plan:
        messages = history[step["conversation"]] + [
            {"role": "user", "content": step["passage"] + "\n\n" + QUESTION}
        ]
        record = server.stream_chat(
            messages,
            {
                "temperature": args.temperature,
                "seed": args.seed + step["conversation"] * 1000 + step["turn"],
            }
            if sampled
            else None,
            tools=(BASE_TOOL,) if args.shared_tools else None,
        )
        evidence = record["evidence"] or {}
        if record["error"] is not None or record["wall_seconds"] is None:
            raise SystemExit(f"turn {step} failed: {record['error']}")
        if args.shared_tools and (record["tool_calls_seen"] or not record["text"].strip()):
            raise SystemExit(
                f"turn c{step['conversation']}t{step['turn']} did not produce a text-only answer"
            )
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
        if (policy is None) != (evidence.get("prefix_cache_profile_id") is None):
            raise SystemExit("the listener's prefix cache does not match the root kind")
        if policy is not None and evidence.get("prefix_cache_profile_id") != policy["profile_id"]:
            raise SystemExit("the listener's prefix cache profile differs from the requested policy")
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
            "trace_captures": evidence.get("trace_captures"),
            "tail_token_latency_seconds": evidence.get("tail_token_latency_seconds"),
            "serving_route": evidence.get("serving_route"),
            "host_cpu_pressure": record["host_cpu_pressure"],
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
            captures=turns[-1]["trace_captures"],
            cpu_stall=(record["host_cpu_pressure"] or {}).get("some_stall_fraction"),
        )
    change_probe = None
    if args.shared_tools:
        record = server.stream_chat(
            [
                {"role": "system", "content": opened[0]["system"]},
                {"role": "user", "content": workload["changed_probe"]},
            ],
            tools=(CHANGED_TOOL,),
        )
        evidence = record["evidence"] or {}
        if record["error"] is not None or record["wall_seconds"] is None:
            raise SystemExit(f"changed-tool probe failed: {record['error']}")
        if record["tool_calls_seen"] or not record["text"].strip():
            raise SystemExit("changed-tool probe did not produce a text-only answer")
        if (
            evidence.get("decode_path") != "continuous-slot"
            or evidence.get("simulated") is not False
            or list(evidence.get("physical_pcie_device_ids") or ()) != EXPECTED_DEVICE_IDS
            or (
                evidence.get("package_identity"),
                evidence.get("runtime_identity"),
                evidence.get("prefix_cache_profile_id"),
            ) != identity
        ):
            raise SystemExit("changed-tool probe ran on a different listener")
        change_probe = {
            "baseline_tools_blake3": blake3_bytes(args.b3sum, json.dumps([BASE_TOOL], sort_keys=True).encode()),
            "changed_tools_blake3": blake3_bytes(args.b3sum, json.dumps([CHANGED_TOOL], sort_keys=True).encode()),
            "prompt_tokens": (record["usage"] or {}).get("prompt_tokens"),
            "token_ids": evidence["generated_token_ids"],
            "prefix_hit_tokens": evidence.get("prefix_hit_tokens", 0),
        }
        log("tool-change", hit=change_probe["prefix_hit_tokens"], prompt_tokens=change_probe["prompt_tokens"])
    finished_utc = utc_now()
    policy_blake3 = (
        blake3_bytes(args.b3sum, args.prefix_policy.read_bytes())
        if args.prefix_policy
        else None
    )
    root = {
        "schema_version": 2,
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
            "workload_id": (
                SHARED_TOOL_WORKLOAD_ID if args.shared_tools else
                SHARED_WORKLOAD_ID if args.shared_system else
                SAMPLED_WORKLOAD_ID if sampled else WORKLOAD_ID
            ),
            "blake3": workload_blake3,
            "conversations": args.conversations,
            "turns": args.turns,
        },
        "measured_at": {"started_utc": started_utc, "finished_utc": finished_utc},
        "startup": {"device_dram_free_bytes": args.device_dram_free_bytes},
        "requests": turns,
    }
    if change_probe is not None:
        root["tool_change"] = change_probe
    args.output.write_text(
        json.dumps(root, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    log("root-written", path=str(args.output))


if __name__ == "__main__":
    main()
