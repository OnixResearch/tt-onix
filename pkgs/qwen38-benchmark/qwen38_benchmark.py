#!/usr/bin/env python3
"""Repeatable client-side benchmark for OpenAI-compatible Qwen services.

The harness measures what a client sees: time to first content token, steady
decode rate, delivered tokens per second, and batch aggregate throughput for a
frozen corpus. It never starts a server, touches a Tenstorrent device, or reads
server-private evidence. Point one run at the onix service and another at a
vLLM endpoint; ``compare`` checks that both ran the same workload and that
greedy text stayed identical.

The ``agentic`` mode instead runs concurrent closed-loop agents through a
frozen multi-call trace that reuses its growing history, and reports the
largest concurrency meeting per-call TTFT/ITL and task-time SLOs. It records
client-known prefix overlap separately from any endpoint-reported cached
tokens.

The default corpus is the admitted Qwen3.8-27B B=8 corpus: eight prompts and 64
generated tokens. Greedy decoding, streaming, and a fixed request order keep
the two JSON results comparable.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import platform
import socket
import ssl
import statistics
import subprocess
import sys
import threading
import time
import urllib.parse
from dataclasses import dataclass
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

SCHEMA = "onix.qwen38-benchmark.v1"
COMPARISON_SCHEMA = "onix.qwen38-benchmark-comparison.v1"
AGENTIC_SCHEMA = "onix.qwen38-agentic-benchmark.v1"
DEFAULT_MODEL = None
DEFAULT_CORPUS = Path(
    os.environ.get("QWEN38_BENCH_CORPUS", Path(__file__).with_name("corpus.json"))
)
DEFAULT_AGENTIC_TRACE = Path(
    os.environ.get(
        "QWEN38_BENCH_AGENTIC_TRACE", Path(__file__).with_name("agentic_trace.json")
    )
)
CACHE_MODES = ("cold", "warm", "disabled", "unknown")
POWER_BOUNDARIES = ("cards", "host", "wall", "unspecified")
ENVIRONMENT_KEYS = (
    "HF_MODEL",
    "MESH_DEVICE",
    "QWEN35_GDN_DECODE_FUSION",
    "QWEN36_GDN_BF4",
    "QWEN36_LM_HEAD_BF4",
    "QWEN36_MLP_DOWN_BF4",
    "QWEN_SDPA_BF8",
    "TT_METAL_DISABLE_SFPLOADMACRO",
    "TT_METAL_HOME",
    "TT_METAL_VERSION",
    "VLLM_ATTENTION_BACKEND",
    "VLLM_TARGET_DEVICE",
    "VLLM_USE_V1",
)


class BenchmarkError(RuntimeError):
    """A target, request, or result cannot satisfy the benchmark contract."""


def validate_header(name: Any, value: Any) -> None:
    if not isinstance(name, str) or not isinstance(value, str):
        raise BenchmarkError("header names and values must be strings")
    if not name or any(character in name + value for character in "\r\n"):
        raise BenchmarkError("header names must be non-empty and contain no newlines")
    try:
        name.encode("latin-1")
        value.encode("latin-1")
    except UnicodeEncodeError as error:
        raise BenchmarkError("headers must be Latin-1 encodable") from error


@dataclass(frozen=True)
class Endpoint:
    label: str
    base_url: str
    model: str | None
    api_key: str | None
    headers: tuple[tuple[str, str], ...]
    timeout: float

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise BenchmarkError("target label must be a non-empty string")
        if (
            not isinstance(self.timeout, (int, float))
            or isinstance(self.timeout, bool)
            or not math.isfinite(self.timeout)
            or self.timeout <= 0
        ):
            raise BenchmarkError(
                f"{self.label}: request timeout must be finite and positive"
            )
        if self.api_key is not None:
            validate_header("Authorization", f"Bearer {self.api_key}")
        if not isinstance(self.headers, tuple):
            raise BenchmarkError(f"{self.label}: headers must be a tuple")
        for pair in self.headers:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise BenchmarkError(f"{self.label}: invalid header pair")
            validate_header(pair[0], pair[1])

    @property
    def origin(self) -> urllib.parse.SplitResult:
        if not isinstance(self.base_url, str):
            raise BenchmarkError(f"{self.label}: base URL must be a string")
        parsed = urllib.parse.urlsplit(self.base_url)
        try:
            hostname = parsed.hostname
            parsed.port
        except ValueError as error:
            raise BenchmarkError(
                f"{self.label}: invalid base URL {self.base_url!r}: {error}"
            ) from error
        if parsed.scheme not in {"http", "https"} or not hostname:
            raise BenchmarkError(f"{self.label}: invalid base URL {self.base_url!r}")
        if parsed.username or parsed.password:
            raise BenchmarkError(
                f"{self.label}: put credentials in --api-key, not the base URL"
            )
        if parsed.query or parsed.fragment:
            raise BenchmarkError(
                f"{self.label}: base URL must not carry a query or fragment"
            )
        return parsed

    def api_path(self, suffix: str) -> str:
        parsed = self.origin
        prefix = parsed.path.rstrip("/")
        if not prefix.endswith("/v1"):
            prefix += "/v1"
        return prefix + suffix

    def root_path(self, suffix: str) -> str:
        parsed = self.origin
        return parsed.path.rstrip("/") + suffix


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * fraction
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - rank) + ordered[upper] * (rank - lower)


def open_connection(
    endpoint: Endpoint, timeout: float | None = None
) -> http.client.HTTPConnection:
    parsed = endpoint.origin
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    effective_timeout = endpoint.timeout if timeout is None else timeout
    if parsed.scheme == "https":
        connection: http.client.HTTPConnection = http.client.HTTPSConnection(
            parsed.hostname,
            port,
            timeout=effective_timeout,
            context=ssl.create_default_context(),
        )
    else:
        connection = http.client.HTTPConnection(
            parsed.hostname, port, timeout=effective_timeout
        )
    connection.connect()
    return connection


def request_headers(endpoint: Endpoint, stream: bool) -> dict[str, str]:
    headers = {
        "Accept": "text/event-stream" if stream else "application/json",
        "Accept-Encoding": "identity",
        "Content-Type": "application/json",
        "Connection": "close",
    }
    if endpoint.api_key:
        headers["Authorization"] = f"Bearer {endpoint.api_key}"
    for name, value in endpoint.headers:
        headers[name] = value
    return headers


def read_http_json(
    endpoint: Endpoint,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: float | None = None,
) -> tuple[int, Any]:
    connection = open_connection(endpoint, timeout)
    try:
        body = None if payload is None else json.dumps(payload).encode()
        connection.request(
            method,
            path,
            body=body,
            headers=request_headers(endpoint, False),
        )
        response = connection.getresponse()
        raw = response.read()
        try:
            decoded = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            decoded = raw.decode("utf-8", "replace")[:2048]
        return response.status, decoded
    finally:
        connection.close()


def endpoint_probe(endpoint: Endpoint) -> dict[str, Any]:
    """Best-effort target metadata. It never gates a run."""
    probe: dict[str, Any] = {"base_url": endpoint.base_url}
    probe_timeout = min(endpoint.timeout, 30.0)
    try:
        status, payload = read_http_json(
            endpoint, "GET", endpoint.api_path("/models"), timeout=probe_timeout
        )
        probe["models_status"] = status
        if status == 200 and isinstance(payload, dict):
            data = payload.get("data")
            if isinstance(data, list):
                probe["models"] = [
                    item.get("id")
                    for item in data
                    if isinstance(item, dict) and isinstance(item.get("id"), str)
                ]
    except (OSError, http.client.HTTPException) as error:
        probe["models_error"] = f"{type(error).__name__}: {error}"

    for suffix in ("/healthz", "/health"):
        try:
            status, payload = read_http_json(
                endpoint, "GET", endpoint.root_path(suffix), timeout=probe_timeout
            )
        except (OSError, http.client.HTTPException):
            continue
        if status == 200:
            probe["health_path"] = suffix
            probe["health"] = payload
            break
    return probe


def resolve_model(endpoint: Endpoint) -> tuple[str, dict[str, Any]]:
    probe = endpoint_probe(endpoint)
    models = probe.get("models")
    if endpoint.model:
        return endpoint.model, probe
    if isinstance(models, list) and models and isinstance(models[0], str):
        return models[0], probe
    raise BenchmarkError(
        f"{endpoint.label}: cannot discover a model from /v1/models; pass --model"
    )


def sse_payloads(response: http.client.HTTPResponse):
    """Yield one JSON document per SSE event, including multi-line data fields."""
    data_lines: list[str] = []
    while True:
        raw = response.readline()
        if not raw:
            if data_lines:
                data = "\n".join(data_lines).strip()
                data_lines.clear()
                if data and data != "[DONE]":
                    yield data
            return
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if not line:
            if not data_lines:
                continue
            data = "\n".join(data_lines).strip()
            data_lines.clear()
            if data == "[DONE]":
                return
            if data:
                yield data
            continue
        if line.startswith(":") or not line.startswith("data:"):
            continue
        data_lines.append(line[5:].lstrip())


def content_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    text = first.get("text")
    if isinstance(text, str):
        return text
    delta = first.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        if isinstance(content, str):
            return content
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    return ""


def record_from_usage(
    text: str,
    usage: dict[str, Any] | None,
    chunk_tokens: int,
    start: float,
    end: float,
    first_token: float | None,
    last_token: float | None,
) -> dict[str, Any]:
    completion_tokens = None
    prompt_tokens = None
    token_count_source = "chunks"
    if isinstance(usage, dict):
        raw_completion = usage.get("completion_tokens")
        raw_prompt = usage.get("prompt_tokens")
        if isinstance(raw_completion, int) and not isinstance(raw_completion, bool):
            completion_tokens = raw_completion
            token_count_source = "usage"
        if isinstance(raw_prompt, int) and not isinstance(raw_prompt, bool):
            prompt_tokens = raw_prompt
    if completion_tokens is None:
        completion_tokens = chunk_tokens

    ttft = None if first_token is None else first_token - start
    last = None if last_token is None else last_token - start
    wall = end - start
    decode_rate = None
    if ttft is not None and last is not None and last > ttft and completion_tokens > 1:
        decode_rate = (completion_tokens - 1) / (last - ttft)
    e2e_rate = completion_tokens / wall if completion_tokens > 0 and wall > 0 else None
    return {
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text_characters": len(text),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "token_count_source": token_count_source,
        "stream_chunks": chunk_tokens,
        "ttft_seconds": ttft,
        "last_token_seconds": last,
        "wall_seconds": wall,
        "decode_tokens_per_second": decode_rate,
        "e2e_tokens_per_second": e2e_rate,
    }


def stream_completion(
    connection: http.client.HTTPConnection,
    endpoint: Endpoint,
    path: str,
    payload: dict[str, Any],
    context: str,
    start: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Stream one completion and measure it. Returns (record, usage)."""
    connection.request(
        "POST",
        path,
        body=json.dumps(payload).encode(),
        headers=request_headers(endpoint, True),
    )
    response = connection.getresponse()
    if response.status != 200:
        body = response.read().decode("utf-8", "replace")
        raise BenchmarkError(f"{endpoint.label}: HTTP {response.status}: {body[:1000]}")

    first_token: float | None = None
    last_token: float | None = None
    text_parts: list[str] = []
    chunk_tokens = 0
    usage: dict[str, Any] | None = None
    completed = False
    for data in sse_payloads(response):
        try:
            decoded = json.loads(data)
        except json.JSONDecodeError as error:
            raise BenchmarkError(
                f"{endpoint.label}: invalid SSE payload {data[:300]!r}"
            ) from error
        if not isinstance(decoded, dict):
            continue
        if isinstance(decoded.get("error"), dict):
            raise BenchmarkError(f"{endpoint.label}: {decoded['error']}")
        if isinstance(decoded.get("usage"), dict):
            usage = decoded["usage"]
        text = content_text(decoded)
        if text:
            now = time.perf_counter()
            if first_token is None:
                first_token = now
            last_token = now
            text_parts.append(text)
            chunk_tokens += 1
        choices = decoded.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict) and first.get("finish_reason") is not None:
                completed = True
    end = time.perf_counter()
    response.close()

    if first_token is None:
        raise BenchmarkError(
            f"{endpoint.label}: stream produced no content for {context}"
        )
    if (
        not isinstance(usage, dict)
        or not _is_int(usage.get("completion_tokens"))
        or usage["completion_tokens"] <= 0
    ):
        raise BenchmarkError(
            f"{endpoint.label}: stream did not report a positive usage.completion_tokens"
        )
    record = record_from_usage(
        "".join(text_parts),
        usage,
        chunk_tokens,
        start,
        end,
        first_token,
        last_token,
    )
    record["finish_reason_seen"] = completed
    return record, usage


def perform_request(
    connection: http.client.HTTPConnection,
    endpoint: Endpoint,
    model: str,
    prompt: str,
    max_tokens: int,
    stream: bool,
    start: float | None = None,
) -> dict[str, Any]:
    request_start = time.perf_counter() if start is None else start
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}

    if stream:
        record, _usage = stream_completion(
            connection,
            endpoint,
            endpoint.api_path("/completions"),
            payload,
            f"prompt {prompt[:60]!r}",
            request_start,
        )
        return record

    connection.request(
        "POST",
        endpoint.api_path("/completions"),
        body=json.dumps(payload).encode(),
        headers=request_headers(endpoint, False),
    )
    response = connection.getresponse()
    if response.status != 200:
        body = response.read().decode("utf-8", "replace")
        raise BenchmarkError(f"{endpoint.label}: HTTP {response.status}: {body[:1000]}")

    raw = response.read()
    end = time.perf_counter()
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as error:
        raise BenchmarkError(
            f"{endpoint.label}: non-JSON response: {raw[:300]!r}"
        ) from error
    if not isinstance(decoded, dict):
        raise BenchmarkError(f"{endpoint.label}: response is not an object")
    if isinstance(decoded.get("error"), dict):
        raise BenchmarkError(f"{endpoint.label}: {decoded['error']}")
    usage = decoded.get("usage")
    if not isinstance(usage, dict) or not _is_int(usage.get("completion_tokens")):
        raise BenchmarkError(
            f"{endpoint.label}: non-stream response did not report completion_tokens"
        )
    text = content_text(decoded)
    if not text:
        raise BenchmarkError(
            f"{endpoint.label}: non-stream response had no completion text"
        )
    record = record_from_usage(
        text,
        usage,
        1,
        request_start,
        end,
        None,
        None,
    )
    if record["completion_tokens"] <= 0:
        raise BenchmarkError(
            f"{endpoint.label}: non-stream response reported no completion tokens"
        )
    return record


def run_single(
    endpoint: Endpoint,
    model: str,
    prompts: list[str],
    max_tokens: int,
    stream: bool,
) -> list[dict[str, Any]]:
    records = []
    for index, prompt in enumerate(prompts):
        connection = open_connection(endpoint)
        try:
            record = perform_request(
                connection, endpoint, model, prompt, max_tokens, stream
            )
        finally:
            connection.close()
        record["prompt_index"] = index
        record["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
        record["start_offset_seconds"] = 0.0
        records.append(record)
    return records


def run_batch_with_offsets(
    endpoint: Endpoint,
    model: str,
    prompts: list[str],
    concurrency: int,
    max_tokens: int,
    stream: bool,
) -> list[dict[str, Any]]:
    """Batch with a shared barrier and per-record offsets from the group start."""
    connections: list[http.client.HTTPConnection] = []
    try:
        for _ in range(concurrency):
            connections.append(open_connection(endpoint))
    except BaseException:
        for connection in connections:
            connection.close()
        raise
    barrier = threading.Barrier(concurrency)
    records: list[dict[str, Any] | None] = [None] * concurrency
    starts: list[float | None] = [None] * concurrency
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        try:
            barrier.wait(timeout=endpoint.timeout)
        except BaseException as error:  # noqa: BLE001 - fail the whole group.
            errors.append(error)
            barrier.abort()
            return
        prompt_index = index % len(prompts)
        try:
            start = time.perf_counter()
            starts[index] = start
            record = perform_request(
                connections[index],
                endpoint,
                model,
                prompts[prompt_index],
                max_tokens,
                stream,
                start=start,
            )
            record["prompt_index"] = prompt_index
            record["prompt_sha256"] = hashlib.sha256(
                prompts[prompt_index].encode()
            ).hexdigest()
            record["_start"] = start
            records[index] = record
        except BaseException as error:  # noqa: BLE001 - report every group member.
            errors.append(error)
        finally:
            connections[index].close()

    threads = [
        threading.Thread(target=worker, args=(index,), name=f"bench-{index}")
        for index in range(concurrency)
    ]
    started: list[threading.Thread] = []
    try:
        for thread in threads:
            thread.start()
            started.append(thread)
    except BaseException:
        barrier.abort()
        for thread in started:
            thread.join()
        for connection in connections:
            connection.close()
        raise
    for thread in threads:
        thread.join()
    for connection in connections:
        connection.close()

    if errors:
        raise BenchmarkError(f"{endpoint.label}: batch failed: {errors[0]}")
    complete = [record for record in records if record is not None]
    if len(complete) != concurrency:
        raise BenchmarkError(
            f"{endpoint.label}: batch returned {len(complete)}/{concurrency} records"
        )
    start_values = [start for start in starts if start is not None]
    if len(start_values) != concurrency:
        raise BenchmarkError(f"{endpoint.label}: batch start timestamps are incomplete")
    group_start = min(start_values)
    for record in complete:
        record["start_offset_seconds"] = record.pop("_start") - group_start
    return complete


def summarize_records(
    mode: str, records: list[dict[str, Any]], run_summaries: list[dict[str, Any]]
) -> dict[str, Any]:
    ttfts = [
        record["ttft_seconds"]
        for record in records
        if record["ttft_seconds"] is not None
    ]
    walls = [record["wall_seconds"] for record in records]
    completion_tokens = sum(record["completion_tokens"] for record in records)
    summary: dict[str, Any] = {
        "mode": mode,
        "requests": len(records),
        "completion_tokens": completion_tokens,
        "prompt_tokens": sum(record["prompt_tokens"] or 0 for record in records),
        "stream_chunks": sum(record["stream_chunks"] for record in records),
        "token_count_sources": sorted({
            record["token_count_source"] for record in records
        }),
        "ttft_seconds_median": median(ttfts),
        "ttft_seconds_p95": percentile(ttfts, 0.95),
        "wall_seconds_median": median(walls),
        "wall_seconds_p95": percentile(walls, 0.95),
        "wall_seconds_max": max(walls) if walls else None,
    }
    if mode.startswith("batch-"):
        summary["delivered_tokens_per_second"] = median([
            run["delivered_tokens_per_second"]
            for run in run_summaries
            if run["delivered_tokens_per_second"] is not None
        ])
        summary["aggregate_decode_tokens_per_second"] = median([
            run["aggregate_decode_tokens_per_second"]
            for run in run_summaries
            if run["aggregate_decode_tokens_per_second"] is not None
        ])
        summary["per_user_decode_tokens_per_second"] = median([
            record["decode_tokens_per_second"]
            for record in records
            if record["decode_tokens_per_second"] is not None
        ])
    else:
        summary["delivered_tokens_per_second"] = (
            completion_tokens / sum(walls) if sum(walls) > 0 else None
        )
        summary["decode_tokens_per_second"] = median([
            record["decode_tokens_per_second"]
            for record in records
            if record["decode_tokens_per_second"] is not None
        ])
        summary["e2e_tokens_per_second"] = median([
            record["e2e_tokens_per_second"]
            for record in records
            if record["e2e_tokens_per_second"] is not None
        ])
    return summary


def summarize_run(mode: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    walls = [record["wall_seconds"] for record in records]
    completion_tokens = sum(record["completion_tokens"] for record in records)
    summary: dict[str, Any] = {
        "completion_tokens": completion_tokens,
        "wall_seconds_max": max(walls) if walls else None,
    }
    if mode.startswith("batch-"):
        starts = [record["start_offset_seconds"] for record in records]
        ends = [
            record["start_offset_seconds"] + record["wall_seconds"]
            for record in records
        ]
        span = max(ends) - min(starts) if starts else 0.0
        summary["delivered_tokens_per_second"] = (
            completion_tokens / span if span > 0 else None
        )
        last_tokens = [
            record["start_offset_seconds"] + record["last_token_seconds"]
            for record in records
            if record["last_token_seconds"] is not None
        ]
        first_ttfts = [
            record["start_offset_seconds"] + record["ttft_seconds"]
            for record in records
            if record["ttft_seconds"] is not None
        ]
        summary["aggregate_decode_tokens_per_second"] = None
        if last_tokens and first_ttfts:
            last_token = max(last_tokens)
            first_ttft = min(first_ttfts)
            if last_token > first_ttft:
                summary["aggregate_decode_tokens_per_second"] = sum(
                    max(record["completion_tokens"] - 1, 0) for record in records
                ) / (last_token - first_ttft)
        summary["per_user_decode_tokens_per_second"] = median([
            record["decode_tokens_per_second"]
            for record in records
            if record["decode_tokens_per_second"] is not None
        ])
        summary["ttft_seconds_median"] = median([
            record["ttft_seconds"]
            for record in records
            if record["ttft_seconds"] is not None
        ])
    else:
        summary["delivered_tokens_per_second"] = (
            completion_tokens / sum(walls) if sum(walls) > 0 else None
        )
        summary["ttft_seconds_median"] = median([
            record["ttft_seconds"]
            for record in records
            if record["ttft_seconds"] is not None
        ])
    return summary


def agentic_messages(trace: dict[str, Any], index: int) -> list[dict[str, str]]:
    """Build the chat messages for one trace call.

    Each call's message list extends the previous call's list by the frozen
    assistant turn, the frozen tool result, and the next user turn, so the
    previous request text is a strict prefix of the current request text.
    """
    calls = trace["calls"]
    messages = [{"role": "system", "content": trace["system_prompt"]}]
    for call in calls[:index]:
        messages.append({"role": "user", "content": call["user_message"]})
        messages.append({"role": "assistant", "content": call["assistant_history"]})
        messages.append({
            "role": "user",
            "content": "Tool result:\n" + call["tool_result"],
        })
    messages.append({"role": "user", "content": calls[index]["user_message"]})
    return messages


def serialize_messages(messages: list[dict[str, str]]) -> str:
    return "\n".join(f"{message['role']}:{message['content']}" for message in messages)


def common_prefix_length(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def validate_agentic_trace(trace: Any, source: str) -> dict[str, Any]:
    _require(isinstance(trace, dict), f"{source}: trace is not an object")
    _require(
        trace.get("schema_version") == 1,
        f"{source}: trace schema_version is not 1",
    )
    trace_id = trace.get("id")
    _require(
        isinstance(trace_id, str) and bool(trace_id.strip()),
        f"{source}: trace id is invalid",
    )
    system_prompt = trace.get("system_prompt")
    _require(
        isinstance(system_prompt, str) and bool(system_prompt.strip()),
        f"{source}: trace system prompt is invalid",
    )
    calls = trace.get("calls")
    _require(
        isinstance(calls, list) and len(calls) >= 2,
        f"{source}: trace must contain at least two calls",
    )
    previous = ""
    for index, call in enumerate(calls):
        _require(
            isinstance(call, dict), f"{source}: trace call {index} is not an object"
        )
        for name in ("user_message", "assistant_history", "tool_result"):
            value = call.get(name)
            _require(
                isinstance(value, str) and bool(value.strip()),
                f"{source}: trace call {index} {name} is invalid",
            )
        _require(
            _is_int(call.get("max_tokens")) and call["max_tokens"] > 0,
            f"{source}: trace call {index} max_tokens is invalid",
        )
        wait = call.get("tool_wait_seconds")
        _require(
            _is_finite_number(wait) and wait >= 0,
            f"{source}: trace call {index} tool_wait_seconds is invalid",
        )
        serialized = serialize_messages(agentic_messages(trace, index))
        _require(
            serialized.startswith(previous),
            f"{source}: trace call {index} does not extend the previous history",
        )
        previous = serialized
    return trace


def load_agentic_trace(path: Path) -> dict[str, Any]:
    try:
        trace = json.loads(path.read_text())
    except OSError as error:
        raise BenchmarkError(f"cannot read trace {path}: {error}") from error
    except UnicodeError as error:
        raise BenchmarkError(f"trace {path} is not UTF-8: {error}") from error
    except json.JSONDecodeError as error:
        raise BenchmarkError(f"trace {path} is not JSON: {error}") from error
    return validate_agentic_trace(trace, str(path))


def agentic_trace_sha256(trace: dict[str, Any]) -> str:
    return canonical_sha256(trace)


def agentic_trace_workload(
    trace: dict[str, Any], tool_wait_scale: float
) -> dict[str, Any]:
    calls = []
    total_wait = 0.0
    for index, call in enumerate(trace["calls"]):
        scaled_wait = call["tool_wait_seconds"] * tool_wait_scale
        total_wait += scaled_wait
        calls.append({
            "index": index,
            "messages_sha256": hashlib.sha256(
                serialize_messages(agentic_messages(trace, index)).encode()
            ).hexdigest(),
            "max_tokens": call["max_tokens"],
            "tool_wait_seconds": call["tool_wait_seconds"],
            "tool_wait_seconds_scaled": scaled_wait,
        })
    return {
        "trace_id": trace["id"],
        "trace_sha256": agentic_trace_sha256(trace),
        "system_prompt_sha256": hashlib.sha256(
            trace["system_prompt"].encode()
        ).hexdigest(),
        "calls": calls,
        "tool_wait_scale": tool_wait_scale,
        "tool_wait_seconds_total": total_wait,
    }


def _total_or_none(records: list[dict[str, Any]], key: str) -> int | float | None:
    values = [record[key] for record in records]
    if any(value is None for value in values):
        return None
    return sum(values)


def agentic_call_record(
    record: dict[str, Any],
    usage: dict[str, Any] | None,
    agent_index: int,
    call_index: int,
    start_offset: float,
    tool_wait_seconds: float,
    max_tokens: int,
    reused_characters: int,
    prompt_characters: int,
    messages_sha256: str,
) -> dict[str, Any]:
    """Add agentic accounting to a measured chat call.

    Reused-prefix tokens are a client-side apportionment of the endpoint's
    reported prompt_tokens by the exact client-known prefix character fraction.
    They never claim that a server kept those tokens in a cache.
    """
    prompt_tokens = record.get("prompt_tokens")
    fraction = reused_characters / prompt_characters if prompt_characters > 0 else 0.0
    reused_estimate = None
    new_estimate = None
    if prompt_tokens is not None:
        reused_estimate = round(prompt_tokens * fraction)
        new_estimate = prompt_tokens - reused_estimate
    cached_tokens = None
    cached_source = "unreported"
    if isinstance(usage, dict):
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict) and _is_int(details.get("cached_tokens")):
            cached_tokens = details["cached_tokens"]
            cached_source = "usage.prompt_tokens_details.cached_tokens"
    ttft = record["ttft_seconds"]
    last = record["last_token_seconds"]
    completion_tokens = record["completion_tokens"]
    itl = None
    if ttft is not None and last is not None and last > ttft and completion_tokens > 1:
        itl = (last - ttft) / (completion_tokens - 1)
    record.update({
        "agent_index": agent_index,
        "call_index": call_index,
        "messages_sha256": messages_sha256,
        "max_tokens": max_tokens,
        "start_offset_seconds": start_offset,
        "tool_wait_after_seconds": tool_wait_seconds,
        "prompt_characters": prompt_characters,
        "reused_prefix_characters": reused_characters,
        "reused_prefix_fraction": fraction,
        "reused_prefix_tokens_estimate": reused_estimate,
        "new_input_tokens_estimate": new_estimate,
        "server_reported_cached_tokens": cached_tokens,
        "server_reported_cached_tokens_source": cached_source,
        "cache_confirmation": "client-prefix-overlap-only",
        "itl_seconds_mean": itl,
    })
    return record


def _agentic_worker(
    endpoint: Endpoint,
    model: str,
    trace: dict[str, Any],
    agent_index: int,
    tool_wait_scale: float,
    barrier: threading.Barrier,
    results: list[dict[str, Any] | None],
    errors: list[BaseException],
) -> None:
    try:
        barrier.wait(timeout=endpoint.timeout)
    except BaseException as error:  # noqa: BLE001 - fail the whole group.
        errors.append(error)
        barrier.abort()
        return
    calls = trace["calls"]
    task_start = time.perf_counter()
    previous_serialized = ""
    records: list[dict[str, Any]] = []
    total_wait = 0.0
    try:
        for index, call in enumerate(calls):
            messages = agentic_messages(trace, index)
            serialized = serialize_messages(messages)
            reused = common_prefix_length(serialized, previous_serialized)
            previous_serialized = serialized
            wait = call["tool_wait_seconds"] * tool_wait_scale
            connection = open_connection(endpoint)
            call_start = time.perf_counter()
            try:
                record, usage = stream_completion(
                    connection,
                    endpoint,
                    endpoint.api_path("/chat/completions"),
                    {
                        "model": model,
                        "messages": messages,
                        "max_tokens": call["max_tokens"],
                        "temperature": 0,
                        "stream": True,
                        "stream_options": {"include_usage": True},
                    },
                    f"agent {agent_index} call {index}",
                    call_start,
                )
            finally:
                connection.close()
            records.append(
                agentic_call_record(
                    record,
                    usage,
                    agent_index,
                    index,
                    call_start - task_start,
                    wait,
                    call["max_tokens"],
                    reused,
                    len(serialized),
                    hashlib.sha256(serialized.encode()).hexdigest(),
                )
            )
            if wait > 0:
                time.sleep(wait)
            total_wait += wait
        task_end = time.perf_counter()
        results[agent_index] = {
            "agent_index": agent_index,
            "call_count": len(calls),
            "task_seconds": task_end - task_start,
            "start_offset_seconds": 0.0,
            "tool_wait_seconds": total_wait,
            "prompt_tokens": _total_or_none(records, "prompt_tokens"),
            "completion_tokens": _total_or_none(records, "completion_tokens"),
            "reused_prefix_characters": _total_or_none(
                records, "reused_prefix_characters"
            ),
            "reused_prefix_tokens_estimate": _total_or_none(
                records, "reused_prefix_tokens_estimate"
            ),
            "new_input_tokens_estimate": _total_or_none(
                records, "new_input_tokens_estimate"
            ),
            "server_reported_cached_tokens": _total_or_none(
                records, "server_reported_cached_tokens"
            ),
            "calls": records,
            "_task_start": task_start,
        }
    except BaseException as error:  # noqa: BLE001 - report every agent failure.
        errors.append(error)


def run_agentic_group(
    endpoint: Endpoint,
    model: str,
    trace: dict[str, Any],
    agents: int,
    tool_wait_scale: float,
) -> list[dict[str, Any]]:
    """Run ``agents`` closed-loop traces released from one barrier."""
    barrier = threading.Barrier(agents)
    results: list[dict[str, Any] | None] = [None] * agents
    errors: list[BaseException] = []
    threads = [
        threading.Thread(
            target=_agentic_worker,
            args=(
                endpoint,
                model,
                trace,
                agent_index,
                tool_wait_scale,
                barrier,
                results,
                errors,
            ),
            name=f"agent-{agent_index}",
        )
        for agent_index in range(agents)
    ]
    started: list[threading.Thread] = []
    try:
        for thread in threads:
            thread.start()
            started.append(thread)
    except BaseException:
        barrier.abort()
        for thread in started:
            thread.join()
        raise
    for thread in threads:
        thread.join()
    if errors:
        raise BenchmarkError(f"{endpoint.label}: agentic group failed: {errors[0]}")
    tasks = [task for task in results if task is not None]
    if len(tasks) != agents:
        raise BenchmarkError(
            f"{endpoint.label}: agentic group returned {len(tasks)}/{agents} tasks"
        )
    group_start = min(task["_task_start"] for task in tasks)
    for task in tasks:
        start = task.pop("_task_start")
        task["start_offset_seconds"] = start - group_start
    return tasks


def agentic_slo_flags(
    summary: dict[str, Any], slo: dict[str, float]
) -> dict[str, bool]:
    ttft = summary["ttft_seconds_p95"]
    itl = summary["itl_seconds_p95"]
    task = summary["task_seconds_p95"]
    meets_ttft = ttft is not None and ttft <= slo["ttft_p95_seconds"]
    meets_itl = itl is not None and itl <= slo["itl_p95_seconds"]
    meets_task = task is not None and task <= slo["task_p95_seconds"]
    return {
        "meets_ttft_slo": meets_ttft,
        "meets_itl_slo": meets_itl,
        "meets_task_slo": meets_task,
        "meets_all_slo": meets_ttft and meets_itl and meets_task,
    }


def summarize_agentic_run(
    agents: int, tasks: list[dict[str, Any]], slo: dict[str, float]
) -> dict[str, Any]:
    calls = [call for task in tasks for call in task["calls"]]
    starts = [task["start_offset_seconds"] for task in tasks]
    ends = [task["start_offset_seconds"] + task["task_seconds"] for task in tasks]
    span = max(ends) - min(starts) if starts else 0.0
    output_tokens = _total_or_none(calls, "completion_tokens")
    task_seconds = [task["task_seconds"] for task in tasks]
    summary: dict[str, Any] = {
        "agents": agents,
        "tasks": len(tasks),
        "calls": len(calls),
        "task_seconds_median": median(task_seconds),
        "task_seconds_p95": percentile(task_seconds, 0.95),
        "task_seconds_max": max(task_seconds) if task_seconds else None,
        "ttft_seconds_median": median([
            call["ttft_seconds"] for call in calls if call["ttft_seconds"] is not None
        ]),
        "ttft_seconds_p95": percentile(
            [
                call["ttft_seconds"]
                for call in calls
                if call["ttft_seconds"] is not None
            ],
            0.95,
        ),
        "itl_seconds_median": median([
            call["itl_seconds_mean"]
            for call in calls
            if call["itl_seconds_mean"] is not None
        ]),
        "itl_seconds_p95": percentile(
            [
                call["itl_seconds_mean"]
                for call in calls
                if call["itl_seconds_mean"] is not None
            ],
            0.95,
        ),
        "tasks_per_second": len(tasks) / span if span > 0 else None,
        "output_tokens_per_second": (
            output_tokens / span if span > 0 and output_tokens is not None else None
        ),
    }
    summary.update(agentic_slo_flags(summary, slo))
    return summary


def summarize_agentic_width(
    agents: int,
    tasks: list[dict[str, Any]],
    run_summaries: list[dict[str, Any]],
    slo: dict[str, float],
) -> dict[str, Any]:
    calls = [call for task in tasks for call in task["calls"]]
    task_seconds = [task["task_seconds"] for task in tasks]
    ttfts = [call["ttft_seconds"] for call in calls if call["ttft_seconds"] is not None]
    itls = [
        call["itl_seconds_mean"]
        for call in calls
        if call["itl_seconds_mean"] is not None
    ]
    summary: dict[str, Any] = {
        "agents": agents,
        "tasks": len(tasks),
        "calls": len(calls),
        "task_seconds_median": median(task_seconds),
        "task_seconds_p95": percentile(task_seconds, 0.95),
        "task_seconds_max": max(task_seconds) if task_seconds else None,
        "ttft_seconds_median": median(ttfts),
        "ttft_seconds_p95": percentile(ttfts, 0.95),
        "itl_seconds_median": median(itls),
        "itl_seconds_p95": percentile(itls, 0.95),
        "output_tokens": _total_or_none(calls, "completion_tokens"),
        "prompt_tokens": _total_or_none(calls, "prompt_tokens"),
        "reused_prefix_characters": sum(
            call["reused_prefix_characters"] for call in calls
        ),
        "reused_prefix_tokens_estimate": _total_or_none(
            calls, "reused_prefix_tokens_estimate"
        ),
        "new_input_tokens_estimate": _total_or_none(calls, "new_input_tokens_estimate"),
        "server_reported_cached_tokens": _total_or_none(
            calls, "server_reported_cached_tokens"
        ),
        "tasks_per_second": median([
            run["tasks_per_second"]
            for run in run_summaries
            if run["tasks_per_second"] is not None
        ]),
        "output_tokens_per_second": median([
            run["output_tokens_per_second"]
            for run in run_summaries
            if run["output_tokens_per_second"] is not None
        ]),
        "agents_at_slo_per_watt": None,
    }
    summary.update(agentic_slo_flags(summary, slo))
    return summary


def run_agentic_benchmark(
    endpoint: Endpoint,
    trace: dict[str, Any],
    agents: list[int],
    runs: int,
    warmup_runs: int,
    settle_seconds: float,
    tool_wait_scale: float,
    slo: dict[str, float],
    cache_mode: str,
    power: dict[str, Any],
    revision: str | None,
    note: str | None,
) -> dict[str, Any]:
    validate_agentic_trace(trace, "trace")
    if set(slo) != {"ttft_p95_seconds", "itl_p95_seconds", "task_p95_seconds"}:
        raise BenchmarkError(
            "SLO keys must be ttft_p95_seconds, itl_p95_seconds, task_p95_seconds"
        )
    if (
        not agents
        or any(not _is_int(width) or width < 1 for width in agents)
        or len(set(agents)) != len(agents)
    ):
        raise BenchmarkError("agent widths must be unique integers greater than zero")
    if not _is_int(runs) or runs <= 0 or not _is_int(warmup_runs) or warmup_runs < 0:
        raise BenchmarkError("runs must be positive; warmup_runs non-negative")
    if not math.isfinite(settle_seconds) or settle_seconds < 0:
        raise BenchmarkError("settle seconds must be finite and non-negative")
    if not math.isfinite(tool_wait_scale) or tool_wait_scale < 0:
        raise BenchmarkError("tool wait scale must be finite and non-negative")
    for name, value in slo.items():
        if not _is_finite_number(value) or value <= 0:
            raise BenchmarkError(f"SLO {name} must be finite and positive")
    if cache_mode not in CACHE_MODES:
        raise BenchmarkError(f"cache mode must be one of {', '.join(CACHE_MODES)}")
    boundary = power.get("boundary")
    watts = power.get("watts")
    method = power.get("method")
    if boundary not in POWER_BOUNDARIES:
        raise BenchmarkError(
            f"power boundary must be one of {', '.join(POWER_BOUNDARIES)}"
        )
    if watts is not None and (not _is_finite_number(watts) or watts <= 0):
        raise BenchmarkError("power watts must be finite and positive")
    if watts is not None and boundary == "unspecified":
        raise BenchmarkError("--watts requires a concrete --power-boundary")
    if boundary != "unspecified" and watts is None:
        raise BenchmarkError("--power-boundary requires --watts")
    if method is not None and (not isinstance(method, str) or not method.strip()):
        raise BenchmarkError("power method must be a non-empty string")

    workload = agentic_trace_workload(trace, tool_wait_scale)
    workload.update({
        "agents": agents,
        "measured_runs": runs,
        "warmup_runs": warmup_runs,
        "settle_seconds": settle_seconds,
        "temperature": 0,
        "stream": True,
        "cache_mode": cache_mode,
    })
    workload_sha256 = canonical_sha256(workload)
    trace_sha256 = workload["trace_sha256"]
    model, probe = resolve_model(endpoint)

    started = utc_now()
    started_monotonic = time.perf_counter()
    for _ in range(warmup_runs):
        for width in agents:
            run_agentic_group(endpoint, model, trace, width, tool_wait_scale)
    if warmup_runs:
        time.sleep(settle_seconds)

    measured: list[dict[str, Any]] = []
    for run_index in range(runs):
        for width in agents:
            tasks = run_agentic_group(endpoint, model, trace, width, tool_wait_scale)
            measured.append({
                "agents": width,
                "index": run_index,
                "summary": summarize_agentic_run(width, tasks, slo),
                "tasks": tasks,
            })
            time.sleep(settle_seconds)

    curve: list[dict[str, Any]] = []
    for width in agents:
        width_tasks = [
            task for run in measured if run["agents"] == width for task in run["tasks"]
        ]
        width_runs = [run["summary"] for run in measured if run["agents"] == width]
        point = summarize_agentic_width(width, width_tasks, width_runs, slo)
        if watts is not None and point["meets_all_slo"]:
            point["agents_at_slo_per_watt"] = width / watts
        curve.append(point)

    meeting = [point["agents"] for point in curve if point["meets_all_slo"]]
    max_agents = max(meeting) if meeting else None
    throughput_points = [
        point for point in curve if point["tasks_per_second"] is not None
    ]
    saturated = (
        max(throughput_points, key=lambda point: point["tasks_per_second"])["agents"]
        if throughput_points
        else None
    )
    slo_point = (
        next((point for point in curve if point["agents"] == max_agents), None)
        if max_agents is not None
        else None
    )
    power_record = {
        "boundary": boundary,
        "watts": watts,
        "method": method,
        "agents_at_slo_per_watt": (
            max_agents / watts if max_agents is not None and watts is not None else None
        ),
        "output_tokens_per_second_per_watt": (
            slo_point["output_tokens_per_second"] / watts
            if slo_point is not None
            and watts is not None
            and slo_point["output_tokens_per_second"] is not None
            else None
        ),
    }

    finished = utc_now()
    return {
        "schema": AGENTIC_SCHEMA,
        "generated_utc": finished,
        "started_utc": started,
        "duration_seconds": time.perf_counter() - started_monotonic,
        "note": note,
        "workload": workload,
        "workload_sha256": workload_sha256,
        "trace": trace,
        "trace_sha256": trace_sha256,
        "target": {
            "label": endpoint.label,
            "base_url": endpoint.base_url,
            "model": model,
            "probe": probe,
        },
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "source_revision": source_revision(revision),
            "selected_environment": selected_environment(),
        },
        "conditions": {
            "slo": slo,
            "cache_mode": cache_mode,
            "power": power_record,
            "temperature": 0,
            "stream": True,
            "request_timeout_seconds": endpoint.timeout,
        },
        "summary": {
            "slo": slo,
            "max_agents_meeting_all_slo": max_agents,
            "saturated_agents": saturated,
            "curve": curve,
        },
        "runs": measured,
    }


def validate_agentic_result(result: Any, source: str = "result") -> dict[str, Any]:
    _require(isinstance(result, dict), f"{source}: result is not an object")
    _require(
        result.get("schema") == AGENTIC_SCHEMA,
        f"{source}: schema is not {AGENTIC_SCHEMA}",
    )
    target = result.get("target")
    _require(
        isinstance(target, dict)
        and isinstance(target.get("label"), str)
        and bool(target["label"]),
        f"{source}: target label is missing",
    )
    _require(
        isinstance(target.get("base_url"), str) and bool(target["base_url"]),
        f"{source}: target base_url is missing",
    )
    _require(
        isinstance(target.get("model"), str) and bool(target["model"]),
        f"{source}: target model is missing",
    )
    _require(
        isinstance(result.get("environment"), dict),
        f"{source}: environment is not an object",
    )

    trace = validate_agentic_trace(result.get("trace"), source)
    trace_sha256 = result.get("trace_sha256")
    _require(
        isinstance(trace_sha256, str)
        and len(trace_sha256) == 64
        and canonical_sha256(trace) == trace_sha256,
        f"{source}: trace digest does not match the trace",
    )
    workload = result.get("workload")
    _require(isinstance(workload, dict), f"{source}: workload is not an object")
    _require(
        isinstance(result.get("workload_sha256"), str)
        and canonical_sha256(workload) == result["workload_sha256"],
        f"{source}: workload digest does not match its workload",
    )
    _require(
        workload.get("trace_sha256") == trace_sha256,
        f"{source}: workload trace digest does not match the trace",
    )
    _require(
        workload.get("trace_id") == trace["id"],
        f"{source}: workload trace id does not match the trace",
    )
    agents = workload.get("agents")
    _require(
        isinstance(agents, list)
        and bool(agents)
        and all(_is_int(width) and width > 0 for width in agents)
        and len(set(agents)) == len(agents),
        f"{source}: workload agents are invalid",
    )
    measured_runs = workload.get("measured_runs")
    _require(
        _is_int(measured_runs) and measured_runs > 0,
        f"{source}: workload measured_runs is invalid",
    )
    _require(
        _is_int(workload.get("warmup_runs")) and workload["warmup_runs"] >= 0,
        f"{source}: workload warmup_runs is invalid",
    )
    _require(
        _is_finite_number(workload.get("settle_seconds"))
        and workload["settle_seconds"] >= 0,
        f"{source}: workload settle_seconds is invalid",
    )
    tool_wait_scale = workload.get("tool_wait_scale")
    _require(
        _is_finite_number(tool_wait_scale) and tool_wait_scale >= 0,
        f"{source}: workload tool_wait_scale is invalid",
    )
    _require(
        workload.get("temperature") == 0,
        f"{source}: workload temperature is not greedy",
    )
    _require(
        workload.get("stream") is True,
        f"{source}: workload stream flag is not true",
    )
    cache_mode = workload.get("cache_mode")
    _require(
        cache_mode in CACHE_MODES,
        f"{source}: workload cache mode is invalid",
    )
    expected_calls = agentic_trace_workload(trace, tool_wait_scale)["calls"]
    _require(
        workload.get("calls") == expected_calls,
        f"{source}: workload calls do not match the trace",
    )
    _require(
        _summary_value_matches(
            sum(call["tool_wait_seconds_scaled"] for call in expected_calls),
            workload.get("tool_wait_seconds_total"),
        ),
        f"{source}: workload tool wait total is invalid",
    )
    _require(
        workload.get("system_prompt_sha256")
        == hashlib.sha256(trace["system_prompt"].encode()).hexdigest(),
        f"{source}: workload system prompt digest is invalid",
    )

    conditions = result.get("conditions")
    _require(isinstance(conditions, dict), f"{source}: conditions are invalid")
    slo = conditions.get("slo")
    _require(isinstance(slo, dict), f"{source}: SLO is invalid")
    _require(
        set(slo) == {"ttft_p95_seconds", "itl_p95_seconds", "task_p95_seconds"},
        f"{source}: SLO keys are invalid",
    )
    for name, value in slo.items():
        _require(
            _is_finite_number(value) and value > 0,
            f"{source}: SLO {name} is invalid",
        )
    _require(
        conditions.get("cache_mode") == cache_mode,
        f"{source}: condition cache mode does not match the workload",
    )
    _require(
        conditions.get("temperature") == 0 and conditions.get("stream") is True,
        f"{source}: conditions are not greedy streaming",
    )
    _require(
        _is_finite_number(conditions.get("request_timeout_seconds"))
        and conditions["request_timeout_seconds"] > 0,
        f"{source}: request timeout condition is invalid",
    )
    power = conditions.get("power")
    _require(isinstance(power, dict), f"{source}: power conditions are invalid")
    boundary = power.get("boundary")
    watts = power.get("watts")
    _require(
        boundary in POWER_BOUNDARIES,
        f"{source}: power boundary is invalid",
    )
    _require(
        watts is None or (_is_finite_number(watts) and watts > 0),
        f"{source}: power watts are invalid",
    )
    _require(
        (watts is None) == (boundary == "unspecified"),
        f"{source}: power boundary and watts are inconsistent",
    )

    summary = result.get("summary")
    _require(isinstance(summary, dict), f"{source}: summary is invalid")
    _require(summary.get("slo") == slo, f"{source}: summary SLO mismatch")
    curve = summary.get("curve")
    _require(
        isinstance(curve, list) and len(curve) == len(agents),
        f"{source}: summary curve does not match the agent widths",
    )
    _require(
        [point.get("agents") for point in curve] == agents,
        f"{source}: summary curve is not in workload order",
    )

    runs = result.get("runs")
    _require(
        isinstance(runs, list) and len(runs) == measured_runs * len(agents),
        f"{source}: run count does not match the workload",
    )
    tasks_by_width: dict[int, list[dict[str, Any]]] = {width: [] for width in agents}
    run_summaries_by_width: dict[int, list[dict[str, Any]]] = {
        width: [] for width in agents
    }
    seen: set[tuple[int, int]] = set()
    for run in runs:
        _require(isinstance(run, dict), f"{source}: run is not an object")
        width = run.get("agents")
        index = run.get("index")
        _require(width in agents, f"{source}: run agent width is not in the workload")
        _require(
            _is_int(index) and 0 <= index < measured_runs,
            f"{source}: run index is out of range",
        )
        key = (width, index)
        _require(
            key not in seen, f"{source}: duplicate run agents {width} index {index}"
        )
        seen.add(key)
        tasks = run.get("tasks")
        _require(
            isinstance(tasks, list) and len(tasks) == width,
            f"{source}: run {width} index {index} has the wrong task count",
        )
        _require(
            {task.get("agent_index") for task in tasks} == set(range(width)),
            f"{source}: run {width} index {index} did not cover every agent",
        )
        for task in tasks:
            _validate_agentic_task(task, trace, source, width, tool_wait_scale)
            tasks_by_width[width].append(task)
        recomputed = summarize_agentic_run(width, tasks, slo)
        _require(
            run.get("summary") == recomputed,
            f"{source}: run {width} index {index} summary does not match its tasks",
        )
        run_summaries_by_width[width].append(recomputed)

    for point in curve:
        _require(isinstance(point, dict), f"{source}: curve point is not an object")
        width = point.get("agents")
        _require(width in agents, f"{source}: curve agent width is not in the workload")
        recomputed = summarize_agentic_width(
            width,
            tasks_by_width[width],
            run_summaries_by_width[width],
            slo,
        )
        if watts is not None and recomputed["meets_all_slo"]:
            recomputed["agents_at_slo_per_watt"] = width / watts
        _require(
            set(point) == set(recomputed),
            f"{source}: curve point keys do not match its runs",
        )
        for metric, expected_value in recomputed.items():
            _require(
                _summary_value_matches(expected_value, point[metric]),
                f"{source}: curve agent {width} metric {metric} does not match its runs",
            )

    meeting = [point["agents"] for point in curve if point["meets_all_slo"]]
    expected_max = max(meeting) if meeting else None
    _require(
        summary.get("max_agents_meeting_all_slo") == expected_max,
        f"{source}: maximum agents at SLO does not match the curve",
    )
    throughput_points = [
        point for point in curve if point["tasks_per_second"] is not None
    ]
    expected_saturated = (
        max(throughput_points, key=lambda point: point["tasks_per_second"])["agents"]
        if throughput_points
        else None
    )
    _require(
        summary.get("saturated_agents") == expected_saturated,
        f"{source}: saturated agents do not match the curve",
    )

    slo_point = (
        next((point for point in curve if point["agents"] == expected_max), None)
        if expected_max is not None
        else None
    )
    expected_power = {
        "boundary": boundary,
        "watts": watts,
        "method": power.get("method"),
        "agents_at_slo_per_watt": (
            expected_max / watts
            if expected_max is not None and watts is not None
            else None
        ),
        "output_tokens_per_second_per_watt": (
            slo_point["output_tokens_per_second"] / watts
            if slo_point is not None
            and watts is not None
            and slo_point["output_tokens_per_second"] is not None
            else None
        ),
    }
    _require(
        set(power) == set(expected_power),
        f"{source}: power condition keys are invalid",
    )
    for key, expected_value in expected_power.items():
        _require(
            _summary_value_matches(expected_value, power.get(key)),
            f"{source}: power condition {key} does not match the curve",
        )
    return result


def _validate_agentic_task(
    task: Any,
    trace: dict[str, Any],
    source: str,
    agents: int,
    tool_wait_scale: float,
) -> None:
    _require(isinstance(task, dict), f"{source}: task is not an object")
    agent_index = task.get("agent_index")
    _require(
        _is_int(agent_index) and 0 <= agent_index < agents,
        f"{source}: task agent_index is invalid",
    )
    call_count = len(trace["calls"])
    _require(
        task.get("call_count") == call_count,
        f"{source}: task call_count does not match the trace",
    )
    task_seconds = task.get("task_seconds")
    _require(
        _is_finite_number(task_seconds) and task_seconds > 0,
        f"{source}: task_seconds is invalid",
    )
    start_offset = task.get("start_offset_seconds")
    _require(
        _is_finite_number(start_offset) and start_offset >= 0,
        f"{source}: task start_offset_seconds is invalid",
    )
    scaled_waits = [
        call["tool_wait_seconds"] * tool_wait_scale for call in trace["calls"]
    ]
    _require(
        _summary_value_matches(sum(scaled_waits), task.get("tool_wait_seconds")),
        f"{source}: task tool_wait_seconds does not match the trace",
    )
    calls = task.get("calls")
    _require(
        isinstance(calls, list) and len(calls) == call_count,
        f"{source}: task calls do not match the trace",
    )
    previous_serialized = ""
    for index, call in enumerate(calls):
        _require(isinstance(call, dict), f"{source}: call is not an object")
        _require(
            call.get("agent_index") == agent_index,
            f"{source}: call agent_index does not match its task",
        )
        _require(
            call.get("call_index") == index,
            f"{source}: call_index does not match its position",
        )
        _require(
            call.get("max_tokens") == trace["calls"][index]["max_tokens"],
            f"{source}: call max_tokens does not match the trace",
        )
        _require(
            _summary_value_matches(
                scaled_waits[index], call.get("tool_wait_after_seconds")
            ),
            f"{source}: call tool wait does not match the trace",
        )
        serialized = serialize_messages(agentic_messages(trace, index))
        expected_reused = common_prefix_length(serialized, previous_serialized)
        previous_serialized = serialized
        _require(
            call.get("prompt_characters") == len(serialized),
            f"{source}: call prompt_characters does not match the trace",
        )
        _require(
            call.get("reused_prefix_characters") == expected_reused,
            f"{source}: call reused_prefix_characters does not match the trace",
        )
        _require(
            call.get("messages_sha256")
            == hashlib.sha256(serialized.encode()).hexdigest(),
            f"{source}: call messages_sha256 does not match the trace",
        )
        _validate_agentic_call(call, source)
        _require(
            call["start_offset_seconds"] <= task_seconds,
            f"{source}: call start offset exceeds its task time",
        )
    _require(
        _summary_value_matches(
            _total_or_none(calls, "prompt_tokens"), task.get("prompt_tokens")
        ),
        f"{source}: task prompt_tokens does not match its calls",
    )
    for key in (
        "completion_tokens",
        "reused_prefix_characters",
        "reused_prefix_tokens_estimate",
        "new_input_tokens_estimate",
        "server_reported_cached_tokens",
    ):
        _require(
            _summary_value_matches(_total_or_none(calls, key), task.get(key)),
            f"{source}: task {key} does not match its calls",
        )


def _validate_agentic_call(call: dict[str, Any], source: str) -> None:
    completion_tokens = call.get("completion_tokens")
    _require(
        _is_int(completion_tokens) and completion_tokens > 0,
        f"{source}: call completion_tokens is invalid",
    )
    prompt_tokens = call.get("prompt_tokens")
    _require(
        prompt_tokens is None or (_is_int(prompt_tokens) and prompt_tokens >= 0),
        f"{source}: call prompt_tokens is invalid",
    )
    prompt_characters = call.get("prompt_characters")
    _require(
        _is_int(prompt_characters) and prompt_characters > 0,
        f"{source}: call prompt_characters is invalid",
    )
    reused = call.get("reused_prefix_characters")
    _require(
        _is_int(reused) and 0 <= reused <= prompt_characters,
        f"{source}: call reused_prefix_characters is invalid",
    )
    fraction = call.get("reused_prefix_fraction")
    _require(
        _is_finite_number(fraction)
        and math.isclose(
            fraction, reused / prompt_characters, rel_tol=1e-9, abs_tol=1e-9
        ),
        f"{source}: call reused_prefix_fraction is invalid",
    )
    reused_estimate = call.get("reused_prefix_tokens_estimate")
    new_estimate = call.get("new_input_tokens_estimate")
    if prompt_tokens is None:
        _require(
            reused_estimate is None and new_estimate is None,
            f"{source}: call token estimates exist without prompt_tokens",
        )
    else:
        _require(
            _is_int(reused_estimate)
            and _is_int(new_estimate)
            and reused_estimate + new_estimate == prompt_tokens
            and reused_estimate == round(prompt_tokens * reused / prompt_characters),
            f"{source}: call token estimates are invalid",
        )
    cached = call.get("server_reported_cached_tokens")
    _require(
        cached is None or (_is_int(cached) and cached >= 0),
        f"{source}: call server_reported_cached_tokens is invalid",
    )
    cached_source = call.get("server_reported_cached_tokens_source")
    if cached is None:
        _require(
            cached_source == "unreported",
            f"{source}: call cached token source is invalid",
        )
    else:
        _require(
            cached_source == "usage.prompt_tokens_details.cached_tokens",
            f"{source}: call cached token source is invalid",
        )
    _require(
        call.get("cache_confirmation") == "client-prefix-overlap-only",
        f"{source}: call cache confirmation is invalid",
    )
    ttft = call.get("ttft_seconds")
    last = call.get("last_token_seconds")
    wall = call.get("wall_seconds")
    _require(
        _is_finite_number(wall) and wall > 0,
        f"{source}: call wall_seconds is invalid",
    )
    for name in ("ttft_seconds", "last_token_seconds"):
        value = call.get(name)
        _require(
            _is_finite_number(value) and 0 <= value <= wall,
            f"{source}: call {name} is invalid",
        )
    _require(last >= ttft, f"{source}: call token times are inverted")
    expected_itl = (
        (last - ttft) / (completion_tokens - 1)
        if last > ttft and completion_tokens > 1
        else None
    )
    _require(
        _summary_value_matches(expected_itl, call.get("itl_seconds_mean")),
        f"{source}: call itl_seconds_mean does not match its timings",
    )
    expected_decode = expected_itl and 1 / expected_itl
    _require(
        _summary_value_matches(expected_decode, call.get("decode_tokens_per_second")),
        f"{source}: call decode_tokens_per_second does not match its timings",
    )
    _require(
        call.get("token_count_source") == "usage",
        f"{source}: call token_count_source is not usage",
    )
    text_hash = call.get("text_sha256")
    _require(
        isinstance(text_hash, str)
        and len(text_hash) == 64
        and all(character in "0123456789abcdef" for character in text_hash),
        f"{source}: call text_sha256 is not a SHA-256",
    )
    _require(
        _is_int(call.get("text_characters")) and call["text_characters"] > 0,
        f"{source}: call text_characters is invalid",
    )
    _require(
        _is_int(call.get("stream_chunks")) and call["stream_chunks"] >= 0,
        f"{source}: call stream_chunks is invalid",
    )
    _require(
        isinstance(call.get("finish_reason_seen"), bool),
        f"{source}: call finish_reason_seen is invalid",
    )
    start_offset = call.get("start_offset_seconds")
    _require(
        _is_finite_number(start_offset) and start_offset >= 0,
        f"{source}: call start_offset_seconds is invalid",
    )


def load_agentic_result(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text())
    except OSError as error:
        raise BenchmarkError(f"cannot read {path}: {error}") from error
    except UnicodeError as error:
        raise BenchmarkError(f"{path} is not UTF-8: {error}") from error
    except json.JSONDecodeError as error:
        raise BenchmarkError(f"{path} is not JSON: {error}") from error
    return validate_agentic_result(result, str(path))


def parse_agent_widths(value: str) -> list[int]:
    widths = []
    for item in value.split(","):
        try:
            width = int(item)
        except ValueError as error:
            raise BenchmarkError(f"invalid agent width {item!r}") from error
        if width < 1:
            raise BenchmarkError("agent widths must be at least one")
        if width not in widths:
            widths.append(width)
    if not widths:
        raise BenchmarkError("at least one agent width is required")
    return widths


def load_corpus(path: Path) -> dict[str, Any]:
    try:
        corpus = json.loads(path.read_text())
    except OSError as error:
        raise BenchmarkError(f"cannot read corpus {path}: {error}") from error
    except UnicodeError as error:
        raise BenchmarkError(f"corpus {path} is not UTF-8: {error}") from error
    except json.JSONDecodeError as error:
        raise BenchmarkError(f"corpus {path} is not JSON: {error}") from error
    if not isinstance(corpus, dict):
        raise BenchmarkError(f"corpus {path} must be an object")
    prompts = corpus.get("prompts")
    if (
        not isinstance(prompts, list)
        or not prompts
        or not all(isinstance(prompt, str) and prompt.strip() for prompt in prompts)
    ):
        raise BenchmarkError(f"corpus {path} must contain non-empty prompts")
    return corpus


def redact_path(value: str) -> str:
    if value.startswith("/") and not value.startswith("/nix/store/"):
        return "<absolute-path>"
    return value


def selected_environment() -> dict[str, str]:
    selected = {}
    for key in ENVIRONMENT_KEYS:
        value = os.environ.get(key)
        if not value:
            continue
        selected[key] = redact_path(value)
    return selected


def source_revision(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    configured = os.environ.get("QWEN38_BENCH_SOURCE_REVISION")
    if configured:
        return configured
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path.cwd(),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def run_benchmark(
    endpoint: Endpoint,
    corpus: dict[str, Any],
    max_tokens: int,
    concurrency: list[int],
    runs: int,
    warmup_runs: int,
    stream: bool,
    settle_seconds: float,
    revision: str | None,
    note: str | None,
) -> dict[str, Any]:
    if (
        not _is_int(max_tokens)
        or not _is_int(runs)
        or not _is_int(warmup_runs)
        or max_tokens <= 0
        or runs <= 0
        or warmup_runs < 0
    ):
        raise BenchmarkError(
            "max_tokens and runs must be positive integers; warmup_runs non-negative"
        )
    if (
        not concurrency
        or any(not _is_int(width) or width <= 1 for width in concurrency)
        or len(set(concurrency)) != len(concurrency)
    ):
        raise BenchmarkError(
            "concurrency widths must be unique integers greater than one"
        )
    if not math.isfinite(settle_seconds) or settle_seconds < 0:
        raise BenchmarkError("settle seconds must be finite and non-negative")
    prompts = corpus.get("prompts")
    if (
        not isinstance(prompts, list)
        or not prompts
        or not all(isinstance(prompt, str) and prompt.strip() for prompt in prompts)
    ):
        raise BenchmarkError("corpus prompts are invalid")
    corpus_id = corpus.get("id", "unnamed")
    if not isinstance(corpus_id, str) or not corpus_id.strip():
        raise BenchmarkError("corpus id must be a non-empty string")
    model, probe = resolve_model(endpoint)
    workload = {
        "corpus_id": corpus_id,
        "corpus_chunk_sha256": canonical_sha256(prompts),
        "prompts": prompts,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
        "warmup_runs": warmup_runs,
        "measured_runs": runs,
        "settle_seconds": settle_seconds,
        "concurrency": concurrency,
        "modes": ["single"] + [f"batch-{width}" for width in concurrency],
    }
    workload_sha256 = canonical_sha256(workload)

    started = utc_now()
    started_monotonic = time.perf_counter()
    for _ in range(warmup_runs):
        run_single(endpoint, model, prompts, max_tokens, stream)
        for width in concurrency:
            run_batch_with_offsets(endpoint, model, prompts, width, max_tokens, stream)
    if warmup_runs:
        time.sleep(settle_seconds)

    measured: list[dict[str, Any]] = []
    run_summaries: dict[str, list[dict[str, Any]]] = {
        "single": [],
        **{f"batch-{width}": [] for width in concurrency},
    }
    for run_index in range(runs):
        records = run_single(endpoint, model, prompts, max_tokens, stream)
        mode = "single"
        summary = summarize_run(mode, records)
        measured.append({
            "mode": mode,
            "index": run_index,
            "summary": summary,
            "records": records,
        })
        run_summaries[mode].append(summary)
        time.sleep(settle_seconds)
        for width in concurrency:
            mode = f"batch-{width}"
            records = run_batch_with_offsets(
                endpoint, model, prompts, width, max_tokens, stream
            )
            summary = summarize_run(mode, records)
            measured.append({
                "mode": mode,
                "index": run_index,
                "summary": summary,
                "records": records,
            })
            run_summaries[mode].append(summary)
            time.sleep(settle_seconds)

    summaries = {
        mode: summarize_records(
            mode,
            [
                record
                for run in measured
                if run["mode"] == mode
                for record in run["records"]
            ],
            run_summaries[mode],
        )
        for mode in run_summaries
    }
    finished = utc_now()
    return {
        "schema": SCHEMA,
        "generated_utc": finished,
        "started_utc": started,
        "duration_seconds": time.perf_counter() - started_monotonic,
        "note": note,
        "workload": workload,
        "workload_sha256": workload_sha256,
        "target": {
            "label": endpoint.label,
            "base_url": endpoint.base_url,
            "model": model,
            "probe": probe,
        },
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "source_revision": source_revision(revision),
            "selected_environment": selected_environment(),
        },
        "summary": summaries,
        "runs": measured,
    }


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BenchmarkError(message)


def _summary_value_matches(expected: Any, stored: Any) -> bool:
    if expected is None or stored is None:
        return expected is stored
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return _is_finite_number(stored) and math.isclose(
            stored, expected, rel_tol=1e-9, abs_tol=1e-9
        )
    return stored == expected


def _validate_record(record: Any, source: str, prompts: list[str]) -> None:
    _require(isinstance(record, dict), f"{source}: run record is not an object")
    prompt_index = record.get("prompt_index")
    _require(
        _is_int(prompt_index) and 0 <= prompt_index < len(prompts),
        f"{source}: record prompt_index is out of range",
    )
    prompt_hash = hashlib.sha256(prompts[prompt_index].encode()).hexdigest()
    _require(
        record.get("prompt_sha256") == prompt_hash,
        f"{source}: record prompt_sha256 does not match the corpus",
    )
    text_hash = record.get("text_sha256")
    _require(
        isinstance(text_hash, str)
        and len(text_hash) == 64
        and all(character in "0123456789abcdef" for character in text_hash),
        f"{source}: record text_sha256 is not a SHA-256",
    )
    completion_tokens = record.get("completion_tokens")
    _require(
        _is_int(completion_tokens) and completion_tokens > 0,
        f"{source}: record completion_tokens is invalid",
    )
    text_characters = record.get("text_characters")
    _require(
        _is_int(text_characters) and text_characters > 0,
        f"{source}: record text_characters is invalid",
    )
    start_offset = record.get("start_offset_seconds")
    _require(
        _is_finite_number(start_offset) and start_offset >= 0,
        f"{source}: record start_offset_seconds is invalid",
    )
    wall = record.get("wall_seconds")
    _require(
        _is_finite_number(wall) and wall >= 0,
        f"{source}: record wall_seconds is invalid",
    )
    if completion_tokens:
        _require(wall > 0, f"{source}: record wall_seconds is not positive")
    for name in ("ttft_seconds", "last_token_seconds"):
        value = record.get(name)
        if value is not None:
            _require(
                _is_finite_number(value) and value >= 0 and value <= wall,
                f"{source}: record {name} is invalid",
            )
    ttft = record.get("ttft_seconds")
    last_token = record.get("last_token_seconds")
    if ttft is not None and last_token is not None:
        _require(last_token >= ttft, f"{source}: record token times are inverted")

    expected_decode = None
    if (
        ttft is not None
        and last_token is not None
        and last_token > ttft
        and completion_tokens > 1
    ):
        expected_decode = (completion_tokens - 1) / (last_token - ttft)
    _require(
        _summary_value_matches(expected_decode, record.get("decode_tokens_per_second")),
        f"{source}: record decode_tokens_per_second does not match its timings",
    )
    expected_e2e = completion_tokens / wall if wall > 0 else None
    _require(
        _summary_value_matches(expected_e2e, record.get("e2e_tokens_per_second")),
        f"{source}: record e2e_tokens_per_second does not match its timings",
    )

    prompt_tokens = record.get("prompt_tokens")
    _require(
        prompt_tokens is None or (_is_int(prompt_tokens) and prompt_tokens >= 0),
        f"{source}: record prompt_tokens is invalid",
    )
    stream_chunks = record.get("stream_chunks")
    _require(
        _is_int(stream_chunks) and stream_chunks >= 0,
        f"{source}: record stream_chunks is invalid",
    )
    _require(
        record.get("token_count_source") == "usage",
        f"{source}: record token_count_source is not usage",
    )
    if "finish_reason_seen" in record:
        _require(
            isinstance(record["finish_reason_seen"], bool),
            f"{source}: record finish_reason_seen is invalid",
        )


def validate_result(result: Any, source: str = "result") -> dict[str, Any]:
    _require(isinstance(result, dict), f"{source}: result is not an object")
    _require(result.get("schema") == SCHEMA, f"{source}: schema is not {SCHEMA}")
    target = result.get("target")
    _require(
        isinstance(target, dict)
        and isinstance(target.get("label"), str)
        and bool(target["label"]),
        f"{source}: target label is missing",
    )
    _require(
        isinstance(target.get("base_url"), str) and bool(target["base_url"]),
        f"{source}: target base_url is missing",
    )
    _require(
        isinstance(target.get("model"), str) and bool(target["model"]),
        f"{source}: target model is missing",
    )
    _require(
        isinstance(result.get("environment"), dict),
        f"{source}: environment is not an object",
    )
    workload = result.get("workload")
    _require(isinstance(workload, dict), f"{source}: workload is not an object")
    digest = result.get("workload_sha256")
    _require(
        isinstance(digest, str)
        and len(digest) == 64
        and canonical_sha256(workload) == digest,
        f"{source}: workload digest does not match its workload",
    )
    _require(
        isinstance(workload.get("corpus_id"), str)
        and bool(workload["corpus_id"].strip()),
        f"{source}: workload corpus id is invalid",
    )
    prompts = workload.get("prompts")
    _require(
        isinstance(prompts, list)
        and bool(prompts)
        and all(isinstance(prompt, str) and bool(prompt.strip()) for prompt in prompts),
        f"{source}: workload prompts are invalid",
    )
    measured_runs = workload.get("measured_runs")
    _require(
        _is_int(measured_runs) and measured_runs > 0,
        f"{source}: workload measured_runs is invalid",
    )
    _require(
        _is_int(workload.get("warmup_runs")) and workload["warmup_runs"] >= 0,
        f"{source}: workload warmup_runs is invalid",
    )
    _require(
        _is_int(workload.get("max_tokens")) and workload["max_tokens"] > 0,
        f"{source}: workload max_tokens is invalid",
    )
    _require(
        workload.get("corpus_chunk_sha256") == canonical_sha256(prompts),
        f"{source}: workload corpus digest does not match its prompts",
    )
    _require(
        workload.get("temperature") == 0,
        f"{source}: workload temperature is not greedy",
    )
    _require(
        isinstance(workload.get("stream"), bool),
        f"{source}: workload stream flag is invalid",
    )
    _require(
        _is_finite_number(workload.get("settle_seconds"))
        and workload["settle_seconds"] >= 0,
        f"{source}: workload settle_seconds is invalid",
    )
    concurrency = workload.get("concurrency")
    _require(
        isinstance(concurrency, list)
        and bool(concurrency)
        and all(_is_int(width) and width > 1 for width in concurrency)
        and len(set(concurrency)) == len(concurrency),
        f"{source}: workload concurrency is invalid",
    )
    expected_modes = ["single"] + [f"batch-{width}" for width in concurrency]
    _require(
        workload.get("modes") == expected_modes,
        f"{source}: workload modes do not match its concurrency",
    )

    summary = result.get("summary")
    _require(
        isinstance(summary, dict) and set(summary) == set(expected_modes),
        f"{source}: summary modes do not match the workload",
    )
    runs = result.get("runs")
    _require(
        isinstance(runs, list) and len(runs) == measured_runs * len(expected_modes),
        f"{source}: run count does not match the workload",
    )
    records_by_mode: dict[str, list[dict[str, Any]]] = {
        mode: [] for mode in expected_modes
    }
    run_summaries: dict[str, list[dict[str, Any]]] = {
        mode: [] for mode in expected_modes
    }
    seen: set[tuple[str, int]] = set()
    for run in runs:
        _require(isinstance(run, dict), f"{source}: run is not an object")
        mode = run.get("mode")
        index = run.get("index")
        _require(
            mode in expected_modes,
            f"{source}: run mode is not in the workload",
        )
        _require(
            _is_int(index) and 0 <= index < measured_runs,
            f"{source}: run index is out of range",
        )
        key = (mode, index)
        _require(key not in seen, f"{source}: duplicate run {mode} index {index}")
        seen.add(key)
        expected_records = (
            len(prompts) if mode == "single" else int(mode.removeprefix("batch-"))
        )
        records = run.get("records")
        _require(
            isinstance(records, list) and len(records) == expected_records,
            f"{source}: run {mode} index {index} has the wrong record count",
        )
        for record in records:
            _validate_record(record, source, prompts)
            records_by_mode[mode].append(record)
        if mode == "single":
            _require(
                len({record["prompt_index"] for record in records}) == len(prompts),
                f"{source}: a single-stream run did not cover every prompt",
            )
        run_summaries[mode].append(summarize_run(mode, records))

    for mode in expected_modes:
        recomputed = summarize_records(mode, records_by_mode[mode], run_summaries[mode])
        stored = summary[mode]
        _require(
            isinstance(stored, dict) and set(stored) == set(recomputed),
            f"{source}: {mode} summary keys do not match its runs",
        )
        for metric, expected_value in recomputed.items():
            _require(
                _summary_value_matches(expected_value, stored[metric]),
                f"{source}: {mode} summary metric {metric} does not match its runs",
            )
    return result


def load_result(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text())
    except OSError as error:
        raise BenchmarkError(f"cannot read {path}: {error}") from error
    except UnicodeError as error:
        raise BenchmarkError(f"{path} is not UTF-8: {error}") from error
    except json.JSONDecodeError as error:
        raise BenchmarkError(f"{path} is not JSON: {error}") from error
    return validate_result(result, str(path))


def text_hashes(result: dict[str, Any]) -> dict[tuple[str, int], set[str]]:
    hashes: dict[tuple[str, int], set[str]] = {}
    for run in result.get("runs", []):
        if not isinstance(run, dict):
            continue
        mode = run.get("mode")
        for record in run.get("records", []):
            if not isinstance(record, dict):
                continue
            key = (str(mode), int(record["prompt_index"]))
            hashes.setdefault(key, set()).add(str(record["text_sha256"]))
    return hashes


def compare_results(
    results: list[dict[str, Any]], baseline_label: str | None = None
) -> dict[str, Any]:
    if len(results) < 2:
        raise BenchmarkError("compare needs at least two result files")
    results = [validate_result(result, "comparison input") for result in results]
    labels = [result["target"]["label"] for result in results]
    if len(set(labels)) != len(labels):
        raise BenchmarkError(f"duplicate target labels: {labels}")
    baseline_label = baseline_label or labels[0]
    if baseline_label not in labels:
        raise BenchmarkError(f"baseline label {baseline_label!r} is not in {labels}")
    ordered = [results[labels.index(baseline_label)]] + [
        result for result in results if result["target"]["label"] != baseline_label
    ]
    workload_hash = ordered[0]["workload_sha256"]
    for result in ordered[1:]:
        if result["workload_sha256"] != workload_hash:
            raise BenchmarkError(
                f"{result['target']['label']}: workload differs from {baseline_label}"
            )

    baseline = ordered[0]
    baseline_hashes = text_hashes(baseline)
    parity_mismatches: list[dict[str, Any]] = []
    deterministic = True
    for key, hashes in baseline_hashes.items():
        if len(hashes) != 1:
            deterministic = False
    for result in ordered[1:]:
        observed = text_hashes(result)
        for key in sorted(set(baseline_hashes) | set(observed)):
            reference = baseline_hashes.get(key, set())
            current = observed.get(key, set())
            if len(reference) != 1 or len(current) != 1:
                deterministic = False
            if reference != current:
                parity_mismatches.append({
                    "mode": key[0],
                    "prompt_index": key[1],
                    "target": result["target"]["label"],
                    "baseline_sha256": sorted(reference),
                    "observed_sha256": sorted(current),
                })

    metric_paths = [
        ("single", "delivered_tokens_per_second", "higher"),
        ("single", "decode_tokens_per_second", "higher"),
        ("single", "e2e_tokens_per_second", "higher"),
        ("single", "ttft_seconds_median", "lower"),
        ("single", "wall_seconds_p95", "lower"),
    ]
    for mode in baseline["summary"]:
        if mode.startswith("batch-"):
            metric_paths.extend([
                (mode, "delivered_tokens_per_second", "higher"),
                (mode, "aggregate_decode_tokens_per_second", "higher"),
                (mode, "per_user_decode_tokens_per_second", "higher"),
                (mode, "ttft_seconds_median", "lower"),
                (mode, "wall_seconds_p95", "lower"),
            ])

    ratios: dict[str, dict[str, Any]] = {}
    for result in ordered[1:]:
        label = result["target"]["label"]
        ratios[label] = {}
        for mode, metric, direction in metric_paths:
            base_value = baseline["summary"].get(mode, {}).get(metric)
            candidate_value = result["summary"].get(mode, {}).get(metric)
            ratio = None
            if (
                isinstance(base_value, (int, float))
                and isinstance(candidate_value, (int, float))
                and base_value != 0
            ):
                ratio = candidate_value / base_value
            ratios[label][f"{mode}.{metric}"] = {
                "baseline": base_value,
                "candidate": candidate_value,
                "ratio": ratio,
                "direction": direction,
            }

    return {
        "schema": COMPARISON_SCHEMA,
        "generated_utc": utc_now(),
        "baseline": baseline_label,
        "targets": [result["target"]["label"] for result in ordered],
        "workload_sha256": workload_hash,
        "workload": baseline["workload"],
        "summaries": {
            result["target"]["label"]: result["summary"] for result in ordered
        },
        "ratios": ratios,
        "text_parity": {
            "deterministic": deterministic,
            "matched": not parity_mismatches and deterministic,
            "mismatches": parity_mismatches,
        },
        "provenance": {
            result["target"]["label"]: {
                "base_url": result["target"]["base_url"],
                "model": result["target"]["model"],
                "source_revision": result["environment"].get("source_revision"),
                "duration_seconds": result.get("duration_seconds"),
            }
            for result in ordered
        },
    }


def write_json(payload: dict[str, Any], path: str | None) -> None:
    try:
        text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except ValueError as error:
        raise BenchmarkError(f"refusing to write non-finite JSON: {error}") from error
    if path in (None, "-"):
        sys.stdout.write(text)
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)


class _FakeHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI endpoint for the device-free self-test."""

    server_version = "qwen38-benchmark-fake/1"

    def log_message(self, *_args: Any) -> None:
        return

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": "fake-model", "object": "model", "owned_by": "test"}
                    ],
                },
            )
            return
        if self.path in {"/healthz", "/health"}:
            self._json(200, {"status": "ok", "model": "fake-model"})
            return
        self._json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(request, dict):
            self._json(400, {"error": {"message": "body must be an object"}})
            return
        if self.path.endswith("/chat/completions"):
            messages = request.get("messages")
            if not isinstance(messages, list) or not messages:
                self._json(400, {"error": {"message": "messages are required"}})
                return
            prompt_text = "\n".join(
                str(message.get("content", ""))
                for message in messages
                if isinstance(message, dict)
            )
            prompt_tokens = max(1, len(prompt_text) // 4 + len(messages))
        elif self.path.endswith("/completions"):
            prompt = request.get("prompt")
            if not isinstance(prompt, str):
                self._json(400, {"error": {"message": "prompt is required"}})
                return
            prompt_tokens = max(1, len(prompt) // 4)
        else:
            self._json(404, {"error": {"message": "not found"}})
            return
        tokens = [" Paris", ".", " The", " capital"]
        if not request.get("stream"):
            self._json(
                200,
                {
                    "id": "cmpl-fake",
                    "object": "text_completion",
                    "choices": [
                        {
                            "index": 0,
                            "text": "".join(tokens),
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": len(tokens),
                        "total_tokens": prompt_tokens + len(tokens),
                    },
                },
            )
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for index, token in enumerate(tokens):
            payload = {
                "id": "cmpl-fake",
                "object": "text_completion",
                "choices": [
                    {
                        "index": 0,
                        "text": token,
                        "finish_reason": "length" if index == len(tokens) - 1 else None,
                    }
                ],
            }
            self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.001)
        usage = {
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(tokens),
                "total_tokens": prompt_tokens + len(tokens),
            },
        }
        self.wfile.write(f"data: {json.dumps(usage)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def self_test() -> None:
    corpus = {
        "id": "self-test",
        "prompts": ["The capital of France is", "Count from one to five in words:"],
    }
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}"
        first = run_benchmark(
            Endpoint(
                label="fake-a",
                base_url=base_url,
                model=None,
                api_key=None,
                headers=(),
                timeout=10.0,
            ),
            corpus,
            max_tokens=4,
            concurrency=[2],
            runs=1,
            warmup_runs=0,
            stream=True,
            settle_seconds=0.0,
            revision="self-test",
            note=None,
        )
        second = run_benchmark(
            Endpoint(
                label="fake-b",
                base_url=base_url,
                model=None,
                api_key=None,
                headers=(),
                timeout=10.0,
            ),
            corpus,
            max_tokens=4,
            concurrency=[2],
            runs=1,
            warmup_runs=0,
            stream=True,
            settle_seconds=0.0,
            revision="self-test",
            note=None,
        )
        validate_result(first, "self-test")
        incomplete = json.loads(json.dumps(first))
        incomplete["runs"].pop()
        try:
            validate_result(incomplete, "self-test-incomplete")
        except BenchmarkError:
            pass
        else:
            raise BenchmarkError("self-test: an incomplete result passed validation")

        tampered = json.loads(json.dumps(first))
        tampered["summary"]["single"]["delivered_tokens_per_second"] = 1e9
        try:
            validate_result(tampered, "self-test-summary")
        except BenchmarkError:
            pass
        else:
            raise BenchmarkError("self-test: an edited summary passed validation")

        missing_target = json.loads(json.dumps(first))
        del missing_target["target"]["base_url"]
        try:
            validate_result(missing_target, "self-test-target")
        except BenchmarkError:
            pass
        else:
            raise BenchmarkError("self-test: a missing target URL passed validation")

        empty_text = json.loads(json.dumps(first))
        empty_text["runs"][0]["records"][0]["text_characters"] = 0
        try:
            validate_result(empty_text, "self-test-text")
        except BenchmarkError:
            pass
        else:
            raise BenchmarkError("self-test: an empty-text record passed validation")

        try:
            parse_headers(["X-Test: ☃"])
        except BenchmarkError:
            pass
        else:
            raise BenchmarkError("self-test: a non-Latin-1 header was accepted")

        other_corpus = {
            "id": "self-test-other",
            "prompts": corpus["prompts"] + ["A different corpus changes the workload."],
        }
        different = run_benchmark(
            Endpoint(
                label="fake-other",
                base_url=base_url,
                model=None,
                api_key=None,
                headers=(),
                timeout=10.0,
            ),
            other_corpus,
            max_tokens=4,
            concurrency=[2],
            runs=1,
            warmup_runs=0,
            stream=True,
            settle_seconds=0.0,
            revision="self-test",
            note=None,
        )
        try:
            compare_results([first, different])
        except BenchmarkError as error:
            if "workload differs" not in str(error):
                raise BenchmarkError(
                    "self-test: a different corpus failed for the wrong reason"
                ) from error
        else:
            raise BenchmarkError("self-test: a different corpus was accepted")

        single = first["summary"]["single"]
        batch = first["summary"]["batch-2"]
        if not (single["ttft_seconds_median"] or 0) > 0:
            raise BenchmarkError("self-test: single TTFT was not measured")
        if not (single["decode_tokens_per_second"] or 0) > 0:
            raise BenchmarkError("self-test: single decode rate was not measured")
        if not (batch["delivered_tokens_per_second"] or 0) > 0:
            raise BenchmarkError("self-test: batch throughput was not measured")
        comparison = compare_results([first, second], baseline_label="fake-a")
        if not comparison["text_parity"]["matched"]:
            raise BenchmarkError("self-test: identical fake targets failed parity")
        if "fake-b" not in comparison["ratios"]:
            raise BenchmarkError("self-test: comparison omitted the candidate")

        plain = run_benchmark(
            Endpoint(
                label="fake-plain",
                base_url=base_url,
                model=None,
                api_key=None,
                headers=(),
                timeout=10.0,
            ),
            corpus,
            max_tokens=4,
            concurrency=[2],
            runs=1,
            warmup_runs=0,
            stream=False,
            settle_seconds=0.0,
            revision="self-test",
            note=None,
        )
        if plain["summary"]["single"]["ttft_seconds_median"] is not None:
            raise BenchmarkError("self-test: non-stream TTFT was reported")
        if not (plain["summary"]["single"]["delivered_tokens_per_second"] or 0) > 0:
            raise BenchmarkError("self-test: non-stream throughput was not measured")

        mismatch = json.loads(json.dumps(second))
        mismatch["runs"][0]["records"][0]["text_sha256"] = "0" * 64
        if compare_results([first, mismatch])["text_parity"]["matched"]:
            raise BenchmarkError("self-test: a text mismatch passed parity")
        wrong_workload = json.loads(json.dumps(second))
        wrong_workload["workload_sha256"] = "1" * 64
        try:
            compare_results([first, wrong_workload])
        except BenchmarkError:
            pass
        else:
            raise BenchmarkError("self-test: a workload mismatch was accepted")

        trace = load_agentic_trace(DEFAULT_AGENTIC_TRACE)
        slo = {
            "ttft_p95_seconds": 5.0,
            "itl_p95_seconds": 5.0,
            "task_p95_seconds": 60.0,
        }
        agentic = run_agentic_benchmark(
            Endpoint(
                label="fake-agentic",
                base_url=base_url,
                model=None,
                api_key=None,
                headers=(),
                timeout=10.0,
            ),
            trace,
            agents=[1, 2],
            runs=1,
            warmup_runs=0,
            settle_seconds=0.0,
            tool_wait_scale=0.0,
            slo=slo,
            cache_mode="disabled",
            power={"boundary": "cards", "watts": 10.0, "method": "self-test"},
            revision="self-test",
            note=None,
        )
        validate_agentic_result(agentic, "self-test-agentic")
        if agentic["summary"]["max_agents_meeting_all_slo"] != 2:
            raise BenchmarkError("self-test: agentic SLO maximum was not 2")
        if agentic["summary"]["saturated_agents"] != 2:
            raise BenchmarkError("self-test: agentic saturation width was not 2")
        if agentic["conditions"]["power"]["agents_at_slo_per_watt"] != 0.2:
            raise BenchmarkError("self-test: agents at SLO per watt was not derived")
        calls = [call for task in agentic["runs"][0]["tasks"] for call in task["calls"]]
        if not any((call["reused_prefix_tokens_estimate"] or 0) > 0 for call in calls):
            raise BenchmarkError("self-test: reused-prefix tokens were not recorded")
        if calls[0]["reused_prefix_characters"] != 0:
            raise BenchmarkError("self-test: the first call reported prefix reuse")
        if any(
            call["cache_confirmation"] != "client-prefix-overlap-only" for call in calls
        ):
            raise BenchmarkError("self-test: cache confirmation overclaimed")
        if agentic["workload"]["tool_wait_seconds_total"] != 0.0:
            raise BenchmarkError("self-test: scaled tool waits were not recorded")
        try:
            validate_result(agentic, "self-test-agentic-schema")
        except BenchmarkError:
            pass
        else:
            raise BenchmarkError("self-test: agentic result passed the run schema")

        edited_call = json.loads(json.dumps(agentic))
        edited_call["runs"][0]["tasks"][0]["calls"][1][
            "reused_prefix_tokens_estimate"
        ] += 1
        try:
            validate_agentic_result(edited_call, "self-test-agentic-call")
        except BenchmarkError:
            pass
        else:
            raise BenchmarkError("self-test: an edited reused-token count passed")

        edited_curve = json.loads(json.dumps(agentic))
        edited_curve["summary"]["curve"][0]["ttft_seconds_p95"] = 0.0
        try:
            validate_agentic_result(edited_curve, "self-test-agentic-curve")
        except BenchmarkError:
            pass
        else:
            raise BenchmarkError("self-test: an edited SLO curve passed")

        edited_trace = json.loads(json.dumps(agentic))
        edited_trace["trace"]["system_prompt"] += " changed"
        try:
            validate_agentic_result(edited_trace, "self-test-agentic-trace")
        except BenchmarkError:
            pass
        else:
            raise BenchmarkError("self-test: an edited trace passed validation")

        missing_run = json.loads(json.dumps(agentic))
        missing_run["runs"].pop()
        try:
            validate_agentic_result(missing_run, "self-test-agentic-runs")
        except BenchmarkError:
            pass
        else:
            raise BenchmarkError("self-test: a missing agentic run passed")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def parse_headers(values: list[str] | None) -> tuple[tuple[str, str], ...]:
    headers = []
    for value in values or []:
        name, separator, content = value.partition(":")
        if not separator or not name.strip():
            raise BenchmarkError(f"invalid --header {value!r}; expected NAME: VALUE")
        name = name.strip()
        content = content.strip()
        try:
            validate_header(name, content)
        except BenchmarkError as error:
            raise BenchmarkError(f"invalid --header {value!r}: {error}") from error
        headers.append((name, content))
    return tuple(headers)


def parse_concurrency(value: str) -> list[int]:
    widths = []
    for item in value.split(","):
        try:
            width = int(item)
        except ValueError as error:
            raise BenchmarkError(f"invalid concurrency {item!r}") from error
        if width <= 1:
            raise BenchmarkError("concurrency widths must be greater than one")
        if width not in widths:
            widths.append(width)
    if not widths:
        raise BenchmarkError("at least one concurrency width is required")
    return widths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repeatable OpenAI-compatible benchmark for Qwen serving stacks"
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the device-free fake-server self-test and exit",
    )
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="measure one endpoint")
    run.add_argument("--label", required=True, help="short target name, e.g. ours")
    run.add_argument("--base-url", required=True, help="http://host:port")
    run.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="served model name; discovered from /v1/models when omitted",
    )
    run.add_argument(
        "--api-key",
        default=os.environ.get("QWEN38_BENCH_API_KEY"),
        help="bearer token, never written to the result (default: QWEN38_BENCH_API_KEY)",
    )
    run.add_argument(
        "--header",
        action="append",
        help="additional request header as NAME: VALUE (repeatable)",
    )
    run.add_argument(
        "--corpus",
        default=str(DEFAULT_CORPUS),
        help="corpus JSON (default: the packaged B=8 corpus)",
    )
    run.add_argument("--max-tokens", type=int, default=64)
    run.add_argument(
        "--concurrency",
        default="8",
        help="comma-separated batch widths greater than one (default: 8)",
    )
    run.add_argument("--runs", type=int, default=3)
    run.add_argument("--warmup-runs", type=int, default=1)
    run.add_argument(
        "--no-stream",
        action="store_true",
        help="do not stream; TTFT and decode rate are then unavailable",
    )
    run.add_argument("--settle-seconds", type=float, default=1.0)
    run.add_argument("--request-timeout", type=float, default=600.0)
    run.add_argument(
        "--source-revision",
        help="repository revision recorded in the result; auto-detected when omitted",
    )
    run.add_argument("--note", help="free-form context recorded in the result")
    run.add_argument(
        "--output",
        default="-",
        help="result path, or - for stdout (default: -)",
    )

    agentic = subparsers.add_parser(
        "agentic",
        help="measure concurrent closed-loop agents on a frozen multi-call trace",
    )
    agentic.add_argument("--label", required=True, help="short target name, e.g. ours")
    agentic.add_argument("--base-url", required=True, help="http://host:port")
    agentic.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="served model name; discovered from /v1/models when omitted",
    )
    agentic.add_argument(
        "--api-key",
        default=os.environ.get("QWEN38_BENCH_API_KEY"),
        help="bearer token, never written to the result (default: QWEN38_BENCH_API_KEY)",
    )
    agentic.add_argument(
        "--header",
        action="append",
        help="additional request header as NAME: VALUE (repeatable)",
    )
    agentic.add_argument(
        "--trace",
        default=str(DEFAULT_AGENTIC_TRACE),
        help="agentic trace JSON (default: the packaged eight-call trace)",
    )
    agentic.add_argument(
        "--agents",
        default="1,2,4,8",
        help="comma-separated concurrent agent counts (default: 1,2,4,8)",
    )
    agentic.add_argument("--runs", type=int, default=2)
    agentic.add_argument("--warmup-runs", type=int, default=1)
    agentic.add_argument("--settle-seconds", type=float, default=1.0)
    agentic.add_argument(
        "--tool-wait-scale",
        type=float,
        default=1.0,
        help="multiply every trace tool wait; recorded in the workload digest",
    )
    agentic.add_argument(
        "--ttft-p95",
        type=float,
        required=True,
        help="per-call TTFT p95 SLO in seconds",
    )
    agentic.add_argument(
        "--itl-p95",
        type=float,
        required=True,
        help="per-call inter-token latency p95 SLO in seconds",
    )
    agentic.add_argument(
        "--task-p95",
        type=float,
        required=True,
        help="agent task end-to-end p95 SLO in seconds",
    )
    agentic.add_argument(
        "--cache-mode",
        choices=CACHE_MODES,
        default="unknown",
        help="operator-declared cache state, recorded and compared (default: unknown)",
    )
    agentic.add_argument(
        "--power-boundary",
        choices=POWER_BOUNDARIES,
        default="unspecified",
        help="what --watts measures (default: unspecified)",
    )
    agentic.add_argument(
        "--watts",
        type=float,
        help="measured power inside --power-boundary, in watts",
    )
    agentic.add_argument(
        "--power-method",
        help="how power was measured, e.g. wall meter model and sample interval",
    )
    agentic.add_argument("--request-timeout", type=float, default=600.0)
    agentic.add_argument(
        "--source-revision",
        help="repository revision recorded in the result; auto-detected when omitted",
    )
    agentic.add_argument("--note", help="free-form context recorded in the result")
    agentic.add_argument(
        "--output",
        default="-",
        help="result path, or - for stdout (default: -)",
    )

    compare = subparsers.add_parser("compare", help="compare result files")
    compare.add_argument("results", nargs="+", help="result JSON paths")
    compare.add_argument(
        "--baseline-label", help="target label to use as baseline (default: first)"
    )
    compare.add_argument(
        "--output",
        default="-",
        help="comparison path, or - for stdout (default: -)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.self_test:
            self_test()
            print("self-test: PASS")
            return 0
        if arguments.command == "run":
            if arguments.max_tokens <= 0:
                raise BenchmarkError("--max-tokens must be positive")
            if arguments.runs <= 0:
                raise BenchmarkError("--runs must be positive")
            if arguments.warmup_runs < 0:
                raise BenchmarkError("--warmup-runs must be non-negative")
            if (
                not math.isfinite(arguments.settle_seconds)
                or arguments.settle_seconds < 0
            ):
                raise BenchmarkError("--settle-seconds must be finite and non-negative")
            if (
                not math.isfinite(arguments.request_timeout)
                or arguments.request_timeout <= 0
            ):
                raise BenchmarkError("--request-timeout must be finite and positive")
            widths = parse_concurrency(arguments.concurrency)
            corpus = load_corpus(Path(arguments.corpus))
            endpoint = Endpoint(
                label=arguments.label,
                base_url=arguments.base_url,
                model=arguments.model,
                api_key=arguments.api_key,
                headers=parse_headers(arguments.header),
                timeout=arguments.request_timeout,
            )
            result = run_benchmark(
                endpoint,
                corpus,
                max_tokens=arguments.max_tokens,
                concurrency=widths,
                runs=arguments.runs,
                warmup_runs=arguments.warmup_runs,
                stream=not arguments.no_stream,
                settle_seconds=arguments.settle_seconds,
                revision=arguments.source_revision,
                note=arguments.note,
            )
            write_json(result, arguments.output)
            return 0
        if arguments.command == "agentic":
            if arguments.runs <= 0:
                raise BenchmarkError("--runs must be positive")
            if arguments.warmup_runs < 0:
                raise BenchmarkError("--warmup-runs must be non-negative")
            if (
                not math.isfinite(arguments.settle_seconds)
                or arguments.settle_seconds < 0
            ):
                raise BenchmarkError("--settle-seconds must be finite and non-negative")
            if (
                not math.isfinite(arguments.tool_wait_scale)
                or arguments.tool_wait_scale < 0
            ):
                raise BenchmarkError(
                    "--tool-wait-scale must be finite and non-negative"
                )
            if (
                not math.isfinite(arguments.request_timeout)
                or arguments.request_timeout <= 0
            ):
                raise BenchmarkError("--request-timeout must be finite and positive")
            slo = {
                "ttft_p95_seconds": arguments.ttft_p95,
                "itl_p95_seconds": arguments.itl_p95,
                "task_p95_seconds": arguments.task_p95,
            }
            for name, value in slo.items():
                if not math.isfinite(value) or value <= 0:
                    raise BenchmarkError(f"SLO {name} must be finite and positive")
            widths = parse_agent_widths(arguments.agents)
            trace = load_agentic_trace(Path(arguments.trace))
            endpoint = Endpoint(
                label=arguments.label,
                base_url=arguments.base_url,
                model=arguments.model,
                api_key=arguments.api_key,
                headers=parse_headers(arguments.header),
                timeout=arguments.request_timeout,
            )
            result = run_agentic_benchmark(
                endpoint,
                trace,
                agents=widths,
                runs=arguments.runs,
                warmup_runs=arguments.warmup_runs,
                settle_seconds=arguments.settle_seconds,
                tool_wait_scale=arguments.tool_wait_scale,
                slo=slo,
                cache_mode=arguments.cache_mode,
                power={
                    "boundary": arguments.power_boundary,
                    "watts": arguments.watts,
                    "method": arguments.power_method,
                },
                revision=arguments.source_revision,
                note=arguments.note,
            )
            validate_agentic_result(result, "agentic benchmark")
            write_json(result, arguments.output)
            return 0
        if arguments.command == "compare":
            results = [load_result(Path(path)) for path in arguments.results]
            comparison = compare_results(results, arguments.baseline_label)
            write_json(comparison, arguments.output)
            return 0 if comparison["text_parity"]["matched"] else 1
        parser.error("a subcommand is required (run, agentic, or compare)")
    except BenchmarkError as error:
        print(f"qwen38-benchmark: {error}", file=sys.stderr)
        return 2
    except (OSError, http.client.HTTPException) as error:
        print(
            f"qwen38-benchmark: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
