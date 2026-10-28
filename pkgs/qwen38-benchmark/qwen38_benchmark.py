#!/usr/bin/env python3
"""Repeatable client-side benchmark for OpenAI-compatible Qwen services.

The harness measures what a client sees: time to first content token, steady
decode rate, delivered tokens per second, and batch aggregate throughput for a
frozen corpus. It never starts a server, touches a Tenstorrent device, or reads
server-private evidence. Point one run at the onix service and another at a
vLLM endpoint; ``compare`` checks that both ran the same workload and that
greedy text stayed identical.

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
DEFAULT_MODEL = None
DEFAULT_CORPUS = Path(
    os.environ.get("QWEN38_BENCH_CORPUS", Path(__file__).with_name("corpus.json"))
)
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

    connection.request(
        "POST",
        endpoint.api_path("/completions"),
        body=json.dumps(payload).encode(),
        headers=request_headers(endpoint, stream),
    )
    response = connection.getresponse()
    if response.status != 200:
        body = response.read().decode("utf-8", "replace")
        raise BenchmarkError(f"{endpoint.label}: HTTP {response.status}: {body[:1000]}")

    if not stream:
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
            f"{endpoint.label}: stream produced no content for prompt {prompt[:60]!r}"
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
        request_start,
        end,
        first_token,
        last_token,
    )
    record["finish_reason_seen"] = completed
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
                        "prompt_tokens": 5,
                        "completion_tokens": len(tokens),
                        "total_tokens": 5 + len(tokens),
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
                "prompt_tokens": 5,
                "completion_tokens": len(tokens),
                "total_tokens": 5 + len(tokens),
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
        if arguments.command == "compare":
            results = [load_result(Path(path)) for path in arguments.results]
            comparison = compare_results(results, arguments.baseline_label)
            write_json(comparison, arguments.output)
            return 0 if comparison["text_parity"]["matched"] else 1
        parser.error("a subcommand is required (run or compare)")
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
