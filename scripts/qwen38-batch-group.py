#!/usr/bin/env python3
"""Release a batch of completions requests so they land inside one collection window.

The service collects a batch with a 10 ms window and, if the window closes early, it
dispatches whatever arrived as a group. A group whose width has no warmed trace then
fails with `BatchDecodeAdapterError`, which costs a full prefill retry and can
contaminate a whole measurement session. Starting eight separate `curl` processes,
as the shell harness does, adds process and connection setup to every request, which
is enough to miss the window under load.

This driver removes that setup from the critical path: it opens every connection
first, then releases all requests from a barrier, so the requests arrive within a
millisecond of each other.

Written in Python rather than a Rust cargo script because it is a short-lived
measurement client that needs only the standard library (sockets, threads, JSON) and
must not pull crates into an offline environment. It is a harness tool: it makes no
measurement claim and changes no served behavior.
"""

from __future__ import annotations

import argparse
import http.client
import json
import threading
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", required=True, help="base URL, e.g. http://127.0.0.1:8000"
    )
    parser.add_argument(
        "--request", action="append", required=True, help="request JSON path, in order"
    )
    parser.add_argument(
        "--output-directory", required=True, help="directory for response files"
    )
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--attempts", type=int, default=4, help="group attempts before giving up"
    )
    return parser.parse_args()


def open_connection(
    base_url: str, timeout_seconds: float
) -> http.client.HTTPConnection:
    if "://" in base_url:
        scheme, remainder = base_url.split("://", 1)
    else:
        scheme, remainder = "http", base_url
    if scheme != "http":
        raise SystemExit(f"only http is supported, got {scheme}")
    host, _, port_text = remainder.partition("/")[0].partition(":")
    port = int(port_text) if port_text else 80
    connection = http.client.HTTPConnection(host, port, timeout=timeout_seconds)
    connection.connect()
    return connection


def usable_response(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and "choices" in payload


def attempt_once(arguments: argparse.Namespace) -> int:
    output_directory = Path(arguments.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    requests = [Path(path).read_bytes() for path in arguments.request]
    width = len(requests)

    connections = [
        open_connection(arguments.url, arguments.timeout_seconds) for _ in range(width)
    ]
    barrier = threading.Barrier(width)
    errors: list[str] = []

    def send(index: int) -> None:
        target = output_directory / f"response-{index}.json"
        target.unlink(missing_ok=True)
        try:
            barrier.wait()
            connection = connections[index]
            connection.request(
                "POST",
                "/v1/completions",
                body=requests[index],
                headers={"Content-Type": "application/json", "Connection": "close"},
            )
            response = connection.getresponse()
            body = response.read()
            target.write_bytes(body)
            if response.status != 200:
                errors.append(f"request {index}: HTTP {response.status}")
        except (
            OSError,
            http.client.HTTPException,
            threading.BrokenBarrierError,
        ) as error:
            errors.append(f"request {index}: {type(error).__name__}: {error}")

    threads = [threading.Thread(target=send, args=(index,)) for index in range(width)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for connection in connections:
        connection.close()

    complete = sum(
        1
        for index in range(width)
        if usable_response(output_directory / f"response-{index}.json")
    )
    if errors:
        print("; ".join(errors[:4]))
    return complete


def main() -> int:
    arguments = parse_arguments()
    width = len(arguments.request)
    for attempt in range(1, arguments.attempts + 1):
        complete = attempt_once(arguments)
        print(f"group attempt {attempt}: {complete}/{width} usable responses")
        if complete == width:
            return 0
        if attempt < arguments.attempts:
            # A rejected group can leave the adapter needing a reset; the retry is
            # what the shell harness did as well.
            downtime = 2.0
            threading.Event().wait(downtime)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
