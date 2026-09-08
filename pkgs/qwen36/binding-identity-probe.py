"""Compare binding accessors on real TT-NN tensors; this is not a decode benchmark."""

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import ttnn

TILE_SIZE = 32
DEFAULT_ITERATIONS = 10000
WARMUP_ITERATIONS = 100
MESH_ROWS = 1
MESH_COLUMNS = 2
PHYSICAL_DEVICE_IDS = [0, 1]
MODEL_PROMPT_COUNT = 4
MODEL_GENERATED_TOKENS = 64
INVALID_TOKEN_ID = -1
MEASURED_PATTERN = "*-measured-0.json"


def compare_response(baseline, candidate):
    before = baseline["x_onix_decode_evidence"]
    after = candidate["x_onix_decode_evidence"]
    for evidence in (before, after):
        for key in ("clean_reset", "trace_binding_stable", "trace_reused"):
            if evidence[key] is not True:
                raise AssertionError(f"missing {key}")
        if evidence["simulated"] is not False:
            raise AssertionError("model response is not physical evidence")
        if evidence["physical_pcie_device_ids"] != PHYSICAL_DEVICE_IDS:
            raise AssertionError("physical devices differ")
        if evidence["batch_width"] != MODEL_PROMPT_COUNT:
            raise AssertionError("batch width differs")
        if len(evidence["generated_token_ids"]) != MODEL_GENERATED_TOKENS:
            raise AssertionError("generated token count differs")
    for key in ("generated_token_ids", "runtime_identity", "model_identity"):
        if before[key] != after[key]:
            raise AssertionError(f"response {key} differs")
    if baseline["choices"] != candidate["choices"]:
        raise AssertionError("response choices differ")


def compare_model_roots(baseline_root, candidate_root):
    files = sorted(candidate_root.glob(MEASURED_PATTERN))
    if len(files) != MODEL_PROMPT_COUNT:
        raise AssertionError("model smoke must contain every prompt")
    identities = set()
    for path in files:
        baseline = json.loads((baseline_root / path.name).read_text())
        candidate = json.loads(path.read_text())
        compare_response(baseline, candidate)
        identities.add(candidate["x_onix_decode_evidence"]["package_identity"])
    if len(identities) != 1:
        raise AssertionError("candidate package identities differ")
    corrupted = json.loads(json.dumps(candidate))
    corrupted["x_onix_decode_evidence"]["generated_token_ids"][0] = INVALID_TOKEN_ID
    try:
        compare_response(baseline, corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError("token corruption was accepted")
    return {
        "schema_version": 1,
        "simulated": False,
        "prompt_count": len(files),
        "generated_tokens_per_prompt": MODEL_GENERATED_TOKENS,
        "exact_token_and_text_parity": True,
        "negative_control_rejected": True,
        "candidate_package_identity": identities.pop(),
        "non_claims": ["not a controlled decode-speed or broad accuracy benchmark"],
    }


def load_accessor(name, path):
    spec = importlib.util.spec_from_file_location(f"qwen36_service.{name}", path)
    if spec is None or spec.loader is None:
        raise ValueError("binding accessor module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module._tensor_identity


def measure(accessor, tensor, iterations, expected):
    started = time.perf_counter_ns()
    for _ in range(iterations):
        observed = accessor(tensor)
    elapsed = time.perf_counter_ns() - started
    if observed != expected:
        raise AssertionError("binding changed during timing")
    return elapsed / iterations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("accessor", "model-responses"), default="accessor"
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("iterations must be positive")
    if args.mode == "model-responses":
        result = compare_model_roots(args.baseline, args.candidate)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        return
    baseline = load_accessor("binding_baseline", args.baseline)
    candidate = load_accessor("binding_candidate", args.candidate)
    mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(MESH_ROWS, MESH_COLUMNS),
        physical_device_ids=PHYSICAL_DEVICE_IDS,
    )
    try:
        shape = (1, 1, TILE_SIZE, TILE_SIZE)
        tensor = ttnn.ones(
            shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
        )
        replacement = ttnn.ones(
            shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
        )
        host = ttnn.from_torch(torch.ones(shape))
        for value in (tensor, replacement, host, None):
            if baseline(value) != candidate(value):
                raise AssertionError("baseline and candidate identities differ")
        expected = baseline(tensor)
        if candidate(replacement) == expected:
            raise AssertionError("replacement buffer did not change the identity")
        before_release = candidate(replacement)
        ttnn.deallocate(replacement)
        if baseline(replacement) != candidate(replacement):
            raise AssertionError("released tensor fallback differs")
        if candidate(replacement) == before_release:
            raise AssertionError("released tensor kept its allocated identity")
        for accessor in (baseline, candidate):
            measure(accessor, tensor, WARMUP_ITERATIONS, expected)
        samples = {"baseline": [], "candidate": []}
        for name, accessor in (
            ("baseline", baseline),
            ("candidate", candidate),
            ("candidate", candidate),
            ("baseline", baseline),
        ):
            samples[name].append(measure(accessor, tensor, args.iterations, expected))
        baseline_ns = statistics.median(samples["baseline"])
        candidate_ns = statistics.median(samples["candidate"])
        result = {
            "schema_version": 1,
            "simulated": bool(os.environ.get("TT_METAL_SIMULATOR")),
            "requested_device_ids": PHYSICAL_DEVICE_IDS,
            "baseline_source": str(args.baseline),
            "candidate_source": str(args.candidate),
            "iterations_per_sample": args.iterations,
            "nanoseconds_per_call": samples,
            "speedup_ratio": baseline_ns / candidate_ns,
            "exact_identity_parity": True,
            "replacement_detected": True,
            "deallocation_detected": True,
            "non_claims": ["not a full-model correctness or decode-speed benchmark"],
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
