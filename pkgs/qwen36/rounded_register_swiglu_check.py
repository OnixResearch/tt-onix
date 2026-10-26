"""Installed behavioral qualification command for rounded register SwiGLU."""

import argparse
import importlib.util
import json
import os
import statistics
import time
from pathlib import Path


def operation():
    path = Path(__file__).with_name("rounded_register_swiglu.py")
    spec = importlib.util.spec_from_file_location("rounded_register_swiglu", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def semantic(op):
    import torch

    original = os.environ.get(op.ENV)
    try:
        os.environ.pop(op.ENV, None)
        assert not op.select_decode("decode")
        os.environ[op.ENV] = "1"
        assert not op.select_decode("prefill")
        assert op.select_decode("decode") and op.select_decode("verify")
        os.environ[op.ENV] = "yes"
        try:
            op.requested()
        except ValueError:
            pass
        else:
            raise AssertionError("malformed opt-in accepted")
    finally:
        if original is None:
            os.environ.pop(op.ENV, None)
        else:
            os.environ[op.ENV] = original
    for batch in (1, 8):
        assert (
            op.validate_geometry((1, 1, batch, op.K), (op.K, op.N), (op.K, op.N))
            == batch
        )
    for shape in (
        (1, 1, 2, op.K),
        (1, 1, 16, op.K),
        (1, 1, 32, op.K),
        (1, 1, 8, 4096),
        (8, 1, op.K),
    ):
        try:
            op.validate_geometry(shape, (op.K, op.N), (op.K, op.N))
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsupported decode shape accepted: {shape}")
    # Known BF16 ties: gate is large positive, so SiLU is exactly gate in FP32.
    # Native matmul pack ties away: 1.00390625 -> 1.0078125, not even 1.0.
    gate = torch.tensor([128.5, 129.5, -128.0, 0.0])
    up = torch.tensor([1.00390625, 1.01171875, 3.0, -3.0])
    actual = op.rounded_reference(gate, up)
    expected = torch.tensor([130.0, 132.0, 0.0, 0.0], dtype=torch.bfloat16)
    assert torch.equal(actual, expected), (actual, expected)
    # Rounding gate/up before multiplying is observably different from a
    # single final BF16 pack. This catches elimination of either boundary.
    generator = torch.Generator().manual_seed(4517)
    gate = torch.randn(4096, generator=generator) * 4
    up = torch.randn(4096, generator=generator)
    rounded = op.rounded_reference(gate, up)
    unrounded = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
    assert not torch.equal(rounded, unrounded)
    return {
        "semantic": "passed",
        "batches": [1, 8],
        "unsupported_decode": "rejected",
        "prefill": "unchanged",
        "rounding": "gate_after_silu_and_up_BF16_then_BF16_product",
    }


def device_check(op, args):
    import torch
    import ttnn
    from models.demos.blackhole.qwen36.tt import tp_common as tpc

    device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 1), physical_device_ids=[args.device_id]
    )
    try:
        device.enable_program_cache()
        torch.manual_seed(4517)
        x_host = torch.randn((1, 1, args.batch, op.K), dtype=torch.bfloat16) * 0.1
        # Real local TP2 geometry and original BF4 weight format, not a BF16
        # post-projection surrogate or the upstream simulator-only T16 example.
        weights = [
            torch.randn((op.K, op.N), dtype=torch.bfloat16) * 0.02 for _ in range(2)
        ]
        x = ttnn.from_torch(
            x_host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        gate, up = [
            ttnn.from_torch(
                w,
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for w in weights
        ]
        ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        grid_width = device.compute_with_storage_grid_size().x
        gate_pc = tpc.create_matmul_1d_decode_progcfg(
            32,
            op.K,
            op.N,
            num_cores=44,
            fused_activation=ttnn.UnaryOpType.SILU,
            grid_w=grid_width,
        )
        up_pc = tpc.create_matmul_1d_decode_progcfg(
            32, op.K, op.N, num_cores=44, grid_w=grid_width
        )

        def baseline():
            g = ttnn.linear(
                x,
                gate,
                compute_kernel_config=ckc,
                program_config=gate_pc,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            u = ttnn.linear(
                x,
                up,
                compute_kernel_config=ckc,
                program_config=up_pc,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            product = ttnn.mul(g, u, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(g)
            ttnn.deallocate(u)
            return product

        # Compile and execute the actual candidate first, so simulator gaps in
        # the original multicast baseline cannot hide a candidate JIT failure.
        cold_candidate = op.gate_up(x, gate, up)
        ttnn.synchronize_device(device)
        ttnn.deallocate(cold_candidate)
        golden = baseline()
        expected = ttnn.to_torch(golden).float()
        ttnn.deallocate(golden)
        timings = {"baseline_host_seconds": [], "candidate_host_seconds": []}
        errors = []
        for _ in range(args.iterations):
            for label, run in (
                ("baseline", baseline),
                ("candidate", lambda: op.gate_up(x, gate, up)),
            ):
                ttnn.synchronize_device(device)
                start = time.perf_counter()
                result = run()
                ttnn.synchronize_device(device)
                timings[label + "_host_seconds"].append(time.perf_counter() - start)
                observed = ttnn.to_torch(result).float()
                delta = observed - expected
                relative_l2 = (delta.norm() / expected.norm()).item()
                maximum = delta.abs().max().item()
                errors.append({
                    "path": label,
                    "relative_l2": relative_l2,
                    "max_abs": maximum,
                    "unequal_elements": (observed != expected).sum().item(),
                })
                # No guessed relaxed tolerance: the candidate intends the same
                # BF16 product as the original fused-SiLU BF4 1D composition.
                if (
                    not torch.isfinite(observed).all()
                    or not torch.isfinite(expected).all()
                    or not torch.equal(observed, expected)
                ):
                    raise AssertionError(
                        f"{label} differs from original fused-SiLU BF4 1D path: {errors[-1]}"
                    )
                ttnn.deallocate(result)
        return {
            "device_component": "passed",
            "simulated": bool(os.environ.get("TT_METAL_SIMULATOR")),
            "scope": "single-chip TP2-local BF4 gate/up, not full-model or mesh admission",
            "batch": args.batch,
            "K": op.K,
            "N": op.N,
            "errors": errors,
            "host_wall_medians": {
                key: statistics.median(value) for key, value in timings.items()
            },
            "timing_scope": "synchronized host wall, includes Python/dispatch; not device profiler",
            "serving_admitted": False,
        }
    finally:
        ttnn.close_mesh_device(device)


def model_check(op, args):
    """Actual packaged prefill + traced recurrent continuation, not admission."""
    os.environ["HF_MODEL"] = str(Path(args.model_path).resolve())
    os.environ["TT_CACHE_PATH"] = os.environ["HF_MODEL"]
    import torch
    import ttnn
    from models.demos.blackhole.qwen36.tt.model import Qwen36Model
    from qwen36_service import decode_component_probe as dcp
    from qwen36_service.batch_decode_adapter import BatchedTracedDecodeAdapter
    from qwen36_service.decode_core import DecodeProfile
    from transformers import AutoTokenizer

    with open(args.decode_profile, encoding="utf-8") as stream:
        profile = DecodeProfile.from_mapping(json.load(stream))
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )
    mesh = None
    adapter = None
    try:
        mesh = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(1, 2),
            physical_device_ids=[0, 1],
            l1_small_size=dcp.L1_SMALL_SIZE_BYTES,
            num_command_queues=dcp.COMMAND_QUEUE_COUNT,
            trace_region_size=profile.trace_region_size_bytes,
        )
        mesh.enable_program_cache()
        model = Qwen36Model.from_pretrained(
            mesh,
            max_batch_size=8,
            max_seq_len=profile.maximum_sequence_length,
        )
        adapter = BatchedTracedDecodeAdapter(
            model=model, profile=profile, maximum_width=8
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, local_files_only=True
        )
        prompts = (
            "The capital of France is",
            "The square root of nine is",
            "A Python list begins with",
            "Water freezes at a temperature of",
            "The planet nearest the Sun is",
            "The chemical symbol for gold is",
            "The opposite of north is",
            "Two plus three equals",
        )[: args.batch]
        tokens = tuple(
            tokenizer(prompt, return_tensors="pt")["input_ids"].to(torch.int32)
            for prompt in prompts
        )
        first = adapter.prefill_users(tokens, generation_tokens=args.iterations + 1)
        positions = tuple(int(ids.shape[1]) for ids in tokens)
        current = first.first_token_ids
        adapter.prepare_width(token_ids=current, positions=positions)
        continuation = []
        for _ in range(args.iterations):
            current = adapter.decode_traced(token_ids=current, positions=positions)
            continuation.append(current)
            positions = tuple(position + 1 for position in positions)
        return {
            "model_smoke": "completed",
            "batch": args.batch,
            "candidate_requested": op.requested(),
            "physical_device_ids": [0, 1],
            "model": str(Path(args.model_path).resolve()),
            "decode_profile": str(Path(args.decode_profile).resolve()),
            "first_token_ids": first.first_token_ids,
            "continuation_token_ids": continuation,
            "positions_after": positions,
            "prefill_route": "unchanged",
            "scope": "TP2 actual model prefill and traced recurrent continuation; not retention or performance admission",
            "serving_admitted": False,
        }
    finally:
        if adapter is not None:
            adapter.close()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--semantic",
        action="store_true",
        help="device-free arithmetic and strict selection proof",
    )
    mode.add_argument(
        "--device",
        action="store_true",
        help="execute real local BF4 projections against original 1D fused-SiLU route",
    )
    mode.add_argument(
        "--model-device",
        action="store_true",
        help="actual TP2 model prefill and recurrent traced decode continuation",
    )
    parser.add_argument("--model-path")
    parser.add_argument("--decode-profile")
    parser.add_argument("--batch", type=int, choices=(1, 8), default=1)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.model_device and (not args.model_path or not args.decode_profile):
        parser.error("--model-device requires --model-path and --decode-profile")
    op = operation()
    result = (
        semantic(op)
        if args.semantic
        else model_check(op, args)
        if args.model_device
        else device_check(op, args)
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
