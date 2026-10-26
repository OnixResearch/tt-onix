"""Opt-in Qwen3.8 TP2 decode gate/up matmul with a rounded Dst SwiGLU.

This is not the runtime's independently gated prefill fuse_swiglu operation.
No gate/up intermediate is packed or written to L1/DRAM. Only the BF16
product is packed. Qualification and model admission remain separate.
"""

import os
from pathlib import Path

ENV = "QWEN38_ROUNDED_REGISTER_SWIGLU"
K = 5120
N = 8704  # Qwen3.8 intermediate 17408 / TP2


def requested():
    value = os.environ.get(ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{ENV} must be 0 or 1")
    return value == "1"


def select_decode(mode):
    """A process-level opt-in never changes prefill, including short prefills."""
    enabled = requested()
    if mode not in ("prefill", "decode", "verify"):
        raise ValueError(f"unknown rounded register SwiGLU consumer mode: {mode!r}")
    return enabled and mode != "prefill"


def validate_geometry(x_shape, gate_shape, up_shape):
    """Logical local tensor shapes, not padded tile height, select B1/B8."""
    shape = tuple(x_shape)
    if len(shape) not in (3, 4) or any(d != 1 for d in shape[:-2]):
        raise ValueError(
            f"rounded register SwiGLU requires [1,B,5120] or [1,1,B,5120], got {shape}"
        )
    if shape[-2] not in (1, 8) or shape[-1] != K:
        raise ValueError(
            f"rounded register SwiGLU supports only B1/B8 K={K}, got {shape}"
        )
    for name, weight in (("gate", gate_shape), ("up", up_shape)):
        if tuple(weight) not in ((K, N), (1, K, N), (1, 1, K, N)):
            raise ValueError(
                f"rounded register SwiGLU requires TP2 local {name} [{K},{N}], got {tuple(weight)}"
            )
    return shape[-2]


def rounded_reference(gate, up):
    """Finite-value arithmetic oracle; SFPU SiLU approximation is device-owned."""
    import torch

    def native_pack(value):
        bits = value.float().contiguous().view(torch.int32)
        return ((bits + 0x8000) & -65536).view(torch.float32).to(torch.bfloat16)

    activated = native_pack(torch.nn.functional.silu(gate.float()))
    rounded_up = native_pack(up)
    return (activated.float() * rounded_up.float()).to(torch.bfloat16)


def gate_up(x, gate, up):
    """Run on one chip (qualification) or an SPMD TP2 mesh (actual MLP).

    Weight tensors must already use the existing BF4 interleaved TP2 cache.
    Conversion of the activation's L1 shard to interleaved L1 is the same
    conversion performed by the packaged 1D decode path, not a precision cast.
    """
    import ttnn

    validate_geometry(x.shape, gate.shape, up.shape)
    device = x.device()
    if device.arch() != ttnn.Arch.BLACKHOLE:
        raise ValueError("rounded register SwiGLU requires Blackhole")
    if device.get_num_devices() not in (1, 2):
        raise ValueError(
            "rounded register SwiGLU requires one qualification chip or TP2"
        )
    if x.dtype != ttnn.bfloat16 or x.layout != ttnn.TILE_LAYOUT:
        raise ValueError("rounded register SwiGLU requires tiled BF16 activation")
    for weight in (gate, up):
        if (
            weight.device() != device
            or weight.dtype != ttnn.bfloat4_b
            or weight.layout != ttnn.TILE_LAYOUT
        ):
            raise ValueError(
                "rounded register SwiGLU requires same-device tiled BF4 gate/up"
            )
        if weight.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            raise ValueError(
                "rounded register SwiGLU requires interleaved DRAM gate/up (1D decode cache)"
            )
    x_il = (
        x
        if x.memory_config() == ttnn.L1_MEMORY_CONFIG
        else ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
    )
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([*tuple(x.shape)[:-1], N]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.L1_MEMORY_CONFIG,
    )
    grid = device.compute_with_storage_grid_size()
    width, height = min(grid.x, 8), min(grid.y, 8)
    cores = ttnn.CoreRangeSet([
        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(width - 1, height - 1))
    ])
    reader, writer, compute = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    count = width * height
    for index in range(count):
        cx, cy = index % width, index // width
        tiles = (N // 32 - index + count - 1) // count
        reader[cx][cy] = [
            x_il.buffer_address(),
            gate.buffer_address(),
            up.buffer_address(),
            index,
            tiles,
            count,
        ]
        writer[cx][cy] = [output.buffer_address(), index, tiles, count]
        compute[cx][cy] = [tiles]
    kernel_root = Path(__file__).with_name("rounded_register_kernels")

    def kernel(name, config, args, runtime):
        return ttnn.KernelDescriptor(
            kernel_source=str(kernel_root / name),
            core_ranges=cores,
            compile_time_args=args,
            runtime_args=runtime,
            config=config,
        )

    cb_descriptors = []
    # BF4 tile: 512 mantissa bytes + 64 shared-exponent bytes. Double-buffer
    # two gate/up tiles together, so reader and compute never split a pair.
    for cb, dtype, page, pages in (
        (0, ttnn.bfloat16, 2048, 2),
        (1, ttnn.bfloat4_b, 576, 4),
        (16, ttnn.bfloat16, 2048, 2),
    ):
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=cb, data_format=dtype, page_size=page
                    )
                ],
            )
        )
    accessor_args = []
    for tensor in (x_il, gate, up):
        accessor_args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    program = ttnn.ProgramDescriptor(
        kernels=[
            kernel("reader.cpp", ttnn.ReaderConfigDescriptor(), accessor_args, reader),
            kernel(
                "writer.cpp",
                ttnn.WriterConfigDescriptor(),
                ttnn.TensorAccessorArgs(output).get_compile_time_args(),
                writer,
            ),
            kernel(
                "compute.cpp",
                ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                    fp32_dest_acc_en=True,
                    math_approx_mode=True,
                    dst_full_sync_en=False,
                ),
                [],
                compute,
            ),
        ],
        cbs=cb_descriptors,
        semaphores=[],
    )
    ttnn.generic_op([x_il, gate, up, output], program)
    if x_il is not x:
        ttnn.deallocate(x_il)
    return output
