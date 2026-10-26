"""Explicit hardware-only export using the existing packaged traced-decode adapters.

No devices are opened by importing this module. The caller owns service isolation.
"""

import os
from contextlib import contextmanager
from pathlib import Path

import retention


@contextmanager
def decode_flags(flags):
    old = {name: os.environ.get(name) for name in flags}
    os.environ.update(flags)
    try:
        yield
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def collect(corpus_path, model_path, package, runtime, cache, output, flags):
    import qwen36_service
    import torch
    import ttnn
    from models.demos.blackhole.qwen36.tt.model import Qwen36Model
    from qwen36_service import decode_component_probe as dcp
    from qwen36_service.decode_adapter import TracedDecodeAdapter
    from qwen36_service.decode_component_adapter import DecodeComponentProbeAdapter

    frozen = retention.corpus(corpus_path)
    model_path, package, runtime, cache = [
        str(Path(p).resolve(strict=True)) for p in (model_path, package, runtime, cache)
    ]
    if not Path(qwen36_service.__file__).resolve().is_relative_to(package) or not Path(
        ttnn.__file__
    ).resolve().is_relative_to(runtime):
        raise ValueError(
            "collector imports do not match supplied immutable package/runtime"
        )
    if os.environ.get("QWEN35_GDN_DECODE_FUSION") != "step":
        raise ValueError("collector requires QWEN35_GDN_DECODE_FUSION=step")
    if os.environ.get("QWEN38_ROUNDED_REGISTER_SWIGLU", "0") != "0":
        raise ValueError(
            "decode-only epilogue flag must be supplied via --decode-flags, not inherited during prefill"
        )
    if (
        retention.digest(Path(model_path) / "config.json")
        != frozen["model_config_sha256"]
    ):
        raise ValueError(
            "collector model configuration differs from frozen scoring domain"
        )
    for name, sha in frozen["tokenizer_assets_sha256"].items():
        if retention.digest(Path(model_path) / name) != sha:
            raise ValueError("collector tokenizer asset mismatch: " + name)
    weights = retention.weight_manifest(model_path)
    share = Path(package) / "share/qwen36"
    profile = dcp._load_component_profile(share / "decode-component-profile.json")
    decode_profile = dcp._load_decode_profile(share / "decode-profile.json")
    if tuple(profile.mesh_shape) != (1, 2) or tuple(ttnn.get_pcie_device_ids()) != (
        0,
        1,
    ):
        raise ValueError(
            "collector requires exact P150x2 device IDs 0,1 and 1x2 profile"
        )
    identity = dcp.DecodeComponentIdentity(
        runtime_identity=runtime,
        package_identity=package,
        model_identity=f"{dcp.MODEL_ALIAS}:{profile.checkpoint_revision}",
        authority_blake3=profile.authority_blake3,
        checkpoint_revision=profile.checkpoint_revision,
        model_source_revision=profile.model_source_revision,
        tokenizer_revision=profile.tokenizer_revision,
    )
    if frozen["checkpoint"] != profile.checkpoint_revision:
        raise ValueError("frozen checkpoint must match component profile")
    # All effective precision flags are retained, including explicit zero values.
    effective_flags = {
        k: v
        for k, v in os.environ.items()
        if k.startswith(("QWEN35_", "QWEN36_", "QWEN38_"))
    }
    effective_flags.update(flags)
    prov = {
        "model": retention.weight_identity(weights),
        "checkpoint": profile.checkpoint_revision,
        "runtime": runtime,
        "package": package,
        "cache": cache,
        "precision_flags": effective_flags,
        "execution": "hardware:P150x2:devices-0,1:B1:step",
    }
    os.environ.update(HF_MODEL=model_path, TT_CACHE_PATH=cache)
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
    model = None
    try:
        mesh = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(*profile.mesh_shape),
            l1_small_size=dcp.L1_SMALL_SIZE_BYTES,
            num_command_queues=1,
            trace_region_size=profile.trace_region_size_bytes,
        )
        mesh.enable_program_cache()
        # Treatment is decode-only: prefill and model setup stay on baseline route.
        model = Qwen36Model.from_pretrained(
            mesh, max_batch_size=1, max_seq_len=profile.maximum_sequence_length
        )
        adapter = TracedDecodeAdapter(
            model=model,
            profile=decode_profile,
            runtime_identity=runtime,
            package_identity=package,
            model_identity=identity.model_identity,
            slot_id="retention",
            buffer_generation=1,
        )
        probe = DecodeComponentProbeAdapter(
            model=model,
            decode_adapter=adapter,
            profile=profile,
            identity=identity,
            physical_pcie_device_ids=(0, 1),
        )
        arrays = []
        for passage in frozen["passages"]:
            ids = passage["input_ids"]
            probe._prefill(
                torch.tensor([ids[:12]], dtype=torch.int32),
                generation_tokens=max(296, len(ids)),
            )
            rows = []
            with decode_flags(flags):
                for position in range(12, len(ids) - 1):
                    adapter.decode_traced(token=ids[position], position=position)
                    rows.append(adapter.read_trace_logits().float().cpu().clone())
            arrays.append(torch.stack(rows))
        if retention.weight_manifest(model_path) != weights:
            raise ValueError("checkpoint content changed during model collection")
        torch.save(
            {
                "corpus_sha256": retention.digest(corpus_path),
                "provenance": prov,
                "weight_manifest": weights,
                "target_indices": [p["target_indices"] for p in frozen["passages"]],
                "rows": arrays,
            },
            output,
        )
    finally:
        if adapter is not None:
            adapter.close()
        if model is not None:
            model.free_kv_caches()
        if mesh is not None:
            for sub in mesh.get_submeshes():
                ttnn.close_mesh_device(sub)
            ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    return {
        "logits": str(output),
        "provenance": prov,
        "corpus_sha256": retention.digest(corpus_path),
    }
