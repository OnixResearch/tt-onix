# r[impl tenstorrent.qwen36.device_free_check]
# Device-free adoption gate for the tt-metal PR-53543 qwen36 stack.
#
# This checker runs without a Tenstorrent device. It imports the adopted
# qwen36 modules against the packaged tt-metal runtime, verifies the PR's
# opt-in BF4 env gates keep their documented defaults, and verifies the vLLM
# adapter imports when vLLM is absent (the PR's text-only 0.10 guard). It makes
# no silicon, placement, profiler, or performance claim.
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

FAILURES = []

_require_failed = {"value": False}


def require(condition, message):
    if not condition:
        _require_failed["value"] = True
        FAILURES.append(message)


def probe(label, fn):
    _require_failed["value"] = False
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        _require_failed["value"] = True
        FAILURES.append(f"{label}: {type(exc).__name__}: {exc}")
        print(f"FAIL {label}: {type(exc).__name__}: {exc}")
        return
    if _require_failed["value"]:
        print(f"FAIL {label} (invariant)")
    else:
        print(f"PASS {label}")


def import_ttnn():
    import ttnn  # noqa: F401


def import_transformer_config():
    import models.tt_transformers.tt.model_config  # noqa: F401


def import_gdn_ops():
    import models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops  # noqa: F401


def import_qwen36_common():
    import models.demos.blackhole.qwen36.tt.common  # noqa: F401


def import_qwen36_model_config():
    import models.demos.blackhole.qwen36.tt.model_config  # noqa: F401


def import_qwen36_model():
    import models.demos.blackhole.qwen36.tt.model  # noqa: F401


def import_qwen36_vllm_adapter():
    # The PR adds a try/except ImportError guard so text-only vLLM 0.10 works
    # without the Qwen3.5 multimodal helpers. Importing must succeed with vLLM
    # absent and _HAS_QWEN35_MM must be False.
    import importlib

    mod = importlib.import_module("models.demos.blackhole.qwen36.tt.qwen36_vllm")
    require(
        getattr(mod, "_HAS_QWEN35_MM", True) is False,
        "qwen36_vllm imported with vLLM absent but _HAS_QWEN35_MM is True",
    )


def probe_mlp_down_gate():
    import ttnn
    from models.demos.blackhole.qwen36.tt.mlp import _mlp_down_dtype_and_tag

    os.environ.pop("QWEN36_MLP_DOWN_BF4", None)
    dtype_default, tag_default = _mlp_down_dtype_and_tag()
    require(
        dtype_default == ttnn.bfloat8_b,
        "QWEN36_MLP_DOWN_BF4 default dtype is not bfloat8_b",
    )
    require(
        tag_default == "",
        "QWEN36_MLP_DOWN_BF4 default cache tag is not empty (would collide with BFP8)",
    )

    os.environ["QWEN36_MLP_DOWN_BF4"] = "1"
    dtype_bf4, tag_bf4 = _mlp_down_dtype_and_tag()
    require(dtype_bf4 == ttnn.bfloat4_b, "QWEN36_MLP_DOWN_BF4=1 dtype is not bfloat4_b")
    require(tag_bf4 == ".bfp4", "QWEN36_MLP_DOWN_BF4=1 cache tag is not .bfp4")
    require(
        tag_bf4 != tag_default, "BF4 cache tag collides with the default BFP8 cache tag"
    )
    os.environ.pop("QWEN36_MLP_DOWN_BF4", None)


def probe_lm_head_gate():
    import models.demos.blackhole.qwen36.tt.model as qwen36_model
    import ttnn

    source = open(qwen36_model.__file__, encoding="utf-8").read()
    require(
        "QWEN36_LM_HEAD_BF4" in source, "model.py lacks the QWEN36_LM_HEAD_BF4 env gate"
    )
    require(".bfp4" in source, "model.py lacks the distinct .bfp4 LM-head cache name")
    require(ttnn.bfloat8_b is not None, "ttnn.bfloat8_b missing")


def probe_source_env_gates():
    import models.demos.blackhole.qwen36.tt.gdn.tp as gdn_tp
    import models.demos.blackhole.qwen36.tt.gdn.weights as gdn_weights
    from models.demos.blackhole.qwen36.tt import mlp

    for module, name in (
        (gdn_weights, "gdn_proj_dtype_and_tag"),
        (mlp, "_mlp_down_dtype_and_tag"),
    ):
        require(hasattr(module, name), f"{module.__name__} lacks {name}")
    # The GDN env gate lives in gdn/weights.py (its sole owner). gdn/tp.py must
    # consume the tagged dtype helper instead of hardcoding BFP8.
    require(
        "QWEN36_GDN_BF4" in open(gdn_weights.__file__, encoding="utf-8").read(),
        "gdn/weights.py lacks the QWEN36_GDN_BF4 env gate",
    )
    require(
        "gdn_proj_dtype_and_tag" in open(gdn_tp.__file__, encoding="utf-8").read(),
        "gdn/tp.py does not consume gdn_proj_dtype_and_tag",
    )
    require(
        "bfloat8_b" not in open(gdn_tp.__file__, encoding="utf-8").read(),
        "gdn/tp.py hardcodes a projection dtype instead of using the tagged helper",
    )
    import models.tt_transformers.tt.model_config as ttc

    ttc_source = open(ttc.__file__, encoding="utf-8").read()
    require(
        "Qwen3.8-27B" in ttc_source,
        "model_config.py lacks the Qwen3.8-27B prefill-chunk entry",
    )
    require(
        "Qwen3.8-27B" in open(ttc.__file__, encoding="utf-8").read(),
        "model_config.py lacks the Qwen3.8-27B tokenizer entry",
    )


# r[verify tenstorrent.qwen38_p150x2.layer_type_authority]
def probe_layer_type_authority():
    """Positive and negative tests for checkpoint-owned hybrid dispatch."""
    from models.demos.blackhole.qwen36.tt import model_config as qwen_config
    from models.demos.blackhole.qwen36.tt import weight_mapping

    authority = json.loads(Path(__file__).with_name("authority.json").read_text())
    release = authority["release"]
    declared_layer_count = release["layer_count"]
    declared_full_attention_count = release["full_attention_layer_count"]
    declared_linear_attention_count = release["linear_attention_layer_count"]
    declared_full_attention_indices = tuple(release["full_attention_layer_indices"])
    full_attention_indices = frozenset(declared_full_attention_indices)
    checkpoint_layer_types = tuple(
        qwen_config.FULL_ATTENTION_TYPE
        if layer_index in full_attention_indices
        else qwen_config.LINEAR_ATTENTION_TYPE
        for layer_index in range(declared_layer_count)
    )

    resolved = qwen_config.resolve_attention_type_list(
        checkpoint_layer_types,
        declared_layer_count,
    )
    require(resolved == checkpoint_layer_types, "valid checkpoint layer types changed")
    require(
        resolved.count(qwen_config.FULL_ATTENTION_TYPE)
        == declared_full_attention_count,
        "full-attention layer count differs from authority",
    )
    require(
        resolved.count(qwen_config.LINEAR_ATTENTION_TYPE)
        == declared_linear_attention_count,
        "linear-attention layer count differs from authority",
    )
    require(
        tuple(
            layer_index
            for layer_index, layer_type in enumerate(resolved)
            if layer_type == qwen_config.FULL_ATTENTION_TYPE
        )
        == declared_full_attention_indices,
        "full-attention layer indices differ from authority",
    )

    legacy_layer_count = len(qwen_config.LEGACY_QWEN35_9B_LAYER_TYPES)
    require(
        qwen_config.resolve_attention_type_list(None, legacy_layer_count)
        == qwen_config.LEGACY_QWEN35_9B_LAYER_TYPES,
        "the exact legacy layer fallback changed",
    )

    unknown_layer_types = list(checkpoint_layer_types)
    unknown_layer_types[0] = "unknown_attention"
    rejected_cases = (
        (None, declared_layer_count, "required"),
        (checkpoint_layer_types[:-1], declared_layer_count, "entries"),
        (unknown_layer_types, declared_layer_count, "unknown type"),
        (qwen_config.LINEAR_ATTENTION_TYPE, declared_layer_count, "list or tuple"),
        (checkpoint_layer_types, 0, "positive integer"),
    )
    for layer_types, layer_count, diagnostic in rejected_cases:
        try:
            qwen_config.resolve_attention_type_list(layer_types, layer_count)
        except qwen_config.LayerTypeAuthorityError as error:
            require(
                diagnostic in str(error), f"wrong layer-authority diagnostic: {error}"
            )
        else:
            require(False, f"invalid layer authority was accepted: {layer_types!r}")

    weight_mapping_source = Path(weight_mapping.__file__).read_text()
    require(
        "FULL_ATTENTION_LAYERS" not in weight_mapping_source,
        "weight_mapping.py still declares the obsolete eight-entry mapper set",
    )


def probe_p150x2_gate():
    """TP=2 (P150x2 / P300) invariants carried by p150x2-tp.patch.

    Positive: the adopted tree registers P150x2 in the mesh maps, registers a
    P300 prefill-chunk value for the 27B checkpoints, and advertises host
    sampling for uncertified TP. It also verifies the packaged runtime disables
    unsupported TP=2 GDN and MLP fusions. Negative: the on-device sampling
    certification set must stay limited to 1x4/1x8.
    """
    import pathlib

    qwen36 = pathlib.Path(
        __import__("models.demos.blackhole.qwen36", fromlist=["__path__"]).__path__[0]
    )

    text_demo = (qwen36 / "demo" / "text_demo.py").read_text()
    require(
        '"P150x2": (1, 2)' in text_demo, "text_demo.py lacks the P150x2 mesh map entry"
    )
    require(
        '"P150"' in text_demo and '"P150x8"' in text_demo,
        "text_demo.py lost an existing mesh map entry",
    )

    test_factory = (qwen36 / "tests" / "test_factory.py").read_text()
    require(
        '"P150x2": (1, 2)' in test_factory,
        "test_factory.py lacks the P150x2 mesh map entry",
    )
    require(
        '}.get(os.environ.get("MESH_DEVICE"))' in test_factory,
        "test_factory.py mesh resolver changed shape",
    )

    vllm_adapter = (qwen36 / "tt" / "qwen36_vllm.py").read_text()
    require(
        "_supports_on_device_sampling" in vllm_adapter,
        "qwen36_vllm.py lacks the host-sampling override for uncertified TP",
    )
    require(
        '"supports_sample_on_device": True' in vllm_adapter,
        "qwen36_vllm.py lost the certified-topology on-device default",
    )
    require(
        "self.model_capabilities" in vllm_adapter,
        "qwen36_vllm.py lacks the instance capabilities override",
    )

    model_source = (qwen36 / "tt" / "model.py").read_text()
    require(
        "(1, 4), (1, 8)" in model_source,
        "on-device sampling certification set must stay 1x4/1x8 for TP=2 host sampling",
    )

    # Real-path probe of get_max_prefill_chunk_size: unbound call with a minimal
    # stub, MAX_PREFILL_CHUNK_SIZE unset so the table is exercised.
    ttc = __import__(
        "models.tt_transformers.tt.model_config",
        fromlist=["ModelArgs"],
    )
    require(ttc.ModelArgs is not None, "ModelArgs missing from model_config")
    os.environ.pop("MAX_PREFILL_CHUNK_SIZE", None)

    for model_name in ("Qwen3.5-27B", "Qwen3.6-27B", "Qwen3.8-27B"):
        stub_p300 = SimpleNamespace(
            base_model_name=model_name, device_name="P300", model_name=model_name
        )
        chunk_p300 = ttc.ModelArgs.get_max_prefill_chunk_size(stub_p300)
        require(
            chunk_p300 == 2 * 1024,
            f"{model_name} P300 prefill chunk != 2K ({chunk_p300})",
        )

        stub_x4 = SimpleNamespace(
            base_model_name=model_name, device_name="P150x4", model_name=model_name
        )
        chunk_x4 = ttc.ModelArgs.get_max_prefill_chunk_size(stub_x4)
        require(
            chunk_x4 == 2 * 1024,
            f"{model_name} P150x4 prefill chunk changed ({chunk_x4})",
        )

        # Negative: unknown device must not crash; the table falls back to 4K.
        stub_unknown = SimpleNamespace(
            base_model_name=model_name, device_name="P150x16", model_name=model_name
        )
        chunk_unknown = ttc.ModelArgs.get_max_prefill_chunk_size(stub_unknown)
        require(
            chunk_unknown == 4 * 1024,
            f"{model_name} unknown-device fallback changed ({chunk_unknown})",
        )

    # 2-device Blackhole meshes resolve to the P300 device name by construct.
    tt_transformers_config = (
        qwen36.parent.parent.parent / "tt_transformers" / "tt" / "model_config.py"
    ).read_text()
    require(
        '2: "P300"' in tt_transformers_config,
        "device-name mapping lost the 2-device P300 entry",
    )

    # generate-tp-reset: reset_tp must dispatch by layer attention API (the
    # single-device full-attention layer only exposes reset_cache).
    require(
        "attention.reset_cache()" in model_source,
        "reset_tp lost its reset_cache fallback for single-device full attention",
    )
    require(
        'hasattr(attention, "reset_state")' in model_source,
        "reset_tp lost its reset_state dispatch",
    )

    # gdn-seq-fallback: fused_chunk_enabled must probe the runtime for the
    # fused op instead of hardcoding True (the packaged 0.74 bindings lack
    # ttnn.transformer.chunk_gated_delta_rule and must use the seq adapter).
    fused_chunk = (qwen36 / "tt" / "gdn" / "fused_chunk.py").read_text()
    require(
        'hasattr(ttnn.transformer, "chunk_gated_delta_rule")' in fused_chunk,
        "fused_chunk_enabled lost its runtime probe for chunk_gated_delta_rule",
    )
    fn_body = fused_chunk.split("def fused_chunk_enabled", 1)[1].split("\ndef ", 1)[0]
    require(
        "return True" not in fn_body,
        "fused_chunk_enabled must not return a constant",
    )
    import ttnn as _ttnn

    require(
        not hasattr(_ttnn.transformer, "chunk_gated_delta_rule"),
        "packaged runtime unexpectedly exposes chunk_gated_delta_rule; revisit the seq fallback",
    )
    require(
        hasattr(_ttnn.transformer, "gated_delta_attn_seq"),
        "packaged runtime lacks gated_delta_attn_seq for the seq fallback",
    )

    # tp2-runtime-compat: the TP=4-tuned fused GDN output reduction waits on
    # P150x2. The packaged AGMM binding also rejects its newer fuse_swiglu
    # keyword. Keep both safe fallbacks capability- and topology-gated.
    import models.demos.blackhole.qwen36.tt.gdn.tp as gdn_tp
    from models.demos.blackhole.qwen36.tt import tp_common

    p150x2_device_count = 2
    p150x4_device_count = 4
    require(
        gdn_tp.MIN_FUSED_PREFILL_REDUCTION_DEVICES == p150x4_device_count,
        "GDN fused prefill reduction must start at the admitted P150x4 topology",
    )
    require(
        p150x2_device_count < gdn_tp.MIN_FUSED_PREFILL_REDUCTION_DEVICES,
        "P150x2 must not enter the hanging fused GDN output reduction",
    )
    require(
        tp_common._runtime_supports_fused_swiglu("(..., fuse_swiglu=True)"),
        "fused-SwiGLU capability probe rejects a positive signature",
    )
    require(
        not tp_common._runtime_supports_fused_swiglu("(..., fused_activation=None)"),
        "fused-SwiGLU capability probe accepts a signature without fuse_swiglu",
    )
    require(
        not tp_common._runtime_supports_fused_swiglu(),
        "packaged 0.74 runtime unexpectedly advertises fuse_swiglu; revisit the fallback",
    )
    require(
        not tp_common.mlp_gateup_agmm_enabled(p150x2_device_count),
        "P150x2 enables MLP gate/up AGMM despite the missing runtime keyword",
    )


# r[verify tenstorrent.qwen38_p150x2.traced_decode]
# r[verify tenstorrent.qwen38_p150x2.persistent_decode_buffers]
# r[verify tenstorrent.qwen38_p150x2.shard_local_argmax]
# r[verify tenstorrent.qwen38_p150x2.decode_performance_admission]
# r[verify tenstorrent.qwen38_p150x2.traced_decode_ttft]
# r[verify tenstorrent.qwen38_p150x2.traced_decode_ttft_evidence]
def probe_traced_decode_contract():
    """Positive and negative tests for traced-decode policy and admission."""
    from qwen36_service.decode_core import (
        DecodeEvidence,
        DecodeMetrics,
        DecodePolicyError,
        DecodeProfile,
        DecodeSlot,
        ShardWinner,
        SlotEvent,
        SlotState,
        TraceBinding,
        admit_decode_candidate,
        request_deadline_expired,
        required_page_table_blocks,
        select_global_winner,
        transition_slot,
        validate_trace_binding,
    )

    profile_path = Path(__file__).with_name("decode-profile.json")
    with profile_path.open(encoding="utf-8") as profile_file:
        profile = DecodeProfile.from_mapping(json.load(profile_file))
    require(profile.trace_enabled, "decode profile disabled trace capture")
    require(
        not profile.physically_admitted,
        "unmeasured decode candidate is marked physically admitted",
    )

    initial_generation = 0
    initial_buffer_generation = 1
    test_token_index = 2
    alternate_token_index = 6
    test_vocabulary_size = 8
    test_shard_boundary = test_vocabulary_size // 2
    test_tie_logit = 7.0
    baseline_decode_rate = 4.0
    candidate_decode_rate = 5.0
    baseline_ttft_seconds = 0.5
    candidate_ttft_seconds = 0.5
    baseline_tail_seconds = 0.3
    candidate_tail_seconds = 0.3
    slot = DecodeSlot(
        slot_id="slot-test",
        ownership_generation=initial_generation,
        buffer_generation=initial_buffer_generation,
        state=SlotState.EMPTY,
    )
    request_id = "request-test"
    slot = transition_slot(slot, SlotEvent.ADMIT, request_id=request_id)
    slot = transition_slot(slot, SlotEvent.ACTIVATE, request_id=request_id)
    slot = transition_slot(slot, SlotEvent.BEGIN_RESET, request_id=request_id)
    slot = transition_slot(slot, SlotEvent.COMPLETE_RESET)
    require(slot.state is SlotState.EMPTY, "slot did not return to empty")
    require(
        slot.buffer_generation == initial_buffer_generation,
        "clean reset changed the stable buffer generation",
    )

    failed_slot = transition_slot(slot, SlotEvent.ADMIT, request_id=request_id)
    failed_slot = transition_slot(
        failed_slot,
        SlotEvent.BEGIN_RESET,
        request_id=request_id,
    )
    failed_slot = transition_slot(
        failed_slot,
        SlotEvent.RESET_FAILED,
        quarantine_reason="negative-fixture",
    )
    require(
        failed_slot.state is SlotState.QUARANTINED,
        "failed reset did not quarantine the slot",
    )

    try:
        transition_slot(slot, SlotEvent.ACTIVATE, request_id=request_id)
    except DecodePolicyError as error:
        require("admitted owner" in str(error), "wrong invalid-transition diagnostic")
    else:
        require(False, "invalid slot transition was accepted")

    binding = TraceBinding(
        runtime_identity="runtime",
        package_identity="package",
        model_identity="model",
        profile_id=profile.profile_id,
        slot_id="slot-test",
        buffer_generation=initial_buffer_generation,
        token_buffer_identity="token",
        position_buffer_identity="position",
        rope_buffer_identity="rope",
        page_table_buffer_identity="page-table",
        kv_buffer_identity="kv",
        gdn_buffer_identity="gdn",
        output_buffer_identity="output",
        output_shape=(1, 1),
        output_dtype="bfloat16",
    )
    require(not validate_trace_binding(binding, binding), "equal trace bindings differ")
    stale_binding = TraceBinding(**{
        **binding.__dict__,
        "buffer_generation": binding.buffer_generation + 1,
    })
    require(
        validate_trace_binding(binding, stale_binding) == ("buffer_generation",),
        "stale trace binding did not name the generation mismatch",
    )

    selected = select_global_winner(
        (
            ShardWinner(0, test_shard_boundary, test_token_index, test_tie_logit),
            ShardWinner(
                test_shard_boundary,
                test_vocabulary_size,
                alternate_token_index,
                test_tie_logit,
            ),
        ),
        test_vocabulary_size,
    )
    require(
        selected == test_token_index,
        "equal shard maxima did not select the lowest token",
    )
    try:
        select_global_winner(
            (
                ShardWinner(0, test_shard_boundary, test_token_index, math.nan),
                ShardWinner(
                    test_shard_boundary,
                    test_vocabulary_size,
                    alternate_token_index,
                    test_tie_logit,
                ),
            ),
            test_vocabulary_size,
        )
    except DecodePolicyError as error:
        require("finite" in str(error), "wrong non-finite shard diagnostic")
    else:
        require(False, "non-finite shard maximum was accepted")

    request_start = 10.0
    request_deadline = 11.0
    require(
        not request_deadline_expired(request_start, request_deadline),
        "request timed out before its deadline",
    )
    require(
        request_deadline_expired(request_deadline, request_deadline),
        "request did not time out at its deadline",
    )

    baseline = DecodeMetrics(
        median_decode_tokens_per_second=baseline_decode_rate,
        median_ttft_seconds=baseline_ttft_seconds,
        p99_token_latency_seconds=baseline_tail_seconds,
    )
    candidate = DecodeMetrics(
        median_decode_tokens_per_second=candidate_decode_rate,
        median_ttft_seconds=candidate_ttft_seconds,
        p99_token_latency_seconds=candidate_tail_seconds,
    )
    admitted = admit_decode_candidate(
        profile,
        DecodeEvidence(
            physical=True,
            identities_match=True,
            exact_tokens_match=True,
            clean_reset=True,
            baseline=baseline,
            candidate=candidate,
        ),
    )
    require(
        admitted.admitted, f"passing decode evidence was rejected: {admitted.reasons}"
    )
    rejected = admit_decode_candidate(
        profile,
        DecodeEvidence(
            physical=False,
            identities_match=True,
            exact_tokens_match=True,
            clean_reset=False,
            baseline=baseline,
            candidate=candidate,
        ),
    )
    require(not rejected.admitted, "device-free dirty evidence was admitted")
    require(
        rejected.reasons == ("physical-evidence-missing", "cleanup-failed"),
        f"wrong decode rejection reasons: {rejected.reasons}",
    )

    test_kv_block_size = 64
    test_capacity_blocks = 32
    exact_prompt_tokens = 32
    exact_generation_tokens = 32
    rounded_prompt_tokens = 5
    rounded_generation_tokens = 64
    rounded_required_blocks = 2
    maximum_prompt_tokens = 1984
    maximum_generation_tokens = 64
    require(
        required_page_table_blocks(
            prompt_tokens=exact_prompt_tokens,
            generation_tokens=exact_generation_tokens,
            kv_block_size=test_kv_block_size,
            capacity_blocks=test_capacity_blocks,
        )
        == 1,
        "exact page-table request did not use one block",
    )
    require(
        required_page_table_blocks(
            prompt_tokens=rounded_prompt_tokens,
            generation_tokens=rounded_generation_tokens,
            kv_block_size=test_kv_block_size,
            capacity_blocks=test_capacity_blocks,
        )
        == rounded_required_blocks,
        "rounded page-table request did not use two blocks",
    )
    require(
        required_page_table_blocks(
            prompt_tokens=maximum_prompt_tokens,
            generation_tokens=maximum_generation_tokens,
            kv_block_size=test_kv_block_size,
            capacity_blocks=test_capacity_blocks,
        )
        == test_capacity_blocks,
        "maximum page-table request did not use full capacity",
    )

    invalid_block_inputs = (
        ({"prompt_tokens": 0}, "prompt token count"),
        ({"generation_tokens": 0}, "generation token count"),
        ({"kv_block_size": 0}, "KV block size"),
        ({"capacity_blocks": 0}, "page-table capacity"),
        ({"prompt_tokens": True}, "prompt token count"),
    )
    base_block_inputs = {
        "prompt_tokens": rounded_prompt_tokens,
        "generation_tokens": rounded_generation_tokens,
        "kv_block_size": test_kv_block_size,
        "capacity_blocks": test_capacity_blocks,
    }
    for override, diagnostic in invalid_block_inputs:
        try:
            required_page_table_blocks(**{**base_block_inputs, **override})
        except DecodePolicyError as error:
            require(diagnostic in str(error), f"wrong block-count diagnostic: {error}")
        else:
            require(False, f"invalid block-count input was accepted: {override!r}")

    overflow_prompt_tokens = test_kv_block_size * test_capacity_blocks
    try:
        required_page_table_blocks(
            prompt_tokens=overflow_prompt_tokens,
            generation_tokens=1,
            kv_block_size=test_kv_block_size,
            capacity_blocks=test_capacity_blocks,
        )
    except DecodePolicyError as error:
        require("capacity" in str(error), "wrong page-table capacity diagnostic")
    else:
        require(False, "capacity-exceeding page-table request was accepted")

    import torch
    from qwen36_service.decode_adapter import TracedDecodeAdapter

    adapter = object.__new__(TracedDecodeAdapter)
    adapter.profile = SimpleNamespace(kv_block_size=test_kv_block_size)
    adapter.page_table = torch.arange(test_capacity_blocks, dtype=torch.int32).reshape(
        1, test_capacity_blocks
    )
    adapter._trace_id = None
    adapter._trace_prefill_page_table_blocks = None
    stable_table_identity = id(adapter.page_table)
    bounded_table = adapter.prefill_page_table(
        prompt_tokens=rounded_prompt_tokens,
        generation_tokens=rounded_generation_tokens,
    )
    require(
        tuple(bounded_table.shape) == (1, rounded_required_blocks),
        "adapter prefill table has the wrong bounded shape",
    )
    require(
        torch.equal(bounded_table, adapter.page_table[:, :rounded_required_blocks]),
        "adapter prefill table is not the stable table prefix",
    )
    require(
        id(adapter.page_table) == stable_table_identity,
        "adapter replaced the stable decode page table",
    )
    require(
        adapter.decode_page_table_blocks == test_capacity_blocks,
        "adapter decode table capacity changed after prefill slicing",
    )
    adapter._trace_id = "test-trace"
    trace_release_count = []

    def release_test_trace():
        trace_release_count.append(1)
        adapter._trace_id = None
        adapter._trace_prefill_page_table_blocks = None

    adapter._release_trace = release_test_trace
    different_prompt_tokens = rounded_prompt_tokens + test_kv_block_size
    resized_table = adapter.prefill_page_table(
        prompt_tokens=different_prompt_tokens,
        generation_tokens=rounded_generation_tokens,
    )
    require(
        trace_release_count == [1],
        "post-capture prefill-width change did not release the parked trace",
    )
    require(
        tuple(resized_table.shape) == (1, rounded_required_blocks + 1),
        "post-capture prefill-width change selected the wrong prefix",
    )
    require(
        id(adapter.page_table) == stable_table_identity,
        "width transition replaced the stable decode page table",
    )


# r[verify tenstorrent.qwen38_p150x2.prefill_program_profile]
# r[verify tenstorrent.qwen38_p150x2.host_timeline_profile]
def probe_prefill_program_profile():
    """Positive and negative tests for prefill profiling core and adapter."""
    from qwen36_service.prefill_profile_adapter import (
        PrefillProfilerAdapterError,
        RealtimePrefillProfiler,
    )
    from qwen36_service.prefill_profile_core import (
        MEASUREMENT_MODE_ADMISSION,
        MEASUREMENT_MODE_DIAGNOSTIC_HOST_TIMELINE,
        MEASUREMENT_MODE_DIAGNOSTIC_PREFILL_PROFILE,
        PrefillProfileError,
        RealtimeProgramObservation,
        select_measurement_mode,
        summarize_prefill_programs,
    )

    admission_mode = select_measurement_mode(
        prefill_program_profiled=False,
        host_timeline_profiled=False,
    )
    require(
        admission_mode == (MEASUREMENT_MODE_ADMISSION, True),
        "unprofiled requests lost admission eligibility",
    )
    host_mode = select_measurement_mode(
        prefill_program_profiled=False,
        host_timeline_profiled=True,
    )
    require(
        host_mode == (MEASUREMENT_MODE_DIAGNOSTIC_HOST_TIMELINE, False),
        "host timeline requests were not marked diagnostic",
    )
    program_mode = select_measurement_mode(
        prefill_program_profiled=True,
        host_timeline_profiled=False,
    )
    require(
        program_mode == (MEASUREMENT_MODE_DIAGNOSTIC_PREFILL_PROFILE, False),
        "prefill program requests lost their diagnostic mode",
    )
    try:
        select_measurement_mode(
            prefill_program_profiled=True,
            host_timeline_profiled=True,
        )
    except PrefillProfileError:
        pass
    else:
        require(False, "conflicting prefill diagnostics were accepted")

    expected_chip_ids = (0, 1)
    first_runtime_id = 10
    second_runtime_id = 11
    first_start_timestamp = 100
    first_end_timestamp = 300
    second_start_timestamp = 200
    second_end_timestamp = 500
    third_start_timestamp = 0
    third_end_timestamp = 100
    frequency_cycles_per_ns = 2.0
    first_chip_expected_total_ns = 200.0
    second_chip_expected_total_ns = 150.0
    receiver_drain_seconds = 0.25
    observations = (
        RealtimeProgramObservation(
            runtime_id=first_runtime_id,
            chip_id=expected_chip_ids[0],
            start_timestamp=first_start_timestamp,
            end_timestamp=first_end_timestamp,
            frequency_cycles_per_ns=frequency_cycles_per_ns,
            kernel_sources=("kernel-b.cpp", "kernel-a.cpp", "kernel-a.cpp"),
        ),
        RealtimeProgramObservation(
            runtime_id=first_runtime_id,
            chip_id=expected_chip_ids[1],
            start_timestamp=second_start_timestamp,
            end_timestamp=second_end_timestamp,
            frequency_cycles_per_ns=frequency_cycles_per_ns,
            kernel_sources=("kernel-a.cpp", "kernel-b.cpp"),
        ),
        RealtimeProgramObservation(
            runtime_id=second_runtime_id,
            chip_id=expected_chip_ids[0],
            start_timestamp=third_start_timestamp,
            end_timestamp=third_end_timestamp,
            frequency_cycles_per_ns=1.0,
            kernel_sources=("kernel-c.cpp",),
        ),
    )
    summary = summarize_prefill_programs(
        observations,
        expected_chip_ids=expected_chip_ids,
    )
    require(summary["record_count"] == len(observations), "profile lost records")
    require(
        summary["runtime_ids"] == [first_runtime_id, second_runtime_id],
        "profile runtime IDs are not deterministic",
    )
    require(
        summary["by_chip"][0]["total_duration_ns"] == first_chip_expected_total_ns,
        "profile computed the wrong first-chip total",
    )
    require(
        summary["by_chip"][1]["total_duration_ns"] == second_chip_expected_total_ns,
        "profile computed the wrong second-chip total",
    )
    require(
        summary["signatures"][0]["kernel_sources"] == ["kernel-a.cpp", "kernel-b.cpp"],
        "profile did not normalize or rank the leading signature",
    )

    invalid_observations = (
        (),
        (
            RealtimeProgramObservation(
                runtime_id=first_runtime_id,
                chip_id=2,
                start_timestamp=first_start_timestamp,
                end_timestamp=first_end_timestamp,
                frequency_cycles_per_ns=frequency_cycles_per_ns,
                kernel_sources=("kernel.cpp",),
            ),
        ),
        (
            RealtimeProgramObservation(
                runtime_id=0,
                chip_id=expected_chip_ids[0],
                start_timestamp=first_start_timestamp,
                end_timestamp=first_end_timestamp,
                frequency_cycles_per_ns=frequency_cycles_per_ns,
                kernel_sources=("kernel.cpp",),
            ),
        ),
        (
            RealtimeProgramObservation(
                runtime_id=first_runtime_id,
                chip_id=expected_chip_ids[0],
                start_timestamp=first_start_timestamp,
                end_timestamp=first_start_timestamp,
                frequency_cycles_per_ns=frequency_cycles_per_ns,
                kernel_sources=("kernel.cpp",),
            ),
        ),
        (
            RealtimeProgramObservation(
                runtime_id=first_runtime_id,
                chip_id=expected_chip_ids[0],
                start_timestamp=first_start_timestamp,
                end_timestamp=first_end_timestamp,
                frequency_cycles_per_ns=math.inf,
                kernel_sources=("kernel.cpp",),
            ),
        ),
        (
            RealtimeProgramObservation(
                runtime_id=first_runtime_id,
                chip_id=expected_chip_ids[0],
                start_timestamp=first_start_timestamp,
                end_timestamp=first_end_timestamp,
                frequency_cycles_per_ns=frequency_cycles_per_ns,
                kernel_sources=(),
            ),
        ),
    )
    for invalid in invalid_observations:
        try:
            summarize_prefill_programs(
                invalid,
                expected_chip_ids=expected_chip_ids,
            )
        except PrefillProfileError:
            pass
        else:
            require(False, f"invalid profile observations were accepted: {invalid!r}")

    class FakeDeviceApi:
        def __init__(self, *, active):
            self.active = active
            self.callback = None
            self.unregistered_handle = None

        def IsProgramRealtimeProfilerActive(self):
            return self.active

        def RegisterProgramRealtimeProfilerCallback(self, callback):
            self.callback = callback
            return first_runtime_id

        def UnregisterProgramRealtimeProfilerCallback(self, handle):
            self.unregistered_handle = handle

    fake_api = FakeDeviceApi(active=True)
    sleep_calls = []
    profiler = RealtimePrefillProfiler(
        fake_api,
        expected_chip_ids=expected_chip_ids,
        receiver_drain_seconds=receiver_drain_seconds,
        sleep=sleep_calls.append,
    )
    profiler.begin()
    require(fake_api.callback is not None, "profile adapter did not register callback")
    fake_api.callback(
        SimpleNamespace(
            runtime_id=first_runtime_id,
            chip_id=expected_chip_ids[0],
            start_timestamp=first_start_timestamp,
            end_timestamp=first_end_timestamp,
            frequency=frequency_cycles_per_ns,
            kernel_sources=("kernel.cpp",),
        )
    )
    adapter_summary = profiler.finish()
    require(adapter_summary["record_count"] == 1, "profile adapter lost callback data")
    require(
        sleep_calls == [receiver_drain_seconds, receiver_drain_seconds],
        "profile adapter did not bound both receiver drains",
    )
    profiler.begin()
    try:
        profiler.begin()
    except PrefillProfilerAdapterError:
        pass
    else:
        require(False, "profile adapter accepted a nested region")
    profiler.cancel()
    profiler.close()
    require(
        fake_api.unregistered_handle == first_runtime_id,
        "profile adapter did not unregister its callback",
    )

    try:
        RealtimePrefillProfiler(
            FakeDeviceApi(active=False),
            expected_chip_ids=expected_chip_ids,
            receiver_drain_seconds=receiver_drain_seconds,
            sleep=sleep_calls.append,
        )
    except PrefillProfilerAdapterError as error:
        require("inactive" in str(error), "inactive profiler had the wrong error")
    else:
        require(False, "inactive real-time profiler was accepted")


# r[verify tenstorrent.qwen38_p150x2.mtp_verification_probe]
# r[verify tenstorrent.qwen38_p150x2.native_mtp_verifier]
def probe_mtp_target_verification():
    """Positive and negative tests for target-only MTP evidence."""
    from dataclasses import replace
    from types import SimpleNamespace

    import torch
    from qwen36_service.mtp_probe_adapter import (
        MtpProbeAdapterError,
        compare_tensor_collections,
        compare_tp_gdn_states,
    )
    from qwen36_service.mtp_probe_core import (
        MtpProbeIdentity,
        MtpProbePolicyError,
        MtpProbeProfile,
        MtpProbeSample,
        evaluate_mtp_probe,
    )
    from qwen36_service.native_mtp_verifier_adapter import (
        NativeMtpVerifierError,
        validate_native_verifier_model,
    )

    profile_path = Path(__file__).with_name("mtp-probe-profile.json")
    with profile_path.open(encoding="utf-8") as profile_file:
        profile_source = json.load(profile_file)
    profile = MtpProbeProfile.from_mapping(profile_source)
    require(profile.diagnostic_only, "MTP target probe became admission eligible")
    require(
        profile.execution_path == "traced-hybrid-packed-decode",
        "MTP target probe lost the native execution path",
    )

    state_source_values = tuple(
        float(value) for value in range(profile.native_width_limit)
    )
    state_source = torch.tensor(state_source_values, dtype=torch.float32)
    exact_snapshot = ((state_source, (state_source,)),)
    recurrent_comparison, convolution_comparison = compare_tp_gdn_states(
        exact_snapshot,
        exact_snapshot,
    )
    require(recurrent_comparison.pcc == 1.0, "exact recurrent state lost PCC")
    require(recurrent_comparison.nmse == 0.0, "exact recurrent state gained NMSE")
    require(convolution_comparison.pcc == 1.0, "exact convolution state lost PCC")
    tensor_comparison = compare_tensor_collections(
        (state_source,),
        (state_source,),
    )
    require(tensor_comparison.pcc == 1.0, "exact tensor collection lost PCC")
    require(tensor_comparison.nmse == 0.0, "exact tensor collection gained NMSE")
    try:
        compare_tp_gdn_states(
            exact_snapshot,
            ((state_source[:-1], (state_source,)),),
        )
    except MtpProbeAdapterError as error:
        require("shapes" in str(error), "wrong state-shape diagnostic")
    else:
        require(False, "mismatched state shapes were accepted")

    fake_layer = SimpleNamespace(
        args=object(),
        attention=object(),
        attention_norm=object(),
        feed_forward=object(),
        ffn_norm=object(),
        is_full_attention=False,
    )
    fake_model = SimpleNamespace(
        args=object(),
        embd=object(),
        layers=[fake_layer],
        mesh_device=object(),
        prepare_inputs_decode=object(),
        _final_norm_decode=object(),
        _lm_head=object(),
    )
    validate_native_verifier_model(fake_model)
    invalid_model = SimpleNamespace(**{
        key: value for key, value in vars(fake_model).items() if key != "_lm_head"
    })
    try:
        validate_native_verifier_model(invalid_model)
    except NativeMtpVerifierError as error:
        require("_lm_head" in str(error), "wrong missing-seam diagnostic")
    else:
        require(False, "native verifier accepted a missing model seam")

    native_source = Path(
        __import__(
            "qwen36_service.native_mtp_verifier_adapter",
            fromlist=["__file__"],
        ).__file__
    ).read_text()
    require(
        "recurrent_gated_delta_rule_decode_ttnn" in native_source,
        "native verifier lost ordered high-precision GDN recurrence",
    )
    require(
        "for row in range(width)" in native_source
        and "paged_update_cache" in native_source,
        "native verifier lost serialized paged-KV writes",
    )
    require(
        "layer.feed_forward.forward" in native_source,
        "native verifier lost width-batched MLP execution",
    )
    require(
        "begin_trace_capture" in native_source and "execute_trace" in native_source,
        "native verifier lost traced replay",
    )

    identity = MtpProbeIdentity(
        runtime_identity="runtime-test",
        package_identity="package-test",
        model_identity="model-test",
    )
    baseline_token_seconds = 0.060
    width_one_verification_seconds = 0.065
    width_two_verification_seconds = 0.080
    width_four_verification_seconds = 0.110
    readback_seconds = 0.002
    width_one, width_two, width_four = profile.verification_widths
    verification_seconds = {
        width_one: width_one_verification_seconds,
        width_two: width_two_verification_seconds,
        width_four: width_four_verification_seconds,
    }
    samples = []
    for width in profile.verification_widths:
        token_ids = tuple(range(width))
        for sample_index in range(1, profile.minimum_samples_per_width + 1):
            samples.append(
                MtpProbeSample(
                    width=width,
                    sample_index=sample_index,
                    identity=identity,
                    physical_pcie_device_ids=profile.expected_pcie_device_ids,
                    sequential_token_ids=token_ids,
                    verification_token_ids=token_ids,
                    sequential_seconds=baseline_token_seconds * width,
                    verification_trace_capture_seconds=baseline_token_seconds,
                    verification_device_seconds=verification_seconds[width],
                    verification_readback_seconds=readback_seconds,
                    logits_pcc=1.0,
                    logits_nmse=0.0,
                    recurrent_state_pcc=1.0,
                    recurrent_state_nmse=0.0,
                    convolution_state_pcc=1.0,
                    convolution_state_nmse=0.0,
                    paged_kv_state_pcc=1.0,
                    paged_kv_state_nmse=0.0,
                )
            )

    decision = evaluate_mtp_probe(profile, identity, samples)
    require(
        decision.target_verification_valid,
        f"valid target-verification evidence was rejected: {decision.reasons}",
    )
    require(not decision.mtp_selected, "target-only evidence selected MTP")
    require(
        not decision.admission_eligible,
        "target-only evidence became admission eligible",
    )
    require(
        decision.width_summaries[-1].device_amortization_ratio > 1.0,
        "width-four fixture did not report device amortization",
    )

    alternate_token_id = profile.native_width_limit + 1
    token_mismatch = list(samples)
    token_mismatch[0] = replace(
        token_mismatch[0], verification_token_ids=(alternate_token_id,)
    )
    mismatch_decision = evaluate_mtp_probe(profile, identity, token_mismatch)
    require(
        "width-1-token-parity" in mismatch_decision.reasons,
        "target-token mismatch was accepted",
    )

    pcc_failure_margin = 0.1
    logit_mismatch = list(samples)
    logit_mismatch[0] = replace(
        logit_mismatch[0],
        logits_pcc=profile.minimum_logits_pcc - pcc_failure_margin,
    )
    logit_decision = evaluate_mtp_probe(profile, identity, logit_mismatch)
    require(
        "width-1-logits-pcc" in logit_decision.reasons,
        "full-logit mismatch was accepted",
    )

    state_mismatch = list(samples)
    state_mismatch[0] = replace(
        state_mismatch[0],
        recurrent_state_pcc=profile.minimum_state_pcc - pcc_failure_margin,
    )
    state_decision = evaluate_mtp_probe(profile, identity, state_mismatch)
    require(
        "width-1-recurrent-state-pcc" in state_decision.reasons,
        "recurrent-state mismatch was accepted",
    )
    kv_mismatch = list(samples)
    kv_mismatch[0] = replace(
        kv_mismatch[0],
        paged_kv_state_pcc=profile.minimum_state_pcc - pcc_failure_margin,
    )
    kv_decision = evaluate_mtp_probe(profile, identity, kv_mismatch)
    require(
        "width-1-paged-kv-state-pcc" in kv_decision.reasons,
        "paged-KV mismatch was accepted",
    )

    largest_width = profile.verification_widths[-1]
    incomplete = [
        sample
        for sample in samples
        if not (
            sample.width == largest_width
            and sample.sample_index == profile.minimum_samples_per_width
        )
    ]
    incomplete_decision = evaluate_mtp_probe(profile, identity, incomplete)
    require(
        f"width-{largest_width}-sample-count" in incomplete_decision.reasons,
        "incomplete width evidence was accepted",
    )

    unstable_multiplier = 2.0
    unstable = list(samples)
    unstable[-1] = replace(
        unstable[-1],
        verification_device_seconds=(
            unstable[-1].verification_device_seconds * unstable_multiplier
        ),
    )
    unstable_decision = evaluate_mtp_probe(profile, identity, unstable)
    require(
        f"width-{largest_width}-sample-instability" in unstable_decision.reasons,
        "unstable width evidence was accepted",
    )
    noisy_readback = list(samples)
    noisy_readback[-1] = replace(
        noisy_readback[-1],
        verification_readback_seconds=(
            noisy_readback[-1].verification_readback_seconds * unstable_multiplier
        ),
    )
    noisy_readback_decision = evaluate_mtp_probe(
        profile,
        identity,
        noisy_readback,
    )
    require(
        f"width-{largest_width}-sample-instability"
        not in noisy_readback_decision.reasons,
        "diagnostic host readback jitter invalidated traced replay stability",
    )

    slow_verification_seconds = baseline_token_seconds * largest_width
    slow = [
        replace(
            sample,
            verification_device_seconds=slow_verification_seconds,
        )
        if sample.width == largest_width
        else sample
        for sample in samples
    ]
    slow_decision = evaluate_mtp_probe(profile, identity, slow)
    require(
        f"width-{largest_width}-no-amortization" in slow_decision.reasons,
        "non-amortized native verification was accepted",
    )

    invalid_profile = {
        **profile_source,
        "verification_widths": [width_one, width_four],
    }
    try:
        MtpProbeProfile.from_mapping(invalid_profile)
    except MtpProbePolicyError as error:
        require("widths 1, 2, and 4" in str(error), "wrong width-set diagnostic")
    else:
        require(False, "invalid target-verification width set was accepted")


# r[verify tenstorrent.qwen38_p150x2.mtp_drafter_probe]
def probe_mtp_drafter():
    """Positive and negative tests for MTP drafter evidence."""
    from dataclasses import replace

    from qwen36_service.mtp_drafter_core import (
        EXPECTED_MTP_WEIGHT_SHAPES,
        MtpDrafterIdentity,
        MtpDrafterPolicyError,
        MtpDrafterProfile,
        MtpDrafterSample,
        accepted_prefix_length,
        evaluate_mtp_drafter,
        validate_mtp_weight_shapes,
    )

    profile_path = Path(__file__).with_name("mtp-drafter-profile.json")
    with profile_path.open(encoding="utf-8") as profile_file:
        profile_source = json.load(profile_file)
    profile = MtpDrafterProfile.from_mapping(profile_source)
    require(profile.diagnostic_only, "MTP drafter profile became admission eligible")
    validate_mtp_weight_shapes(EXPECTED_MTP_WEIGHT_SHAPES)

    missing_weights = dict(EXPECTED_MTP_WEIGHT_SHAPES)
    missing_weights.pop(next(iter(missing_weights)))
    try:
        validate_mtp_weight_shapes(missing_weights)
    except MtpDrafterPolicyError as error:
        require("missing=" in str(error), "wrong missing-weight diagnostic")
    else:
        require(False, "missing MTP weight was accepted")

    shape_drift = dict(EXPECTED_MTP_WEIGHT_SHAPES)
    drift_name = "mtp.fc.weight"
    drift_shape = shape_drift[drift_name]
    shape_drift[drift_name] = (drift_shape[0], drift_shape[1] + 1)
    try:
        validate_mtp_weight_shapes(shape_drift)
    except MtpDrafterPolicyError as error:
        require(drift_name in str(error), "wrong weight-shape diagnostic")
    else:
        require(False, "drifted MTP weight shape was accepted")

    draft_tokens = tuple(range(profile.draft_token_count))
    require(
        accepted_prefix_length(draft_tokens, draft_tokens) == profile.draft_token_count,
        "matching drafts lost acceptance",
    )
    mismatched_targets = list(draft_tokens)
    mismatched_targets[1] = profile.draft_token_count + 1
    require(
        accepted_prefix_length(draft_tokens, mismatched_targets) == 1,
        "draft mismatch used the wrong prefix",
    )

    identity = MtpDrafterIdentity(
        runtime_identity="runtime-test",
        package_identity="package-test",
        model_identity="model-test",
    )
    mtp_prefill_seconds = 0.5
    draft_seconds = 0.04
    target_sequential_seconds = 0.24
    target_verification_seconds = 0.11
    target_readback_seconds = 0.008
    samples = [
        MtpDrafterSample(
            prompt_index=prompt_index,
            identity=identity,
            physical_pcie_device_ids=profile.expected_pcie_device_ids,
            draft_token_ids=draft_tokens,
            target_token_ids=draft_tokens,
            accepted_prefix_length=profile.draft_token_count,
            mtp_prefill_seconds=mtp_prefill_seconds,
            draft_seconds=draft_seconds,
            target_sequential_seconds=target_sequential_seconds,
            target_verification_seconds=target_verification_seconds,
            target_verification_readback_seconds=target_readback_seconds,
            rollback_seconds=0.0,
            mtp_cache_reprefill_seconds=0.0,
            rollback_valid=True,
            target_output_valid=True,
        )
        for prompt_index in range(1, profile.minimum_prompt_count + 1)
    ]
    decision = evaluate_mtp_drafter(profile, identity, samples)
    require(
        decision.drafter_valid, f"valid drafter evidence failed: {decision.reasons}"
    )
    require(decision.mtp_selected, "valid drafter evidence was not selected")
    require(
        decision.aggregate_emitted_token_speedup
        > profile.minimum_emitted_token_speedup,
        "valid aggregate economics did not pass",
    )
    require(not decision.admission_eligible, "drafter evidence admitted production")

    rollback_failure = list(samples)
    rollback_failure[0] = replace(rollback_failure[0], rollback_valid=False)
    rollback_decision = evaluate_mtp_drafter(profile, identity, rollback_failure)
    require(
        "prompt-1-rollback" in rollback_decision.reasons,
        "rollback failure was accepted",
    )

    alignment_failure = list(samples)
    alignment_failure[0] = replace(alignment_failure[0], accepted_prefix_length=0)
    alignment_decision = evaluate_mtp_drafter(profile, identity, alignment_failure)
    require(
        "prompt-1-acceptance-alignment" in alignment_decision.reasons,
        "misaligned acceptance was accepted",
    )

    unstable_multiplier = 2.0
    unstable = list(samples)
    unstable[-1] = replace(
        unstable[-1],
        draft_seconds=unstable[-1].draft_seconds * unstable_multiplier,
    )
    unstable_decision = evaluate_mtp_drafter(profile, identity, unstable)
    require(
        "sample-instability" in unstable_decision.reasons,
        "unstable drafts were accepted",
    )

    slow = [
        replace(sample, draft_seconds=target_sequential_seconds) for sample in samples
    ]
    slow_decision = evaluate_mtp_drafter(profile, identity, slow)
    require(
        "emitted-token-economics" in slow_decision.reasons,
        "slow emitted-token economics were accepted",
    )

    aggregate_rollback_seconds = 1.0
    zero_accept_targets = list(draft_tokens)
    zero_accept_targets[0] = profile.draft_token_count + 1
    aggregate_failure = list(samples)
    aggregate_failure[-1] = replace(
        aggregate_failure[-1],
        target_token_ids=tuple(zero_accept_targets),
        accepted_prefix_length=0,
        rollback_seconds=aggregate_rollback_seconds,
    )
    aggregate_decision = evaluate_mtp_drafter(
        profile,
        identity,
        aggregate_failure,
    )
    require(
        "emitted-token-economics" in aggregate_decision.reasons,
        "median-positive but aggregate-negative economics were accepted",
    )

    invalid_rollback_seconds = -0.01
    invalid_duration = list(samples)
    invalid_duration[0] = replace(
        invalid_duration[0],
        rollback_seconds=invalid_rollback_seconds,
    )
    invalid_duration_decision = evaluate_mtp_drafter(
        profile,
        identity,
        invalid_duration,
    )
    require(
        "prompt-1-duration" in invalid_duration_decision.reasons,
        "negative rollback duration was accepted",
    )

    invalid_profile = {**profile_source, "draft_token_count": 2}
    try:
        MtpDrafterProfile.from_mapping(invalid_profile)
    except MtpDrafterPolicyError as error:
        require("three drafts" in str(error), "wrong draft-count diagnostic")
    else:
        require(False, "invalid MTP drafter profile was accepted")

    adapter_source = Path(
        __import__(
            "qwen36_service.mtp_drafter_adapter",
            fromlist=["__file__"],
        ).__file__
    ).read_text()
    require(
        "safe_open" in adapter_source, "MTP drafter lost bounded safetensor loading"
    )
    require(
        "pre_hidden_norm" in adapter_source, "MTP drafter lost hidden normalization"
    )
    require("shifted" in adapter_source, "MTP drafter lost shifted token inputs")
    require("reset_state" in adapter_source, "MTP drafter lost its cache reset")
    require(
        "self._read_token(" not in adapter_source,
        "MTP drafter retained the removed full-token helper",
    )
    require(
        "_rebuild_mtp_cache" in adapter_source,
        "MTP drafter lost rejected-prefix cache re-prefill",
    )
    verifier_source = Path(
        __import__(
            "qwen36_service.native_mtp_verifier_adapter",
            fromlist=["__file__"],
        ).__file__
    ).read_text()
    require(
        "_GdnPrefixSnapshot" in verifier_source,
        "target verifier lost trace-time prefix snapshots",
    )


# r[verify tenstorrent.qwen38_p150x2.mtp_draft_selection]
def probe_mtp_draft_selection():
    """Positive and negative tests for paired MTP selector evidence."""
    from dataclasses import replace

    from qwen36_service.mtp_drafter_core import MtpDrafterIdentity
    from qwen36_service.mtp_selection_core import (
        FULL_LOGIT_READBACK,
        SHARD_LOCAL_TOP1,
        MtpSelectionPolicyError,
        MtpSelectionProfile,
        MtpSelectionSample,
        evaluate_mtp_selection,
    )

    profile_source = json.loads(
        Path(__file__).with_name("mtp-selection-profile.json").read_text()
    )
    profile = MtpSelectionProfile.from_mapping(profile_source)
    identity = MtpDrafterIdentity(
        runtime_identity="runtime-test",
        package_identity="package-test",
        model_identity="model-test",
    )
    draft_tokens = (10, 11, 12)
    samples = []
    root_variation = 0.01
    prompt_variation = 0.005
    full_prefill_seconds = 0.03
    full_decode_seconds = 0.05
    full_selection_seconds = 0.03
    bounded_prefill_seconds = 0.02
    bounded_decode_seconds = 0.03
    bounded_selection_seconds = 0.01
    target_sequential_seconds = 0.32
    target_verification_seconds = 0.09
    target_readback_seconds = 0.005
    for prompt_index in range(1, profile.minimum_prompt_count + 1):
        for root_index, route in enumerate(profile.route_schedule, start=1):
            scale = (
                1.0
                + root_variation * (root_index - 1)
                + prompt_variation * (prompt_index - 1)
            )
            if route == FULL_LOGIT_READBACK:
                mtp_prefill_seconds = full_prefill_seconds * scale
                draft_decode_seconds = full_decode_seconds * scale
                draft_selection_seconds = full_selection_seconds * scale
            else:
                mtp_prefill_seconds = bounded_prefill_seconds * scale
                draft_decode_seconds = bounded_decode_seconds * scale
                draft_selection_seconds = bounded_selection_seconds * scale
            samples.append(
                MtpSelectionSample(
                    prompt_index=prompt_index,
                    root_index=root_index,
                    route=route,
                    identity=identity,
                    physical_pcie_device_ids=profile.expected_pcie_device_ids,
                    draft_token_ids=draft_tokens,
                    target_token_ids=draft_tokens,
                    accepted_prefix_length=profile.draft_token_count,
                    mtp_prefill_seconds=mtp_prefill_seconds,
                    draft_decode_seconds=draft_decode_seconds,
                    draft_selection_seconds=draft_selection_seconds,
                    target_sequential_seconds=target_sequential_seconds * scale,
                    target_verification_seconds=target_verification_seconds * scale,
                    target_verification_readback_seconds=(
                        target_readback_seconds * scale
                    ),
                    rollback_seconds=0.0,
                    mtp_cache_reprefill_seconds=0.0,
                    rollback_valid=True,
                    target_output_valid=True,
                )
            )

    decision = evaluate_mtp_selection(profile, identity, samples)
    require(decision.selector_valid, f"valid selector rejected: {decision.reasons}")
    require(decision.bounded_selector_selected, "bounded selector was not selected")
    require(not decision.admission_eligible, "selector evidence admitted production")
    require(
        decision.selector_ratio < profile.maximum_selector_ratio,
        "bounded selector did not satisfy the gain gate",
    )
    require(
        decision.aggregate_emitted_token_speedup > 1.0,
        "bounded selector did not satisfy aggregate economics",
    )

    mismatched = list(samples)
    mismatch_index = 1
    mismatched[mismatch_index] = replace(
        mismatched[mismatch_index],
        draft_token_ids=(13, 11, 12),
    )
    mismatch_decision = evaluate_mtp_selection(profile, identity, mismatched)
    require(
        any("route-parity" in reason for reason in mismatch_decision.reasons),
        "selector route token mismatch was accepted",
    )

    unstable = list(samples)
    unstable_index = 2
    unstable[unstable_index] = replace(
        unstable[unstable_index],
        draft_selection_seconds=bounded_selection_seconds * 2,
    )
    unstable_decision = evaluate_mtp_selection(profile, identity, unstable)
    require(
        any("instability" in reason for reason in unstable_decision.reasons),
        "unstable selector roots were accepted",
    )

    slow_target_seconds = 0.04
    uneconomic = [
        replace(sample, target_sequential_seconds=slow_target_seconds)
        for sample in samples
    ]
    uneconomic_decision = evaluate_mtp_selection(profile, identity, uneconomic)
    require(
        "emitted-token-economics" in uneconomic_decision.reasons,
        "uneconomic bounded selector was accepted",
    )

    impossible_selection = list(samples)
    impossible_selection[0] = replace(
        impossible_selection[0],
        draft_selection_seconds=1.0,
    )
    impossible_decision = evaluate_mtp_selection(
        profile,
        identity,
        impossible_selection,
    )
    require(
        any("selection-duration" in reason for reason in impossible_decision.reasons),
        "selector duration above total draft time was accepted",
    )

    invalid_profile = {
        **profile_source,
        "route_schedule": [SHARD_LOCAL_TOP1, FULL_LOGIT_READBACK],
    }
    try:
        MtpSelectionProfile.from_mapping(invalid_profile)
    except MtpSelectionPolicyError as error:
        require("route schedule" in str(error), "wrong selector schedule diagnostic")
    else:
        require(False, "invalid selector route schedule was accepted")

    adapter_source = Path(
        __import__(
            "qwen36_service.decode_adapter",
            fromlist=["__file__"],
        ).__file__
    ).read_text()
    require(
        "select_sharded_logits" in adapter_source,
        "bounded selector adapter boundary is missing",
    )
    require(
        "select_global_winner" in adapter_source,
        "bounded selector lost deterministic global tie policy",
    )


# r[verify tenstorrent.qwen38_p150x2.mtp_draft_trace]
def probe_mtp_draft_trace():
    """Positive and negative tests for matched MTP draft traces."""
    from dataclasses import replace

    from qwen36_service.mtp_drafter_core import MtpDrafterIdentity
    from qwen36_service.mtp_trace_core import (
        EAGER_DRAFT,
        TRACED_DRAFT,
        MtpTraceBinding,
        MtpTracePolicyError,
        MtpTraceProfile,
        MtpTraceSample,
        evaluate_mtp_trace,
        validate_mtp_trace_binding,
    )

    profile_source = json.loads(
        Path(__file__).with_name("mtp-trace-profile.json").read_text()
    )
    profile = MtpTraceProfile.from_mapping(profile_source)
    binding = MtpTraceBinding(
        token_buffer_identity="token",
        position_buffer_identity="position",
        rope_buffer_identity="rope",
        feedback_buffer_identity="feedback",
        kv_buffer_identity="kv",
        feedback_output_identity="feedback-output",
        index_output_identity="index-output",
        value_output_identity="value-output",
        feedback_output_shape=(1, 1, 1, 1),
        feedback_output_dtype="bfloat16",
        index_output_shape=(1,),
        index_output_dtype="uint32",
        value_output_shape=(1,),
        value_output_dtype="bfloat16",
    )
    require(
        not validate_mtp_trace_binding(binding, binding),
        "stable MTP trace binding reported drift",
    )
    drifted_binding = replace(binding, token_buffer_identity="changed-token")
    require(
        validate_mtp_trace_binding(binding, drifted_binding)
        == ("token_buffer_identity",),
        "MTP trace binding drift used the wrong field",
    )
    identity = MtpDrafterIdentity(
        runtime_identity="runtime-test",
        package_identity="package-test",
        model_identity="model-test",
    )
    first_draft_token_id = 10
    draft_tokens = tuple(
        range(first_draft_token_id, first_draft_token_id + profile.draft_token_count)
    )
    root_variation = 0.005
    prompt_variation = 0.002
    eager_prefill_seconds = 0.02
    eager_draft_seconds = 0.06
    eager_selection_seconds = 0.012
    traced_prefill_seconds = 0.02
    traced_draft_seconds = 0.03
    traced_selection_seconds = 0.008
    trace_capture_seconds = 5.0
    target_sequential_seconds = 0.40
    target_verification_seconds = 0.07
    target_readback_seconds = 0.003
    samples = []
    for prompt_index in range(1, profile.minimum_prompt_count + 1):
        for root_index, route in enumerate(profile.route_schedule, start=1):
            scale = (
                1.0
                + root_variation * (root_index - 1)
                + prompt_variation * (prompt_index - 1)
            )
            if route == EAGER_DRAFT:
                mtp_prefill_seconds = eager_prefill_seconds * scale
                draft_decode_seconds = eager_draft_seconds * scale
                draft_selection_seconds = eager_selection_seconds * scale
                capture_seconds = 0.0
            else:
                mtp_prefill_seconds = traced_prefill_seconds * scale
                draft_decode_seconds = traced_draft_seconds * scale
                draft_selection_seconds = traced_selection_seconds * scale
                capture_seconds = trace_capture_seconds * scale
            samples.append(
                MtpTraceSample(
                    prompt_index=prompt_index,
                    root_index=root_index,
                    route=route,
                    identity=identity,
                    physical_pcie_device_ids=profile.expected_pcie_device_ids,
                    draft_token_ids=draft_tokens,
                    target_token_ids=draft_tokens,
                    accepted_prefix_length=profile.draft_token_count,
                    mtp_prefill_seconds=mtp_prefill_seconds,
                    draft_decode_seconds=draft_decode_seconds,
                    draft_selection_seconds=draft_selection_seconds,
                    trace_capture_seconds=capture_seconds,
                    target_sequential_seconds=target_sequential_seconds * scale,
                    target_verification_seconds=target_verification_seconds * scale,
                    target_verification_readback_seconds=(
                        target_readback_seconds * scale
                    ),
                    rollback_seconds=0.0,
                    mtp_cache_recovery_seconds=0.0,
                    trace_binding_valid=True,
                    capture_idempotence_valid=True,
                    mtp_kv_prefix_valid=True,
                    rollback_valid=True,
                    target_output_valid=True,
                )
            )

    decision = evaluate_mtp_trace(profile, identity, samples)
    require(decision.trace_valid, f"valid MTP trace rejected: {decision.reasons}")
    require(decision.traced_drafting_selected, "valid MTP trace was not selected")
    require(not decision.admission_eligible, "MTP trace evidence admitted production")
    require(
        decision.traced_draft_ratio < profile.maximum_traced_draft_ratio,
        "traced MTP draft did not satisfy its gain gate",
    )
    require(
        decision.aggregate_emitted_token_speedup
        > profile.minimum_emitted_token_speedup,
        "traced MTP draft did not satisfy aggregate economics",
    )

    incomplete = samples[:-1]
    require(
        "sample-count" in evaluate_mtp_trace(profile, identity, incomplete).reasons,
        "incomplete MTP trace evidence was accepted",
    )

    binding_failure = list(samples)
    binding_failure[1] = replace(binding_failure[1], trace_binding_valid=False)
    require(
        any(
            "trace-binding" in reason
            for reason in evaluate_mtp_trace(profile, identity, binding_failure).reasons
        ),
        "drifted MTP trace binding was accepted",
    )

    idempotence_failure = list(samples)
    idempotence_failure[1] = replace(
        idempotence_failure[1], capture_idempotence_valid=False
    )
    require(
        any(
            "capture-idempotence" in reason
            for reason in evaluate_mtp_trace(
                profile, identity, idempotence_failure
            ).reasons
        ),
        "non-idempotent MTP trace capture was accepted",
    )

    cache_failure = list(samples)
    cache_failure[1] = replace(cache_failure[1], mtp_kv_prefix_valid=False)
    require(
        any(
            "mtp-kv-prefix" in reason
            for reason in evaluate_mtp_trace(profile, identity, cache_failure).reasons
        ),
        "MTP KV-prefix mismatch was accepted",
    )

    route_mismatch = list(samples)
    route_mismatch[1] = replace(
        route_mismatch[1],
        draft_token_ids=(first_draft_token_id + 1, *draft_tokens[1:]),
    )
    require(
        any(
            "route-parity" in reason
            for reason in evaluate_mtp_trace(profile, identity, route_mismatch).reasons
        ),
        "MTP trace token mismatch was accepted",
    )

    instability_multiplier = 2.0
    unstable = list(samples)
    unstable[1] = replace(
        unstable[1],
        draft_decode_seconds=unstable[1].draft_decode_seconds * instability_multiplier,
    )
    require(
        any(
            "instability" in reason
            for reason in evaluate_mtp_trace(profile, identity, unstable).reasons
        ),
        "unstable MTP trace roots were accepted",
    )

    slow_traced_seconds = eager_draft_seconds
    slow = [
        replace(sample, draft_decode_seconds=slow_traced_seconds)
        if sample.route == TRACED_DRAFT
        else sample
        for sample in samples
    ]
    require(
        "traced-draft-gain" in evaluate_mtp_trace(profile, identity, slow).reasons,
        "slow traced MTP draft was accepted",
    )

    uneconomic_target_seconds = 0.05
    uneconomic = [
        replace(sample, target_sequential_seconds=uneconomic_target_seconds)
        for sample in samples
    ]
    require(
        "emitted-token-economics"
        in evaluate_mtp_trace(profile, identity, uneconomic).reasons,
        "uneconomic traced MTP draft was accepted",
    )

    eager_capture = list(samples)
    eager_index = next(
        index
        for index, sample in enumerate(eager_capture)
        if sample.route == EAGER_DRAFT
    )
    eager_capture[eager_index] = replace(
        eager_capture[eager_index], trace_capture_seconds=trace_capture_seconds
    )
    require(
        any(
            "eager-trace-capture" in reason
            for reason in evaluate_mtp_trace(profile, identity, eager_capture).reasons
        ),
        "eager root with trace capture was accepted",
    )

    invalid_profile = {**profile_source, "minimum_samples_per_route": 5}
    try:
        MtpTraceProfile.from_mapping(invalid_profile)
    except MtpTracePolicyError as error:
        require("sample count" in str(error), "wrong trace sample-count diagnostic")
    else:
        require(False, "weakened MTP trace sample count was accepted")

    unknown_profile = {**profile_source, "unknown": True}
    try:
        MtpTraceProfile.from_mapping(unknown_profile)
    except MtpTracePolicyError as error:
        require("keys differ" in str(error), "wrong closed-profile diagnostic")
    else:
        require(False, "unknown MTP trace profile key was accepted")

    adapter_source = Path(
        __import__(
            "qwen36_service.mtp_trace_adapter",
            fromlist=["__file__"],
        ).__file__
    ).read_text()
    for required_symbol in (
        "begin_trace_capture",
        "end_trace_capture",
        "execute_trace",
        "release_trace",
        "buffer_address",
        "_snapshot_cache_row",
    ):
        require(
            required_symbol in adapter_source,
            f"MTP trace adapter lost {required_symbol}",
        )


# r[verify tenstorrent.qwen38_p150x2.mtp_kv_rollback]
# r[verify tenstorrent.qwen38_p150x2.mtp_recovery_stability]
def probe_mtp_kv_rollback():
    """Positive and negative tests for matched MTP K/V rollback."""
    from dataclasses import replace

    from qwen36_service.mtp_drafter_core import MtpDrafterIdentity
    from qwen36_service.mtp_rollback_core import (
        REPREFILL_RECOVERY,
        SNAPSHOT_RECOVERY,
        MtpRollbackPolicyError,
        MtpRollbackProfile,
        MtpRollbackSample,
        evaluate_mtp_rollback,
    )

    profile_source = json.loads(
        Path(__file__).with_name("mtp-rollback-profile.json").read_text()
    )
    profile = MtpRollbackProfile.from_mapping(profile_source)
    identity = MtpDrafterIdentity(
        runtime_identity="runtime-test",
        package_identity="package-test",
        model_identity="model-test",
    )
    zero_accept_prompt_index = 2
    first_draft_token_id = 11
    second_draft_token_id = 12
    third_draft_token_id = 13
    mismatched_target_token_id = 21
    committed_bonus_token_id = 31
    next_draft_token_id = 41
    reprefill_recovery_seconds = 0.04
    snapshot_recovery_seconds = 0.01
    draft_tokens = (
        first_draft_token_id,
        second_draft_token_id,
        third_draft_token_id,
    )
    samples = []
    for prompt_index in range(1, profile.minimum_prompt_count + 1):
        zero_accept = prompt_index == zero_accept_prompt_index
        target_tokens = (
            (mismatched_target_token_id, second_draft_token_id, third_draft_token_id)
            if zero_accept
            else draft_tokens
        )
        accepted = 0 if zero_accept else profile.draft_token_count
        committed_token = (
            mismatched_target_token_id if zero_accept else committed_bonus_token_id
        )
        for root_index, route in enumerate(profile.route_schedule, start=1):
            samples.append(
                MtpRollbackSample(
                    prompt_index=prompt_index,
                    root_index=root_index,
                    route=route,
                    identity=identity,
                    physical_pcie_device_ids=profile.expected_pcie_device_ids,
                    draft_token_ids=draft_tokens,
                    target_token_ids=target_tokens,
                    accepted_prefix_length=accepted,
                    committed_token_id=committed_token,
                    next_draft_token_id=next_draft_token_id,
                    recovery_seconds=(
                        snapshot_recovery_seconds
                        if route == SNAPSHOT_RECOVERY
                        else reprefill_recovery_seconds
                    ),
                    recovery_execution_mode=profile.recovery_execution_mode,
                    process_trace_capture_count=(
                        profile.required_process_trace_capture_count
                    ),
                    trace_binding_identity="persistent-trace-binding-test",
                    trace_binding_valid=True,
                    snapshot_row_count=(
                        profile.snapshot_row_count if route == SNAPSHOT_RECOVERY else 0
                    ),
                    committed_advance_count=profile.committed_advance_count,
                    next_draft_valid=True,
                    feedback_state_valid=True,
                    kv_prefix_valid=True,
                    target_output_valid=True,
                    target_rollback_valid=True,
                )
            )

    decision = evaluate_mtp_rollback(profile, identity, samples)
    require(decision.rollback_valid, f"valid MTP rollback rejected: {decision.reasons}")
    require(
        decision.snapshot_rollback_selected,
        "valid MTP snapshot rollback was not selected",
    )
    require(not decision.admission_eligible, "MTP rollback admitted production")
    require(
        decision.observed_acceptance_lengths == profile.required_acceptance_lengths,
        "MTP rollback lost natural acceptance coverage",
    )
    require(
        decision.snapshot_recovery_ratio < profile.maximum_snapshot_recovery_ratio,
        "MTP rollback did not satisfy its gain gate",
    )

    incomplete = samples[:-1]
    require(
        "sample-count" in evaluate_mtp_rollback(profile, identity, incomplete).reasons,
        "incomplete MTP rollback evidence was accepted",
    )

    capture_failure = list(samples)
    capture_failure[1] = replace(capture_failure[1], process_trace_capture_count=2)
    require(
        any(
            "process-trace-capture-count" in reason
            for reason in evaluate_mtp_rollback(
                profile, identity, capture_failure
            ).reasons
        ),
        "repeated MTP recovery trace capture was accepted",
    )

    binding_failure = list(samples)
    binding_failure[1] = replace(
        binding_failure[1], trace_binding_identity="changed-trace-binding"
    )
    require(
        "process-trace-binding-parity"
        in evaluate_mtp_rollback(profile, identity, binding_failure).reasons,
        "changed MTP recovery trace binding was accepted",
    )

    invalid_binding = list(samples)
    invalid_binding[1] = replace(invalid_binding[1], trace_binding_valid=False)
    require(
        any(
            "trace-binding" in reason
            for reason in evaluate_mtp_rollback(
                profile, identity, invalid_binding
            ).reasons
        ),
        "invalid MTP recovery trace binding was accepted",
    )

    state_failure = list(samples)
    state_failure[1] = replace(state_failure[1], feedback_state_valid=False)
    require(
        any(
            "feedback-state" in reason
            for reason in evaluate_mtp_rollback(
                profile, identity, state_failure
            ).reasons
        ),
        "MTP rollback feedback mismatch was accepted",
    )

    instability_multiplier = 2.0
    unstable = list(samples)
    unstable[1] = replace(
        unstable[1],
        recovery_seconds=unstable[1].recovery_seconds * instability_multiplier,
    )
    require(
        any(
            "instability" in reason
            for reason in evaluate_mtp_rollback(profile, identity, unstable).reasons
        ),
        "unstable MTP rollback roots were accepted",
    )

    slow = [
        replace(sample, recovery_seconds=reprefill_recovery_seconds)
        if sample.route == SNAPSHOT_RECOVERY
        else sample
        for sample in samples
    ]
    require(
        "snapshot-recovery-gain"
        in evaluate_mtp_rollback(profile, identity, slow).reasons,
        "slow MTP snapshot rollback was accepted",
    )

    invalid_profile = {**profile_source, "snapshot_row_count": 1}
    try:
        MtpRollbackProfile.from_mapping(invalid_profile)
    except MtpRollbackPolicyError as error:
        require("snapshot row count" in str(error), "wrong rollback row diagnostic")
    else:
        require(False, "weakened MTP rollback row count was accepted")

    unknown_profile = {**profile_source, "unknown": True}
    try:
        MtpRollbackProfile.from_mapping(unknown_profile)
    except MtpRollbackPolicyError as error:
        require("keys differ" in str(error), "wrong closed rollback diagnostic")
    else:
        require(False, "unknown MTP rollback profile key was accepted")

    require(
        REPREFILL_RECOVERY in profile.route_schedule,
        "MTP rollback profile lost the re-prefill reference route",
    )
    adapter_source = Path(
        __import__(
            "qwen36_service.mtp_rollback_adapter",
            fromlist=["__file__"],
        ).__file__
    ).read_text()
    for required_symbol in (
        "_snapshot_update_input",
        "paged_update_cache",
        "record_prefix_feedback",
        "recover_from_snapshot",
        "recover_from_reprefill",
        "recovery_state_components_equal",
        "_drain_device",
    ):
        require(
            required_symbol in adapter_source,
            f"MTP rollback adapter lost {required_symbol}",
        )
    trace_adapter_source = Path(
        __import__(
            "qwen36_service.mtp_trace_adapter",
            fromlist=["__file__"],
        ).__file__
    ).read_text()
    for required_symbol in ("replay_token_inputs", "trace_capture_count"):
        require(
            required_symbol in trace_adapter_source,
            f"MTP persistent trace adapter lost {required_symbol}",
        )
    drafter_adapter_source = Path(
        __import__(
            "qwen36_service.mtp_drafter_adapter",
            fromlist=["__file__"],
        ).__file__
    ).read_text()
    for required_symbol in ("_reset_mtp_cache", "_mtp_zero_caches"):
        require(
            required_symbol in drafter_adapter_source,
            f"MTP stable cache adapter lost {required_symbol}",
        )


# r[verify tenstorrent.qwen38_p150x2.decode_component_attribution]
def probe_decode_component_attribution():
    """Positive and negative tests for pure component attribution."""
    from dataclasses import replace

    from qwen36_service.decode_component_core import (
        EMBEDDING_COMPONENT,
        FULL_ATTENTION_MIXER_COMPONENT,
        FULL_ATTENTION_TYPE,
        GDN_MIXER_COMPONENT,
        LINEAR_ATTENTION_TYPE,
        LM_HEAD_ARGMAX_COMPONENT,
        MLP_COMPONENT,
        ComponentReplaySample,
        DecodeComponentEvidence,
        DecodeComponentIdentity,
        DecodeComponentPolicyError,
        DecodeComponentProfile,
        FullReplaySample,
        HostBoundarySample,
        LayerTypeObservation,
        evaluate_decode_components,
    )

    profile_source = json.loads(
        Path(__file__).with_name("decode-component-profile.json").read_text()
    )
    profile = DecodeComponentProfile.from_mapping(profile_source)
    identity = DecodeComponentIdentity(
        runtime_identity="runtime",
        package_identity="package",
        model_identity=f"Qwen3.8-27B:{profile.checkpoint_revision}",
        authority_blake3=profile.authority_blake3,
        checkpoint_revision=profile.checkpoint_revision,
        model_source_revision=profile.model_source_revision,
        tokenizer_revision=profile.tokenizer_revision,
    )
    full_attention_indices = set(profile.full_attention_representatives)
    representative_indices = set(profile.gdn_representatives)
    representative_indices.update(profile.full_attention_representatives)
    representative_indices.update(profile.mlp_representatives)
    layer_types = tuple(
        LayerTypeObservation(
            layer_index=layer_index,
            mixer_type=(
                FULL_ATTENTION_TYPE
                if layer_index in full_attention_indices
                else LINEAR_ATTENTION_TYPE
            ),
        )
        for layer_index in sorted(representative_indices)
    )

    embedding_seconds = 0.0002
    gdn_seconds = 0.0003
    mlp_seconds = 0.0005
    full_attention_seconds = 0.00025
    lm_head_seconds = 0.006
    input_update_seconds = 0.001
    winner_readback_seconds = 0.0005
    sample_step_ratio = 0.001
    exact_closure_tolerance = 1.0e-12
    instability_multiplier = 2.0
    closure_failure_multiplier = 2.0
    invalid_duration_seconds = -1.0
    component_groups = [(EMBEDDING_COMPONENT, None, embedding_seconds)]
    component_groups.extend(
        (GDN_MIXER_COMPONENT, layer_index, gdn_seconds)
        for layer_index in profile.gdn_representatives
    )
    component_groups.extend(
        (MLP_COMPONENT, layer_index, mlp_seconds)
        for layer_index in profile.mlp_representatives
    )
    component_groups.extend(
        (
            FULL_ATTENTION_MIXER_COMPONENT,
            layer_index,
            full_attention_seconds,
        )
        for layer_index in profile.full_attention_representatives
    )
    component_groups.append((LM_HEAD_ARGMAX_COMPONENT, None, lm_head_seconds))
    component_samples = tuple(
        ComponentReplaySample(
            component=component,
            layer_index=layer_index,
            sample_index=sample_index,
            seconds=seconds * (1.0 + (sample_index - 1) * sample_step_ratio),
        )
        for component, layer_index, seconds in component_groups
        for sample_index in range(1, profile.samples_per_group + 1)
    )
    modeled_seconds = (
        embedding_seconds
        + gdn_seconds * profile.gdn_layer_count
        + mlp_seconds * profile.mlp_layer_count
        + full_attention_seconds * profile.full_attention_layer_count
        + lm_head_seconds
    )
    repeated_token_id = 42
    full_samples = tuple(
        FullReplaySample(
            sample_index=sample_index,
            device_seconds=modeled_seconds
            * (1.0 + (sample_index - 1) * sample_step_ratio),
            token_id=repeated_token_id,
        )
        for sample_index in range(1, profile.samples_per_group + 1)
    )
    host_samples = tuple(
        HostBoundarySample(
            sample_index=sample_index,
            input_update_seconds=input_update_seconds
            * (1.0 + (sample_index - 1) * sample_step_ratio),
            winner_readback_seconds=winner_readback_seconds
            * (1.0 + (sample_index - 1) * sample_step_ratio),
        )
        for sample_index in range(1, profile.samples_per_group + 1)
    )
    evidence = DecodeComponentEvidence(
        identity=identity,
        physical_pcie_device_ids=profile.expected_pcie_device_ids,
        mesh_shape=profile.mesh_shape,
        host_cpu_affinity=profile.expected_cpu_affinity,
        layer_types=layer_types,
        component_samples=component_samples,
        full_samples=full_samples,
        host_samples=host_samples,
        cleanup_complete=True,
    )
    decision = evaluate_decode_components(profile, evidence)
    require(
        decision.attribution_valid,
        f"valid component evidence failed: {decision.reasons}",
    )
    require(
        decision.selected_component == MLP_COMPONENT,
        "valid evidence did not select MLP",
    )
    require(
        decision.repeated_token_id == repeated_token_id, "full replay token changed"
    )
    require(
        decision.closure_ratio is not None
        and math.isclose(decision.closure_ratio, 1.0, rel_tol=exact_closure_tolerance),
        "additive component model did not close",
    )

    missing = replace(evidence, component_samples=component_samples[:-1])
    require(
        "lm_head_argmax-samples"
        in evaluate_decode_components(profile, missing).reasons,
        "missing component samples were accepted",
    )
    wrong_layer_types = list(layer_types)
    wrong_layer_types[0] = replace(
        wrong_layer_types[0],
        mixer_type=FULL_ATTENTION_TYPE,
    )
    wrong_type = replace(evidence, layer_types=tuple(wrong_layer_types))
    require(
        "gdn-layer-0-type" in evaluate_decode_components(profile, wrong_type).reasons,
        "wrong representative layer type was accepted",
    )
    unstable_samples = list(component_samples)
    unstable_samples[0] = replace(
        unstable_samples[0],
        seconds=unstable_samples[0].seconds * instability_multiplier,
    )
    unstable = replace(evidence, component_samples=tuple(unstable_samples))
    require(
        "embedding-instability"
        in evaluate_decode_components(profile, unstable).reasons,
        "unstable component samples were accepted",
    )
    closure_failure = replace(
        evidence,
        full_samples=tuple(
            replace(
                sample,
                device_seconds=sample.device_seconds * closure_failure_multiplier,
            )
            for sample in full_samples
        ),
    )
    require(
        "additive-model-closure"
        in evaluate_decode_components(profile, closure_failure).reasons,
        "non-closing component model was accepted",
    )
    invalid_samples = list(component_samples)
    invalid_samples[0] = replace(
        invalid_samples[0],
        seconds=invalid_duration_seconds,
    )
    invalid_duration = replace(evidence, component_samples=tuple(invalid_samples))
    require(
        "component-duration"
        in evaluate_decode_components(profile, invalid_duration).reasons,
        "negative component duration was accepted",
    )
    require(
        "cleanup-incomplete"
        in evaluate_decode_components(
            profile,
            replace(evidence, cleanup_complete=False),
        ).reasons,
        "incomplete cleanup was accepted",
    )
    mismatched_identity = replace(
        evidence,
        identity=replace(
            identity,
            authority_blake3="0" * len(identity.authority_blake3),
        ),
    )
    require(
        "authority-identity"
        in evaluate_decode_components(profile, mismatched_identity).reasons,
        "mismatched authority was accepted",
    )

    invalid_profile = {**profile_source, "diagnostic_only": False}
    try:
        DecodeComponentProfile.from_mapping(invalid_profile)
    except DecodeComponentPolicyError as error:
        require("diagnostic-only" in str(error), "wrong component profile diagnostic")
    else:
        require(False, "non-diagnostic component profile was accepted")


# r[verify tenstorrent.qwen38_p150x2.traced_ttft_control]
def probe_ttft_control():
    """Positive and negative tests for the same-process TTFT control."""
    from dataclasses import replace

    from qwen36_service.decode_core import DecodeProfile
    from qwen36_service.ttft_control_core import (
        CACHE_CAPACITY_MECHANISM,
        CAPACITY_CACHE_TRACED,
        NO_MATERIAL_REGRESSION,
        PARKED_TRACE_MECHANISM,
        REQUEST_CACHE_UNTRACED,
        TtftControlEvidence,
        TtftControlIdentity,
        TtftControlPolicyError,
        TtftControlProfile,
        TtftControlSample,
        evaluate_ttft_control,
    )
    from qwen36_service.ttft_control_probe import _validate_profile_binding

    profile_source = json.loads(
        Path(__file__).with_name("ttft-control-profile.json").read_text()
    )
    profile = TtftControlProfile.from_mapping(profile_source)
    decode_profile = DecodeProfile.from_mapping(
        json.loads(Path(__file__).with_name("decode-profile.json").read_text())
    )
    model_path = Path("/model") / profile.checkpoint_revision
    _validate_profile_binding(model_path, decode_profile, profile)
    mismatched_source_profile = replace(
        profile,
        model_source_revision="0" * len(profile.model_source_revision),
    )
    try:
        _validate_profile_binding(
            model_path,
            decode_profile,
            mismatched_source_profile,
        )
    except SystemExit as error:
        require("source identities differ" in str(error), "wrong source diagnostic")
    else:
        require(False, "mismatched TTFT source identity was accepted")

    identity = TtftControlIdentity(
        runtime_identity="runtime-test",
        package_identity="package-test",
        model_identity=f"Qwen3.8-27B:{profile.checkpoint_revision}",
        authority_blake3=profile.authority_blake3,
        checkpoint_revision=profile.checkpoint_revision,
        model_source_revision=profile.model_source_revision,
        tokenizer_revision=profile.tokenizer_revision,
    )
    test_first_token_id = 13
    request_prefill_seconds = 1.0
    capacity_prefill_seconds = 1.08
    traced_prefill_seconds = 1.20
    readback_seconds = 0.10
    sample_variation = 0.005
    invalid_duration = -0.01
    instability_multiplier = 2.0

    samples = []
    for root_index, route in enumerate(profile.route_schedule, start=1):
        if route == REQUEST_CACHE_UNTRACED:
            base_prefill = request_prefill_seconds
            cache_blocks = profile.request_cache_blocks
        elif route == CAPACITY_CACHE_TRACED:
            base_prefill = traced_prefill_seconds
            cache_blocks = profile.trace_capacity_blocks
        else:
            base_prefill = capacity_prefill_seconds
            cache_blocks = profile.trace_capacity_blocks
        for sample_index in range(1, profile.retained_samples + 1):
            prefill_seconds = base_prefill * (1.0 + sample_variation * sample_index)
            samples.append(
                TtftControlSample(
                    root_index=root_index,
                    sample_index=sample_index,
                    route=route,
                    cache_blocks=cache_blocks,
                    page_table_blocks=profile.page_table_blocks,
                    trace_parked=route == CAPACITY_CACHE_TRACED,
                    first_token_id=test_first_token_id,
                    prefill_device_seconds=prefill_seconds,
                    first_token_readback_seconds=readback_seconds,
                    ttft_seconds=prefill_seconds + readback_seconds,
                )
            )
    evidence = TtftControlEvidence(
        identity=identity,
        physical_pcie_device_ids=profile.expected_pcie_device_ids,
        mesh_shape=profile.mesh_shape,
        host_cpu_affinity=profile.expected_cpu_affinity,
        samples=tuple(samples),
        cleanup_complete=True,
    )
    decision = evaluate_ttft_control(profile, evidence)
    require(decision.control_valid, f"valid TTFT control failed: {decision.reasons}")
    require(
        decision.selected_mechanisms
        == (CACHE_CAPACITY_MECHANISM, PARKED_TRACE_MECHANISM),
        "valid material TTFT mechanisms were not selected",
    )
    require(not decision.admission_eligible, "TTFT control admitted production")

    equal_samples = tuple(
        replace(
            sample,
            prefill_device_seconds=request_prefill_seconds,
            first_token_readback_seconds=readback_seconds,
            ttft_seconds=request_prefill_seconds + readback_seconds,
        )
        for sample in samples
    )
    equal_decision = evaluate_ttft_control(
        profile,
        replace(evidence, samples=equal_samples),
    )
    require(equal_decision.control_valid, "equal TTFT control evidence failed")
    require(
        equal_decision.disposition == NO_MATERIAL_REGRESSION,
        "equal TTFT routes reported a material regression",
    )
    require(
        not equal_decision.selected_mechanisms,
        "equal TTFT routes selected a mechanism",
    )

    wrong_route = list(samples)
    wrong_route[0] = replace(wrong_route[0], route=CAPACITY_CACHE_TRACED)
    require(
        "root-1-route"
        in evaluate_ttft_control(
            profile,
            replace(evidence, samples=tuple(wrong_route)),
        ).reasons,
        "wrong TTFT route was accepted",
    )

    token_mismatch = list(samples)
    token_mismatch[0] = replace(
        token_mismatch[0], first_token_id=test_first_token_id + 1
    )
    require(
        "first-token-parity"
        in evaluate_ttft_control(
            profile,
            replace(evidence, samples=tuple(token_mismatch)),
        ).reasons,
        "TTFT first-token mismatch was accepted",
    )

    width_mismatch = list(samples)
    width_mismatch[0] = replace(
        width_mismatch[0], cache_blocks=profile.trace_capacity_blocks
    )
    require(
        "root-1-cache-width"
        in evaluate_ttft_control(
            profile,
            replace(evidence, samples=tuple(width_mismatch)),
        ).reasons,
        "TTFT cache-width mismatch was accepted",
    )

    bad_duration = list(samples)
    bad_duration[0] = replace(bad_duration[0], prefill_device_seconds=invalid_duration)
    require(
        "prefill-device-duration"
        in evaluate_ttft_control(
            profile,
            replace(evidence, samples=tuple(bad_duration)),
        ).reasons,
        "negative TTFT phase duration was accepted",
    )

    unstable = list(equal_samples)
    unstable[0] = replace(
        unstable[0],
        prefill_device_seconds=(
            unstable[0].prefill_device_seconds * instability_multiplier
        ),
        ttft_seconds=(
            unstable[0].prefill_device_seconds * instability_multiplier
            + unstable[0].first_token_readback_seconds
        ),
    )
    require(
        "root-1-prefill-device-instability"
        in evaluate_ttft_control(
            profile,
            replace(evidence, samples=tuple(unstable)),
        ).reasons,
        "unstable TTFT root was accepted",
    )

    duplicate_unstable = tuple(
        replace(
            sample,
            prefill_device_seconds=(
                sample.prefill_device_seconds * instability_multiplier
            ),
            first_token_readback_seconds=(
                sample.first_token_readback_seconds * instability_multiplier
            ),
            ttft_seconds=sample.ttft_seconds * instability_multiplier,
        )
        if sample.root_index == len(profile.route_schedule)
        else sample
        for sample in equal_samples
    )
    require(
        f"{REQUEST_CACHE_UNTRACED}-ttft-duplicate-instability"
        in evaluate_ttft_control(
            profile,
            replace(evidence, samples=duplicate_unstable),
        ).reasons,
        "unstable duplicate TTFT roots were accepted",
    )

    require(
        f"root-{len(profile.route_schedule)}-sample-count"
        in evaluate_ttft_control(
            profile,
            replace(evidence, samples=tuple(samples[:-1])),
        ).reasons,
        "incomplete TTFT root was accepted",
    )
    require(
        "cleanup-incomplete"
        in evaluate_ttft_control(
            profile,
            replace(evidence, cleanup_complete=False),
        ).reasons,
        "incomplete TTFT cleanup was accepted",
    )
    mismatched_identity = replace(
        evidence,
        identity=replace(
            identity,
            authority_blake3="0" * len(identity.authority_blake3),
        ),
    )
    require(
        "authority-identity"
        in evaluate_ttft_control(profile, mismatched_identity).reasons,
        "mismatched TTFT authority was accepted",
    )

    invalid_profile = {**profile_source, "diagnostic_only": False}
    try:
        TtftControlProfile.from_mapping(invalid_profile)
    except TtftControlPolicyError as error:
        require("diagnostic-only" in str(error), "wrong TTFT profile diagnostic")
    else:
        require(False, "non-diagnostic TTFT control profile was accepted")

    wrong_route_schedule = list(profile.route_schedule)
    wrong_route_schedule[0] = CAPACITY_CACHE_TRACED
    wrong_schedule = {
        **profile_source,
        "roots": {
            **profile_source["roots"],
            "route_schedule": wrong_route_schedule,
        },
    }
    try:
        TtftControlProfile.from_mapping(wrong_schedule)
    except TtftControlPolicyError as error:
        require("route schedule" in str(error), "wrong TTFT schedule diagnostic")
    else:
        require(False, "wrong TTFT route schedule was accepted")

    too_few_samples = {
        **profile_source,
        "roots": {
            **profile_source["roots"],
            "retained_samples": profile.retained_samples - 1,
        },
    }
    try:
        TtftControlProfile.from_mapping(too_few_samples)
    except TtftControlPolicyError as error:
        require("at least" in str(error), "wrong sample-count diagnostic")
    else:
        require(False, "too few TTFT control samples were accepted")


def probe_service_contract():
    """Positive and negative tests for the pure HTTP request policy."""
    from qwen36_service.core import (
        Endpoint,
        RequestValidationError,
        build_completion_response,
        parse_completion_request,
        parse_endpoint,
    )

    admitted_model = "Qwen3.8-27B"
    maximum_generation_tokens = 64
    requested_generation_tokens = 4

    require(
        parse_endpoint("/v1/completions") is Endpoint.COMPLETIONS,
        "completion route did not resolve",
    )
    require(
        parse_endpoint("/v1/chat/completions") is Endpoint.CHAT_COMPLETIONS,
        "chat-completion route did not resolve",
    )

    completion = parse_completion_request(
        {
            "max_tokens": requested_generation_tokens,
            "model": admitted_model,
            "prompt": "The capital of France is",
            "temperature": 0,
        },
        Endpoint.COMPLETIONS,
        admitted_model,
        maximum_generation_tokens,
    )
    require(
        completion.prompt == "The capital of France is", "completion prompt changed"
    )
    require(
        completion.max_tokens == requested_generation_tokens,
        "completion max_tokens changed",
    )

    chat = parse_completion_request(
        {
            "max_tokens": requested_generation_tokens,
            "messages": [{"content": "Name France's capital.", "role": "user"}],
            "model": admitted_model,
        },
        Endpoint.CHAT_COMPLETIONS,
        admitted_model,
        maximum_generation_tokens,
    )
    require(len(chat.messages) == 1, "valid chat request lost its message")
    prompt_token_count = 5
    generated_token_count = 1
    response = build_completion_response(
        request=chat,
        request_id="cmpl-test",
        created=0,
        generated_text="Paris",
        prompt_token_count=prompt_token_count,
        generated_token_count=generated_token_count,
    )
    require(
        response["choices"][0]["message"]["content"] == "Paris",
        "chat response changed generated text",
    )
    require(
        response["usage"]["total_tokens"] == prompt_token_count + generated_token_count,
        "chat response token total changed",
    )

    rejected = (
        ([], Endpoint.COMPLETIONS, "JSON object"),
        ({"prompt": ""}, Endpoint.COMPLETIONS, "non-empty"),
        ({"prompt": "ok", "max_tokens": 0}, Endpoint.COMPLETIONS, "between"),
        (
            {"prompt": "ok", "max_tokens": maximum_generation_tokens + 1},
            Endpoint.COMPLETIONS,
            "between",
        ),
        ({"prompt": "ok", "stream": True}, Endpoint.COMPLETIONS, "streaming"),
        ({"prompt": "ok", "temperature": 1}, Endpoint.COMPLETIONS, "greedy"),
        ({"prompt": "ok", "n": 2}, Endpoint.COMPLETIONS, "n must be 1"),
        ({"prompt": "ok", "model": "wrong"}, Endpoint.COMPLETIONS, "model must be"),
        (
            {"messages": [{"content": "answer", "role": "assistant"}]},
            Endpoint.CHAT_COMPLETIONS,
            "user message",
        ),
        (
            {"messages": [{"content": [{"type": "text"}], "role": "user"}]},
            Endpoint.CHAT_COMPLETIONS,
            "non-empty text",
        ),
    )
    for payload, endpoint, diagnostic in rejected:
        try:
            parse_completion_request(
                payload,
                endpoint,
                admitted_model,
                maximum_generation_tokens,
            )
        except RequestValidationError as error:
            require(diagnostic in str(error), f"wrong rejection diagnostic: {error}")
        else:
            require(False, f"invalid service request was accepted: {payload!r}")

    try:
        parse_endpoint("/unsupported")
    except RequestValidationError as error:
        require(
            "unsupported endpoint" in str(error), "wrong route rejection diagnostic"
        )
    else:
        require(False, "unsupported service route was accepted")


# r[verify tenstorrent.qwen38_p150x2.mtp_recovery_stage_probe]
def probe_mtp_recovery_stage():
    """Positive and negative tests for ordered staged progress."""
    from qwen36_service import mtp_stage_core as stage_core
    from qwen36_service.mtp_rollback_core import REPREFILL_RECOVERY, SNAPSHOT_RECOVERY

    progress = stage_core.new_mtp_stage_progress(
        prompt_index=stage_core.MINIMUM_PROMPT_INDEX,
        route=REPREFILL_RECOVERY,
    )
    success_stages = (
        stage_core.COMMAND_START,
        stage_core.MESH_OPEN_START,
        stage_core.MESH_OPEN_COMPLETE,
        stage_core.MODEL_LOAD_START,
        stage_core.MODEL_LOAD_COMPLETE,
        stage_core.ADAPTER_CREATE_START,
        stage_core.ADAPTER_CREATE_COMPLETE,
        stage_core.PILOT_START,
        stage_core.TARGET_PREFILL_START,
        stage_core.TARGET_PREFILL_COMPLETE,
        stage_core.DRAFT_START,
        stage_core.MTP_PREFILL_START,
        stage_core.MTP_PREFILL_COMPLETE,
        stage_core.TRACE_RUN_START,
        stage_core.TRACE_INPUT_ALLOCATION_START,
        stage_core.TRACE_INPUT_ALLOCATION_COMPLETE,
        stage_core.TRACE_CAPTURE_START,
        stage_core.TRACE_CAPTURE_COMPLETE,
        stage_core.TRACE_DRAFT_REPLAY_STEP_1_START,
        stage_core.TRACE_DRAFT_REPLAY_STEP_1_COMPLETE,
        stage_core.TRACE_DRAFT_REPLAY_STEP_2_START,
        stage_core.TRACE_DRAFT_REPLAY_STEP_2_COMPLETE,
        stage_core.TRACE_RUN_COMPLETE,
        stage_core.DRAFT_COMPLETE,
        stage_core.TARGET_TRACE_START,
        stage_core.TARGET_TRACE_COMPLETE,
        stage_core.TARGET_SEQUENTIAL_START,
        stage_core.TARGET_SEQUENTIAL_COMPLETE,
        stage_core.TARGET_TRACE_RELEASE_COMPLETE,
        stage_core.TARGET_REFERENCE_STATE_START,
        stage_core.TARGET_REFERENCE_STATE_COMPLETE,
        stage_core.VERIFICATION_PREFILL_START,
        stage_core.VERIFICATION_PREFILL_COMPLETE,
        stage_core.RECOVERY_START,
        stage_core.RECOVERY_QUEUE_DRAIN_START,
        stage_core.RECOVERY_QUEUE_DRAIN_COMPLETE,
        stage_core.RECOVERY_MTP_PREFILL_START,
        stage_core.RECOVERY_MTP_PREFILL_COMPLETE,
        stage_core.CONTROLLED_REPLAY_START,
        stage_core.CONTROLLED_REPLAY_STEP_1_START,
        stage_core.CONTROLLED_REPLAY_STEP_1_COMPLETE,
        stage_core.CONTROLLED_REPLAY_STEP_2_START,
        stage_core.CONTROLLED_REPLAY_STEP_2_COMPLETE,
        stage_core.CONTROLLED_REPLAY_STEP_3_START,
        stage_core.CONTROLLED_REPLAY_STEP_3_COMPLETE,
        stage_core.CONTROLLED_REPLAY_STEP_4_START,
        stage_core.CONTROLLED_REPLAY_STEP_4_COMPLETE,
        stage_core.CONTROLLED_REPLAY_COMPLETE,
        stage_core.RECOVERY_SYNCHRONIZE_START,
        stage_core.RECOVERY_SYNCHRONIZE_COMPLETE,
        stage_core.RECOVERY_STATE_READBACK_START,
        stage_core.RECOVERY_STATE_READBACK_COMPLETE,
        stage_core.RECOVERY_COMPLETE,
        stage_core.NATIVE_VERIFIER_START,
        stage_core.NATIVE_VERIFIER_COMPLETE,
        stage_core.TARGET_STATE_COMPARE_START,
        stage_core.TARGET_STATE_COMPARE_COMPLETE,
        stage_core.PILOT_COMPLETE,
        stage_core.CLEANUP_START,
        stage_core.CLEANUP_COMPLETE,
        stage_core.RECEIPT_WRITE_START,
        stage_core.COMPLETE,
    )
    for event_index, stage in enumerate(success_stages, start=1):
        progress = stage_core.advance_mtp_stage_progress(
            progress,
            stage=stage,
            elapsed_seconds=float(event_index),
        )
    require(progress.complete, "valid staged progress did not complete")
    require(
        progress.to_mapping()["last_stage"] == stage_core.COMPLETE,
        "valid staged progress lost its terminal stage",
    )

    failed = stage_core.new_mtp_stage_progress(
        prompt_index=stage_core.MINIMUM_PROMPT_INDEX,
        route=SNAPSHOT_RECOVERY,
    )
    for event_index, stage in enumerate(
        (
            stage_core.COMMAND_START,
            stage_core.PILOT_START,
            stage_core.ROLLBACK_CONTEXT_START,
            stage_core.FAILURE_RECORDED,
            stage_core.CLEANUP_START,
            stage_core.CLEANUP_COMPLETE,
        ),
        start=1,
    ):
        failed = stage_core.advance_mtp_stage_progress(
            failed,
            stage=stage,
            elapsed_seconds=float(event_index),
        )
    require(not failed.complete, "failed staged progress became complete")
    require(
        failed.last_stage == stage_core.CLEANUP_COMPLETE,
        "failed staged progress lost cleanup",
    )

    rejected = []
    try:
        stage_core.new_mtp_stage_progress(prompt_index=0, route=REPREFILL_RECOVERY)
    except stage_core.MtpStageProgressError:
        pass
    else:
        rejected.append("non-positive prompt index")

    initial = stage_core.new_mtp_stage_progress(
        prompt_index=stage_core.MINIMUM_PROMPT_INDEX,
        route=SNAPSHOT_RECOVERY,
    )
    initial = stage_core.advance_mtp_stage_progress(
        initial,
        stage=stage_core.COMMAND_START,
        elapsed_seconds=1.0,
    )
    later_elapsed_seconds = initial.events[-1].elapsed_seconds + 1.0
    invalid_cases = (
        (stage_core.COMMAND_START, later_elapsed_seconds, "repeated stage"),
        (
            stage_core.RECOVERY_MTP_PREFILL_START,
            later_elapsed_seconds,
            "route-incompatible stage",
        ),
        (stage_core.COMPLETE, later_elapsed_seconds, "early completion"),
        ("unknown-stage", later_elapsed_seconds, "unknown stage"),
        (stage_core.MESH_OPEN_START, 0.0, "regressed time"),
    )
    for stage, elapsed_seconds, label in invalid_cases:
        try:
            stage_core.advance_mtp_stage_progress(
                initial,
                stage=stage,
                elapsed_seconds=elapsed_seconds,
            )
        except stage_core.MtpStageProgressError:
            continue
        rejected.append(label)
    require(not rejected, f"invalid staged progress was accepted: {rejected}")


# r[verify tenstorrent.qwen38_p150x2.mtp_recovery_prewarmed_control]
def probe_mtp_recovery_prewarmed_control():
    """Positive and negative tests for the full prewarmed recovery ledger."""
    from tempfile import TemporaryDirectory

    from qwen36_service import mtp_recovery_control_core as recovery_core
    from qwen36_service import mtp_target_prewarm_core as prewarm_core
    from qwen36_service.mtp_drafter import (
        MtpDrafterCommandError,
        _MtpRecoveryControlProgressWriter,
    )
    from qwen36_service.mtp_drafter_core import EXPECTED_DRAFT_TOKEN_COUNT
    from qwen36_service.mtp_rollback_core import (
        EXPECTED_ROUTE_SCHEDULE,
        SNAPSHOT_RECOVERY,
    )

    prompt_count = recovery_core.EXPECTED_MINIMUM_PROMPT_COUNT
    route_schedule = EXPECTED_ROUTE_SCHEDULE
    progress = recovery_core.new_mtp_recovery_control_progress(
        prompt_count=prompt_count,
        route_schedule=route_schedule,
    )
    token_start = recovery_core.MINIMUM_PROMPT_INDEX
    draft_token_ids = tuple(
        range(token_start, token_start + EXPECTED_DRAFT_TOKEN_COUNT)
    )
    target_token_ids = draft_token_ids
    target_verification_token_ids = (*target_token_ids, target_token_ids[-1])
    signature = prewarm_core.make_target_validation_signature(
        draft_token_ids=draft_token_ids,
        target_token_ids=target_token_ids,
        target_verification_token_ids=target_verification_token_ids,
        accepted_prefix_length=prewarm_core.MINIMUM_ACCEPTED_PREFIX_LENGTH,
        rollback_applied=True,
        target_output_valid=True,
        rollback_valid=True,
    )
    changed_signature = prewarm_core.make_target_validation_signature(
        draft_token_ids=draft_token_ids,
        target_token_ids=target_token_ids,
        target_verification_token_ids=(
            *target_token_ids,
            target_token_ids[-1] + recovery_core.MINIMUM_PROMPT_INDEX,
        ),
        accepted_prefix_length=prewarm_core.MINIMUM_ACCEPTED_PREFIX_LENGTH,
        rollback_applied=True,
        target_output_valid=True,
        rollback_valid=True,
    )
    event_index = 0
    cache_entries = recovery_core.MINIMUM_PROGRAM_CACHE_ENTRIES

    def record(
        stage,
        *,
        observed_cache_entries=cache_entries,
        prompt_index=None,
        root_index=None,
        route=None,
        observed_signature=None,
    ):
        nonlocal event_index, progress
        event_index += recovery_core.MINIMUM_PROMPT_INDEX
        progress = recovery_core.advance_mtp_recovery_control_progress(
            progress,
            stage=stage,
            elapsed_seconds=float(event_index),
            program_cache_entries=observed_cache_entries,
            prompt_index=prompt_index,
            root_index=root_index,
            route=route,
            signature=observed_signature,
        )

    record(recovery_core.COMMAND_START, observed_cache_entries=None)
    record(recovery_core.MESH_OPEN_START, observed_cache_entries=None)
    record(recovery_core.MESH_OPEN_COMPLETE)
    record(recovery_core.MODEL_LOAD_START)
    record(recovery_core.MODEL_LOAD_COMPLETE)
    record(recovery_core.ADAPTER_CREATE_START)
    record(recovery_core.ADAPTER_CREATE_COMPLETE)
    record(recovery_core.PREWARM_START)
    first_prewarm_start = None
    for prompt_index in range(
        recovery_core.MINIMUM_PROMPT_INDEX,
        prompt_count + recovery_core.MINIMUM_PROMPT_INDEX,
    ):
        record(recovery_core.PREWARM_PROMPT_START, prompt_index=prompt_index)
        if first_prewarm_start is None:
            first_prewarm_start = progress
        record(
            recovery_core.PREWARM_PROMPT_COMPLETE,
            prompt_index=prompt_index,
            observed_signature=signature,
        )
    record(recovery_core.PREWARM_COMPLETE)
    record(recovery_core.PERSISTENT_WARMUP_START)
    first_warmup_start = None
    for prompt_index in range(
        recovery_core.MINIMUM_PROMPT_INDEX,
        prompt_count + recovery_core.MINIMUM_PROMPT_INDEX,
    ):
        for root_index, route in enumerate(
            recovery_core.WARMUP_ROUTES,
            start=recovery_core.MINIMUM_ROOT_INDEX,
        ):
            record(
                recovery_core.WARMUP_ROOT_START,
                prompt_index=prompt_index,
                root_index=root_index,
                route=route,
            )
            if first_warmup_start is None:
                first_warmup_start = progress
            record(
                recovery_core.WARMUP_ROOT_COMPLETE,
                prompt_index=prompt_index,
                root_index=root_index,
                route=route,
                observed_signature=signature,
            )
    record(recovery_core.PERSISTENT_WARMUP_COMPLETE)
    record(recovery_core.MEASUREMENT_START)
    measurement_started = progress
    first_measured_start = None
    for prompt_index in range(
        recovery_core.MINIMUM_PROMPT_INDEX,
        prompt_count + recovery_core.MINIMUM_PROMPT_INDEX,
    ):
        for root_index, route in enumerate(
            route_schedule,
            start=recovery_core.MINIMUM_ROOT_INDEX,
        ):
            record(
                recovery_core.MEASURED_ROOT_START,
                prompt_index=prompt_index,
                root_index=root_index,
                route=route,
            )
            if first_measured_start is None:
                first_measured_start = progress
            record(
                recovery_core.MEASURED_ROOT_COMPLETE,
                prompt_index=prompt_index,
                root_index=root_index,
                route=route,
                observed_signature=signature,
            )
    record(recovery_core.MEASUREMENT_COMPLETE)
    record(recovery_core.CLEANUP_START)
    record(recovery_core.CLEANUP_COMPLETE)
    record(recovery_core.EVALUATION_START)
    record(recovery_core.EVALUATION_COMPLETE)
    record(recovery_core.RECEIPT_WRITE_START)
    record(recovery_core.COMPLETE)
    require(progress.complete, "valid recovery control progress did not complete")
    progress_mapping = progress.to_mapping()
    require(
        progress_mapping["measurement_program_cache_entries"] == cache_entries,
        "recovery control lost its frozen program-cache count",
    )
    require(
        progress_mapping["admission_eligible"] is False,
        "recovery control progress became admission eligible",
    )

    failed = recovery_core.new_mtp_recovery_control_progress(
        prompt_count=prompt_count,
        route_schedule=route_schedule,
    )
    failure_elapsed_seconds = float(recovery_core.MINIMUM_PROMPT_INDEX)
    failed = recovery_core.advance_mtp_recovery_control_progress(
        failed,
        stage=recovery_core.COMMAND_START,
        elapsed_seconds=failure_elapsed_seconds,
        program_cache_entries=None,
    )
    failure_elapsed_seconds += recovery_core.MINIMUM_PROMPT_INDEX
    failed = recovery_core.advance_mtp_recovery_control_progress(
        failed,
        stage=recovery_core.FAILURE_RECORDED,
        elapsed_seconds=failure_elapsed_seconds,
        program_cache_entries=None,
    )
    failure_elapsed_seconds += recovery_core.MINIMUM_PROMPT_INDEX
    failed = recovery_core.advance_mtp_recovery_control_progress(
        failed,
        stage=recovery_core.CLEANUP_START,
        elapsed_seconds=failure_elapsed_seconds,
        program_cache_entries=None,
    )
    failure_elapsed_seconds += recovery_core.MINIMUM_PROMPT_INDEX
    failed = recovery_core.advance_mtp_recovery_control_progress(
        failed,
        stage=recovery_core.CLEANUP_COMPLETE,
        elapsed_seconds=failure_elapsed_seconds,
        program_cache_entries=None,
    )
    require(failed.failed, "recovery control failure was not retained")
    require(not failed.complete, "failed recovery control became complete")

    require(first_prewarm_start is not None, "prewarm test boundary is missing")
    require(first_warmup_start is not None, "warmup test boundary is missing")
    require(first_measured_start is not None, "measured test boundary is missing")
    rejected = []
    invalid_route_schedule = (
        SNAPSHOT_RECOVERY,
        *route_schedule[recovery_core.MINIMUM_PROMPT_INDEX :],
    )
    try:
        recovery_core.new_mtp_recovery_control_progress(
            prompt_count=prompt_count,
            route_schedule=invalid_route_schedule,
        )
    except recovery_core.MtpRecoveryControlProgressError:
        pass
    else:
        rejected.append("changed route schedule")
    try:
        recovery_core.advance_mtp_recovery_control_progress(
            first_prewarm_start,
            stage=recovery_core.PREWARM_PROMPT_COMPLETE,
            elapsed_seconds=first_prewarm_start.events[-1].elapsed_seconds
            + recovery_core.MINIMUM_PROMPT_INDEX,
            program_cache_entries=cache_entries,
            prompt_index=recovery_core.MINIMUM_PROMPT_INDEX,
        )
    except recovery_core.MtpRecoveryControlProgressError:
        pass
    else:
        rejected.append("missing prewarm signature")
    first_warmup_event = first_warmup_start.events[-1]
    try:
        recovery_core.advance_mtp_recovery_control_progress(
            first_warmup_start,
            stage=recovery_core.WARMUP_ROOT_COMPLETE,
            elapsed_seconds=first_warmup_event.elapsed_seconds
            + recovery_core.MINIMUM_PROMPT_INDEX,
            program_cache_entries=cache_entries,
            prompt_index=first_warmup_event.prompt_index,
            root_index=first_warmup_event.root_index,
            route=first_warmup_event.route,
            signature=changed_signature,
        )
    except recovery_core.MtpRecoveryControlProgressError:
        pass
    else:
        rejected.append("false warmup signature match")
    first_measured_event = first_measured_start.events[-1]
    try:
        recovery_core.advance_mtp_recovery_control_progress(
            first_measured_start,
            stage=recovery_core.MEASURED_ROOT_COMPLETE,
            elapsed_seconds=first_measured_event.elapsed_seconds
            + recovery_core.MINIMUM_PROMPT_INDEX,
            program_cache_entries=cache_entries + recovery_core.MINIMUM_PROMPT_INDEX,
            prompt_index=first_measured_event.prompt_index,
            root_index=first_measured_event.root_index,
            route=first_measured_event.route,
            signature=signature,
        )
    except recovery_core.MtpRecoveryControlProgressError:
        pass
    else:
        rejected.append("measured program-cache growth")
    try:
        recovery_core.advance_mtp_recovery_control_progress(
            measurement_started,
            stage=recovery_core.MEASURED_ROOT_START,
            elapsed_seconds=measurement_started.events[-1].elapsed_seconds
            - recovery_core.MINIMUM_PROMPT_INDEX,
            program_cache_entries=cache_entries,
            prompt_index=recovery_core.MINIMUM_PROMPT_INDEX,
            root_index=recovery_core.MINIMUM_ROOT_INDEX,
            route=route_schedule[0],
        )
    except recovery_core.MtpRecoveryControlProgressError:
        pass
    else:
        rejected.append("regressed elapsed time")
    require(not rejected, f"invalid recovery control progress was accepted: {rejected}")

    class FakeMesh:
        def __init__(self):
            self.entries = cache_entries

        def num_program_cache_entries(self):
            return self.entries

    with TemporaryDirectory() as temporary_directory:
        progress_path = Path(temporary_directory) / "progress.json"
        writer = _MtpRecoveryControlProgressWriter(
            path=progress_path,
            prompt_count=prompt_count,
            route_schedule=route_schedule,
        )
        writer.record(recovery_core.COMMAND_START)
        writer.record(recovery_core.MESH_OPEN_START)
        fake_mesh = FakeMesh()
        writer.bind_program_cache(fake_mesh)
        writer.record(recovery_core.MESH_OPEN_COMPLETE)
        fake_mesh.entries += recovery_core.MINIMUM_PROMPT_INDEX
        writer.record(recovery_core.MODEL_LOAD_START)
        writer.freeze_program_cache()
        fake_mesh.entries += recovery_core.MINIMUM_PROMPT_INDEX
        writer.record(recovery_core.MODEL_LOAD_COMPLETE)
        persisted = json.loads(progress_path.read_text())
        require(
            persisted["events"][-1]["program_cache_entries"]
            == cache_entries + recovery_core.MINIMUM_PROMPT_INDEX,
            "recovery control writer did not freeze its cache observation",
        )
        require(
            not progress_path.with_name(progress_path.name + ".tmp").exists(),
            "recovery control writer left its temporary file",
        )
        invalid_writer = _MtpRecoveryControlProgressWriter(
            path=Path(temporary_directory) / "invalid.json",
            prompt_count=prompt_count,
            route_schedule=route_schedule,
        )
        try:
            invalid_writer.bind_program_cache(object())
        except MtpDrafterCommandError:
            pass
        else:
            require(False, "recovery control accepted a missing cache reader")


# r[verify tenstorrent.qwen38_p150x2.mtp_target_validation_prewarm]
def probe_mtp_target_validation_prewarm():
    """Positive and negative tests for exact target prewarm evidence."""
    from tempfile import TemporaryDirectory

    from qwen36_service import mtp_stage_core as stage_core
    from qwen36_service import mtp_target_prewarm_core as prewarm_core
    from qwen36_service.mtp_drafter import (
        MtpDrafterCommandError,
        _MtpStageProgressWriter,
    )
    from qwen36_service.mtp_rollback_core import REPREFILL_RECOVERY

    progress = stage_core.new_mtp_stage_progress(
        prompt_index=stage_core.MINIMUM_PROMPT_INDEX,
        route=REPREFILL_RECOVERY,
        prewarm_target_validation=True,
    )
    initial_cache_entries = stage_core.MINIMUM_PROGRAM_CACHE_ENTRIES
    warmed_cache_entries = initial_cache_entries + stage_core.MINIMUM_PROMPT_INDEX
    progress_events = (
        (stage_core.COMMAND_START, None),
        (stage_core.MESH_OPEN_START, None),
        (stage_core.MESH_OPEN_COMPLETE, initial_cache_entries),
        (stage_core.MODEL_LOAD_START, initial_cache_entries),
        (stage_core.MODEL_LOAD_COMPLETE, warmed_cache_entries),
        (stage_core.ADAPTER_CREATE_START, warmed_cache_entries),
        (stage_core.ADAPTER_CREATE_COMPLETE, warmed_cache_entries),
        (stage_core.TARGET_VALIDATION_PREWARM_START, warmed_cache_entries),
        (
            stage_core.target_validation_prewarm_stage(stage_core.TARGET_PREFILL_START),
            warmed_cache_entries,
        ),
        (
            stage_core.target_validation_prewarm_stage(
                stage_core.NATIVE_VERIFIER_START
            ),
            warmed_cache_entries,
        ),
        (
            stage_core.target_validation_prewarm_stage(
                stage_core.NATIVE_VERIFIER_COMPLETE
            ),
            warmed_cache_entries,
        ),
        (
            stage_core.target_validation_prewarm_stage(stage_core.PILOT_COMPLETE),
            warmed_cache_entries,
        ),
        (stage_core.TARGET_VALIDATION_PREWARM_COMPLETE, warmed_cache_entries),
        (stage_core.PILOT_START, warmed_cache_entries),
        (stage_core.RECOVERY_START, warmed_cache_entries),
        (stage_core.RECOVERY_COMPLETE, warmed_cache_entries),
        (stage_core.NATIVE_VERIFIER_START, warmed_cache_entries),
        (stage_core.NATIVE_VERIFIER_COMPLETE, warmed_cache_entries),
        (stage_core.PILOT_COMPLETE, warmed_cache_entries),
        (stage_core.PREWARM_SIGNATURE_COMPARE_START, warmed_cache_entries),
        (stage_core.PREWARM_SIGNATURE_COMPARE_COMPLETE, warmed_cache_entries),
        (stage_core.CLEANUP_START, warmed_cache_entries),
        (stage_core.CLEANUP_COMPLETE, warmed_cache_entries),
        (stage_core.RECEIPT_WRITE_START, warmed_cache_entries),
        (stage_core.COMPLETE, warmed_cache_entries),
    )
    for event_index, (stage, cache_entries) in enumerate(progress_events, start=1):
        progress = stage_core.advance_mtp_stage_progress(
            progress,
            stage=stage,
            elapsed_seconds=float(event_index),
            program_cache_entries=cache_entries,
        )
    require(progress.complete, "valid target-prewarm progress did not complete")
    progress_mapping = progress.to_mapping()
    require(
        progress_mapping["prewarm_target_validation"] is True,
        "target-prewarm progress lost its mode",
    )
    require(
        progress_mapping["events"][-1]["program_cache_entries"] == warmed_cache_entries,
        "target-prewarm progress lost its program-cache observation",
    )

    class FakeMesh:
        def __init__(self):
            self.entries = initial_cache_entries

        def num_program_cache_entries(self):
            return self.entries

    with TemporaryDirectory() as temporary_directory:
        progress_path = Path(temporary_directory) / "progress.json"
        writer = _MtpStageProgressWriter(
            path=progress_path,
            prompt_index=stage_core.MINIMUM_PROMPT_INDEX,
            route=REPREFILL_RECOVERY,
            prewarm_target_validation=False,
        )
        writer.record(stage_core.COMMAND_START)
        writer.record(stage_core.MESH_OPEN_START)
        fake_mesh = FakeMesh()
        writer.bind_program_cache(fake_mesh)
        writer.record(stage_core.MESH_OPEN_COMPLETE)
        fake_mesh.entries = warmed_cache_entries
        writer.record(stage_core.MODEL_LOAD_START)
        writer.freeze_program_cache()
        fake_mesh.entries = initial_cache_entries
        writer.record(stage_core.FAILURE_RECORDED)
        writer.record(stage_core.CLEANUP_START)
        writer.record(stage_core.CLEANUP_COMPLETE)
        persisted = json.loads(progress_path.read_text())
        require(
            persisted["last_stage"] == stage_core.CLEANUP_COMPLETE,
            "stage writer lost its terminal boundary",
        )
        require(
            persisted["events"][-1]["program_cache_entries"] == warmed_cache_entries,
            "stage writer did not freeze its last program-cache observation",
        )
        require(
            not progress_path.with_name(progress_path.name + ".tmp").exists(),
            "stage writer left its temporary file",
        )
        invalid_writer = _MtpStageProgressWriter(
            path=Path(temporary_directory) / "invalid.json",
            prompt_index=stage_core.MINIMUM_PROMPT_INDEX,
            route=REPREFILL_RECOVERY,
            prewarm_target_validation=False,
        )
        try:
            invalid_writer.bind_program_cache(object())
        except MtpDrafterCommandError:
            pass
        else:
            require(False, "stage writer accepted a missing program-cache reader")

    token_start = prewarm_core.MINIMUM_TOKEN_ID + stage_core.MINIMUM_PROMPT_INDEX
    draft_count = 3
    draft_token_ids = tuple(range(token_start, token_start + draft_count))
    target_token_ids = draft_token_ids
    target_verification_token_ids = (*target_token_ids, target_token_ids[-1])
    signature = prewarm_core.make_target_validation_signature(
        draft_token_ids=draft_token_ids,
        target_token_ids=target_token_ids,
        target_verification_token_ids=target_verification_token_ids,
        accepted_prefix_length=stage_core.MINIMUM_PROMPT_INDEX,
        rollback_applied=True,
        target_output_valid=True,
        rollback_valid=True,
    )
    same_signature = prewarm_core.make_target_validation_signature(
        draft_token_ids=draft_token_ids,
        target_token_ids=target_token_ids,
        target_verification_token_ids=target_verification_token_ids,
        accepted_prefix_length=stage_core.MINIMUM_PROMPT_INDEX,
        rollback_applied=True,
        target_output_valid=True,
        rollback_valid=True,
    )
    require(
        prewarm_core.target_validation_signatures_match(signature, same_signature),
        "equal target-prewarm signatures did not match",
    )
    changed_verification = (
        *target_token_ids,
        target_token_ids[-1] + stage_core.MINIMUM_PROMPT_INDEX,
    )
    changed_signature = prewarm_core.make_target_validation_signature(
        draft_token_ids=draft_token_ids,
        target_token_ids=target_token_ids,
        target_verification_token_ids=changed_verification,
        accepted_prefix_length=stage_core.MINIMUM_PROMPT_INDEX,
        rollback_applied=True,
        target_output_valid=True,
        rollback_valid=True,
    )
    require(
        not prewarm_core.target_validation_signatures_match(
            signature,
            changed_signature,
        ),
        "changed target-prewarm signature matched",
    )

    rejected = []
    invalid_signature_cases = (
        {
            "target_output_valid": False,
            "rollback_valid": True,
            "rollback_applied": True,
            "accepted_prefix_length": stage_core.MINIMUM_PROMPT_INDEX,
            "label": "inexact target output",
        },
        {
            "target_output_valid": True,
            "rollback_valid": False,
            "rollback_applied": True,
            "accepted_prefix_length": stage_core.MINIMUM_PROMPT_INDEX,
            "label": "inexact rollback",
        },
        {
            "target_output_valid": True,
            "rollback_valid": True,
            "rollback_applied": False,
            "accepted_prefix_length": stage_core.MINIMUM_PROMPT_INDEX,
            "label": "wrong rollback flag",
        },
    )
    for case in invalid_signature_cases:
        label = case["label"]
        try:
            prewarm_core.make_target_validation_signature(
                draft_token_ids=draft_token_ids,
                target_token_ids=target_token_ids,
                target_verification_token_ids=target_verification_token_ids,
                accepted_prefix_length=case["accepted_prefix_length"],
                rollback_applied=case["rollback_applied"],
                target_output_valid=case["target_output_valid"],
                rollback_valid=case["rollback_valid"],
            )
        except prewarm_core.MtpTargetPrewarmError:
            continue
        rejected.append(label)

    try:
        stage_core.target_validation_prewarm_stage(stage_core.RECOVERY_START)
    except stage_core.MtpStageProgressError:
        pass
    else:
        rejected.append("recovery stage in prewarm namespace")

    try:
        stage_core.new_mtp_stage_progress(
            prompt_index=stage_core.MINIMUM_PROMPT_INDEX,
            route=REPREFILL_RECOVERY,
            prewarm_target_validation="yes",
        )
    except stage_core.MtpStageProgressError:
        pass
    else:
        rejected.append("non-boolean prewarm mode")

    cache_progress = stage_core.new_mtp_stage_progress(
        prompt_index=stage_core.MINIMUM_PROMPT_INDEX,
        route=REPREFILL_RECOVERY,
    )
    cache_progress = stage_core.advance_mtp_stage_progress(
        cache_progress,
        stage=stage_core.MESH_OPEN_COMPLETE,
        elapsed_seconds=float(stage_core.MINIMUM_PROMPT_INDEX),
        program_cache_entries=warmed_cache_entries,
    )
    invalid_cache_cases = (
        (initial_cache_entries, "regressing program-cache count"),
        (None, "missing program-cache count"),
    )
    for cache_entries, label in invalid_cache_cases:
        try:
            stage_core.advance_mtp_stage_progress(
                cache_progress,
                stage=stage_core.MODEL_LOAD_START,
                elapsed_seconds=float(warmed_cache_entries),
                program_cache_entries=cache_entries,
            )
        except stage_core.MtpStageProgressError:
            continue
        rejected.append(label)
    try:
        stage_core.advance_mtp_stage_progress(
            stage_core.new_mtp_stage_progress(
                prompt_index=stage_core.MINIMUM_PROMPT_INDEX,
                route=REPREFILL_RECOVERY,
            ),
            stage=stage_core.MESH_OPEN_COMPLETE,
            elapsed_seconds=float(stage_core.MINIMUM_PROMPT_INDEX),
            program_cache_entries=-stage_core.MINIMUM_PROMPT_INDEX,
        )
    except stage_core.MtpStageProgressError:
        pass
    else:
        rejected.append("negative program-cache count")
    require(not rejected, f"invalid target-prewarm evidence was accepted: {rejected}")


# r[verify tenstorrent.qwen38_p150x2.mtp_exact_rejection_sampling]
def probe_mtp_sampled_acceptance():
    """Positive and negative tests for distribution-exact MTP acceptance."""
    import math

    from qwen36_service.mtp_sampler_core import (
        ACCEPTANCE_RULES,
        RULE_EXACT,
        RULE_GREEDY,
        RULE_LENIENT,
        RULE_RELAXED,
        MtpSamplerPolicyError,
        acceptance_probability,
        admitted_production_rule,
        realized_total_variation,
        resolve_rule,
        resolve_speculative_round,
        sample_from_distribution,
        truncated_target_distribution,
    )

    # Candidate row with a known distribution: 0.5 / 0.3 / 0.2.
    values = [math.log(0.5), math.log(0.3), math.log(0.2)]
    token_ids = [7, 8, 9]
    probabilities, ids = truncated_target_distribution(values, token_ids)
    expected_probabilities = (0.5, 0.3, 0.2)
    for computed, expected in zip(probabilities, expected_probabilities, strict=True):
        require(
            math.isclose(computed, expected, rel_tol=1e-12),
            f"truncated distribution drifted: {probabilities}",
        )
    require(ids == tuple(token_ids), "distribution ordering changed")

    # Exact rule: accept probability equals the draft's own mass; TV is zero.
    draft_mass, accepted = acceptance_probability(
        RULE_EXACT, probabilities, ids, 8
    )
    require(
        math.isclose(draft_mass, 0.3, rel_tol=1e-12),
        f"exact rule draft mass drifted: {draft_mass}",
    )
    require(
        math.isclose(accepted, draft_mass, rel_tol=1e-12),
        "exact rule acceptance probability differs from p(x^)",
    )
    require(
        realized_total_variation(accepted, draft_mass) == 0.0,
        "exact rule reported nonzero total variation",
    )

    # Lenient reduces to exact at lenience 1.0 and interpolates below it.
    _, lenient_exact = acceptance_probability(
        RULE_LENIENT,
        probabilities,
        ids,
        8,
        lenience=1.0,
    )
    require(
        math.isclose(lenient_exact, accepted, rel_tol=1e-12),
        "lenient rule at lenience 1.0 differs from exact",
    )
    _, lenient_half = acceptance_probability(
        RULE_LENIENT,
        probabilities,
        ids,
        8,
        lenience=0.5,
    )
    require(
        math.isclose(lenient_half, min(1.0, draft_mass / 0.5), rel_tol=1e-12),
        "lenient rule interpolation changed",
    )

    # Greedy accepts only the argmax choice.
    _, greedy_top = acceptance_probability(RULE_GREEDY, probabilities, ids, 7)
    _, greedy_other = acceptance_probability(RULE_GREEDY, probabilities, ids, 9)
    require(greedy_top == 1.0, "greedy rule rejected the argmax draft")
    require(greedy_other == 0.0, "greedy rule accepted a non-argmax draft")

    # Relaxed typical acceptance is deterministic against its threshold.
    _, relaxed = acceptance_probability(RULE_RELAXED, probabilities, ids, 8)
    entropy = -sum(p * math.log(p) for p in probabilities)
    threshold = min(0.09, 0.3 * math.exp(-entropy))
    require(
        relaxed == float(0.3 >= threshold),
        "relaxed rule diverged from its typical-acceptance threshold",
    )

    # Residual sampling excludes the rejected draft and renormalises.
    residual = sample_from_distribution(probabilities, ids, 0.99, exclude_token_id=7)
    require(residual in (8, 9), "residual sample escaped the renormalised support")
    require(
        sample_from_distribution((1.0,), (7,), 0.5, exclude_token_id=7) == 7,
        "degenerate residual support raised instead of falling back",
    )

    # Full-accept round emits every draft plus the bonus token.
    gamma = 2
    row_values = [values] * (gamma + 1)
    row_ids = [token_ids] * (gamma + 1)
    drafts = (7, 8)

    def fixed_uniforms(sequence):
        iterator = iter(sequence)
        return lambda: next(iterator)

    outcome = resolve_speculative_round(
        row_values,
        row_ids,
        drafts,
        fixed_uniforms((0.0, 0.0, 0.95)),
        rule=RULE_EXACT,
    )
    require(
        outcome.emitted_token_ids == drafts + (9,),
        f"full-accept round emitted {outcome.emitted_token_ids}",
    )
    require(outcome.bonus_emitted, "full-accept round lost its bonus flag")
    require(
        outcome.accepted_draft_count == gamma,
        "full-accept round miscounted accepted drafts",
    )

    # Early rejection stops the walk and samples the residual.
    rejecting = resolve_speculative_round(
        row_values,
        row_ids,
        (9, 9),
        fixed_uniforms((0.999, 0.0, 0.0)),
        rule=RULE_EXACT,
    )
    require(
        rejecting.accepted_draft_count == 0,
        "rejected draft was counted as accepted",
    )
    require(
        rejecting.emitted_token_ids[0] != 9 or len(rejecting.emitted_token_ids) == 2,
        "rejected round kept walking later rows",
    )

    # Truncation exactness: penalising a candidate outside the kept top-k set
    # leaves the kept distribution identical to no penalty at all.
    wide_values = [3.0, 2.0, 1.0, 0.5]
    wide_ids = [11, 12, 13, 14]
    unpenalised = truncated_target_distribution(
        wide_values,
        wide_ids,
        top_k=2,
    )
    penalised = truncated_target_distribution(
        wide_values,
        wide_ids,
        top_k=2,
        presence_penalty=5.0,
        penalised_token_ids=(13, 14),
    )
    require(
        unpenalised == penalised,
        "out-of-support penalties leaked into the kept top-k distribution",
    )

    # Production admission stays closed to sampled rules.
    require(admitted_production_rule() == RULE_GREEDY, "production rule moved")
    require(
        resolve_rule(RULE_GREEDY, diagnostic_only=False) == RULE_GREEDY,
        "greedy rule lost production eligibility",
    )
    try:
        resolve_rule(RULE_EXACT, diagnostic_only=False)
    except MtpSamplerPolicyError as error:
        require("diagnostic-only" in str(error), "wrong sampled-rule diagnostic")
    else:
        require(False, "sampled rule was admitted to production")
    for rule in ACCEPTANCE_RULES:
        resolve_rule(rule, diagnostic_only=True)

    # Fail-closed matrix over malformed inputs.
    negative_calls = (
        (
            lambda: truncated_target_distribution([], []),
            "candidate list is empty",
        ),
        (
            lambda: truncated_target_distribution([0.0], [1, 2]),
            "differ in length",
        ),
        (
            lambda: truncated_target_distribution([float("nan")], [1]),
            "must be finite",
        ),
        (
            lambda: truncated_target_distribution(values, token_ids, temperature=0.0),
            "finite and positive",
        ),
        (
            lambda: truncated_target_distribution(values, token_ids, top_k=-1),
            "non-negative integer",
        ),
        (
            lambda: truncated_target_distribution(values, token_ids, top_p=0.0),
            "top_p must lie",
        ),
        (
            lambda: truncated_target_distribution(
                values,
                token_ids,
                presence_penalty=-1.0,
            ),
            "non-negative",
        ),
        (
            lambda: acceptance_probability("typical", probabilities, ids, 7),
            "acceptance rule must be one of",
        ),
        (
            lambda: acceptance_probability(RULE_EXACT, [], [], 7),
            "distribution is empty",
        ),
        (
            lambda: acceptance_probability(RULE_LENIENT, probabilities, ids, 7, lenience=0.0),
            "lenience must be finite and positive",
        ),
        (
            lambda: resolved_round_exhausted_uniforms(),
            "uniform stream exhausted",
        ),
        (
            lambda: resolve_speculative_round(row_values[:-1], row_ids, drafts, iter([])),
            "draft count plus one",
        ),
        (
            lambda: resolve_speculative_round(row_values, row_ids, (), iter([])),
            "at least one draft",
        ),
    )

    def resolved_round_exhausted_uniforms():
        return resolve_speculative_round(
            row_values,
            row_ids,
            drafts,
            fixed_uniforms((0.0, 0.0)),
            rule=RULE_EXACT,
        )

    for negative_call, diagnostic in negative_calls:
        try:
            negative_call()
        except MtpSamplerPolicyError as error:
            require(diagnostic in str(error), f"wrong sampler diagnostic: {error}")
        else:
            require(False, f"invalid sampler input was accepted: {negative_call!r}")


def main():
    probe("import ttnn (packaged runtime)", import_ttnn)
    probe("import models.tt_transformers.tt.model_config", import_transformer_config)
    probe("import gated_attention_gated_deltanet tt ops", import_gdn_ops)
    probe("import qwen36.tt.common", import_qwen36_common)
    probe("import qwen36.tt.model_config", import_qwen36_model_config)
    probe("import qwen36.tt.model", import_qwen36_model)
    probe("import qwen36.tt.qwen36_vllm (vLLM absent)", import_qwen36_vllm_adapter)
    probe("MLP down-proj BF4 gate defaults", probe_mlp_down_gate)
    probe("LM head BF4 gate + .bfp4 cache name", probe_lm_head_gate)
    probe("source env gates + Qwen3.8-27B entries", probe_source_env_gates)
    probe("checkpoint layer-type authority", probe_layer_type_authority)
    probe("TP=2 / P300 gate invariants", probe_p150x2_gate)
    probe("traced decode pure core", probe_traced_decode_contract)
    probe("prefill program profile", probe_prefill_program_profile)
    probe("MTP target-verification pure core", probe_mtp_target_verification)
    probe("MTP drafter pure core", probe_mtp_drafter)
    probe("MTP draft selection", probe_mtp_draft_selection)
    probe("MTP draft trace", probe_mtp_draft_trace)
    probe("MTP K/V rollback", probe_mtp_kv_rollback)
    probe("MTP recovery stage progress", probe_mtp_recovery_stage)
    probe("MTP prewarmed recovery control", probe_mtp_recovery_prewarmed_control)
    probe("MTP target-validation prewarm", probe_mtp_target_validation_prewarm)
    probe("MTP sampled acceptance", probe_mtp_sampled_acceptance)
    probe("decode component attribution", probe_decode_component_attribution)
    probe("same-process TTFT control", probe_ttft_control)
    probe("serialized service request contract", probe_service_contract)

    if FAILURES:
        print("qwen36 device-free check: FAIL")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("qwen36 device-free check: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
