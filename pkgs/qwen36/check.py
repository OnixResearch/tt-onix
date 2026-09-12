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

    # Runtime availability does not admit a new numerical prefill path.
    # Keep the seq adapter until model-level evidence admits fused prefill.
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
        hasattr(_ttnn.transformer, "chunk_gated_delta_rule"),
        "packaged 0.77 runtime lacks chunk_gated_delta_rule",
    )
    from unittest import mock

    from models.demos.blackhole.qwen36.tt.gdn import fused_chunk as fused_chunk_module

    require(
        not fused_chunk_module.FUSED_CHUNK_PREFILL_ADMITTED
        and not fused_chunk_module.fused_chunk_enabled(),
        "a runtime upgrade enabled unadmitted fused GDN prefill",
    )
    with mock.patch.object(fused_chunk_module, "FUSED_CHUNK_PREFILL_ADMITTED", True):
        require(
            fused_chunk_module.fused_chunk_enabled(),
            "admitted fused GDN capability was rejected",
        )
        with mock.patch.object(
            fused_chunk_module.ttnn, "transformer", SimpleNamespace()
        ):
            require(
                not fused_chunk_module.fused_chunk_enabled(),
                "missing fused GDN capability was admitted",
            )
    require(
        hasattr(_ttnn.transformer, "gated_delta_attn_seq"),
        "packaged runtime lacks gated_delta_attn_seq for the seq fallback",
    )

    # tp2-runtime-compat: the TP=4-tuned fused GDN output reduction waits on
    # P150x2. The new AGMM keyword does not establish numerical admission.
    # Keep the safe fallbacks admission-, capability-, and topology-gated.
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
        tp_common._runtime_supports_fused_swiglu(),
        "packaged 0.77 runtime lacks the expected fuse_swiglu keyword",
    )
    require(
        not tp_common.FUSED_MLP_GATEUP_ADMITTED
        and not tp_common.mlp_gateup_agmm_enabled(p150x2_device_count),
        "a runtime upgrade enabled unadmitted MLP gate/up AGMM",
    )
    with mock.patch.object(tp_common, "FUSED_MLP_GATEUP_ADMITTED", True):
        require(
            tp_common.mlp_gateup_agmm_enabled(p150x2_device_count),
            "admitted AGMM capability was rejected",
        )
        require(
            not tp_common.mlp_gateup_agmm_enabled(1), "single-device AGMM was admitted"
        )
        with mock.patch.object(
            tp_common, "_runtime_supports_fused_swiglu", return_value=False
        ):
            require(
                not tp_common.mlp_gateup_agmm_enabled(p150x2_device_count),
                "missing AGMM capability was admitted",
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


# r[verify tenstorrent.qwen38_p150x2.batched_concurrency]
# r[verify tenstorrent.qwen38_p150x2.same_width_trace_reuse]
def probe_batch_decode_scheduler():
    """Positive and negative tests for pure active-batch planning."""
    from qwen36_service.batch_decode_core import (
        MAXIMUM_ACTIVE_BATCH_WIDTH,
        MAXIMUM_RESIDENT_TRACE_COUNT,
        TRACE_REUSE_REASON_BINDING,
        TRACE_REUSE_REASON_DISABLED,
        TRACE_REUSE_REASON_MISSING,
        TRACE_REUSE_REASON_MULTIPLE,
        TRACE_REUSE_REASON_RESET,
        TRACE_REUSE_REASON_WIDTH,
        ActiveDecodeUser,
        BatchDecodePolicyError,
        TracePrefillAction,
        plan_decode_batch,
        plan_same_width_trace_prefill,
    )

    test_vocabulary_size = 128
    test_sequence_length = 2048
    first_token_id = 7
    second_token_id = 11
    first_position = 32
    second_position = 64
    users = (
        ActiveDecodeUser("request-a", first_token_id, first_position, 0),
        ActiveDecodeUser("request-b", second_token_id, second_position, 1),
    )
    plan = plan_decode_batch(
        users,
        vocabulary_size=test_vocabulary_size,
        maximum_sequence_length=test_sequence_length,
    )
    require(plan.width == len(users), "batch planner selected the wrong active width")
    require(
        tuple(row.request_id for row in plan.rows)
        == tuple(user.request_id for user in users),
        "batch planner changed user row ownership",
    )
    require(
        tuple(row.state_slice for row in plan.rows)
        == tuple(user.state_slice for user in users),
        "batch planner changed isolated state slices",
    )
    require(
        all(row.next_position == row.position + 1 for row in plan.rows),
        "batch planner did not advance every user by one token",
    )

    mixed_user_count = 4
    mixed_users = tuple(
        ActiveDecodeUser(
            request_id=f"request-mixed-{index}",
            token_id=first_token_id + index,
            position=first_position + index,
            state_slice=index,
        )
        for index in range(mixed_user_count)
    )
    mixed_plan = plan_decode_batch(
        mixed_users,
        vocabulary_size=test_vocabulary_size,
        maximum_sequence_length=test_sequence_length,
    )
    require(
        mixed_plan.width == mixed_user_count,
        "mixed-user batch planner selected the wrong width",
    )
    require(
        len({row.state_slice for row in mixed_plan.rows}) == mixed_user_count,
        "mixed-user batch planner lost state isolation",
    )

    trace_facts = {
        "requested_width": mixed_user_count,
        "resident_widths": (mixed_user_count,),
        "reuse_enabled": True,
        "prior_reset_complete": True,
        "binding_stable": True,
    }
    reuse_plan = plan_same_width_trace_prefill(**trace_facts)
    require(
        reuse_plan.action is TracePrefillAction.PRESERVE,
        "same-width stable trace was not preserved",
    )
    require(reuse_plan.reuse_after_prefill, "preserved trace was not reusable")
    require(reuse_plan.reason is None, "valid trace reuse has a rejection reason")

    release_cases = (
        ({"reuse_enabled": False}, TRACE_REUSE_REASON_DISABLED),
        ({"resident_widths": ()}, TRACE_REUSE_REASON_MISSING),
        (
            {"resident_widths": (mixed_user_count, mixed_user_count)},
            TRACE_REUSE_REASON_MULTIPLE,
        ),
        ({"resident_widths": (1,)}, TRACE_REUSE_REASON_WIDTH),
        ({"prior_reset_complete": False}, TRACE_REUSE_REASON_RESET),
        ({"binding_stable": False}, TRACE_REUSE_REASON_BINDING),
    )
    for override, expected_reason in release_cases:
        release_plan = plan_same_width_trace_prefill(**{**trace_facts, **override})
        require(
            release_plan.action is TracePrefillAction.RELEASE,
            f"unsafe trace was preserved: {expected_reason}",
        )
        require(
            not release_plan.reuse_after_prefill,
            f"unsafe trace became reusable: {expected_reason}",
        )
        require(
            release_plan.reason == expected_reason,
            f"wrong trace release reason: {release_plan.reason}",
        )

    invalid_trace_facts = (
        ({"requested_width": 0}, "requested trace width"),
        ({"requested_width": True}, "requested trace width"),
        ({"resident_widths": [mixed_user_count]}, "resident trace widths"),
        ({"resident_widths": (True,)}, "resident trace width"),
        ({"reuse_enabled": 1}, "reuse enabled"),
        ({"prior_reset_complete": None}, "prior reset complete"),
        ({"binding_stable": "yes"}, "binding stable"),
    )
    for override, diagnostic in invalid_trace_facts:
        try:
            plan_same_width_trace_prefill(**{**trace_facts, **override})
        except BatchDecodePolicyError as error:
            require(diagnostic in str(error), f"wrong trace-policy diagnostic: {error}")
        else:
            require(False, f"invalid trace facts were accepted: {override!r}")

    too_wide_users = tuple(
        ActiveDecodeUser(
            request_id=f"request-wide-{index}",
            token_id=first_token_id,
            position=first_position,
            state_slice=index % MAXIMUM_ACTIVE_BATCH_WIDTH,
        )
        for index in range(MAXIMUM_ACTIVE_BATCH_WIDTH + 1)
    )
    rejected_cases = (
        ((), "active batch width"),
        (too_wide_users, "active batch width"),
        (
            (
                ActiveDecodeUser(
                    "request-collision-a", first_token_id, first_position, 0
                ),
                ActiveDecodeUser(
                    "request-collision-b", second_token_id, second_position, 0
                ),
            ),
            "state slice collision",
        ),
        (
            (
                ActiveDecodeUser("request-gap-a", first_token_id, first_position, 0),
                ActiveDecodeUser("request-gap-b", second_token_id, second_position, 2),
            ),
            "active prefix",
        ),
    )
    for rejected_users, diagnostic in rejected_cases:
        try:
            plan_decode_batch(
                rejected_users,
                vocabulary_size=test_vocabulary_size,
                maximum_sequence_length=test_sequence_length,
            )
        except BatchDecodePolicyError as error:
            require(diagnostic in str(error), f"wrong batch diagnostic: {error}")
        else:
            require(False, f"invalid active batch was accepted: {diagnostic}")

    from qwen36_service.batch_decode_adapter import (
        BatchDecodeAdapterError,
        BatchedTracedDecodeAdapter,
    )

    adapter = object.__new__(BatchedTracedDecodeAdapter)
    adapter.maximum_width = MAXIMUM_ACTIVE_BATCH_WIDTH
    adapter._closed = False
    adapter._require_width(1)
    adapter._require_width(MAXIMUM_ACTIVE_BATCH_WIDTH)
    for invalid_width in (0, MAXIMUM_ACTIVE_BATCH_WIDTH + 1, True):
        try:
            adapter._require_width(invalid_width)
        except BatchDecodeAdapterError as error:
            require("width" in str(error), f"wrong adapter-width diagnostic: {error}")
        else:
            require(False, f"invalid adapter width was accepted: {invalid_width!r}")

    resident_trace = object()
    adapter.vocabulary_size = test_vocabulary_size
    adapter.profile = SimpleNamespace(maximum_sequence_length=test_sequence_length)
    adapter._traces = {mixed_user_count: resident_trace}
    adapter._pending_trace_plan = reuse_plan
    adapter._binding_matches = lambda trace: trace is resident_trace
    reused_preparation = adapter.prepare_width(
        token_ids=tuple(user.token_id for user in mixed_users),
        positions=tuple(user.position for user in mixed_users),
    )
    require(reused_preparation.reused, "adapter recaptured a reusable trace")
    require(
        reused_preparation.capture_count == 0,
        "adapter reported a capture during trace reuse",
    )
    require(reused_preparation.binding_stable, "reused trace binding is unstable")
    require(
        reused_preparation.resident_trace_count == MAXIMUM_RESIDENT_TRACE_COUNT,
        "adapter reuse changed the resident trace count",
    )
    adapter._pending_trace_plan = reuse_plan
    try:
        adapter.prepare_width(
            token_ids=(True, *tuple(user.token_id for user in mixed_users[1:])),
            positions=tuple(user.position for user in mixed_users),
        )
    except BatchDecodeAdapterError as error:
        require("token" in str(error), f"wrong reused-token diagnostic: {error}")
    else:
        require(False, "trace reuse bypassed token validation")

    adapter_source = Path(
        __import__(
            "qwen36_service.batch_decode_adapter",
            fromlist=["__file__"],
        ).__file__
    ).read_text()
    require(
        "prefill_paged_slots" in adapter_source,
        "batch adapter does not use per-slot paged prefill",
    )
    require(
        "sharded_lm_head=True" in adapter_source,
        "batch adapter does not preserve shard-local token reduction",
    )
    require(
        "_clone_gdn_state" in adapter_source and "_restore_gdn_state" in adapter_source,
        "batch trace capture does not preserve post-prefill GDN state",
    )
    require(
        "plan_same_width_trace_prefill" in adapter_source,
        "batch adapter bypasses the pure trace-reuse policy",
    )
    server_source = Path(
        __import__("qwen36_service.server", fromlist=["__file__"]).__file__
    ).read_text()
    require(
        "ThreadingHTTPServer" in server_source,
        "service shell cannot accept concurrent requests",
    )
    batch_service_source = Path(
        __import__("qwen36_service.batch_service", fromlist=["__file__"]).__file__
    ).read_text()
    require(
        "plan_decode_batch" in batch_service_source,
        "batch service does not call the pure batch planner",
    )
    require(
        "BatchGenerationCoordinator" in server_source,
        "service composition root does not select the batch coordinator",
    )


# r[verify tenstorrent.qwen38_p150x2.batch_performance_admission]
# r[verify tenstorrent.qwen38_p150x2.batch_ttft_evidence]
# r[verify tenstorrent.qwen38_p150x2.batch_b8_admission]
def probe_batch_admission():
    """Positive and negative tests for batch admission policy and phases."""
    from qwen36_service.batch_admission_core import (
        ADMISSION_MEASUREMENT_MODE,
        CANDIDATE_WIDTH,
        PRODUCTION_MEASUREMENT_MODE,
        BatchAdmissionPolicyError,
        BatchAdmissionProfile,
    )
    from qwen36_service.batch_decode_adapter import BatchPrefillResult

    test_token_id = 7
    test_total_prefill_seconds = 0.4
    test_first_token_seconds = 0.3
    test_page_table_blocks = 8
    test_state_install_seconds = 0.1
    profile_path = Path(__file__).with_name("batch-admission-profile.json")
    with profile_path.open(encoding="utf-8") as profile_file:
        profile_mapping = json.load(profile_file)
    profile = BatchAdmissionProfile.from_mapping(profile_mapping)
    require(
        profile.candidate_width == CANDIDATE_WIDTH,
        "batch profile selected the wrong candidate width",
    )
    production_mode, production_eligible = profile.measurement(
        enabled=False,
        width=profile.candidate_width,
    )
    require(
        production_mode == PRODUCTION_MEASUREMENT_MODE and not production_eligible,
        "normal production became admission-eligible",
    )
    admission_mode, admission_eligible = profile.measurement(
        enabled=True,
        width=profile.candidate_width,
    )
    require(
        admission_mode == ADMISSION_MEASUREMENT_MODE and admission_eligible,
        "explicit batch admission did not become eligible",
    )
    for invalid_width in (0, profile.candidate_width + 1, True):
        try:
            profile.measurement(enabled=True, width=invalid_width)
        except BatchAdmissionPolicyError as error:
            require("width" in str(error), f"wrong admission-width diagnostic: {error}")
        else:
            require(False, f"invalid admission width was accepted: {invalid_width!r}")

    invalid_profile = json.loads(json.dumps(profile_mapping))
    invalid_profile["unexpected"] = True
    try:
        BatchAdmissionProfile.from_mapping(invalid_profile)
    except BatchAdmissionPolicyError as error:
        require("fields differ" in str(error), "wrong unknown-field diagnostic")
    else:
        require(False, "batch profile accepted an unknown field")

    invalid_admission = json.loads(json.dumps(profile_mapping))
    invalid_admission["physically_admitted"] = True
    invalid_admission["identities"]["receipt_blake3"] = "not-measured"
    try:
        BatchAdmissionProfile.from_mapping(invalid_admission)
    except BatchAdmissionPolicyError as error:
        require("no measured receipt" in str(error), "wrong receipt diagnostic")
    else:
        require(False, "unmeasured batch profile became admitted")

    prefill_result = BatchPrefillResult(
        first_token_ids=(test_token_id,),
        elapsed_seconds=test_total_prefill_seconds,
        first_token_ready_seconds=test_first_token_seconds,
        page_table_blocks=test_page_table_blocks,
        state_install_seconds=test_state_install_seconds,
    )
    require(
        math.isclose(
            prefill_result.first_token_ready_seconds
            + prefill_result.state_install_seconds,
            prefill_result.elapsed_seconds,
        ),
        "batch prefill phase split is inconsistent",
    )
    model_source = Path(
        __import__(
            "models.demos.blackhole.qwen36.tt.model",
            fromlist=["__file__"],
        ).__file__
    ).read_text()
    require(
        "first_token_callback(host_logits)" in model_source,
        "model does not expose pre-slot-write first-token evidence",
    )

    # r[verify tenstorrent.qwen38_p150x2.prefill_device_staging]
    require(
        "_rope_decode_tables" in model_source and "tables[0][rows]" in model_source,
        "decode RoPE does not use the cached per-position tables",
    )
    require(
        "device_direct = (" in model_source
        and "self._chunked_trace_id is None" in model_source
        and '"_bucket_trace_id", None' in model_source,
        "device-direct prefill staging lacks the no-parked-trace gate",
    )
    require(
        "_write_gdn_slot_device" in model_source
        and "dn.write_slot(slot, rec_clones[li], conv_clones[li])" in model_source,
        "device-resident slot writes bypass write_slot ownership semantics",
    )
    batch_service_source = Path(
        __import__("qwen36_service.batch_service", fromlist=["__file__"]).__file__
    ).read_text()
    first_token_index = batch_service_source.index("first_token_ready =")
    trace_prepare_index = batch_service_source.index("trace_prepare_started =")
    ttft_index = batch_service_source.index(
        "ttft_seconds = first_token_ready - prefill_started"
    )
    require(
        first_token_index < trace_prepare_index < ttft_index,
        "batch TTFT does not stay separate from trace preparation",
    )

    # r[verify tenstorrent.qwen38_p150x2.batch_b8_admission]
    b8_profile_path = Path(__file__).with_name("batch-admission-b8-profile.json")
    with b8_profile_path.open(encoding="utf-8") as b8_profile_file:
        b8_profile_mapping = json.load(b8_profile_file)
    b8_profile = BatchAdmissionProfile.from_mapping(b8_profile_mapping)
    require(b8_profile.candidate_width == 8, "B=8 profile width changed")
    require(
        len(b8_profile.prompts) == 8,
        "B=8 profile does not carry eight prompts",
    )
    require(
        b8_profile.minimum_aggregate_decode_speedup_ratio == 6.0
        and b8_profile.maximum_ttft_ratio == 9.0
        and b8_profile.minimum_per_user_decode_retention_ratio == 0.75,
        "B=8 profile bounds drifted from the predeclared policy",
    )
    b8_admission_mode, b8_admission_eligible = b8_profile.measurement(
        enabled=True,
        width=8,
    )
    require(
        b8_admission_mode == ADMISSION_MEASUREMENT_MODE and b8_admission_eligible,
        "explicit B=8 measurement did not become eligible",
    )
    b8_single_mode, b8_single_eligible = b8_profile.measurement(
        enabled=True,
        width=1,
    )
    require(
        b8_single_mode == ADMISSION_MEASUREMENT_MODE and b8_single_eligible,
        "B=8 single-stream width lost eligibility",
    )
    for invalid_b8_width in (5, 9, 16):
        try:
            b8_profile.measurement(enabled=True, width=invalid_b8_width)
        except BatchAdmissionPolicyError as error:
            require(
                "outside the profile" in str(error),
                f"wrong B=8 width diagnostic: {error}",
            )
        else:
            require(
                False,
                f"invalid B=8 width was accepted: {invalid_b8_width!r}",
            )

    invalid_b8_mappings = []
    b8_unsupported_width = json.loads(json.dumps(b8_profile_mapping))
    b8_unsupported_width["workload"]["candidate_width"] = 5
    invalid_b8_mappings.append((b8_unsupported_width, "workload widths"))
    b8_admitted_without_receipt = json.loads(json.dumps(b8_profile_mapping))
    b8_admitted_without_receipt["physically_admitted"] = True
    b8_admitted_without_receipt["identities"]["receipt_blake3"] = "not-measured"
    invalid_b8_mappings.append((b8_admitted_without_receipt, "no measured receipt"))
    for invalid_b8_mapping, diagnostic in invalid_b8_mappings:
        try:
            BatchAdmissionProfile.from_mapping(invalid_b8_mapping)
        except BatchAdmissionPolicyError as error:
            require(
                diagnostic in str(error),
                f"wrong B=8 profile diagnostic: {error}",
            )
        else:
            require(
                False,
                f"invalid B=8 profile was accepted: {diagnostic}",
            )

    b4_still_valid = BatchAdmissionProfile.from_mapping(profile_mapping)
    require(
        b4_still_valid.candidate_width == CANDIDATE_WIDTH,
        "B=4 profile stopped loading after B=8 parameterization",
    )


# r[verify tenstorrent.qwen38_p150x2.same_width_trace_reuse]
def probe_tensor_binding_identity():
    """Keep exact identities and failure behavior without temporary shard views."""
    from types import SimpleNamespace

    from qwen36_service import batch_decode_adapter as adapter_module

    first_address = 4096
    second_address = 8192
    shard_count = 2
    shard_reads = []

    class FakeMeshTensor:
        def __init__(self):
            self.address = first_address
            self.coords = tuple(range(shard_count))
            self.fail_address = False
            self.fail_coords = False

        def buffer_address(self):
            if self.fail_address:
                raise RuntimeError("no shared address")
            return self.address

        def device_coords(self):
            if self.fail_coords:
                raise AttributeError("legacy API")
            return self.coords

    def shards(value):
        shard_reads.append(value)
        return (
            SimpleNamespace(buffer_address=lambda: first_address),
            SimpleNamespace(buffer_address=lambda: second_address),
        )

    original_ttnn = adapter_module.ttnn
    adapter_module.ttnn = SimpleNamespace(
        Tensor=FakeMeshTensor, get_device_tensors=shards
    )
    try:
        tensor = FakeMeshTensor()
        initial = adapter_module._tensor_identity(tensor)
        require(
            initial == f"FakeMeshTensor:device:[{first_address},{first_address}]",
            "shared identity changed",
        )
        require(not shard_reads, "shared identity constructed shard views")
        tensor.address = second_address
        require(
            adapter_module._tensor_identity(tensor) != initial,
            "address mutation escaped validation",
        )
        tensor.address = first_address
        tensor.coords = (0,)
        require(
            adapter_module._tensor_identity(tensor) != initial,
            "shard-count mutation escaped validation",
        )
        tensor.coords = ()
        require(
            adapter_module._tensor_identity(tensor) == "FakeMeshTensor:device:[]",
            "empty shard identity changed",
        )
        fallback = f"FakeMeshTensor:device:[{first_address},{second_address}]"
        tensor.fail_address = True
        require(
            adapter_module._tensor_identity(tensor) == fallback,
            "address failure lost shard identities",
        )
        tensor.fail_address = False
        tensor.fail_coords = True
        require(
            adapter_module._tensor_identity(tensor) == fallback,
            "legacy API lost shard identities",
        )

        class CustomMeshTensor(FakeMeshTensor):
            pass

        custom = CustomMeshTensor()
        require(
            adapter_module._tensor_identity(custom)
            == f"CustomMeshTensor:device:[{first_address},{second_address}]",
            "custom tensor type bypassed shard validation",
        )
        host = object()
        adapter_module.ttnn.get_device_tensors = lambda value: (value,)
        require(
            adapter_module._tensor_identity(host) == f"object:host:{id(host)}",
            "host fallback changed",
        )
        adapter_module.ttnn.get_device_tensors = None
        require(
            adapter_module._tensor_identity(host) == f"object:host:{id(host)}",
            "unavailable shard API lost host fallback",
        )
    finally:
        adapter_module.ttnn = original_ttnn


def probe_trace_reuse_profile():
    """Positive and negative tests for closed trace-reuse activation policy."""
    from qwen36_service.trace_reuse_core import (
        BLAKE3_LENGTH,
        MAXIMUM_RESIDENT_TRACE_COUNT,
        PRODUCTION_MEASUREMENT_MODE,
        TRACE_REUSE_MEASUREMENT_MODE,
        TRACE_REUSE_WIDTH,
        TraceReusePolicyError,
        TraceReuseProfile,
    )

    profile_path = Path(__file__).with_name("trace-reuse-profile.json")
    with profile_path.open(encoding="utf-8") as profile_file:
        profile_mapping = json.load(profile_file)
    profile = TraceReuseProfile.from_mapping(profile_mapping)
    require(profile.width == TRACE_REUSE_WIDTH, "trace-reuse profile width changed")
    require(
        profile.maximum_resident_trace_count == MAXIMUM_RESIDENT_TRACE_COUNT,
        "trace-reuse profile permits multiple resident traces",
    )
    production = profile.activation(
        explicit_measurement=False,
        width=profile.width,
    )
    require(not production.enabled, "unadmitted production enabled trace reuse")
    require(
        not production.admission_eligible,
        "normal production became trace-reuse admission evidence",
    )
    require(
        production.measurement_mode == PRODUCTION_MEASUREMENT_MODE,
        "normal trace-reuse measurement mode changed",
    )
    explicit = profile.activation(
        explicit_measurement=True,
        width=profile.width,
    )
    require(explicit.enabled, "explicit trace-reuse measurement stayed disabled")
    require(
        explicit.admission_eligible,
        "explicit trace-reuse measurement is not eligible",
    )
    require(
        explicit.measurement_mode == TRACE_REUSE_MEASUREMENT_MODE,
        "explicit trace-reuse measurement mode changed",
    )

    admitted_mapping = json.loads(json.dumps(profile_mapping))
    admitted_mapping["physically_admitted"] = True
    admitted_mapping["identities"]["receipt_blake3"] = "a" * BLAKE3_LENGTH
    admitted_profile = TraceReuseProfile.from_mapping(admitted_mapping)
    admitted_production = admitted_profile.activation(
        explicit_measurement=False,
        width=admitted_profile.width,
    )
    require(admitted_production.enabled, "admitted production did not enable reuse")
    require(
        not admitted_production.admission_eligible,
        "admitted production became operator evidence",
    )

    invalid_profiles = []
    unknown_profile = json.loads(json.dumps(profile_mapping))
    unknown_profile["unexpected"] = True
    invalid_profiles.append((unknown_profile, "fields differ"))
    missing_receipt = json.loads(json.dumps(profile_mapping))
    missing_receipt["physically_admitted"] = True
    invalid_profiles.append((missing_receipt, "no measured receipt"))
    multiple_traces = json.loads(json.dumps(profile_mapping))
    invalid_resident_trace_count = MAXIMUM_RESIDENT_TRACE_COUNT + 1
    multiple_traces["admission"]["maximum_resident_trace_count"] = (
        invalid_resident_trace_count
    )
    invalid_profiles.append((multiple_traces, "must be one"))
    wrong_order = json.loads(json.dumps(profile_mapping))
    wrong_order["workload"]["execution_order"] = [
        "reuse",
        "recapture",
        "reuse",
        "recapture",
    ]
    invalid_profiles.append((wrong_order, "execution order"))
    for invalid_profile, diagnostic in invalid_profiles:
        try:
            TraceReuseProfile.from_mapping(invalid_profile)
        except TraceReusePolicyError as error:
            require(
                diagnostic in str(error), f"wrong trace profile diagnostic: {error}"
            )
        else:
            require(False, f"invalid trace profile was accepted: {diagnostic}")

    outside_profile_width = profile.width - 1
    server_source = Path(
        __import__("qwen36_service.server", fromlist=["__file__"]).__file__
    ).read_text()
    require(
        "--trace-reuse-profile" in server_source,
        "service does not load the trace-reuse profile",
    )
    require(
        "--trace-reuse-admission" in server_source,
        "service has no explicit trace-reuse evidence mode",
    )

    invalid_activations = (
        (
            {"explicit_measurement": True, "width": outside_profile_width},
            "outside the profile",
        ),
        ({"explicit_measurement": 1, "width": profile.width}, "must be a boolean"),
        ({"explicit_measurement": True, "width": True}, "must be an integer"),
    )
    for activation_input, diagnostic in invalid_activations:
        try:
            profile.activation(**activation_input)
        except TraceReusePolicyError as error:
            require(diagnostic in str(error), f"wrong activation diagnostic: {error}")
        else:
            require(
                False, f"invalid trace activation was accepted: {activation_input!r}"
            )


# r[verify tenstorrent.qwen38_p150x2.startup_readiness]
def probe_startup_readiness():
    """Positive and negative tests for readiness policy and shell adapters."""
    import torch
    from qwen36_service.batch_decode_adapter import (
        BatchDecodeAdapterError,
        TracePreparationResult,
    )
    from qwen36_service.batch_service import BatchGenerationCoordinator
    from qwen36_service.core import CompletionRequest, Endpoint
    from qwen36_service.readiness_core import (
        StartupPhase,
        StartupReadinessError,
        begin_warmup,
        complete_warmup,
        fail_warmup,
        initial_readiness,
        plan_startup_warmup,
        record_warm_width,
    )
    from qwen36_service.systemd_notify_adapter import (
        SystemdNotifyError,
        notify_ready,
    )

    warmup_widths = (1, 4)
    maximum_width = 16
    warmup_prompt = "Warm the test path."
    warmup_generation_tokens = 2
    expected_resident_trace_count = 1
    warmup_token_id = 7
    test_vocabulary_size = 128
    test_maximum_sequence_length = 2048
    test_physical_device_ids = (0, 1)
    first_width_seconds = 0.1
    second_width_seconds = 0.2
    total_seconds = 0.4
    plan = plan_startup_warmup(
        widths=warmup_widths,
        prompt=warmup_prompt,
        generation_tokens=warmup_generation_tokens,
        maximum_width=maximum_width,
    )
    state = begin_warmup(initial_readiness(plan))
    state = record_warm_width(
        state,
        width=warmup_widths[0],
        elapsed_seconds=first_width_seconds,
    )
    state = record_warm_width(
        state,
        width=warmup_widths[1],
        elapsed_seconds=second_width_seconds,
    )
    state = complete_warmup(state, total_elapsed_seconds=total_seconds)
    require(state.ready, "completed startup warm-up did not become ready")
    require(
        state.phase is StartupPhase.READY,
        "completed startup warm-up has the wrong phase",
    )
    require(
        state.completed_widths == warmup_widths,
        "startup readiness lost a completed width",
    )

    invalid_plan_cases = (
        ({"widths": ()}, "must not be empty"),
        ({"widths": (1, 1)}, "unique and increasing"),
        ({"widths": (4, 1)}, "unique and increasing"),
        ({"widths": (1, maximum_width + 1)}, "outside capacity"),
        ({"prompt": ""}, "prompt"),
        ({"generation_tokens": 1}, "at least two"),
        ({"generation_tokens": True}, "at least two"),
    )
    base_plan_inputs = {
        "widths": warmup_widths,
        "prompt": warmup_prompt,
        "generation_tokens": warmup_generation_tokens,
        "maximum_width": maximum_width,
    }
    for override, diagnostic in invalid_plan_cases:
        try:
            plan_startup_warmup(**{**base_plan_inputs, **override})
        except StartupReadinessError as error:
            require(diagnostic in str(error), f"wrong warm-up plan diagnostic: {error}")
        else:
            require(False, f"invalid warm-up plan was accepted: {override!r}")

    incomplete = begin_warmup(initial_readiness(plan))
    try:
        complete_warmup(incomplete, total_elapsed_seconds=total_seconds)
    except StartupReadinessError as error:
        require("incomplete" in str(error), "wrong incomplete warm-up diagnostic")
    else:
        require(False, "incomplete warm-up became ready")
    try:
        record_warm_width(
            incomplete,
            width=warmup_widths[1],
            elapsed_seconds=first_width_seconds,
        )
    except StartupReadinessError as error:
        require("out of order" in str(error), "wrong width-order diagnostic")
    else:
        require(False, "out-of-order warm-up width was accepted")
    failed = fail_warmup(incomplete)
    require(failed.phase is StartupPhase.FAILED, "warm-up failure was not retained")

    class FakeWarmupAdapter:
        fusion_mode = "off"

        def __init__(self):
            self.prefill_widths = []
            self.prepared_widths = []
            self.replay_widths = []
            self.reset_count = 0

        def prefill_users(
            self,
            token_ids,
            *,
            generation_tokens,
            trace_reuse_enabled=False,
        ):
            width = len(token_ids)
            self.prefill_widths.append(width)
            require(
                generation_tokens == warmup_generation_tokens,
                "warm-up shell changed the generation bound",
            )
            require(
                trace_reuse_enabled is False,
                "startup warm-up enabled trace reuse",
            )
            return SimpleNamespace(
                first_token_ids=tuple(warmup_token_id for _ in range(width)),
                elapsed_seconds=first_width_seconds,
                page_table_blocks=1,
            )

        def prepare_width(self, *, token_ids, positions):
            require(len(token_ids) == len(positions), "warm-up trace widths differ")
            self.prepared_widths.append(len(token_ids))
            return TracePreparationResult(
                capture_seconds=first_width_seconds,
                capture_count=1,
                reused=False,
                resident_trace_count=expected_resident_trace_count,
                binding_stable=True,
                transition_reason=None,
            )

        def decode_traced(self, *, token_ids, positions):
            require(len(token_ids) == len(positions), "warm-up replay widths differ")
            self.replay_widths.append(len(token_ids))
            return token_ids

        def reset_all(self):
            self.reset_count += 1

    emitted = []
    fake_adapter = FakeWarmupAdapter()
    coordinator = BatchGenerationCoordinator(
        adapter=fake_adapter,
        tokenizer=object(),
        model=SimpleNamespace(args=SimpleNamespace(vocab_size=test_vocabulary_size)),
        profile=SimpleNamespace(profile_id="profile-test"),
        model_alias="model-test",
        model_identity="model-identity-test",
        package_identity="package-test",
        runtime_identity="runtime-test",
        maximum_sequence_length=test_maximum_sequence_length,
        physical_pcie_device_ids=test_physical_device_ids,
        emitter=lambda stage, **fields: emitted.append((stage, fields)),
    )
    test_request = CompletionRequest(
        endpoint=Endpoint.COMPLETIONS,
        model="model-test",
        max_tokens=warmup_generation_tokens,
        prompt=warmup_prompt,
        messages=(),
    )
    try:
        try:
            coordinator.generate(
                request=test_request,
                request_id="request-before-ready",
                token_ids=torch.tensor([[warmup_token_id]], dtype=torch.int32),
                prompt_token_count=1,
                started=0.0,
                deadline=1.0,
            )
        except BatchDecodeAdapterError as error:
            require("not ready" in str(error), "wrong pre-ready request diagnostic")
        else:
            require(False, "batch worker accepted a request before warm-up")

        observations = coordinator.warm_up(
            plan,
            torch.tensor([[warmup_token_id]], dtype=torch.int32),
        )
        require(
            tuple(item.width for item in observations) == warmup_widths,
            "warm-up shell completed the wrong widths",
        )
        require(
            tuple(fake_adapter.prefill_widths) == warmup_widths,
            "warm-up shell prefills differ from the plan",
        )
        require(
            tuple(fake_adapter.prepared_widths) == warmup_widths,
            "warm-up shell trace captures differ from the plan",
        )
        require(
            tuple(fake_adapter.replay_widths) == warmup_widths,
            "warm-up shell replays differ from the plan",
        )
        require(
            all(not observation.trace_reused for observation in observations),
            "startup warm-up reused a trace",
        )
        require(
            all(
                observation.resident_trace_count == expected_resident_trace_count
                for observation in observations
            ),
            "startup warm-up retained the wrong trace count",
        )
        require(
            fake_adapter.reset_count == len(warmup_widths),
            "warm-up shell did not clean every width",
        )
        try:
            coordinator.warm_up(
                plan,
                torch.tensor([[warmup_token_id]], dtype=torch.int32),
            )
        except BatchDecodeAdapterError as error:
            require("already ready" in str(error), "wrong repeated warm-up diagnostic")
        else:
            require(False, "batch worker accepted repeated warm-up")
    finally:
        coordinator.close()

    class FailingWarmupAdapter(FakeWarmupAdapter):
        def prepare_width(self, *, token_ids, positions):
            raise RuntimeError("negative trace fixture")

    failing_adapter = FailingWarmupAdapter()
    failing_coordinator = BatchGenerationCoordinator(
        adapter=failing_adapter,
        tokenizer=object(),
        model=SimpleNamespace(args=SimpleNamespace(vocab_size=test_vocabulary_size)),
        profile=SimpleNamespace(profile_id="profile-test"),
        model_alias="model-test",
        model_identity="model-identity-test",
        package_identity="package-test",
        runtime_identity="runtime-test",
        maximum_sequence_length=test_maximum_sequence_length,
        physical_pcie_device_ids=test_physical_device_ids,
        emitter=lambda _stage, **_fields: None,
    )
    try:
        try:
            failing_coordinator.warm_up(
                plan,
                torch.tensor([[warmup_token_id]], dtype=torch.int32),
            )
        except BatchDecodeAdapterError as error:
            require("warm-up failed" in str(error), "wrong shell failure diagnostic")
        else:
            require(False, "failed warm-up was accepted")
        require(
            failing_adapter.reset_count == 1,
            "failed warm-up did not attempt a clean reset",
        )
    finally:
        failing_coordinator.close()

    class FakeNotifySocket:
        def __init__(self, *, short_write=False):
            self.short_write = short_write
            self.address = None
            self.payload = None
            self.closed = False

        def connect(self, address):
            self.address = address

        def send(self, payload):
            self.payload = payload
            return len(payload) - 1 if self.short_write else len(payload)

        def close(self):
            self.closed = True

    fake_sockets = []

    def fake_socket_factory(_family, _kind):
        result = FakeNotifySocket()
        fake_sockets.append(result)
        return result

    require(
        notify_ready(
            "ready-test",
            required=True,
            environment={"NOTIFY_SOCKET": "@qwen-test"},
            socket_factory=fake_socket_factory,
        ),
        "systemd adapter did not report a successful notification",
    )
    require(
        fake_sockets[0].address == "\0qwen-test",
        "systemd adapter did not map the abstract socket address",
    )
    require(
        fake_sockets[0].payload == b"READY=1\nSTATUS=ready-test",
        "systemd adapter sent the wrong readiness payload",
    )
    require(fake_sockets[0].closed, "systemd adapter did not close its socket")
    require(
        not notify_ready("ready-test", required=False, environment={}),
        "optional missing systemd socket reported success",
    )
    try:
        notify_ready("ready-test", required=True, environment={})
    except SystemdNotifyError as error:
        require("absent" in str(error), "wrong missing-notify diagnostic")
    else:
        require(False, "required missing systemd socket was accepted")

    short_socket = FakeNotifySocket(short_write=True)
    try:
        notify_ready(
            "ready-test",
            required=True,
            environment={"NOTIFY_SOCKET": "/run/qwen-test"},
            socket_factory=lambda _family, _kind: short_socket,
        )
    except SystemdNotifyError as error:
        require("incomplete" in str(error), "wrong short-notify diagnostic")
    else:
        require(False, "short systemd notification was accepted")

    server_source = Path(
        __import__("qwen36_service.server", fromlist=["__file__"]).__file__
    ).read_text()
    require(
        server_source.index("engine.warm_up()")
        < server_source.index("server = ThreadingHTTPServer"),
        "HTTP server binds before startup warm-up",
    )
    require(
        'self.path in ("/healthz", "/readyz")' in server_source,
        "service does not expose both health and readiness endpoints",
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

    from qwen36_service.mtp_drafter_core import (
        EXPECTED_MTP_SCALE_WEIGHT_SHAPES,
        EXPECTED_MTP_SCALE_WEIGHT_SHAPES_UNION,
    )

    fp8_observed = {
        **EXPECTED_MTP_WEIGHT_SHAPES,
        **EXPECTED_MTP_SCALE_WEIGHT_SHAPES,
    }
    validate_mtp_weight_shapes(fp8_observed)
    require(
        set(EXPECTED_MTP_SCALE_WEIGHT_SHAPES_UNION) == set(fp8_observed),
        "fp8 scale union diverged from its parts",
    )
    for name, shape in EXPECTED_MTP_SCALE_WEIGHT_SHAPES.items():
        base = name.removesuffix(".weight_scale_inv") + ".weight"
        base_shape = EXPECTED_MTP_WEIGHT_SHAPES[base]
        require(
            shape == (base_shape[0] // 128, base_shape[1] // 128),
            f"MTP scale grid {name} left the 128 block bound",
        )

    orphan_scale = {
        **fp8_observed,
        "mtp.layers.0.extra.weight_scale_inv": (1, 1),
    }
    try:
        validate_mtp_weight_shapes(orphan_scale)
    except MtpDrafterPolicyError as error:
        require("extra=" in str(error), "wrong orphan-scale diagnostic")
    else:
        require(False, "orphan MTP scale key was accepted")

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

    below_range_profile = {**profile_source, "draft_token_count": 1}
    try:
        MtpDrafterProfile.from_mapping(below_range_profile)
    except MtpDrafterPolicyError as error:
        require("between 2 and 6" in str(error), "wrong below-range draft diagnostic")
    else:
        require(False, "below-range MTP drafter profile was accepted")

    above_range_profile = {**profile_source, "draft_token_count": 7}
    try:
        MtpDrafterProfile.from_mapping(above_range_profile)
    except MtpDrafterPolicyError as error:
        require("between 2 and 6" in str(error), "wrong above-range draft diagnostic")
    else:
        require(False, "above-range MTP drafter profile was accepted")

    mismatched_counts = ([1, 2], [1, 2, 3])
    try:
        accepted_prefix_length(*mismatched_counts)
    except MtpDrafterPolicyError as error:
        require("differs from draft count" in str(error), "wrong count diagnostic")
    else:
        require(False, "mismatched draft/target counts were accepted")

    for rejected_width in (1, 7):
        try:
            accepted_prefix_length(
                list(range(rejected_width)), list(range(rejected_width))
            )
        except MtpDrafterPolicyError as error:
            require(
                "outside the policy range" in str(error),
                "wrong prefix-length range diagnostic",
            )
        else:
            require(False, f"prefix width {rejected_width} was accepted")

    wide_draft_tokens = tuple(range(5))
    require(
        accepted_prefix_length(wide_draft_tokens, wide_draft_tokens) == 5,
        "wide matching drafts lost acceptance",
    )

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
    from qwen36_service.mtp_rollback_core import (
        EXPECTED_ROLLBACK_DRAFT_COUNT,
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
        range(token_start, token_start + EXPECTED_ROLLBACK_DRAFT_COUNT)
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
    draft_mass, accepted = acceptance_probability(RULE_EXACT, probabilities, ids, 8)
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
            lambda: acceptance_probability(
                RULE_LENIENT, probabilities, ids, 7, lenience=0.0
            ),
            "lenience must be finite and positive",
        ),
        (
            lambda: resolved_round_exhausted_uniforms(),
            "uniform stream exhausted",
        ),
        (
            lambda: resolve_speculative_round(
                row_values[:-1], row_ids, drafts, iter([])
            ),
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


# r[verify tenstorrent.qwen38_p150x2.mtp_trace_capture_hygiene]
def probe_mtp_trace_capture_hygiene():
    """Positive and negative tests for trace-capture provenance gates."""
    from dataclasses import replace

    from qwen36_service.mtp_trace_hygiene_core import (
        REASON_BUFFER_ADDRESS,
        REASON_PRIMING,
        REASON_RESIDENT_TRACE,
        MtpTraceHygienePolicyError,
        TraceCaptureProvenance,
        validate_trace_capture_provenance,
    )

    clean = TraceCaptureProvenance(
        resident_traces_before=0,
        warm_execution_completed=True,
        capture_from_zeroed_priming=False,
        trace_bound_addresses_stable=True,
    )
    require(
        validate_trace_capture_provenance(clean) == (),
        "clean capture provenance was rejected",
    )

    violation_cases = (
        (
            replace(clean, resident_traces_before=1),
            REASON_RESIDENT_TRACE,
            "resident second trace",
        ),
        (
            replace(clean, warm_execution_completed=False),
            REASON_PRIMING,
            "missing warm execution",
        ),
        (
            replace(clean, capture_from_zeroed_priming=True),
            REASON_PRIMING,
            "zeroed priming",
        ),
        (
            replace(clean, trace_bound_addresses_stable=False),
            REASON_BUFFER_ADDRESS,
            "unstable trace-bound addresses",
        ),
    )
    for provenance, expected_reason, label in violation_cases:
        reasons = validate_trace_capture_provenance(provenance)
        require(
            reasons == (expected_reason,),
            f"{label} produced wrong hygiene reasons: {reasons}",
        )

    malformed_cases = (
        (
            replace(clean, resident_traces_before=-1),
            "non-negative integer",
        ),
        (
            replace(clean, resident_traces_before=True),
            "non-negative integer",
        ),
        (
            replace(clean, warm_execution_completed=1),
            "must be boolean",
        ),
        (
            replace(clean, capture_from_zeroed_priming="no"),
            "must be boolean",
        ),
        (
            replace(clean, trace_bound_addresses_stable=None),
            "must be boolean",
        ),
    )
    for provenance, diagnostic in malformed_cases:
        try:
            validate_trace_capture_provenance(provenance)
        except MtpTraceHygienePolicyError as error:
            require(diagnostic in str(error), f"wrong hygiene diagnostic: {error}")
        else:
            require(False, f"malformed provenance was accepted: {provenance!r}")

    try:
        validate_trace_capture_provenance({})
    except MtpTraceHygienePolicyError as error:
        require("TraceCaptureProvenance" in str(error), "wrong type diagnostic")
    else:
        require(False, "non-provenance input was accepted")

    # Wiring: the native verifier capture path must run the gate and must
    # re-check trace-bound identities before replay.
    adapter_source = Path(
        __import__(
            "qwen36_service.native_mtp_verifier_adapter",
            fromlist=["__file__"],
        ).__file__
    ).read_text()
    require(
        "validate_trace_capture_provenance" in adapter_source,
        "native verifier lost its capture-hygiene gate",
    )
    require(
        "capture_input_identities" in adapter_source
        and "trace-bound input addresses drifted" in adapter_source,
        "native verifier lost its pre-replay address stability check",
    )
    require(
        "_hygiene_capture_resident = False" in adapter_source,
        "native verifier never clears its resident-capture flag",
    )


# r[verify tenstorrent.qwen38_p150x2.gdn_decode_projection_layout]
def probe_gdn_decode_layout_candidate():
    """Positive and negative tests for the gated DRAM-sharded decode candidate.

    The candidate must stay default-off, prefill must stay untouched, and
    shape eligibility must live in the pure device-free module. The gated
    branch must precede the production 1D-mcast branch in gdn/tp.py or the
    gate could never engage.
    """
    import pathlib

    qwen36 = pathlib.Path(
        __import__("models.demos.blackhole.qwen36", fromlist=["__path__"]).__path__[0]
    )
    from models.demos.blackhole.qwen36.tt.gdn import (
        decode_layout_candidate as candidate,
    )

    real_shape = dict(
        activation_width=5120,
        projection_width=8240,
        decode_width=1,
        grid_width=11,
    )
    require(
        candidate.select_decode_layout(enabled=False, **real_shape) is None,
        "candidate gate off changed the production path",
    )
    require(
        candidate.select_decode_layout(enabled=candidate.gate_enabled({}), **real_shape)
        is None,
        "absent gate left the candidate enabled",
    )
    require(
        candidate.select_decode_layout(
            enabled=candidate.gate_enabled({"QWEN35_GDN_DECODE_LAYOUT": "0"}),
            **real_shape,
        )
        is None,
        "gate '0' left the candidate enabled",
    )
    gated = candidate.select_decode_layout(
        enabled=candidate.gate_enabled({"QWEN35_GDN_DECODE_LAYOUT": "1"}),
        **real_shape,
    )
    require(
        gated == candidate.LAYOUT_DRAM_SHARDED,
        "eligible decode shape did not select dram_sharded under the gate",
    )
    require(
        candidate.select_decode_layout(
            enabled=True,
            activation_width=5121,
            projection_width=8240,
            decode_width=1,
            grid_width=11,
        )
        is None,
        "non-tile-aligned activation stayed on the candidate path",
    )
    require(
        candidate.select_decode_layout(
            enabled=True,
            activation_width=5120,
            projection_width=8240,
            decode_width=2,
            grid_width=11,
        )
        is None,
        "two-token decode stayed on the candidate path",
    )

    candidate_source = (
        qwen36 / "tt" / "gdn" / "decode_layout_candidate.py"
    ).read_text()
    require(
        "import ttnn" not in candidate_source,
        "candidate core imports ttnn (must be device-free pure policy)",
    )
    require(
        "QWEN35_GDN_DECODE_LAYOUT" in candidate_source,
        "candidate core lost the default-off gate name",
    )

    tp_source = (qwen36 / "tt" / "gdn" / "tp.py").read_text()
    require(
        "select_decode_layout" in tp_source and "gate_enabled" in tp_source,
        "gdn tp.py does not consume the candidate policy",
    )
    require(
        "gdn_qkvzab_progcfg" in tp_source,
        "gdn tp.py lost the DRAM-sharded qkvzab program config route",
    )
    candidate_index = tp_source.index("select_decode_layout")
    one_d_index = tp_source.index("gdn_qkvz_decode_1d_progcfg")
    require(
        candidate_index < one_d_index,
        "candidate branch must precede the default 1D-mcast branch",
    )


# r[verify tenstorrent.qwen38_p150x2.gdn_decode_fusion]
def probe_gdn_decode_fusion_candidate():
    """Test the closed, default-off GDN decode fusion policy."""
    import pathlib

    qwen36 = pathlib.Path(
        __import__("models.demos.blackhole.qwen36", fromlist=["__path__"]).__path__[0]
    )
    from models.demos.blackhole.qwen36.tt.gdn import (
        decode_fusion_candidate as candidate,
    )

    require(
        candidate.selected_mode({}) == candidate.MODE_OFF,
        "missing fusion gate did not select off",
    )
    require(
        candidate.selected_mode({candidate.ENV_GATE: "unknown"}) == candidate.MODE_OFF,
        "unknown fusion mode did not fail closed",
    )
    require(
        candidate.selected_mode({candidate.ENV_GATE: object()}) == candidate.MODE_OFF,
        "non-string fusion mode did not fail closed",
    )
    require(
        candidate.selected_mode({candidate.ENV_GATE: " FULL "}) == candidate.MODE_FULL,
        "closed full mode was not normalized",
    )
    require(
        candidate.uses_fused_norm(candidate.MODE_NORM),
        "norm mode did not enable norm fusion",
    )
    require(
        not candidate.uses_fused_state_update(candidate.MODE_NORM),
        "norm mode enabled state fusion",
    )
    require(
        candidate.uses_fused_state_update(candidate.MODE_STATE),
        "state mode did not enable state fusion",
    )
    require(
        not candidate.uses_fused_norm(candidate.MODE_STATE),
        "state mode enabled norm fusion",
    )
    require(
        candidate.selected_mode({candidate.ENV_GATE: " NORM-STATE "})
        == candidate.MODE_NORM_STATE,
        "combined mode was not normalized",
    )
    require(
        candidate.uses_fused_norm(candidate.MODE_NORM_STATE)
        and candidate.uses_fused_state_update(candidate.MODE_NORM_STATE)
        and not candidate.uses_fused_recurrence(candidate.MODE_NORM_STATE),
        "combined mode must retain the composite recurrence",
    )
    require(
        candidate.selected_mode({candidate.ENV_GATE: "norm_state"})
        == candidate.MODE_OFF,
        "unsupported combined-mode spelling did not fail closed",
    )
    require(
        candidate.uses_fused_norm(candidate.MODE_FULL)
        and not candidate.uses_fused_state_update(candidate.MODE_FULL)
        and candidate.uses_fused_recurrence(candidate.MODE_FULL),
        "full mode did not select the custom recurrence",
    )
    require(
        not candidate.uses_fused_norm(candidate.MODE_OFF)
        and not candidate.uses_fused_state_update(candidate.MODE_OFF)
        and not candidate.uses_fused_recurrence(candidate.MODE_OFF),
        "off mode changed the production path",
    )

    policy_source = (qwen36 / "tt" / "gdn" / "decode_fusion_candidate.py").read_text()
    require("import ttnn" not in policy_source, "fusion policy imports ttnn")
    require(candidate.ENV_GATE in policy_source, "fusion policy lost its gate name")

    tp_source = (qwen36 / "tt" / "gdn" / "tp.py").read_text()
    delta_source = (
        qwen36.parents[2]
        / "experimental"
        / "gated_attention_gated_deltanet"
        / "tt"
        / "ttnn_delta_rule_ops.py"
    ).read_text()
    require(
        "decode_fusion_candidate.selected_mode" in tp_source,
        "GDN runtime does not consume fusion policy",
    )
    require(
        "q_norm_weight=" in tp_source and "k_norm_weight=" in tp_source,
        "GDN runtime lost norm fusion inputs",
    )
    require(
        "fuse_state_update=" in tp_source, "GDN runtime lost state fusion selection"
    )
    require(
        "fused_recurrence=" in tp_source, "GDN runtime lost fused-recurrence selection"
    )
    require("state_output=" in tp_source, "GDN runtime lost trace-stable state output")
    require(
        "ttnn.addcmul" in delta_source, "state-update fusion lost its ternary kernel"
    )
    require(
        "q_head_major = ttnn.permute" in delta_source
        and "beta_vector = ttnn.unsqueeze(beta, -1)" in delta_source
        and "output = ttnn.permute(output_head_major" in delta_source,
        "fused recurrence lost its physical head-major layout conversion",
    )
    require(
        "q_flat = ttnn.reshape(q," not in delta_source,
        "fused recurrence restored the unsafe metadata-only Q reshape",
    )
    import ttnn

    require(
        hasattr(ttnn.transformer, "gated_delta_decode"),
        "packaged TTNN runtime lost gated_delta_decode",
    )
    require(
        "q_norm_weight and k_norm_weight must be provided together" in delta_source,
        "norm fusion lost its paired-input failure",
    )

    import ast

    def fused_call_uses_hifi4(source):
        calls = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "gated_delta_decode"
        ]
        if len(calls) != 1:
            return False
        configs = [
            item.value
            for item in calls[0].keywords
            if item.arg == "compute_kernel_config"
        ]
        if len(configs) != 1 or not isinstance(configs[0], ast.Call):
            return False
        fidelities = [
            item.value for item in configs[0].keywords if item.arg == "math_fidelity"
        ]
        return (
            len(fidelities) == 1
            and isinstance(fidelities[0], ast.Attribute)
            and fidelities[0].attr == "HiFi4"
        )

    require(
        fused_call_uses_hifi4(delta_source), "fused recurrence lost its HiFi4 profile"
    )
    require(
        not fused_call_uses_hifi4(
            delta_source.replace("MathFidelity.HiFi4", "MathFidelity.HiFi2")
        ),
        "fused precision check admitted HiFi2",
    )
    require(
        not fused_call_uses_hifi4(
            delta_source.replace("compute_kernel_config=", "ignored_config=")
        ),
        "fused precision check admitted a missing config",
    )


# r[verify tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission]
def probe_kernel_distribution_admission():
    """Positive and negative tests for the sampled-mode distributional gate."""
    import math

    from qwen36_service.kernel_distribution_admission_core import (
        ADMITTED_SERVING_MODE,
        DistributionAdmissionEvidence,
        PositionSample,
        PromptProbe,
        evaluate_distribution_admission,
    )

    prompt_count = 5
    position_count = 48
    token_base = 100

    def softmax(values):
        peak = max(values)
        weights = [math.exp(value - peak) for value in values]
        total = math.fsum(weights)
        return [weight / total for weight in weights]

    def build_position(index, *, flip_stable=False, break_tv=False, break_shared=False):
        # Stable spacing of 1.0 keeps every non-near-tie margin above the
        # 0.5 epsilon; every tenth position squeezes to a 0.2 near tie.
        near_tie = index % 10 == 0
        gap = 0.2 if near_tie else 1.0
        control_logits = [5.0 - gap * rank for rank in range(20)]
        noise = 1e-3 if index % 2 == 0 else -1e-3
        candidate_logits = [
            value + noise * ((-1) ** rank) for rank, value in enumerate(control_logits)
        ]
        if near_tie and flip_stable:
            # Widen the near tie so the candidate keeps the control order.
            candidate_logits = list(control_logits)
        if not near_tie and flip_stable:
            candidate_logits[0], candidate_logits[1] = (
                candidate_logits[1],
                candidate_logits[0],
            )
        control_tokens = [token_base + offset for offset in range(20)]
        candidate_tokens = (
            [token_base + 40 + offset for offset in range(20)]
            if break_shared
            else control_tokens
        )
        control_probabilities = softmax(control_logits)
        candidate_probabilities = [0.05] * 20 if break_tv else softmax(candidate_logits)
        control = tuple(
            (token, probability, logit)
            for token, probability, logit in zip(
                control_tokens, control_probabilities, control_logits, strict=True
            )
        )
        candidate = tuple(
            (token, probability, logit)
            for token, probability, logit in zip(
                candidate_tokens, candidate_probabilities, candidate_logits, strict=True
            )
        )
        return PositionSample(control=control, candidate=candidate)

    def build_evidence(**kwargs):
        position_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in ("flip_stable", "break_tv", "break_shared")
        }
        prompt_total = kwargs.get("prompt_override", prompt_count)
        prompts = tuple(
            PromptProbe(
                positions=tuple(
                    build_position(
                        prompt_index * position_count + index, **position_kwargs
                    )
                    for index in range(position_count)
                )
            )
            for prompt_index in range(prompt_total)
        )
        return DistributionAdmissionEvidence(
            probe_mode=kwargs.get("probe_mode", "teacher_forced"),
            root_order=kwargs.get(
                "root_order", ("control", "candidate", "candidate", "control")
            ),
            prompts=prompts,
        )

    # Positive: tiny logit noise keeps PCC and TV inside bounds; near-tie
    # positions are stratified and reported without gating the verdict.
    passing = evaluate_distribution_admission(build_evidence())
    require(passing.admitted, f"close candidate rejected: {passing.rejection_reasons}")
    require(
        passing.eligible_modes == (ADMITTED_SERVING_MODE,),
        f"admitted modes must be sampled-only, got {passing.eligible_modes}",
    )
    require(
        passing.stable_positions >= 120,
        f"stable position count too low: {passing.stable_positions}",
    )
    require(
        passing.stable_agreement is not None and passing.stable_agreement >= 0.95,
        "stable agreement fell below the floor for a passing candidate",
    )
    require(
        passing.near_tie_positions > 0 and passing.near_tie_agreement is not None,
        "near-tie positions were not stratified",
    )

    # Negative: flipping the argmax on stable positions trips the agreement
    # floor even though PCC stays high.
    stable_flip = evaluate_distribution_admission(build_evidence(flip_stable=True))
    require(
        not stable_flip.admitted,
        "stable-position argmax flips were admitted",
    )
    require(
        any(
            "stable-position agreement" in reason
            for reason in stable_flip.rejection_reasons
        ),
        f"stable flip missed its bound: {stable_flip.rejection_reasons}",
    )

    # Negative: same argmax, different distribution trips the TV bound.
    tv_break = evaluate_distribution_admission(build_evidence(break_tv=True))
    require(not tv_break.admitted, "flattened top-K distribution was admitted")
    require(
        any("top-K TV" in reason for reason in tv_break.rejection_reasons),
        f"TV break missed its bound: {tv_break.rejection_reasons}",
    )

    # Negative: disjoint top-K sets invalidate the probe.
    shared_break = evaluate_distribution_admission(build_evidence(break_shared=True))
    require(not shared_break.admitted, "disjoint top-K sets were admitted")

    # Negative: corpus, root-order, and mode violations fail closed.
    short_corpus = evaluate_distribution_admission(build_evidence(prompt_override=4))
    require(not short_corpus.admitted, "a four-prompt corpus was admitted")
    wrong_order = evaluate_distribution_admission(
        build_evidence(root_order=("candidate", "control", "control", "candidate"))
    )
    require(not wrong_order.admitted, "an unbalanced root order was admitted")
    wrong_mode = evaluate_distribution_admission(build_evidence(probe_mode="eager"))
    require(not wrong_mode.admitted, "a non-teacher-forced probe was admitted")

    # Negative: non-finite evidence fails closed.
    bad = build_position(1)
    broken = PositionSample(
        control=bad.control,
        candidate=tuple(
            (token, float("nan") if offset == 0 else probability, logit)
            for offset, (token, probability, logit) in enumerate(bad.candidate)
        ),
    )
    nonfinite = evaluate_distribution_admission(
        DistributionAdmissionEvidence(
            probe_mode="teacher_forced",
            root_order=("control", "candidate", "candidate", "control"),
            prompts=(PromptProbe(positions=(broken,)),),
        )
    )
    require(not nonfinite.admitted, "non-finite probabilities were admitted")


# r[verify tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission]
def probe_kernel_distribution_probe():
    """Positive and negative tests for the pure probe builder and receipt."""

    from qwen36_service.kernel_distribution_admission_core import TOP_K_CANDIDATES
    from qwen36_service.kernel_distribution_probe_core import (
        PositionReadback,
        ProbeBuildError,
        canonical_corpus_payload,
        evaluate_probe,
    )

    vocab_size = 32
    prompt_count = 5
    position_count = 48
    tokens = tuple(range(vocab_size))

    def readback(index, side):
        near_tie = index % 10 == 0
        gap = 0.2 if near_tie else 1.0
        logits = tuple(5.0 - gap * rank for rank in range(vocab_size))
        if side == "candidate":
            noise = 1e-3 if index % 2 == 0 else -1e-3
            logits = tuple(
                value + noise * ((-1) ** rank) for rank, value in enumerate(logits)
            )
        return PositionReadback(tokens=tokens, logits=logits)

    def corpus(**overrides):
        prompt_total = overrides.get("prompt_count", prompt_count)
        position_total = overrides.get("position_count", position_count)
        return tuple(
            tuple(
                (
                    readback(index, "control"),
                    readback(index, overrides.get("candidate_side", "candidate")),
                )
                for index in range(position_total)
            )
            for _prompt in range(prompt_total)
        )

    corpus_digest_hex = "a" * 64
    receipt = evaluate_probe(
        probe_mode="teacher_forced",
        root_order=("control", "candidate", "candidate", "control"),
        corpus=corpus(),
        corpus_blake3=corpus_digest_hex,
    )
    require(
        receipt["admitted"],
        f"probe receipt rejected a close candidate: {receipt['rejection_reasons']}",
    )
    require(
        receipt["eligible_modes"] == ["sampled"],
        f"probe receipt modes must be sampled-only: {receipt['eligible_modes']}",
    )
    require(receipt["top_k"] == TOP_K_CANDIDATES, "receipt top-K drifted")
    require(
        receipt["corpus_blake3"] == corpus_digest_hex,
        "receipt did not carry the shell-provided corpus digest",
    )
    for metric in (
        "mean_logit_pcc",
        "min_logit_pcc",
        "mean_topk_tv",
        "stable_agreement",
    ):
        require(
            receipt["metrics"][metric] is not None,
            f"receipt metric {metric} is missing",
        )

    # Determinism: same corpus, same canonical payload; one changed logit, a
    # different payload (the shell hashes this payload with b3sum).
    require(
        canonical_corpus_payload(corpus()) == canonical_corpus_payload(corpus()),
        "identical corpora produced different canonical payloads",
    )
    drifted = list(corpus())
    first_prompt = list(drifted[0])
    first_pair = first_prompt[0]
    drifted_control = PositionReadback(
        tokens=first_pair[0].tokens,
        logits=first_pair[0].logits[:-1] + (first_pair[0].logits[-1] + 0.25,),
    )
    first_prompt[0] = (drifted_control, first_pair[1])
    drifted[0] = tuple(first_prompt)
    require(
        canonical_corpus_payload(tuple(drifted)) != canonical_corpus_payload(corpus()),
        "a changed corpus kept the canonical payload",
    )
    try:
        evaluate_probe(
            probe_mode="teacher_forced",
            root_order=("control", "candidate", "candidate", "control"),
            corpus=corpus(),
            corpus_blake3="not-hex",
        )
    except ProbeBuildError:
        pass
    else:
        raise AssertionError("an invalid corpus digest produced a receipt")

    def expect_rejected(label, **kwargs):
        try:
            evaluate_probe(
                probe_mode=kwargs.get("probe_mode", "teacher_forced"),
                root_order=kwargs.get(
                    "root_order", ("control", "candidate", "candidate", "control")
                ),
                corpus=kwargs["corpus"],
                corpus_blake3=kwargs.get("corpus_blake3", corpus_digest_hex),
            )
        except ProbeBuildError:
            return
        raise AssertionError(f"{label} produced a receipt")

    # Negative: duplicate tokens, short vocab, mismatched vocab, non-finite
    # logits, short corpus, wrong root order, wrong probe mode. Malformed
    # prompts stay full length so the corpus floors pass and the intended
    # readback validation is what rejects them.
    dup = corpus()
    first_pair = tuple(dup[0][0])
    dup_control = PositionReadback(
        tokens=first_pair[0].tokens[:-1] + (first_pair[0].tokens[0],),
        logits=first_pair[0].logits,
    )
    bad_pair = (dup_control, first_pair[1])
    expect_rejected(
        "duplicate token ids",
        corpus=(tuple(bad_pair for _ in range(position_count)),) + dup[1:],
    )
    short_vocab = PositionReadback(
        tokens=tuple(range(TOP_K_CANDIDATES)),
        logits=tuple(1.0 for _ in range(TOP_K_CANDIDATES)),
    )
    short_pair = (short_vocab, short_vocab)
    expect_rejected(
        "top-K-sized readback",
        corpus=(tuple(short_pair for _ in range(position_count)),) + dup[1:],
    )
    mismatched = PositionReadback(
        tokens=tokens[:-1], logits=tuple(1.0 for _ in range(vocab_size - 1))
    )
    mismatch_pair = (first_pair[0], mismatched)
    expect_rejected(
        "mismatched vocab lengths",
        corpus=(tuple(mismatch_pair for _ in range(position_count)),) + dup[1:],
    )
    nonfinite = PositionReadback(
        tokens=tokens,
        logits=(float("inf"),) + tuple(1.0 for _ in range(vocab_size - 1)),
    )
    nonfinite_pair = (nonfinite, first_pair[1])
    expect_rejected(
        "non-finite logit",
        corpus=(tuple(nonfinite_pair for _ in range(position_count)),) + dup[1:],
    )
    expect_rejected("four-prompt corpus", corpus=corpus(prompt_count=4))
    expect_rejected(
        "wrong root order",
        corpus=corpus(),
        root_order=("candidate", "control", "control", "candidate"),
    )
    expect_rejected("eager probe mode", corpus=corpus(), probe_mode="eager")


# r[verify tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission]
def probe_sampled_mode_router():
    """Positive and negative tests for the pure sampled-mode router."""
    from types import SimpleNamespace

    from qwen36_service.decode_adapter import DecodeAdapterError, TracedDecodeAdapter
    from qwen36_service.sampled_mode_router_core import (
        ROUTE_DETERMINISTIC,
        ROUTE_SAMPLED,
        ROUTES,
        RouterPolicyError,
        classify_temperature,
        eligible_routes_for_admitted_modes,
        resolve_batch_route,
        route_request,
    )

    # Classification: missing and zero temperatures stay deterministic.
    require(
        classify_temperature(None) == ROUTE_DETERMINISTIC,
        "missing temperature must route deterministic",
    )
    require(
        classify_temperature(0) == ROUTE_DETERMINISTIC,
        "zero temperature must route deterministic",
    )
    require(
        classify_temperature(0.0) == ROUTE_DETERMINISTIC,
        "zero float temperature must route deterministic",
    )
    require(
        classify_temperature(0.6) == ROUTE_SAMPLED,
        "positive temperature must route sampled",
    )
    require(
        classify_temperature(1) == ROUTE_SAMPLED,
        "integer positive temperature must route sampled",
    )

    # Classification fails closed on invalid temperatures.
    for bad in (-0.1, float("nan"), float("inf"), "0.6", True, []):
        try:
            classify_temperature(bad)
        except RouterPolicyError:
            continue
        raise AssertionError(f"invalid temperature {bad!r} was classified")

    # Switch decisions compare the route against the resident mode.
    stay = route_request(0.6, current_mode=ROUTE_SAMPLED)
    require(
        stay.route == ROUTE_SAMPLED and not stay.switch_required,
        "sampled request on sampled mode must not switch",
    )
    switch = route_request(0, current_mode=ROUTE_SAMPLED)
    require(
        switch.route == ROUTE_DETERMINISTIC and switch.switch_required,
        "greedy request on sampled mode must switch",
    )

    # Only a sampled admission permits the sampled route.
    require(
        eligible_routes_for_admitted_modes(["sampled"]) == ROUTES,
        "a sampled admission must permit both routes",
    )
    require(
        eligible_routes_for_admitted_modes([]) == (ROUTE_DETERMINISTIC,),
        "an empty admission must permit only the deterministic route",
    )

    # Batch routing: any deterministic request forces the whole batch onto
    # the exact-parity path, and a missing admission never routes sampled.
    all_sampled = resolve_batch_route(
        (0.6, 0.9, 1.2),
        sampled_admitted=True,
        current_mode=ROUTE_DETERMINISTIC,
    )
    require(
        all_sampled.route == ROUTE_SAMPLED and all_sampled.switch_required,
        "an all-sampled admitted batch must route sampled",
    )
    mixed = resolve_batch_route(
        (0.6, 0.0),
        sampled_admitted=True,
        current_mode=ROUTE_SAMPLED,
    )
    require(
        mixed.route == ROUTE_DETERMINISTIC and mixed.switch_required,
        "a mixed batch must route deterministic and switch",
    )
    unadmitted = resolve_batch_route(
        (0.6,),
        sampled_admitted=False,
        current_mode=ROUTE_SAMPLED,
    )
    require(
        unadmitted.route == ROUTE_DETERMINISTIC,
        "without a sampled admission every batch must route deterministic",
    )
    try:
        resolve_batch_route((), sampled_admitted=True, current_mode=ROUTE_DETERMINISTIC)
    except RouterPolicyError:
        pass
    else:
        raise AssertionError("an empty batch was routed")
    try:
        resolve_batch_route(
            (-1.0,), sampled_admitted=True, current_mode=ROUTE_DETERMINISTIC
        )
    except RouterPolicyError:
        pass
    else:
        raise AssertionError("an invalid batch temperature was routed")

    # Adapter switch primitive against a stubbed adapter: release happens
    # only with a resident trace, and every GDN layer flips.
    def stub_adapter(trace_id, modes):
        released = []
        layers = tuple(SimpleNamespace(_decode_fusion_mode=mode) for mode in modes)
        stub = SimpleNamespace(
            _trace_id=trace_id,
            _release_trace=lambda: released.append(True),
            _tp_gdn_layers=lambda: layers,
        )
        return stub, released

    stub, released = stub_adapter(None, ("off", "full", "off"))
    TracedDecodeAdapter.switch_fusion_mode(stub, "full")
    require(not released, "no resident trace must not release")
    require(
        all(layer._decode_fusion_mode == "full" for layer in stub._tp_gdn_layers()),
        "switch must flip every GDN layer",
    )

    stub, released = stub_adapter(object(), ("full",))
    TracedDecodeAdapter.switch_fusion_mode(stub, "off")
    require(released == [True], "a resident trace must be released before flipping")
    require(stub._tp_gdn_layers()[0]._decode_fusion_mode == "off", "mode flip failed")

    stub, _ = stub_adapter(None, ("off",))
    try:
        TracedDecodeAdapter.switch_fusion_mode(stub, "state")
    except DecodeAdapterError:
        pass
    else:
        raise AssertionError("an unknown mode was accepted")


# r[verify tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission]
def probe_sampled_decode():
    """Positive and negative tests for host-side sampled decode."""
    from qwen36_service.core import (
        CompletionRequest,
        Endpoint,
        RequestValidationError,
        parse_completion_request,
    )
    from qwen36_service.sampled_decode_core import (
        DEFAULT_TOP_K,
        DEFAULT_TOP_P,
        SampledDecodeError,
        ShardCandidates,
        apply_nucleus_cut,
        merge_shard_candidates,
        sample_candidates,
        temperature_probabilities,
    )

    left = ShardCandidates(values=(4.0, 2.0, 1.0), token_ids=(10, 11, 12))
    right = ShardCandidates(values=(3.0, 2.5, 0.5), token_ids=(110, 111, 112))
    values, tokens = merge_shard_candidates((left, right), top_k=4)
    require(tokens == (10, 110, 111, 11), f"merge order wrong: {tokens}")
    require(
        values == (4.0, 3.0, 2.5, 2.0),
        f"merged values wrong: {values}",
    )
    cut_values, cut_tokens = merge_shard_candidates((left, right), top_k=2)
    require(cut_tokens == (10, 110), "top_k cut kept the wrong tokens")
    require(
        cut_values == values[: len(cut_tokens)],
        "top_k cut kept values that differ from the selected tokens",
    )

    probabilities = temperature_probabilities((4.0, 3.0, 2.0), 1.0)
    require(
        abs(sum(probabilities) - 1.0) < 1e-9,
        f"probabilities do not sum to one: {probabilities}",
    )
    require(
        probabilities[0] > probabilities[1] > probabilities[2],
        "probability ordering must follow logit ordering",
    )

    kept, positions = apply_nucleus_cut(probabilities, top_p=0.5)
    require(kept == (probabilities[0],), "a 0.5 nucleus kept more than the top token")
    require(positions == (0,), "nucleus positions must be a prefix")

    # Deterministic draws: same draw and inputs always pick the same token,
    # and every draw lands inside the candidate set.
    draw_inputs = dict(
        values=(4.0, 3.0, 2.0),
        token_ids=(10, 11, 12),
        temperature=0.6,
        top_k=DEFAULT_TOP_K,
        top_p=DEFAULT_TOP_P,
    )
    first = sample_candidates(draw=0.1, **draw_inputs)
    second = sample_candidates(draw=0.1, **draw_inputs)
    require(first == second, "identical draws produced different tokens")
    seen = {sample_candidates(draw=draw / 100.0, **draw_inputs) for draw in range(100)}
    require(seen <= {10, 11, 12}, f"draws escaped the candidate set: {seen}")
    require(
        len(seen) > 1,
        "a spread of draws never left the top candidate; sampling is broken",
    )
    greedy_cut = sample_candidates(draw=0.99, **dict(draw_inputs, top_p=0.1))
    require(greedy_cut == 10, "a tiny nucleus must always pick the top token")

    # Parse gate: sampled temperatures pass only when the service enables
    # sampled serving.
    sampled_payload = {
        "model": "m",
        "prompt": "hi",
        "max_tokens": 4,
        "temperature": 0.6,
    }
    request = parse_completion_request(
        sampled_payload,
        Endpoint.COMPLETIONS,
        "m",
        64,
        sampled_serving_enabled=True,
    )
    require(
        isinstance(request, CompletionRequest) and request.temperature == 0.6,
        "sampled serving must accept a positive temperature",
    )
    try:
        parse_completion_request(
            sampled_payload,
            Endpoint.COMPLETIONS,
            "m",
            64,
            sampled_serving_enabled=False,
        )
    except RequestValidationError:
        pass
    else:
        raise AssertionError("greedy-only mode accepted a sampled temperature")

    def expect_error(label, **kwargs):
        try:
            sample_candidates(**kwargs)
        except SampledDecodeError:
            return
        raise AssertionError(f"{label} was sampled")

    base = dict(temperature=0.6, top_k=DEFAULT_TOP_K, top_p=DEFAULT_TOP_P)
    expect_error(
        "empty candidates",
        values=(),
        token_ids=(),
        draw=0.1,
        **base,
    )
    expect_error(
        "count mismatch",
        values=(1.0,),
        token_ids=(1, 2),
        draw=0.1,
        **base,
    )
    expect_error(
        "zero temperature",
        values=(1.0,),
        token_ids=(1,),
        draw=0.1,
        temperature=0.0,
        top_k=DEFAULT_TOP_K,
        top_p=DEFAULT_TOP_P,
    )
    expect_error(
        "negative temperature",
        values=(1.0,),
        token_ids=(1,),
        draw=0.1,
        temperature=-0.5,
        top_k=DEFAULT_TOP_K,
        top_p=DEFAULT_TOP_P,
    )
    expect_error(
        "draw at one",
        values=(1.0,),
        token_ids=(1,),
        draw=1.0,
        **base,
    )
    expect_error(
        "invalid top_p",
        values=(1.0,),
        token_ids=(1,),
        draw=0.1,
        temperature=0.6,
        top_k=DEFAULT_TOP_K,
        top_p=1.5,
    )
    try:
        merge_shard_candidates((), top_k=DEFAULT_TOP_K)
    except SampledDecodeError:
        pass
    else:
        raise AssertionError("an empty shard list was merged")
    try:
        merge_shard_candidates(
            (ShardCandidates(values=(1.0, 1.0), token_ids=(5, 5)),),
            top_k=DEFAULT_TOP_K,
        )
    except SampledDecodeError:
        pass
    else:
        raise AssertionError("duplicate tokens were merged")


# r[verify tenstorrent.qwen38_p150x2.mtp_serving_rounds]
def probe_mtp_serving_core():
    """Positive and negative tests for the gated serving round policy."""
    from qwen36_service.mtp_drafter_core import (
        MAXIMUM_DRAFT_TOKEN_COUNT,
        MINIMUM_DRAFT_TOKEN_COUNT,
    )
    from qwen36_service.mtp_serving_core import (
        ROUND_MODE_PLAIN,
        ROUND_MODE_SPECULATIVE,
        MtpServingPolicyError,
        commit_verified_round,
        plan_round,
    )

    # Scheduling: speculative strictly inside the generation tail.
    plan = plan_round(
        tokens_generated=0,
        maximum_generation_tokens=64,
        draft_token_count=3,
    )
    require(
        plan.mode == ROUND_MODE_SPECULATIVE and plan.draft_token_count == 3,
        "early round was not speculative",
    )
    boundary = plan_round(
        tokens_generated=60,
        maximum_generation_tokens=64,
        draft_token_count=3,
    )
    require(
        boundary.mode == ROUND_MODE_SPECULATIVE,
        "remaining == k+1 was not speculative",
    )
    tail = plan_round(
        tokens_generated=61,
        maximum_generation_tokens=64,
        draft_token_count=3,
    )
    require(
        tail.mode == ROUND_MODE_PLAIN and tail.draft_token_count == 0,
        "remaining == k was not plain",
    )
    done = plan_round(
        tokens_generated=64,
        maximum_generation_tokens=64,
        draft_token_count=3,
    )
    require(done.mode == ROUND_MODE_PLAIN, "exhausted request was not plain")
    for width in (MINIMUM_DRAFT_TOKEN_COUNT, MAXIMUM_DRAFT_TOKEN_COUNT):
        edge = plan_round(
            tokens_generated=0,
            maximum_generation_tokens=width + 1,
            draft_token_count=width,
        )
        require(
            edge.mode == ROUND_MODE_SPECULATIVE,
            f"width {width} boundary was not speculative",
        )

    # Scheduling negatives.
    for kwargs, needle in (
        (
            dict(
                tokens_generated=-1, maximum_generation_tokens=64, draft_token_count=3
            ),
            "tokens_generated must not be negative",
        ),
        (
            dict(
                tokens_generated=65, maximum_generation_tokens=64, draft_token_count=3
            ),
            "exceeds maximum_generation_tokens",
        ),
        (
            dict(tokens_generated=0, maximum_generation_tokens=64, draft_token_count=1),
            "between 2 and 6",
        ),
        (
            dict(tokens_generated=0, maximum_generation_tokens=64, draft_token_count=7),
            "between 2 and 6",
        ),
        (
            dict(
                tokens_generated=True, maximum_generation_tokens=64, draft_token_count=3
            ),
            "must be an integer",
        ),
        (
            dict(tokens_generated=0, maximum_generation_tokens=-5, draft_token_count=3),
            "maximum_generation_tokens must not be negative",
        ),
    ):
        try:
            plan_round(**kwargs)
        except MtpServingPolicyError as error:
            require(needle in str(error), f"wrong schedule diagnostic: {error}")
        else:
            require(False, f"invalid schedule was accepted: {kwargs}")

    # Exactness property: committed streams equal a reference sequential
    # stream for every acceptance pattern and every round start. The
    # continuation carries the window's sequential outputs plus one.
    sequential = (101, 102, 103, 104, 105, 106, 107)
    width = 3
    for start in range(0, len(sequential) - width):
        continuation = sequential[start : start + width + 1]
        for corrupt in (None, 0, 1, 2):
            drafts = list(continuation[:width])
            if corrupt is not None:
                drafts[corrupt] = 999999
            commit = commit_verified_round(
                draft_token_ids=tuple(drafts),
                continuation_token_ids=continuation,
            )
            expected_accept = width if corrupt is None else corrupt
            expected = sequential[start : start + expected_accept] + (
                sequential[start + expected_accept],
            )
            require(
                commit.committed_token_ids == expected,
                f"commit at start {start} corrupt {corrupt} diverged",
            )
            require(
                commit.accepted == expected_accept,
                "commit accepted count diverged",
            )
            require(
                commit.rejected == (corrupt is not None),
                "commit rejection flag diverged",
            )

    # Commit negatives.
    for drafts, continuation, needle in (
        ((11, 12), (11, 12, 13, 14), "differs from draft count plus one"),
        ((11, 12, 13), (11, 12, 13), "differs from draft count plus one"),
        ((11,), (11,), "outside the policy range"),
        (
            tuple(range(7)),
            tuple(range(8)),
            "outside the policy range",
        ),
        ((11, -2), (11, -2, 13), "negative token"),
        ((11, True), (11, 12, 13), "non-integer token"),
    ):
        try:
            commit_verified_round(
                draft_token_ids=drafts, continuation_token_ids=continuation
            )
        except MtpServingPolicyError as error:
            require(needle in str(error), f"wrong commit diagnostic: {error}")
        else:
            require(False, f"invalid commit was accepted: {drafts}")

    try:
        commit_verified_round(
            draft_token_ids=[11, 12], continuation_token_ids=[11, 12, 13]
        )
    except MtpServingPolicyError as error:
        require("must be a tuple" in str(error), "wrong tuple diagnostic")
    else:
        require(False, "list inputs were accepted")

    # Watchdog: success passthrough, worker error re-raise, and hang
    # detection fire the three outcomes the serving loop relies on.
    import time as _time

    from qwen36_service.mtp_watchdog_adapter import (
        MtpServingRoundTimeout,
        run_with_watchdog,
    )

    require(
        run_with_watchdog(lambda: 5, seconds=1.0, stage="probe") == 5,
        "watchdog dropped a completed value",
    )

    def _probe_boom():
        raise RuntimeError("probe worker failure")

    try:
        run_with_watchdog(_probe_boom, seconds=1.0, stage="probe")
    except RuntimeError as error:
        require("probe worker failure" in str(error), "worker error wrapped")
    else:
        require(False, "worker error was swallowed")

    def _probe_hang():
        _time.sleep(2.0)

    hang_started = _time.monotonic()
    try:
        run_with_watchdog(_probe_hang, seconds=0.2, stage="probe-hang")
    except MtpServingRoundTimeout as error:
        require("probe-hang" in str(error), "wrong hang diagnostic")
        require(
            _time.monotonic() - hang_started < 1.5,
            "watchdog fired late",
        )
    else:
        require(False, "hanging stage was accepted")

    for args, kwargs, needle in (
        ((lambda: None,), dict(seconds=0, stage="probe"), "positive"),
        ((lambda: None,), dict(seconds=-1, stage="probe"), "positive"),
        ((None,), dict(seconds=1.0, stage="probe"), "callable"),
    ):
        try:
            run_with_watchdog(*args, **kwargs)
        except MtpServingPolicyError as error:
            require(needle in str(error), "wrong watchdog diagnostic")
        else:
            require(False, "invalid watchdog input was accepted")


# r[verify tenstorrent.qwen38_p150x2.gdn_grid_instance_cap]
def probe_gdn_grid_instance_cap():
    """Contiguous-chunk assignment above the compute-core count.

    The fused GDN decode factory refuses any head count above the core count,
    which bounds the admitted fast path at B=4 on a P150x2 mesh. This probe
    covers the assignment that replaces that guard and the validation that
    takes over the removed fatal's job.
    """
    from qwen36_service.gdn_grid_core import (
        Chunk,
        ChunkPlan,
        GridAssignmentError,
        plan_chunks,
        validate_plan,
    )

    # P150x2 TP=2 decode: 24 local heads per device, so instances = 24 * B.
    for batch, want_cores, want_length in ((1, 24, 1), (4, 96, 1), (8, 96, 2)):
        plan = plan_chunks(24 * batch, 110)
        require(plan.active_cores == want_cores, f"B={batch} used the wrong core count")
        require(plan.chunk_length == want_length, f"B={batch} used the wrong chunk length")
        require(
            sum(chunk.count for chunk in plan.chunks) == 24 * batch,
            f"B={batch} plan does not cover every instance",
        )

    # In-range case reproduces the pre-change one-instance-per-core mapping.
    for instances in (1, 16, 24, 96, 110):
        plan = plan_chunks(instances, 110)
        require(plan.chunk_length == 1, "in-range plan used a chunk above one")
        require(plan.active_cores == instances, "in-range plan did not use one core per instance")
        require(
            [chunk.start for chunk in plan.chunks] == list(range(instances)),
            "in-range plan did not reproduce the one-instance-per-core mapping",
        )

    # Overflow case partitions the range with exactly one short final chunk.
    for instances, cores in ((111, 110), (192, 110), (3072, 110)):
        plan = plan_chunks(instances, cores)
        require(plan.active_cores <= cores, "plan used more cores than available")
        require(
            all(
                chunk.start == index * plan.chunk_length
                for index, chunk in enumerate(plan.chunks)
            ),
            "chunk start is not core_index * chunk_length",
        )
        short = [index for index, c in enumerate(plan.chunks) if c.count < plan.chunk_length]
        require(
            short in ([], [len(plan.chunks) - 1]),
            "a chunk other than the final one is short",
        )

    # Invalid inputs.
    for bad in ((0, 110), (-1, 110), (1, 0), (1, -1), (1.5, 110), (110.0, 110), (True, 1)):
        try:
            plan_chunks(*bad)
        except GridAssignmentError:
            pass
        else:
            raise AssertionError(f"invalid instance count was accepted: {bad}")

    # Malformed plans that must not survive validation.
    malformed = {
        "gap": ChunkPlan(4, 4, 1, (Chunk(0, 0, 1), Chunk(1, 2, 1))),
        "overlap": ChunkPlan(4, 4, 1, (Chunk(0, 0, 1), Chunk(1, 0, 1))),
        "short interior": ChunkPlan(4, 4, 1, (Chunk(0, 0, 1), Chunk(1, 1, 1), Chunk(2, 3, 1), Chunk(3, 4, 0))),
        "over core count": ChunkPlan(2, 1, 1, (Chunk(0, 0, 1), Chunk(1, 1, 1))),
        "duplicate core": ChunkPlan(2, 4, 1, (Chunk(0, 0, 1), Chunk(0, 1, 1))),
        "zero chunk length": ChunkPlan(2, 4, 0, (Chunk(0, 0, 1), Chunk(1, 1, 1))),
        "empty": ChunkPlan(2, 4, 1, ()),
        "undercover": ChunkPlan(4, 4, 1, (Chunk(0, 0, 1), Chunk(1, 1, 1))),
    }
    for name, plan in malformed.items():
        try:
            validate_plan(plan)
        except GridAssignmentError:
            pass
        else:
            raise AssertionError(f"malformed plan was accepted: {name}")


# r[verify tenstorrent.qwen38_p150x2.gdn_grid_instance_cap.unchanged]
def probe_gdn_grid_mapping_unchanged():
    """The in-range program mapping equals the pre-change factory mapping.

    Both transcriptions follow the shipped C++ line by line: the decode
    factory before `8bf9f35` and after it, and the prefill `distribute_scan`
    in the pinned source versus the chunked variant. Every in-range launch
    must produce the identical core-to-instance assignment, which is what
    keeps every existing decode and prefill receipt valid.
    """

    def decode_pre(num_heads, grid_x, grid_y):
        """Pre-change factory: one head per core, core {h/grid_y, h%grid_y}."""
        if num_heads > grid_x * grid_y:
            return None  # the removed TT_FATAL refused the launch
        return {
            (head // grid_y, head % grid_y): (head,)
            for head in range(num_heads)
        }

    def decode_post(num_heads, grid_x, grid_y, ceil_div=True):
        """Chunked factory: contiguous ceil(n/k) instances per active core."""
        ncores = grid_x * grid_y
        per_core = -(-num_heads // ncores) if ceil_div else num_heads // ncores
        if per_core < 1:
            return None
        active = -(-num_heads // per_core)
        return {
            (i // grid_y, i % grid_y): tuple(
                range(i * per_core, min(i * per_core + per_core, num_heads))
            )
            for i in range(active)
        }

    def scan_nv(bh, vt, ncores, bounded=True):
        """The NV v-blocking search, identical source before and after."""
        chosen = 1
        for cand in range(vt, 0, -1):
            if vt % cand == 0 and (bounded and bh * cand <= ncores or not bounded):
                chosen = cand
                break
        return chosen

    def scan_pre(bh, vt, grid_x, grid_y):
        """Pre-change distribute_scan: one (head, v-block) pair per core."""
        ncores = grid_x * grid_y
        if bh > ncores:
            return None  # the removed TT_FATAL refused the launch
        nv = scan_nv(bh, vt, ncores)
        total = bh * nv
        return {
            (i % grid_x, i // grid_x): ((i // nv, i % nv),)
            for i in range(total)
        }

    def scan_post(bh, vt, grid_x, grid_y, bounded=True):
        """Chunked distribute_scan with wi_start/wi_count per active core."""
        ncores = grid_x * grid_y
        if ncores < 1:
            return None
        nv = scan_nv(bh, vt, ncores, bounded=bounded)
        total = bh * nv
        per_core = -(-total // ncores)
        active = -(-total // per_core)
        mapping = {}
        for i in range(active):
            start = i * per_core
            count = min(per_core, total - start)
            instances = tuple(
                (wi // nv, wi % nv) for wi in range(start, start + count)
            )
            mapping[(i % grid_x, i // grid_x)] = instances
        return mapping

    grids = ((13, 10), (11, 10))
    head_counts = (1, 2, 4, 12, 24, 25, 48, 96, 100, 109, 110, 129)
    value_tiles = (1, 2, 4, 8)

    for grid_x, grid_y in grids:
        ncores = grid_x * grid_y
        for num_heads in head_counts:
            if num_heads > ncores:
                continue
            require(
                decode_pre(num_heads, grid_x, grid_y)
                == decode_post(num_heads, grid_x, grid_y),
                f"decode mapping changed in range at {num_heads} heads on {grid_x}x{grid_y}",
            )
            for vt in value_tiles:
                require(
                    scan_pre(num_heads, vt, grid_x, grid_y)
                    == scan_post(num_heads, vt, grid_x, grid_y),
                    f"prefill mapping changed in range at BH={num_heads}, Vt={vt} "
                    f"on {grid_x}x{grid_y}",
                )

    # Both pre-change factories refuse the overflow launch; the chunked
    # factories must accept it with a partition instead.
    for grid_x, grid_y in grids:
        require(
            decode_pre(24 * 8, grid_x, grid_y) is None,
            "pre-change decode factory accepted an overflow launch",
        )
        require(
            scan_pre(24 * 8, 4, grid_x, grid_y) is None,
            "pre-change scan factory accepted an overflow launch",
        )
        overflow = decode_post(24 * 8, grid_x, grid_y)
        require(overflow is not None, "chunked decode factory refused B=8")
        covered = sorted(instance for chunk in overflow.values() for instance in chunk)
        require(
            covered == list(range(24 * 8)),
            "chunked decode mapping does not partition B=8 instances",
        )

    # Negative teeth: each mutation the equality could hide must be caught.
    require(
        decode_post(96, 11, 10, ceil_div=False) != decode_pre(96, 11, 10)
        or decode_post(24, 11, 10, ceil_div=False) != decode_pre(24, 11, 10)
        or decode_post(110, 11, 10, ceil_div=False) != decode_pre(110, 11, 10),
        "floor-division chunking was not distinguished from the pre-change mapping",
    )
    require(
        scan_post(25, 8, 13, 10, bounded=False) != scan_pre(25, 8, 13, 10)
        or scan_post(48, 8, 13, 10, bounded=False) != scan_pre(48, 8, 13, 10),
        "an unbounded NV search was not distinguished from the pre-change mapping",
    )

# r[verify tenstorrent.qwen38_p150x2.grouped_prefill_eligibility]
def probe_grouped_prefill_eligibility():
    """Positive and negative tests for the derived grouped-prefill bound.

    The planner must derive the group bound from the linked runtime and the
    local head geometry. Copying the source's TP4 literal (8) into TP2 would
    over-subscribe the grid: at TP2 the local value-head count is 24, so a
    one-instance-per-core scan bounds a group at four users, not eight.
    """
    from qwen36_service.grouped_prefill_core import (
        REASON_BUCKET_ABOVE_SINGLE_PASS,
        REASON_ELIGIBLE,
        REASON_EMPTY_GROUP,
        REASON_HEAD_CAPACITY_EXCEEDED,
        REASON_SCRATCH_BUDGET_EXCEEDED,
        REASON_SINGLE_USER,
        GroupedPrefillPolicyError,
        RuntimeCapability,
        plan_grouped_prefill,
    )

    p150_compute_cores = 110
    tp2_local_value_heads = 24
    tp4_local_value_heads = 12
    gdn_bucket_chunk = 128
    single_pass_bucket = 128
    per_user_scratch_bytes = 8 * 1024 * 1024
    four_users_scratch_bytes = 4 * per_user_scratch_bytes
    eight_users_scratch_bytes = 8 * per_user_scratch_bytes
    eight_lengths = (tp2_local_value_heads,) * 8
    eight_slots = (5, 0, 7, 2, 6, 1, 3, 4)

    one_instance_per_core = RuntimeCapability(
        chunked_scan_instances=False,
        compute_cores=p150_compute_cores,
        instances_per_core=1,
    )
    chunked_scan = RuntimeCapability(
        chunked_scan_instances=True,
        compute_cores=p150_compute_cores,
        instances_per_core=2,
    )

    # Positive: a TP2 group of four fits a one-instance-per-core scan.
    plan = plan_grouped_prefill(
        user_count=4,
        user_lengths=(96, 128, 64, 128),
        slots=(5, 0, 7, 2),
        bucket=single_pass_bucket,
        local_value_heads=tp2_local_value_heads,
        gdn_bucket_chunk=gdn_bucket_chunk,
        capability=one_instance_per_core,
        scratch_bytes=four_users_scratch_bytes,
        bytes_per_user=per_user_scratch_bytes,
    )
    require(plan.grouped, f"a legal TP2 group was rejected: {plan.reason}")
    require(plan.reason == REASON_ELIGIBLE, "legal group reported the wrong reason")
    require(plan.head_bound == 4, f"TP2 head bound is {plan.head_bound}, expected 4")
    require(plan.groups == ((0, 1, 2, 3),), "legal group planned the wrong members")
    require(plan.destinations == (5, 0, 7, 2), "plan lost the explicit slot order")

    # Positive: the chunked scan lifts the head bound on the same geometry.
    chunked_plan = plan_grouped_prefill(
        user_count=8,
        user_lengths=eight_lengths,
        slots=eight_slots,
        bucket=single_pass_bucket,
        local_value_heads=tp2_local_value_heads,
        gdn_bucket_chunk=gdn_bucket_chunk,
        capability=chunked_scan,
        scratch_bytes=eight_users_scratch_bytes,
        bytes_per_user=per_user_scratch_bytes,
    )
    require(chunked_plan.grouped, "the chunked scan did not admit a TP2 group of eight")
    require(
        chunked_plan.head_bound == 9,
        f"chunked TP2 head bound is {chunked_plan.head_bound}, expected 9",
    )
    require(chunked_plan.groups == (tuple(range(8)),), "chunked plan split an eight-user group")
    require(chunked_plan.destinations == eight_slots, "chunked plan reordered the slots")

    # Derived split: eight TP2 users on a one-instance scan exceed a single
    # group of eight (110 // 24 = 4 per group), so the plan must split them
    # rather than copy the TP4 literal and over-subscribe the grid.
    split_plan = plan_grouped_prefill(
        user_count=8,
        user_lengths=eight_lengths,
        slots=eight_slots,
        bucket=single_pass_bucket,
        local_value_heads=tp2_local_value_heads,
        gdn_bucket_chunk=gdn_bucket_chunk,
        capability=one_instance_per_core,
        scratch_bytes=eight_users_scratch_bytes,
        bytes_per_user=per_user_scratch_bytes,
    )
    require(split_plan.grouped, "eight TP2 users were not grouped at all")
    require(split_plan.head_bound == 4, f"TP2 split bound is {split_plan.head_bound}, expected 4")
    require(
        split_plan.groups == ((0, 1, 2, 3), (4, 5, 6, 7)),
        f"TP2 split planned the wrong groups: {split_plan.groups}",
    )
    require(
        split_plan.destinations == eight_slots,
        "TP2 split lost the explicit slot order",
    )

    # Negative: a user wider than the whole grid cannot be grouped even alone,
    # and an empty group and a single user are the per-user route by contract.
    over_subscribed = plan_grouped_prefill(
        user_count=2,
        user_lengths=(64, 64),
        slots=(3, 4),
        bucket=single_pass_bucket,
        local_value_heads=p150_compute_cores + tp2_local_value_heads,
        gdn_bucket_chunk=gdn_bucket_chunk,
        capability=one_instance_per_core,
        scratch_bytes=four_users_scratch_bytes,
        bytes_per_user=per_user_scratch_bytes,
    )
    require(not over_subscribed.grouped, "users wider than the grid were grouped")
    require(
        over_subscribed.reason == REASON_HEAD_CAPACITY_EXCEEDED,
        f"wrong refusal reason: {over_subscribed.reason}",
    )
    require(
        over_subscribed.groups == ((0,), (1,)),
        "refusal did not fall back to one user per group",
    )

    empty = plan_grouped_prefill(
        user_count=0,
        user_lengths=(),
        slots=(),
        bucket=single_pass_bucket,
        local_value_heads=tp2_local_value_heads,
        gdn_bucket_chunk=gdn_bucket_chunk,
        capability=one_instance_per_core,
        scratch_bytes=four_users_scratch_bytes,
        bytes_per_user=per_user_scratch_bytes,
    )
    require(not empty.grouped, "an empty group was planned as grouped")
    require(empty.reason == REASON_EMPTY_GROUP, f"wrong empty-group reason: {empty.reason}")
    require(empty.groups == (), "an empty group produced rows")

    single = plan_grouped_prefill(
        user_count=1,
        user_lengths=(64,),
        slots=(3,),
        bucket=single_pass_bucket,
        local_value_heads=tp2_local_value_heads,
        gdn_bucket_chunk=gdn_bucket_chunk,
        capability=one_instance_per_core,
        scratch_bytes=four_users_scratch_bytes,
        bytes_per_user=per_user_scratch_bytes,
    )
    require(not single.grouped, "a single user was planned as a group")
    require(single.reason == REASON_SINGLE_USER, f"wrong single-user reason: {single.reason}")

    # Negative: the TP4 geometry the source literal was written for derives 9,
    # so the bound may not be taken from the literal at either width.
    tp4_plan = plan_grouped_prefill(
        user_count=8,
        user_lengths=(tp4_local_value_heads,) * 8,
        slots=tuple(range(8)),
        bucket=single_pass_bucket,
        local_value_heads=tp4_local_value_heads,
        gdn_bucket_chunk=gdn_bucket_chunk,
        capability=one_instance_per_core,
        scratch_bytes=eight_users_scratch_bytes,
        bytes_per_user=per_user_scratch_bytes,
    )
    require(tp4_plan.head_bound == 9, f"TP4 head bound is {tp4_plan.head_bound}, expected 9")

    # Negative: a bucket above the single-pass multiple is a per-user candidate.
    long_bucket = plan_grouped_prefill(
        user_count=4,
        user_lengths=(96, 128, 64, 128),
        slots=(5, 0, 7, 2),
        bucket=4 * gdn_bucket_chunk,
        local_value_heads=tp2_local_value_heads,
        gdn_bucket_chunk=gdn_bucket_chunk,
        capability=chunked_scan,
        scratch_bytes=four_users_scratch_bytes,
        bytes_per_user=per_user_scratch_bytes,
    )
    require(not long_bucket.grouped, "an oversized bucket was grouped")
    require(
        long_bucket.reason == REASON_BUCKET_ABOVE_SINGLE_PASS,
        f"wrong oversized-bucket reason: {long_bucket.reason}",
    )

    # Boundary: exactly the maximum multiple of the GDN bucket chunk is still
    # one single-pass group; one chunk more is not.
    boundary_bucket = plan_grouped_prefill(
        user_count=4,
        user_lengths=(96, 128, 64, 128),
        slots=(5, 0, 7, 2),
        bucket=2 * gdn_bucket_chunk,
        local_value_heads=tp2_local_value_heads,
        gdn_bucket_chunk=gdn_bucket_chunk,
        capability=chunked_scan,
        scratch_bytes=four_users_scratch_bytes,
        bytes_per_user=per_user_scratch_bytes,
    )
    require(
        boundary_bucket.grouped,
        f"the single-pass bucket boundary was refused: {boundary_bucket.reason}",
    )

    # Negative: a scratch budget below one user is refused before allocation.
    starved = plan_grouped_prefill(
        user_count=4,
        user_lengths=(96, 128, 64, 128),
        slots=(5, 0, 7, 2),
        bucket=single_pass_bucket,
        local_value_heads=tp2_local_value_heads,
        gdn_bucket_chunk=gdn_bucket_chunk,
        capability=chunked_scan,
        scratch_bytes=per_user_scratch_bytes,
        bytes_per_user=per_user_scratch_bytes,
    )
    require(not starved.grouped, "a single-user scratch budget was grouped")
    require(
        starved.reason == REASON_SCRATCH_BUDGET_EXCEEDED,
        f"wrong scratch reason: {starved.reason}",
    )

    # Rejected shapes: duplicated destinations, a short length list, a negative
    # user count, and a bucket below the longest user.
    rejected_calls = (
        {"user_count": 2, "user_lengths": (64, 64), "slots": (1, 1)},
        {"user_count": 2, "user_lengths": (64,), "slots": (1, 2)},
        {"user_count": -1, "user_lengths": (), "slots": ()},
        {"user_count": 2, "user_lengths": (256, 64), "slots": (1, 2), "bucket": 128},
    )
    for overrides in rejected_calls:
        arguments = dict(
            user_count=2,
            user_lengths=(64, 64),
            slots=(1, 2),
            bucket=single_pass_bucket,
            local_value_heads=tp2_local_value_heads,
            gdn_bucket_chunk=gdn_bucket_chunk,
            capability=one_instance_per_core,
            scratch_bytes=four_users_scratch_bytes,
            bytes_per_user=per_user_scratch_bytes,
        )
        arguments.update(overrides)
        try:
            plan_grouped_prefill(**arguments)
        except GroupedPrefillPolicyError:
            pass
        else:
            raise AssertionError(f"invalid grouped-prefill request was accepted: {overrides}")

    for contradictory in (
        {"chunked_scan_instances": False, "compute_cores": 110, "instances_per_core": 2},
        {"chunked_scan_instances": True, "compute_cores": 110, "instances_per_core": 0},
        {"chunked_scan_instances": True, "compute_cores": 0, "instances_per_core": 1},
    ):
        try:
            RuntimeCapability(**contradictory).validate()
        except GroupedPrefillPolicyError:
            pass
        else:
            raise AssertionError(f"contradictory capability was accepted: {contradictory}")


# r[verify tenstorrent.qwen38_p150x2.mtp_serving_rounds]
def probe_mtp_head_load_core():
    """Positive and negative tests for the head weight integrity verdicts."""
    import torch

    from qwen36_service.mtp_head_load_probe import evaluate_weight_integrity

    expected = {
        "fc": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "norm": torch.tensor([0.5, 1.5]),
    }
    verdicts = evaluate_weight_integrity(
        expected, {k: v.clone() for k, v in expected.items()}
    )
    require(
        all(verdict["passed"] for verdict in verdicts.values()),
        f"identical weights reported drift: {verdicts}",
    )
    drifted = {k: v.clone() for k, v in expected.items()}
    drifted["fc"][1, 1] += 0.5
    verdicts = evaluate_weight_integrity(expected, drifted)
    require(
        not verdicts["fc"]["passed"] and verdicts["norm"]["passed"],
        "integrity verdict flagged the wrong weight",
    )
    require(
        verdicts["fc"]["max_abs_diff"] == 0.5,
        "integrity verdict lost the drift magnitude",
    )
    try:
        evaluate_weight_integrity(expected, {"fc": expected["fc"].clone()})
    except ValueError as error:
        require("key sets differ" in str(error), "wrong key-set diagnostic")
    else:
        require(False, "missing weight key was accepted")
    reshaped = {k: v.clone() for k, v in expected.items()}
    reshaped["norm"] = reshaped["norm"].reshape(1, 2)
    verdicts = evaluate_weight_integrity(expected, reshaped)
    require(
        not verdicts["norm"]["passed"]
        and verdicts["norm"]["reason"] == "shape mismatch",
        "shape mismatch was not flagged",
    )

    from qwen36_service.mtp_drafter_adapter import mtp_cache_directory_name

    require(
        mtp_cache_directory_name(None) == "mtp_drafter_probe",
        "missing runtime identity did not select the legacy cache directory",
    )
    require(
        mtp_cache_directory_name("") == "mtp_drafter_probe",
        "empty runtime identity did not select the legacy cache directory",
    )
    scoped = mtp_cache_directory_name(
        "/nix/store/dr9bj8qwflylq39zl5ajcx42cdrb1bv9-tt-metal-0.77.0"
    )
    require(
        scoped
        == "mtp_drafter_probe-runtime-dr9bj8qwflylq39zl5ajcx42cdrb1bv9-tt-metal-0.77.0",
        f"runtime-scoped cache directory changed: {scoped}",
    )
    require(
        mtp_cache_directory_name("bad id with spaces/and:colons")
        == "mtp_drafter_probe-runtime-bad_id_with_spaces_and_colons",
        "runtime identity sanitization changed",
    )
    require(
        mtp_cache_directory_name("a" * 100) == "mtp_drafter_probe-runtime-" + "a" * 64,
        "runtime identity tail was not bounded to 64 characters",
    )
    require(
        mtp_cache_directory_name(
            "/nix/store/dr9bj8qwflylq39zl5ajcx42cdrb1bv9-tt-metal-0.77.0"
        )
        != mtp_cache_directory_name("/nix/store/other-tt-metal-0.74.0"),
        "distinct runtimes shared one cache directory",
    )


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
    probe("GDN decode layout candidate gate", probe_gdn_decode_layout_candidate)
    probe("GDN decode fusion candidate", probe_gdn_decode_fusion_candidate)
    probe("TP=2 / P300 gate invariants", probe_p150x2_gate)
    probe("traced decode pure core", probe_traced_decode_contract)
    probe("batch decode scheduler", probe_batch_decode_scheduler)
    probe("batch admission policy", probe_batch_admission)
    probe("tensor binding identity", probe_tensor_binding_identity)
    probe("same-width trace-reuse profile", probe_trace_reuse_profile)
    probe("startup warm-up readiness", probe_startup_readiness)
    probe("prefill program profile", probe_prefill_program_profile)
    probe("MTP target-verification pure core", probe_mtp_target_verification)
    probe("MTP drafter pure core", probe_mtp_drafter)
    probe("MTP draft selection", probe_mtp_draft_selection)
    probe("MTP draft trace", probe_mtp_draft_trace)
    probe("MTP K/V rollback", probe_mtp_kv_rollback)
    probe("MTP head load integrity core", probe_mtp_head_load_core)
    probe("MTP recovery stage progress", probe_mtp_recovery_stage)
    probe("MTP prewarmed recovery control", probe_mtp_recovery_prewarmed_control)
    probe("MTP target-validation prewarm", probe_mtp_target_validation_prewarm)
    probe("MTP sampled acceptance", probe_mtp_sampled_acceptance)
    probe("MTP trace capture hygiene", probe_mtp_trace_capture_hygiene)
    probe("decode component attribution", probe_decode_component_attribution)
    probe("same-process TTFT control", probe_ttft_control)
    probe("serialized service request contract", probe_service_contract)
    probe("kernel distribution admission", probe_kernel_distribution_admission)
    probe("kernel distribution probe builder", probe_kernel_distribution_probe)
    probe("sampled-mode router", probe_sampled_mode_router)
    probe("sampled decode", probe_sampled_decode)
    probe("MTP serving round policy", probe_mtp_serving_core)
    probe("GDN grid instance cap assignment", probe_gdn_grid_instance_cap)
    probe("GDN grid in-range mapping unchanged", probe_gdn_grid_mapping_unchanged)
    probe("grouped prefill eligibility", probe_grouped_prefill_eligibility)

    if FAILURES:
        print("qwen36 device-free check: FAIL")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("qwen36 device-free check: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
