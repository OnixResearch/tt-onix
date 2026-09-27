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
    import importlib
    import os as _os

    gate = fused_chunk_module.FUSED_PREFILL_GATE
    try:
        for value, expected in (
            ("1", True),
            (" 1 ", True),
            ("0", False),
            ("yes", False),
            ("", False),
        ):
            with mock.patch.dict(_os.environ, {gate: value}):
                importlib.reload(fused_chunk_module)
                require(
                    fused_chunk_module.FUSED_CHUNK_PREFILL_ADMITTED is expected,
                    f"fused GDN prefill gate value {value!r} did not select {expected}",
                )
    finally:
        with mock.patch.dict(_os.environ, clear=False):
            _os.environ.pop(gate, None)
            importlib.reload(fused_chunk_module)
    require(
        not fused_chunk_module.FUSED_CHUNK_PREFILL_ADMITTED,
        "an unset fused GDN prefill gate enabled fused prefill",
    )
    require(
        hasattr(_ttnn.transformer, "gated_delta_attn_seq"),
        "packaged runtime lacks gated_delta_attn_seq for the seq fallback",
    )
    # A 2048-row masked prefill bucket overflows P150x2 L1 beside the resident
    # decode buffers; untraced prompts above 1024 tokens must chunk.
    from models.demos.blackhole.qwen36.tt import model as model_module

    require(
        model_module.EAGER_PREFILL_CHUNK_TOKENS <= 1024
        and model_module.EAGER_PREFILL_CHUNK_TOKENS
        in model_module.Qwen36Model._PREFILL_MASK_BUCKETS,
        "eager prefill chunk must be a warmed masked bucket of at most 1024 tokens",
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


def probe_kv_pool():
    """Pooled KV rows: prefill coverage, disjoint runs, the pool bound, and batch selection."""
    import threading
    import types
    from collections import deque

    from qwen36_service.batch_decode_adapter import BatchDecodeAdapterError
    from qwen36_service.batch_service import BatchGenerationCoordinator
    from qwen36_service.kv_pool_core import KvPoolError, KvPoolLayout

    pool_blocks = 4096
    row_blocks = 1024
    pool = KvPoolLayout(pool_blocks=pool_blocks, row_blocks=row_blocks, kv_block_size=64)
    prefix = set(range(pool.reserved_prefix_blocks))
    require(pool.scratch_block == pool_blocks - 1, "the scratch block is not the last block")
    require(
        pool.group_capacity_blocks == pool_blocks - len(prefix) - 1,
        "the group capacity counts the read prefix or the scratch block",
    )

    # A row covers its whole prefill buckets and every decoded position.
    for prompt, generation, blocks, reason in (
        (100, 8, 16, "a short prompt did not cover its 1,024-token prefill bucket"),
        (1030, 64, 32, "the tail bucket past 1,024 tokens was not covered"),
        (5000, 8192, 207, "a long generation was not covered"),
        (65536 - 8192, 8192, row_blocks, "a full-length request did not fit one row"),
    ):
        require(
            pool.request_blocks(prompt_tokens=prompt, generation_tokens=generation) == blocks,
            reason,
        )
    try:
        pool.request_blocks(prompt_tokens=65536, generation_tokens=1)
    except KvPoolError as error:
        require("a row holds" in str(error), f"wrong oversized-request diagnostic: {error}")
    else:
        require(False, "a request longer than one row was accepted")

    counts = (16, 32, 207)
    rows = pool.assign_group(counts, width=16)
    require(
        len(rows) == 16 and all(len(row) == row_blocks for row in rows),
        "a group table does not have full-width rows",
    )
    runs = [row[:count] for row, count in zip(rows, counts)]
    require(
        all(run == list(range(run[0], run[0] + len(run))) for run in runs),
        "a group row is not one contiguous run",
    )
    assigned = [set(run) for run in runs]
    require(
        sum(len(run) for run in assigned) == len(set().union(*assigned)),
        "two group rows share a KV block",
    )
    require(
        not any(run & (prefix | {pool.scratch_block}) for run in assigned),
        "a group row took a read-prefix or scratch block",
    )
    require(
        all(set(row[count:]) <= {pool.scratch_block} for row, count in zip(rows, counts))
        and all(set(row) == {pool.scratch_block} for row in rows[len(counts) :]),
        "entries past a row's run do not point at the scratch block",
    )
    for block_counts, diagnostic in (
        ((row_blocks,) * 4, "the pool holds"),
        ((row_blocks + 1,), "a row holds"),
        ((), "between one row"),
    ):
        try:
            pool.assign_group(block_counts, width=16)
        except KvPoolError as error:
            require(diagnostic in str(error), f"wrong group diagnostic: {error}")
        else:
            require(False, f"an invalid group was assigned: {block_counts!r}")

    startup = pool.startup_rows(width=16)
    require(startup[-1] == list(range(row_blocks)), "the read row is not a full run from block 0")
    generation_rows = [set(row) - {pool.scratch_block} for row in startup[:-1]]
    require(
        all(len(row) >= 16 for row in generation_rows)
        and sum(len(row) for row in generation_rows) == len(set().union(*generation_rows))
        and not any(row & set(startup[-1]) for row in generation_rows)
        and max(max(row) for row in generation_rows) < pool.scratch_block,
        "startup rows overlap, exceed the pool, or cannot hold one prefill bucket",
    )
    try:
        KvPoolLayout(pool_blocks=row_blocks, row_blocks=row_blocks, kv_block_size=64)
    except KvPoolError as error:
        require("one full-length row" in str(error), f"wrong small-pool diagnostic: {error}")
    else:
        require(False, "a pool without room for one row, the prefix and scratch was accepted")

    # The batch worker stops adding rows once the pool would overflow.
    class PoolAdapter:
        group_capacity_blocks = pool.group_capacity_blocks

        def request_blocks(self, *, prompt_tokens, generation_tokens):
            try:
                return pool.request_blocks(
                    prompt_tokens=prompt_tokens, generation_tokens=generation_tokens
                )
            except KvPoolError as error:
                raise BatchDecodeAdapterError(str(error)) from error

        def decode_width_supported(self, width):
            return True

    def take(prompt_counts):
        pending = [
            SimpleNamespace(
                request=SimpleNamespace(max_tokens=8192, temperature=0.0),
                prompt_token_count=prompt_count,
                index=index,
            )
            for index, prompt_count in enumerate(prompt_counts)
        ]
        worker = SimpleNamespace(
            _condition=threading.Condition(),
            _queue=deque(pending),
            _closing=False,
            adapter=PoolAdapter(),
            sampled_admission_enabled=False,
        )
        worker._request_blocks = types.MethodType(
            BatchGenerationCoordinator._request_blocks, worker
        )
        batch = BatchGenerationCoordinator._take_batch(worker)
        return [item.index for item in batch], [item.index for item in worker._queue]

    full = 65536 - 8192
    require(take([full] * 4) == ([0, 1, 2], [3]), "a group overflowed the KV pool")
    require(
        take([full, 100, 100]) == ([0, 1, 2], []),
        "short requests did not join a long one while blocks remained",
    )
    require(
        take([65536, 100]) == ([0], [1]),
        "a request no row can hold was grouped with others",
    )


# r[verify tenstorrent.qwen38_p150x2.continuous_slot_policy]
def probe_continuous_slot_policy():
    """Continuous slots: B4 admission, staggered refill, stale events, cancellation, deadlines,
    capacity, fairness, refill budgets, and quarantine, all on explicit inputs."""
    import dataclasses

    from qwen36_service import continuous_slot_core as core
    from qwen36_service.kv_pool_core import KvPoolLayout

    with Path(__file__).with_name("continuous-slot-policy.json").open(encoding="utf-8") as handle:
        mapping = json.load(handle)
    policy = core.ContinuousSlotPolicy.from_mapping(mapping)
    require(policy.slot_capacity == 4, "the packaged policy is not B4")
    for change, reason in (
        ({"extra": 1}, "an extra policy field was accepted"),
        ({"slot_capacity": 8}, "a B8 slot capacity was accepted"),
        ({"refills_per_boundary": 5}, "more refills per boundary than slots were accepted"),
        ({"worker_stall_seconds": 0}, "a zero worker stall bound was accepted"),
        ({"queue_capacity": True}, "a boolean queue capacity was accepted"),
    ):
        try:
            core.ContinuousSlotPolicy.from_mapping({**mapping, **change})
        except core.ContinuousSlotPolicyError:
            pass
        else:
            require(False, reason)

    layout = KvPoolLayout(pool_blocks=4096, row_blocks=1024, kv_block_size=64)
    kv = core.KvBlockRange(
        layout.reserved_prefix_blocks, layout.scratch_block, layout.row_blocks, layout.kv_block_size
    )

    def request(request_id, prompt, max_tokens, *, route="step", arrival=0.0, deadline=1000.0):
        return core.SlotRequest(
            request_id=request_id,
            prompt_tokens=prompt,
            max_tokens=max_tokens,
            route=route,
            blocks=layout.request_blocks(prompt_tokens=prompt, generation_tokens=max_tokens),
            arrival_seconds=arrival,
            deadline_seconds=deadline,
        )

    def queue_all(schedule, requests, use_policy=policy):
        for item in requests:
            schedule, refused = core.submit(schedule, use_policy, kv, item)
            require(refused is None, f"{item.request_id} was refused: {refused}")
        return schedule

    def kinds(effects, kind):
        return [effect for effect in effects if isinstance(effect, kind)]

    def prefill_all(schedule, effects, token=7):
        for effect in kinds(effects, core.PrefillSlot):
            schedule = core.observe_prefill(
                schedule, slot=effect.slot, generation=effect.generation, token_id=token, stopped=False
            )
        return schedule

    def run_window(schedule, stop_slots=frozenset()):
        window = core.plan_window(schedule, policy)
        for step in range(window.steps):
            for row in window.rows:
                if core.slot_running(schedule, row.slot, row.generation):
                    schedule = core.observe_step(
                        schedule,
                        slot=row.slot,
                        generation=row.generation,
                        token_id=100 + step,
                        stopped=row.slot in stop_slots,
                    )
        return schedule, window

    # Initial admission stays within B4: six arrivals start four slots, in order, with
    # distinct generations and disjoint KV runs inside the generation range.
    schedule = queue_all(
        core.initial_schedule(policy), [request(f"r{i}", 100 + i, 32) for i in range(6)]
    )
    schedule, effects = core.plan_boundary(schedule, policy, kv, now=1.0)
    starts = kinds(effects, core.StartGroup)
    prefills = kinds(effects, core.PrefillSlot)
    require(starts == [core.StartGroup("step")], f"group start effects: {starts}")
    require(
        [(p.slot, p.request_id) for p in prefills] == [(i, f"r{i}") for i in range(4)],
        f"initial admission was not four slots in arrival order: {prefills}",
    )
    require(len({p.generation for p in prefills}) == 4, "slot generations repeated")
    runs = sorted((p.block_start, p.block_start + p.block_count) for p in prefills)
    require(
        runs[0][0] >= kv.first_block
        and runs[-1][1] <= kv.end_block
        and all(a[1] <= b[0] for a, b in zip(runs, runs[1:])),
        f"KV runs overlap or leave the generation range: {runs}",
    )
    require([q.request.request_id for q in schedule.queue] == ["r4", "r5"], "the queue lost order")

    # Staggered, unequal lengths: A finishes after 4 tokens while B runs on. C arrives
    # later and refills A's slot under a new generation; B's identity, position and
    # next token are untouched.
    schedule = queue_all(
        core.initial_schedule(policy), [request("A", 100, 4), request("B", 3000, 50)]
    )
    schedule, effects = core.plan_boundary(schedule, policy, kv, now=1.0)
    schedule = prefill_all(schedule, effects)
    schedule, window = run_window(schedule)
    require(window.steps == 3, f"the window ran past A's bound: {window.steps}")
    require(schedule.slots[0].state == core.SLOT_COMPLETE, "A did not complete at its bound")
    require(schedule.slots[0].finish_reason == core.FINISH_LENGTH, "A's finish reason is wrong")
    b_before = schedule.slots[1]
    require(b_before.position == 3003 and b_before.generated == 4, f"B advanced wrongly: {b_before}")
    a_generation = schedule.slots[0].generation
    schedule = queue_all(schedule, [request("C", 200, 8, arrival=2.0)])
    schedule, effects = core.plan_boundary(schedule, policy, kv, now=2.5)
    require(
        kinds(effects, core.RetireSlot) == [core.RetireSlot(0, a_generation, "A", core.FINISH_LENGTH)],
        f"A was not retired once: {effects}",
    )
    refill = kinds(effects, core.PrefillSlot)
    require(
        len(refill) == 1 and refill[0].slot == 0 and refill[0].request_id == "C",
        f"C did not refill A's slot: {refill}",
    )
    require(refill[0].generation > b_before.generation, "the refill reused an old generation")
    require(not kinds(effects, core.StartGroup), "a refill beside an active user restarted the group")
    require(schedule.slots[1] == b_before, "the refill changed B's slot")
    b_run = (b_before.block_start, b_before.block_start + b_before.request.blocks)
    c_run = (refill[0].block_start, refill[0].block_start + refill[0].block_count)
    require(c_run[1] <= b_run[0] or b_run[1] <= c_run[0], "C's KV run overlaps B's")

    # Stale and out-of-order events are rejected and change nothing.
    for event, reason in (
        (lambda: core.observe_step(schedule, slot=0, generation=a_generation, token_id=1, stopped=False),
         "a token for A's old generation reached C's slot"),
        (lambda: core.observe_step(schedule, slot=0, generation=refill[0].generation, token_id=1, stopped=False),
         "a decode token was accepted before C's prefill completed"),
        (lambda: core.observe_prefill(schedule, slot=1, generation=b_before.generation, token_id=1, stopped=False),
         "a second prefill completion was accepted for B"),
        (lambda: core.observe_step(schedule, slot=7, generation=1, token_id=1, stopped=False),
         "an event for a slot outside the schedule was accepted"),
    ):
        snapshot = schedule
        try:
            event()
        except core.SlotEventRejected:
            pass
        else:
            require(False, reason)
        require(schedule == snapshot, "a rejected event changed the schedule")
    # Once C decodes, a late token from A's generation still cannot reach C.
    refilled = core.observe_prefill(
        schedule, slot=0, generation=refill[0].generation, token_id=9, stopped=False
    )
    try:
        core.observe_step(refilled, slot=0, generation=a_generation, token_id=1, stopped=False)
    except core.SlotEventRejected:
        pass
    else:
        require(False, "a token from A's generation was applied to C's active slot")

    # A first token that ends a request completes it without a decode window.
    schedule = queue_all(core.initial_schedule(policy), [request("one", 50, 1), request("eos", 50, 9)])
    schedule, effects = core.plan_boundary(schedule, policy, kv, now=1.0)
    one, eos = kinds(effects, core.PrefillSlot)
    schedule = core.observe_prefill(schedule, slot=one.slot, generation=one.generation, token_id=5, stopped=False)
    schedule = core.observe_prefill(schedule, slot=eos.slot, generation=eos.generation, token_id=2, stopped=True)
    require(
        [slot.finish_reason for slot in schedule.slots[:2]] == [core.FINISH_LENGTH, core.FINISH_STOP],
        "first-token completions were not recorded",
    )
    require(core.plan_window(schedule, policy) is None, "a window was planned with no active slot")

    # An end-of-sequence token mid-window completes that row only.
    schedule = queue_all(core.initial_schedule(policy), [request("x", 50, 40), request("y", 50, 40)])
    schedule, effects = core.plan_boundary(schedule, policy, kv, now=1.0)
    schedule = prefill_all(schedule, effects)
    schedule, window = run_window(schedule, stop_slots=frozenset({1}))
    require(window.steps == policy.decode_window_steps, "the window ignored its step bound")
    require(
        schedule.slots[1].finish_reason == core.FINISH_STOP and schedule.slots[1].generated == 2,
        "the stopped row kept generating",
    )
    require(
        schedule.slots[0].generated == 1 + policy.decode_window_steps,
        "the running row lost steps when its neighbour stopped",
    )

    # Cancellation: a queued request is refused at once, a running one completes as
    # cancelled and is retired at the next boundary, an unknown one changes nothing.
    schedule = queue_all(
        core.initial_schedule(policy), [request(f"c{i}", 64, 20) for i in range(6)]
    )
    schedule, effects = core.plan_boundary(schedule, policy, kv, now=1.0)
    schedule = prefill_all(schedule, effects)
    schedule, refused = core.cancel(schedule, "c5")
    require(refused == core.RejectRequest("c5", core.REJECT_CANCELLED), "a queued cancel was not refused")
    require([q.request.request_id for q in schedule.queue] == ["c4"], "the cancelled request stayed queued")
    schedule, refused = core.cancel(schedule, "c2")
    require(refused is None and schedule.slots[2].finish_reason == core.FINISH_CANCELLED, "a running cancel failed")
    unchanged = schedule
    schedule, refused = core.cancel(schedule, "missing")
    require(refused is None and schedule == unchanged, "an unknown cancel changed the schedule")
    schedule, effects = core.plan_boundary(schedule, policy, kv, now=2.0)
    require(
        [(r.slot, r.finish_reason) for r in kinds(effects, core.RetireSlot)] == [(2, core.FINISH_CANCELLED)],
        f"the cancelled slot was not retired: {effects}",
    )
    require(
        [(p.slot, p.request_id) for p in kinds(effects, core.PrefillSlot)] == [(2, "c4")],
        "the cancelled slot was not refilled",
    )

    # Deadlines: queued requests past their deadline or the queue bound are refused;
    # an active request past its deadline is retired as timed out.
    schedule = queue_all(
        core.initial_schedule(policy),
        [request(f"d{i}", 64, 64, deadline=50.0) for i in range(4)]
        + [request("late", 64, 8, deadline=10.0), request("stale", 64, 8, deadline=5000.0)],
    )
    schedule, effects = core.plan_boundary(schedule, policy, kv, now=1.0)
    schedule = prefill_all(schedule, effects)
    schedule, effects = core.plan_boundary(
        schedule, policy, kv, now=policy.maximum_queue_seconds + 1.0
    )
    require(
        sorted((r.request_id, r.reason) for r in kinds(effects, core.RejectRequest))
        == [("late", core.REJECT_DEADLINE), ("stale", core.REJECT_QUEUE_TIMEOUT)],
        f"queued deadlines were not enforced: {effects}",
    )
    require(
        {r.finish_reason for r in kinds(effects, core.RetireSlot)} == {core.FINISH_TIMEOUT}
        and len(kinds(effects, core.RetireSlot)) == 4,
        "active requests past their deadline were not retired",
    )

    # Capacity: a full queue, a request no row can hold, and a duplicate id are refused
    # without changing the schedule; a request waits while no KV run fits it.
    schedule = core.initial_schedule(policy)
    for index in range(policy.queue_capacity):
        schedule, refused = core.submit(schedule, policy, kv, request(f"q{index}", 64, 8))
    before = schedule
    for item, reason in (
        (request("overflow", 64, 8), core.REJECT_QUEUE_FULL),
        (request("q0", 64, 8), core.REJECT_DUPLICATE),
    ):
        schedule, refused = core.submit(schedule, policy, kv, item)
        require(refused == core.RejectRequest(item.request_id, reason), f"{item.request_id}: {refused}")
        require(schedule == before, "a refused request changed the schedule")
    oversized = dataclasses.replace(request("huge", 64, 8), blocks=kv.row_blocks + 1)
    schedule, refused = core.submit(core.initial_schedule(policy), policy, kv, oversized)
    require(refused == core.RejectRequest("huge", core.REJECT_EXCEEDS_KV_ROW), "an oversized row was queued")
    full = 65536 - 8192
    # Three full-length rows leave less than one row free; a fourth full row waits.
    schedule = queue_all(
        core.initial_schedule(policy),
        [request(f"f{i}", full, 8192) for i in range(4)] + [request("small", 100, 8)],
    )
    schedule, effects = core.plan_boundary(schedule, policy, kv, now=1.0)
    admitted = [p.request_id for p in kinds(effects, core.PrefillSlot)]
    require(admitted[:3] == ["f0", "f1", "f2"] and "f3" not in admitted, f"KV admission: {admitted}")
    require("small" in admitted, "a small request did not pass a head that no KV run fits")
    require(schedule.queue[0].request.request_id == "f3" and schedule.queue[0].bypassed == 1,
            "the blocked head's bypass was not counted")

    # Fairness: while a sampled head waits for the greedy group, later greedy requests
    # pass it at most maximum_head_bypasses times; then admission stops, the group
    # drains, and the head starts the next group.
    schedule = queue_all(core.initial_schedule(policy), [request("g0", 64, 60)])
    schedule, effects = core.plan_boundary(schedule, policy, kv, now=1.0)
    schedule = prefill_all(schedule, effects)
    schedule = queue_all(
        schedule,
        [request("sampled", 64, 4, route="full")] + [request(f"g{i}", 64, 60) for i in range(1, 6)],
    )
    passed = []
    for boundary in range(40):
        schedule, effects = core.plan_boundary(schedule, policy, kv, now=2.0 + boundary)
        passed += [p.request_id for p in kinds(effects, core.PrefillSlot)]
        if "sampled" in passed:
            require(
                kinds(effects, core.StartGroup) == [core.StartGroup("full")],
                "the sampled head did not start its own group",
            )
            break
        schedule = prefill_all(schedule, effects)
        if core.plan_window(schedule, policy) is not None:
            schedule, _ = run_window(schedule)
    require(
        passed[: policy.maximum_head_bypasses] == ["g1", "g2"][: policy.maximum_head_bypasses]
        and passed[policy.maximum_head_bypasses] == "sampled",
        f"the blocked head was passed more than the policy allows: {passed}",
    )

    # Refill budgets beside active users: at most refills_per_boundary prefills, and the
    # prompt-token budget delays (never skips) the next request; the first refill at a
    # boundary is always allowed.
    wide = dataclasses.replace(policy, refills_per_boundary=3, refill_prompt_token_budget=4096)
    schedule = queue_all(core.initial_schedule(wide), [request("long", 64, 400)], wide)
    schedule, effects = core.plan_boundary(schedule, wide, kv, now=1.0)
    schedule = prefill_all(schedule, effects)
    schedule = queue_all(
        schedule, [request("big", 6000, 8), request("mid", 3000, 8), request("tiny", 50, 8)], wide
    )
    schedule, effects = core.plan_boundary(schedule, wide, kv, now=2.0)
    require(
        [p.request_id for p in kinds(effects, core.PrefillSlot)] == ["big"],
        "the token budget let a later request pass, or blocked the first refill",
    )
    schedule = prefill_all(schedule, effects)
    schedule, effects = core.plan_boundary(schedule, wide, kv, now=3.0)
    require(
        [p.request_id for p in kinds(effects, core.PrefillSlot)] == ["mid", "tiny"],
        "the budget did not admit the waiting requests at the next boundary",
    )
    schedule = queue_all(core.initial_schedule(policy), [request("l", 64, 400)])
    schedule, effects = core.plan_boundary(schedule, policy, kv, now=1.0)
    schedule = prefill_all(schedule, effects)
    schedule = queue_all(schedule, [request("m", 64, 8), request("n", 64, 8)])
    schedule, effects = core.plan_boundary(schedule, policy, kv, now=2.0)
    require(len(kinds(effects, core.PrefillSlot)) == policy.refills_per_boundary, "refill count bound broken")

    # A waiting structured read admits nothing, so the group drains for it.
    schedule, effects = core.plan_boundary(schedule, policy, kv, now=3.0, reads_waiting=True)
    require(not kinds(effects, core.PrefillSlot), "a refill was admitted while a read waited")

    # Draining and quarantine: nothing is admitted until the group is empty and reset.
    schedule = prefill_all(schedule, effects)
    drained = core.mark_draining(schedule)
    _, effects = core.plan_boundary(drained, policy, kv, now=4.0)
    require(not kinds(effects, core.PrefillSlot), "a draining group was refilled")
    quarantined, failed = core.quarantine_group(schedule)
    require(
        sorted(r.request_id for r in failed) == ["l", "m"]
        and {r.finish_reason for r in failed} == {core.FINISH_FAILED},
        f"quarantine did not fail every occupied slot: {failed}",
    )
    quarantined, effects = core.plan_boundary(quarantined, policy, kv, now=5.0)
    require(not effects, f"a quarantined group retired or admitted work: {effects}")
    released = core.release_quarantine(quarantined)
    require(released.route is None and not released.draining, "the released schedule kept its group")
    released, effects = core.plan_boundary(released, policy, kv, now=6.0)
    require(
        kinds(effects, core.StartGroup) and [p.request_id for p in kinds(effects, core.PrefillSlot)] == ["n"],
        "a released schedule did not start a new group",
    )
    released, refused = core.drain_queue(queue_all(released, [request("z", 64, 8)]), "closed")
    require(refused == (core.RejectRequest("z", "closed"),) and not released.queue, "drain_queue kept work")


# r[verify tenstorrent.qwen38_p150x2.continuous_slot_effects]
def probe_continuous_slot_worker():
    """Continuous-slot shell over a fake adapter: one device owner, exact refills beside live
    rows, the inactive-row contract, group failure and reset, drift and stall quarantine."""
    import threading
    import time

    import torch
    from qwen36_service import continuous_slot_core as core
    from qwen36_service.batch_decode_adapter import (
        BatchDecodeAdapterError,
        SlotRefillResult,
        TraceBindingDriftError,
    )
    from qwen36_service.continuous_slot_service import (
        ContinuousSlotCoordinator,
        EngineQuarantinedError,
        SlotRequestRefused,
    )
    from qwen36_service.core import CompletionRequest, Endpoint
    from qwen36_service.kv_pool_core import KvPoolError, KvPoolLayout
    from qwen36_service.sampled_decode_core import ShardCandidates

    stop_token = 3

    def expected_token(seed, index):
        """The fake model: a request's index-th generated token depends only on its prompt."""
        if seed % 5 == 0 and index == 6:
            return stop_token
        return 10 + (seed * 31 + index * 17) % 1000

    def reference(seed, max_tokens):
        tokens = []
        for index in range(max_tokens):
            token = expected_token(seed, index)
            if token == stop_token:
                break
            tokens.append(token)
        return tokens

    class FakeSlotAdapter:
        """Row-independent fake device that enforces the adapter's slot contract."""

        maximum_width = 16

        def __init__(self):
            self.kv_pool = KvPoolLayout(pool_blocks=4096, row_blocks=1024, kv_block_size=64)
            self.fusion_mode = "step"
            self.lock = threading.Lock()
            self.log = []
            self.threads = set()
            self.active = 0
            self.overlapped = False
            self.width = None
            self.owned = {}
            self.hooks = {}
            self.release_trace_on_refill = False
            self.slot_trace_captures = 0
            self.prefix_states = 0

        def prefix_cache_available(self):
            return False

        def _enter(self, name, **fields):
            with self.lock:
                self.active += 1
                self.overlapped = self.overlapped or self.active > 1
                self.threads.add(threading.get_ident())
                self.log.append((name, fields))
            try:
                hook = self.hooks.get(name)
                if hook is not None:
                    hook(**fields)
            except BaseException:
                self._exit()
                raise

        def _exit(self):
            with self.lock:
                self.active -= 1

        def request_blocks(self, *, prompt_tokens, generation_tokens):
            # Host arithmetic, not a device effect: callers may use it from any thread.
            try:
                return self.kv_pool.request_blocks(
                    prompt_tokens=prompt_tokens, generation_tokens=generation_tokens
                )
            except KvPoolError as error:
                raise BatchDecodeAdapterError(str(error)) from error

        def begin_slots(self, *, width, route):
            self._enter("begin_slots", width=width, route=route)
            self.width = width
            self.owned = {}
            self.fusion_mode = route
            self._exit()

        def refill_slot(self, *, slot, token_ids, block_start, block_count, **prefix):
            if any(prefix.get(key) for key in ("prefix_ranges", "prefix_tokens", "resume_state", "save_state")):
                raise BatchDecodeAdapterError("a prefix plan reached an adapter without a prefix cache")
            self._enter("refill_slot", slot=slot, block_start=block_start, block_count=block_count)
            try:
                if self.width is None:
                    raise BatchDecodeAdapterError("refill before the slot group was reset")
                if slot in self.owned:
                    raise BatchDecodeAdapterError("slot already owned")
                for other in self.owned.values():
                    if block_start < other["end"] and other["start"] < block_start + block_count:
                        raise BatchDecodeAdapterError("KV runs overlap")
                seed = int(token_ids.sum())
                first = expected_token(seed, 0)
                self.owned[slot] = {
                    "seed": seed,
                    "index": 1,
                    "last": first,
                    "position": int(token_ids.shape[1]),
                    "start": block_start,
                    "end": block_start + block_count,
                }
                return SlotRefillResult(
                    first_token_id=first,
                    elapsed_seconds=0.002,
                    first_token_ready_seconds=0.001,
                    masked_traces_used=True,
                    masked_traces_released=False,
                    decode_trace_released=self.release_trace_on_refill,
                )
            finally:
                self._exit()

        def retire_slot(self, *, slot):
            self._enter("retire_slot", slot=slot)
            try:
                if self.owned.pop(slot, None) is None:
                    raise BatchDecodeAdapterError("retired an unowned slot")
            finally:
                self._exit()

        def abandon_slots(self):
            self._enter("abandon_slots")
            self.width = None
            self.owned = {}
            self._exit()

        def _check_inputs(self, token_ids, positions):
            if len(token_ids) != self.width or len(positions) != self.width:
                raise BatchDecodeAdapterError("decode inputs do not cover every row")
            for row in range(self.width):
                state = self.owned.get(row)
                expected = (0, 0) if state is None else (state["last"], state["position"])
                if (token_ids[row], positions[row]) != expected:
                    raise BatchDecodeAdapterError(
                        f"row {row} input {(token_ids[row], positions[row])} != {expected}"
                    )

        def _advance(self):
            tokens = []
            for row in range(self.width):
                state = self.owned.get(row)
                if state is None:
                    tokens.append(0)
                    continue
                token = expected_token(state["seed"], state["index"])
                state["index"] += 1
                state["position"] += 1
                state["last"] = token
                tokens.append(token)
            return tuple(tokens)

        def decode_slots_greedy(self, *, token_ids, positions, steps):
            self._enter("decode_window", steps=steps)
            try:
                self._check_inputs(token_ids, positions)
            finally:
                self._exit()

            def stream():
                for _ in range(steps):
                    self._enter("decode_step")
                    try:
                        tokens = self._advance()
                    finally:
                        self._exit()
                    yield tokens

            return stream()

        def decode_slots_sampled(self, *, token_ids, positions):
            self._enter("decode_sampled", route=self.fusion_mode)
            try:
                self._check_inputs(token_ids, positions)
                return tuple(
                    (ShardCandidates((0.0,), (token,)), ShardCandidates((-1000.0,), (token + 1,)))
                    for token in self._advance()
                )
            finally:
                self._exit()

    class FakeTokenizer:
        @staticmethod
        def decode(tokens, skip_special_tokens=True):
            return " ".join(str(token) for token in tokens)

    with Path(__file__).with_name("continuous-slot-policy.json").open(encoding="utf-8") as handle:
        mapping = json.load(handle)
    policy = core.ContinuousSlotPolicy.from_mapping({**mapping, "decode_window_steps": 4})

    def make(adapter, *, stall_seconds=None, sampled=False):
        use_policy = policy
        if stall_seconds is not None:
            use_policy = core.ContinuousSlotPolicy.from_mapping(
                {**mapping, "decode_window_steps": 4, "worker_stall_seconds": stall_seconds}
            )
        emitted = []
        coordinator = ContinuousSlotCoordinator(
            slot_policy=use_policy,
            adapter=adapter,
            tokenizer=FakeTokenizer(),
            model=SimpleNamespace(args=SimpleNamespace(vocab_size=2048)),
            profile=SimpleNamespace(profile_id="decode-test"),
            model_alias="model-test",
            model_identity="model-identity-test",
            package_identity="package-test",
            runtime_identity="runtime-test",
            maximum_sequence_length=65536,
            physical_pcie_device_ids=(0, 1),
            emitter=lambda stage, **fields: emitted.append((stage, fields)),
            sampled_admission_enabled=sampled,
            stop_token_ids=frozenset({stop_token}),
        )
        with coordinator._condition:
            coordinator._accepting_requests = True
        return coordinator, emitted

    def prompt(seed, length):
        tokens = torch.ones((1, length), dtype=torch.int64)
        tokens[0, 0] = seed - (length - 1)
        return tokens

    def submit(coordinator, name, seed, length, max_tokens, results, *, temperature=None, deadline=60.0):
        sink = []
        request = CompletionRequest(
            endpoint=Endpoint.COMPLETIONS,
            model="model-test",
            max_tokens=max_tokens,
            prompt=name,
            messages=(),
            temperature=temperature,
            seed=7,
        )

        def run():
            started = time.monotonic()
            try:
                response = coordinator.generate(
                    request=request,
                    request_id=name,
                    token_ids=prompt(seed, length),
                    prompt_token_count=length,
                    started=started,
                    deadline=started + deadline,
                    token_sink=sink.append,
                )
            except Exception as error:  # noqa: BLE001 - reported through results
                results[name] = ("error", error, sink)
            else:
                results[name] = ("ok", response, sink)

        thread = threading.Thread(target=run, name=f"client-{name}")
        thread.start()
        return thread

    # Staggered arrivals with unequal lengths, more requests than slots, and one request
    # that ends early on a stop token. Every answer equals its request decoded alone.
    adapter = FakeSlotAdapter()
    # Slow steps keep the long request running while the later ones arrive.
    adapter.hooks["decode_step"] = lambda **_fields: time.sleep(0.01)
    coordinator, emitted = make(adapter)
    results = {}
    plan = [
        ("long", 101, 3000, 40),
        ("short", 202, 100, 3),
        ("eos", 305, 50, 30),
        ("mid", 404, 700, 12),
        ("late1", 507, 64, 9),
        ("late2", 608, 64, 17),
        ("late3", 709, 2000, 5),
    ]
    threads = []
    try:
        for index, (name, seed, length, max_tokens) in enumerate(plan):
            threads.append(submit(coordinator, name, seed, length, max_tokens, results))
            if index == 0:
                # The long request holds a slot before the others arrive.
                waited = time.monotonic() + 5
                while not any(n == "refill_slot" for n, _ in list(adapter.log)):
                    if time.monotonic() >= waited:
                        require(False, "the first request was never prefilled")
                        break
                    time.sleep(0.001)
            if index in (3, 5):
                time.sleep(0.05)
        for thread in threads:
            thread.join(timeout=10)
    finally:
        coordinator.close()
    for name, seed, _length, max_tokens in plan:
        outcome = results.get(name)
        require(outcome is not None and outcome[0] == "ok", f"{name} failed: {outcome}")
        if outcome is None or outcome[0] != "ok":
            continue
        response, sink = outcome[1], outcome[2]
        expected = reference(seed, max_tokens)
        tokens = response["x_onix_decode_evidence"]["generated_token_ids"]
        require(tokens == expected, f"{name} decoded {tokens}, alone it decodes {expected}")
        require(sink[: len(expected)] == expected, f"{name} streamed tokens out of order")
        require(
            response["choices"][0]["finish_reason"] == ("stop" if len(expected) < max_tokens else "length"),
            f"{name} finish reason is wrong",
        )
        require(
            response["x_onix_decode_evidence"]["decode_path"] == "continuous-slot",
            "the response does not name the continuous-slot path",
        )
    require(len(adapter.threads) == 1, f"device calls came from {len(adapter.threads)} threads")
    require(not adapter.overlapped, "two device effects overlapped")
    refills = [fields for name, fields in adapter.log if name == "refill_slot"]
    require(len(refills) == len(plan), "a request was prefilled more or less than once")
    require(
        sum(1 for name, _ in adapter.log if name == "begin_slots") == 1,
        "the group restarted although slots stayed busy",
    )
    first_window = next(i for i, (name, _) in enumerate(adapter.log) if name == "decode_window")
    require(
        any(name == "refill_slot" for name, _ in adapter.log[first_window:]),
        "no slot was refilled beside a running decode",
    )
    require(
        max(fields["slot"] for fields in refills) < policy.slot_capacity,
        "a refill used a row outside B4",
    )

    # A sampled group decodes step by step with host-staged tokens, on its own route.
    adapter = FakeSlotAdapter()
    coordinator, _ = make(adapter, sampled=True)
    results = {}
    try:
        threads = [
            submit(coordinator, "s1", 811, 64, 11, results, temperature=0.6),
            submit(coordinator, "s2", 912, 128, 6, results, temperature=0.6),
        ]
        for thread in threads:
            thread.join(timeout=10)
    finally:
        coordinator.close()
    for name, seed, max_tokens in (("s1", 811, 11), ("s2", 912, 6)):
        outcome = results.get(name)
        require(
            outcome is not None
            and outcome[0] == "ok"
            and outcome[1]["x_onix_decode_evidence"]["generated_token_ids"] == reference(seed, max_tokens)
            and outcome[1]["x_onix_decode_evidence"]["serving_route"] == "full",
            f"sampled {name} failed: {outcome}",
        )

    # A device error fails the whole group, which is reset before the next one starts.
    adapter = FakeSlotAdapter()
    coordinator, emitted = make(adapter)
    failures = {"left": 1}

    def fail_once(**_fields):
        time.sleep(0.01)
        # Fail once both requests hold a slot, so the failure hits a two-row group.
        if failures["left"] and len(adapter.owned) == 2:
            failures["left"] -= 1
            raise RuntimeError("injected device failure")

    adapter.hooks["decode_step"] = fail_once
    results = {}
    try:
        for thread in [
            submit(coordinator, "f1", 1001, 64, 20, results),
            submit(coordinator, "f2", 1102, 64, 20, results),
        ]:
            thread.join(timeout=10)
        after = {}
        submit(coordinator, "after", 1203, 64, 8, after).join(timeout=10)
    finally:
        coordinator.close()
    require(
        all(results.get(name, ("missing",))[0] == "error" for name in ("f1", "f2")),
        f"the failed group's requests did not fail: {results}",
    )
    names = [name for name, _ in adapter.log]
    require("abandon_slots" in names, "the failed group was not reset")
    require(
        names.index("abandon_slots") < len(names) - 1 - names[::-1].index("begin_slots"),
        "a new group began before the failed one was reset",
    )
    require(
        after.get("after", ("missing",))[0] == "ok"
        and after["after"][1]["x_onix_decode_evidence"]["generated_token_ids"] == reference(1203, 8),
        f"the worker did not serve after a reset: {after}",
    )
    require(coordinator.quarantine_reason is None, "a recoverable failure quarantined the engine")

    # An unsafe refill (decode trace released) drains the group: nothing else is admitted
    # until it is empty, and its live users still finish exactly.
    adapter = FakeSlotAdapter()
    adapter.release_trace_on_refill = True
    adapter.hooks["decode_step"] = lambda **_fields: time.sleep(0.01)
    coordinator, emitted = make(adapter)
    results = {}
    try:
        first = submit(coordinator, "d1", 1301, 64, 24, results)
        time.sleep(0.05)
        second = submit(coordinator, "d2", 1402, 64, 4, results)
        for thread in (first, second):
            thread.join(timeout=10)
    finally:
        coordinator.close()
    begins = [i for i, (name, _) in enumerate(adapter.log) if name == "begin_slots"]
    require(
        all(results.get(name, ("missing",))[0] == "ok" for name in ("d1", "d2")),
        f"a draining group lost a request: {results}",
    )
    require(
        results["d1"][1]["x_onix_decode_evidence"]["generated_token_ids"] == reference(1301, 24),
        "a draining group's live user diverged",
    )
    require(len(begins) == 2, "a request was refilled into a draining group")
    require(
        any(stage == "continuous-slot-group-draining" for stage, _ in emitted),
        "the unsafe refill was not reported",
    )

    # Cancellation withdraws a queued request without device work.
    adapter = FakeSlotAdapter()
    coordinator, _ = make(adapter)
    gate = threading.Event()
    adapter.hooks["begin_slots"] = lambda **_fields: gate.wait(timeout=5)
    results = {}
    try:
        blocker = submit(coordinator, "blocker", 1501, 64, 4, results)
        time.sleep(0.05)
        queued = submit(coordinator, "withdrawn", 1602, 64, 4, results)
        time.sleep(0.05)
        require(coordinator.cancel("withdrawn"), "a queued request could not be cancelled")
        require(not coordinator.cancel("unknown"), "an unknown request was reported cancelled")
        gate.set()
        for thread in (blocker, queued):
            thread.join(timeout=10)
    finally:
        gate.set()
        coordinator.close()
    outcome = results.get("withdrawn")
    require(
        outcome is not None and isinstance(outcome[1], SlotRequestRefused) and outcome[1].reason == "cancelled",
        f"the cancelled request did not fail as cancelled: {outcome}",
    )
    require(
        not [f for name, f in adapter.log if name == "refill_slot" and f["slot"] != 0],
        "the cancelled request was prefilled",
    )

    # Trace-binding drift quarantines the engine: every request fails and no device call
    # follows.
    adapter = FakeSlotAdapter()
    coordinator, emitted = make(adapter)
    refill_count = {"value": 0}

    def drift(**_fields):
        # The second refill observes a moved binding, whichever slot it targets.
        refill_count["value"] += 1
        if refill_count["value"] == 2:
            raise TraceBindingDriftError("batch trace binding changed")

    adapter.hooks["refill_slot"] = drift
    adapter.hooks["decode_step"] = lambda **_fields: time.sleep(0.01)
    results = {}
    try:
        threads = [submit(coordinator, f"q{i}", 1701 + i, 64, 30, results) for i in range(3)]
        for thread in threads:
            thread.join(timeout=10)
        calls_after_drift = len(adapter.log)
        refused = {}
        submit(coordinator, "refused", 1800, 64, 4, refused).join(timeout=5)
        require(
            isinstance(refused["refused"][1], EngineQuarantinedError),
            "a quarantined engine accepted a request",
        )
    finally:
        coordinator.close()
    require(coordinator.quarantine_reason == "trace-binding-drift", "drift did not quarantine the engine")
    require(
        all(isinstance(results[name][1], BatchDecodeAdapterError) for name in ("q0", "q1", "q2")),
        f"requests survived trace drift: {results}",
    )
    require(len(adapter.log) == calls_after_drift, "the device was used after quarantine")
    require(not any(name == "abandon_slots" for name, _ in adapter.log), "a drifted engine reused the device")

    # A device effect that outlives the stall bound quarantines the engine from the waiting
    # caller; the worker touches the device no more once the effect returns.
    adapter = FakeSlotAdapter()
    coordinator, emitted = make(adapter, stall_seconds=0.2)
    release = threading.Event()
    adapter.hooks["decode_step"] = lambda **_fields: release.wait(timeout=10)
    results = {}
    try:
        thread = submit(coordinator, "stalled", 1901, 64, 8, results)
        thread.join(timeout=5)
        stalled_calls = len(adapter.log)
        release.set()
        time.sleep(0.2)
    finally:
        release.set()
        coordinator.close()
    outcome = results.get("stalled")
    require(
        outcome is not None and isinstance(outcome[1], EngineQuarantinedError),
        f"a stalled worker did not fail its caller: {outcome}",
    )
    require(coordinator.quarantine_reason == "worker-stalled", "the stall did not quarantine the engine")
    require(len(adapter.log) == stalled_calls, "the worker used the device after a stall quarantine")
    require(
        any(stage == "engine-quarantined" for stage, _ in emitted),
        "the quarantine was not reported",
    )

    # Construction refuses fixed-group admission measurements.
    try:
        ContinuousSlotCoordinator(
            slot_policy=policy,
            adapter=FakeSlotAdapter(),
            tokenizer=FakeTokenizer(),
            model=None,
            profile=None,
            model_alias="m",
            model_identity="m",
            package_identity="p",
            runtime_identity="r",
            maximum_sequence_length=65536,
            physical_pcie_device_ids=(0, 1),
            emitter=lambda *_args, **_fields: None,
            batch_admission_enabled=True,
        )
    except BatchDecodeAdapterError:
        pass
    else:
        require(False, "continuous slots accepted a fixed-group admission measurement")


# r[verify tenstorrent.qwen38_p150x2.continuous_slot_effects]
def probe_continuous_slot_adapter():
    """The adapter's slot contract on fake TTNN: KV-run ownership, page-table rows, full
    restage after a refill or retirement, drift rejection, and unsafe-refill release."""
    import torch
    from qwen36_service import batch_decode_adapter as adapter_module
    from qwen36_service.batch_decode_adapter import (
        BatchDecodeAdapterError,
        BatchedTracedDecodeAdapter,
        TraceBindingDriftError,
        _BatchTrace,
    )
    from qwen36_service.kv_pool_core import KvPoolLayout

    vocabulary = 64
    first_token = 42
    binding = {"value": ("inputs@1", "kv@1", "gdn@1", "logits@1")}
    released, copies, captures, resets = [], [], [], []

    class FakeMesh:
        programs = 10

        def num_program_cache_entries(self):
            return self.programs

    class FakeModel:
        def __init__(self):
            self.layers = [
                SimpleNamespace(is_full_attention=True, attention=None),
                SimpleNamespace(
                    is_full_attention=False, attention=SimpleNamespace(_decode_fusion_mode="step")
                ),
            ]
            self.prefills = []
            self.buckets = (128, 256)
            self.stale = False
            self.on_prefill = None

        def masked_prefill_traces_stale(self):
            return self.stale

        def release_masked_prefill_traces(self):
            self.buckets = ()
            self.stale = False

        def masked_prefill_trace_buckets(self):
            return self.buckets

        def prefill_paged_slots(self, tokens, page_table, slots, valid_lens, first_token_callback, masked_traces):
            self.prefills.append((page_table.clone(), list(slots), list(valid_lens), masked_traces))
            logits = torch.zeros((1, 1, vocabulary))
            logits[0, 0, first_token] = 1.0
            first_token_callback([logits])
            if self.on_prefill is not None:
                self.on_prefill()
            return [logits]

        @staticmethod
        def prepare_decode_inputs_host(tokens, positions, page_table=None):
            return (
                tokens.clone(),
                positions.clone(),
                "rope",
                None if page_table is None else page_table.clone(),
            )

    layout = KvPoolLayout(pool_blocks=96, row_blocks=16, kv_block_size=64)
    scratch = layout.scratch_block
    model = FakeModel()
    adapter = object.__new__(BatchedTracedDecodeAdapter)
    adapter.model = model
    adapter.profile = SimpleNamespace(maximum_sequence_length=1024, kv_block_size=64)
    adapter.maximum_width = 4
    adapter.mesh = FakeMesh()
    adapter.vocabulary_size = vocabulary
    adapter.blocks_per_user = layout.row_blocks
    adapter.kv_pool = layout
    adapter.page_table = torch.zeros((4, layout.row_blocks), dtype=torch.int32)
    adapter._traces = {}
    adapter._pending_trace_plan = None
    adapter._prior_reset_complete = True
    adapter._closed = False
    adapter._slot_width = None
    adapter._slot_runs = {}
    adapter._slot_inputs_stale = False
    adapter.slot_trace_captures = 0
    adapter.prefix_states = 0
    adapter.reset_all = lambda: resets.append(True)
    adapter._observe_binding = lambda _inputs, _logits: binding["value"]
    streams = []

    def capture(*, token_ids, positions):
        captures.append((token_ids, positions, adapter.page_table[:4].clone()))
        adapter._traces[len(token_ids)] = _BatchTrace(
            width=len(token_ids),
            trace_id=f"trace-{len(captures)}",
            device_inputs=("tokens-in", "positions-in", "rope-in", "page-table-in"),
            logits="logits",
            indices="indices",
            values=None,
            binding=binding["value"],
            capture_seconds=0.1,
        )

    adapter.prepare_width = capture
    adapter.decode_greedy_stream = lambda **fields: streams.append(fields) or iter(())

    def prompt(length):
        return torch.ones((1, length), dtype=torch.int64)

    def rejects(action, reason):
        before = (adapter.page_table.clone(), dict(adapter._slot_runs), len(model.prefills))
        try:
            action()
        except BatchDecodeAdapterError:
            pass
        else:
            require(False, reason)
        after = (adapter.page_table, adapter._slot_runs, len(model.prefills))
        require(
            torch.equal(before[0], after[0]) and before[1] == after[1] and before[2] == after[2],
            f"a rejected slot action changed adapter state: {reason}",
        )

    original_ttnn = adapter_module.ttnn
    original_copy = adapter_module.copy_host_to_device
    adapter_module.ttnn = SimpleNamespace(
        synchronize_device=lambda _mesh: None,
        release_trace=lambda _mesh, trace_id: released.append(trace_id),
        deallocate=lambda _tensor: None,
    )
    adapter_module.copy_host_to_device = lambda host, device_tensors=None, mesh_device=None: copies.append(
        (host, device_tensors)
    )
    try:
        rejects(
            lambda: adapter.refill_slot(slot=0, token_ids=prompt(10), block_start=16, block_count=16),
            "a refill ran without an open slot group",
        )
        adapter.begin_slots(width=4, route="full")
        require(model.layers[1].attention._decode_fusion_mode == "full", "the group route was not selected")
        require(resets, "the slot group began without a reset")
        require(bool((adapter.page_table == scratch).all()), "a new group's rows are not all inactive")

        result = adapter.refill_slot(slot=1, token_ids=prompt(100), block_start=16, block_count=16)
        page_table, prefill_slots, lengths, masked = model.prefills[-1]
        require(prefill_slots == [1] and lengths == [100] and masked, f"wrong slot prefill: {model.prefills[-1]}")
        require(page_table.tolist() == [list(range(16, 32))], "the prefill saw another row")
        require(adapter.page_table[1].tolist() == list(range(16, 32)), "the slot row was not installed")
        require(
            all(adapter.page_table[row].tolist() == [scratch] * 16 for row in (0, 2, 3)),
            "a refill touched another slot's row",
        )
        require(
            result.first_token_id == first_token and not result.decode_trace_released,
            f"wrong refill result: {result}",
        )

        # Ownership and the KV run contract.
        rejects(
            lambda: adapter.refill_slot(slot=1, token_ids=prompt(100), block_start=40, block_count=16),
            "an owned slot was refilled",
        )
        rejects(
            lambda: adapter.refill_slot(slot=2, token_ids=prompt(100), block_start=20, block_count=16),
            "overlapping KV runs were accepted",
        )
        rejects(
            lambda: adapter.refill_slot(slot=2, token_ids=prompt(100), block_start=0, block_count=16),
            "a KV run in the reserved read prefix was accepted",
        )
        rejects(
            lambda: adapter.refill_slot(slot=2, token_ids=prompt(100), block_start=82, block_count=16),
            "a KV run reaching the scratch block was accepted",
        )
        rejects(
            lambda: adapter.refill_slot(slot=2, token_ids=prompt(100), block_start=40, block_count=4),
            "a KV run shorter than the prompt's prefill was accepted",
        )
        rejects(
            lambda: adapter.refill_slot(slot=4, token_ids=prompt(100), block_start=40, block_count=16),
            "a slot outside the group was accepted",
        )

        # The first window captures; an unchanged group replays without restaging.
        tokens, positions = (0, first_token, 0, 0), (0, 100, 0, 0)
        adapter.decode_slots_greedy(token_ids=tokens, positions=positions, steps=4)
        require(len(captures) == 1 and not copies, "the first window did not capture exactly once")
        adapter.decode_slots_greedy(token_ids=tokens, positions=(0, 104, 0, 0), steps=4)
        require(len(captures) == 1 and not copies, "an unchanged group recaptured or restaged")
        require(adapter.slot_trace_captures == 1, "the slot capture was not counted")

        # A refill beside the live row keeps the trace and restages every input.
        adapter.refill_slot(slot=2, token_ids=prompt(100), block_start=32, block_count=16)
        require(not released, "a clean refill released the decode trace")
        adapter.decode_slots_greedy(token_ids=(0, 7, first_token, 0), positions=(0, 108, 100, 0), steps=4)
        require(len(captures) == 1 and len(copies) == 1, "a refill did not restage the resident trace")
        host, device_tensors = copies[-1]
        require(len(host) == 4 and device_tensors == adapter._traces[4].device_inputs, "a partial restage")
        require(
            host[3][2].tolist() == list(range(32, 48)) and host[3][1].tolist() == list(range(16, 32)),
            "the restaged page table does not hold both live rows",
        )

        # Retirement points the row at scratch and restages.
        adapter.retire_slot(slot=1)
        require(adapter.page_table[1].tolist() == [scratch] * 16, "a retired row kept its KV run")
        adapter.decode_slots_greedy(token_ids=(0, 0, 9, 0), positions=(0, 0, 104, 0), steps=4)
        require(len(copies) == 2 and copies[-1][0][3][1].tolist() == [scratch] * 16, "retirement was not restaged")
        rejects(lambda: adapter.retire_slot(slot=1), "an unowned slot was retired")
        rejects(
            lambda: adapter.decode_slots_greedy(token_ids=(0, 0, 9), positions=(0, 0, 104), steps=4),
            "a window that omits a row was accepted",
        )

        # Drift before or during a refill is rejected as drift.
        binding["value"] = ("inputs@2", "kv@1", "gdn@1", "logits@1")
        prefills_before = len(model.prefills)
        try:
            adapter.refill_slot(slot=0, token_ids=prompt(100), block_start=48, block_count=16)
        except TraceBindingDriftError:
            pass
        else:
            require(False, "a drifted trace accepted a refill")
        require(len(model.prefills) == prefills_before, "a refill ran against a drifted trace")
        binding["value"] = adapter._traces[4].binding

        def drift_during_prefill():
            binding["value"] = ("inputs@1", "kv@9", "gdn@1", "logits@1")

        model.on_prefill = drift_during_prefill
        try:
            adapter.refill_slot(slot=0, token_ids=prompt(100), block_start=48, block_count=16)
        except TraceBindingDriftError:
            pass
        else:
            require(False, "drift during a refill was not detected")
        binding["value"] = adapter._traces[4].binding
        model.on_prefill = None
        adapter.abandon_slots()
        require(adapter._slot_width is None and not adapter._traces, "abandoned slots kept a trace")

        # A refill that compiles a program releases the trace instead of replaying it.
        adapter.begin_slots(width=4, route="step")
        adapter.refill_slot(slot=0, token_ids=prompt(100), block_start=16, block_count=16)
        adapter.decode_slots_greedy(token_ids=(first_token, 0, 0, 0), positions=(100, 0, 0, 0), steps=2)
        trace_id = adapter._traces[4].trace_id
        model.on_prefill = lambda: setattr(adapter.mesh, "programs", adapter.mesh.programs + 1)
        result = adapter.refill_slot(slot=1, token_ids=prompt(100), block_start=32, block_count=16)
        model.on_prefill = None
        require(result.decode_trace_released and trace_id in released, "a compiling refill kept the trace")
        captured = len(captures)
        adapter.decode_slots_greedy(token_ids=(5, first_token, 0, 0), positions=(102, 100, 0, 0), steps=2)
        require(len(captures) == captured + 1, "the next window replayed a released trace")

        # Stale masked traces are dropped before the refill, which then runs eagerly.
        model.stale = True
        result = adapter.refill_slot(slot=2, token_ids=prompt(100), block_start=48, block_count=16)
        require(
            result.masked_traces_released and not result.masked_traces_used and result.decode_trace_released,
            f"stale masked traces were not handled: {result}",
        )

        # A structured read closes the slot group.
        adapter._prepare_read()
        rejects(
            lambda: adapter.refill_slot(slot=3, token_ids=prompt(10), block_start=64, block_count=16),
            "a slot group survived a structured read",
        )
    finally:
        adapter_module.ttnn = original_ttnn
        adapter_module.copy_host_to_device = original_copy


# r[verify tenstorrent.qwen38_p150x2.continuous_slot_admission]
def probe_continuous_slot_admission():
    """Queue-inclusive admission: a complete better candidate is admitted; fewer captures,
    starvation, a worse declared tail, parity loss, identity drift, simulation, and
    incomplete roots are each rejected with their own reason."""
    import copy

    from qwen36_service import continuous_slot_admission_core as admission
    from qwen36_service.continuous_slot_receipt import build_receipt

    with Path(__file__).with_name("continuous-slot-admission-policy.json").open(encoding="utf-8") as handle:
        policy_mapping = json.load(handle)
    policy = admission.AdmissionPolicy.from_mapping(policy_mapping)
    require(policy.repetitions == 2, "the packaged policy does not require two passes")
    for path, value, reason in (
        (("gates", "require_exact_candidate_parity"), False, "an admission without parity was accepted"),
        (("gates", "extra"), 1.0, "an undeclared gate was accepted"),
        (("workload", "long_class"), "short", "equal short and long classes were accepted"),
    ):
        broken = copy.deepcopy(policy_mapping)
        broken[path[0]][path[1]] = value
        try:
            admission.AdmissionPolicy.from_mapping(broken)
        except admission.AdmissionPolicyError:
            pass
        else:
            require(False, reason)

    long_tokens = 1024
    shorts = {"s1": 4.0, "s2": 8.0, "s3": 12.0, "s4": 16.0}
    longs = {"l1": 0.0, "l2": 0.0}
    solo_walls = {"l1": 49.0, "l2": 49.0, "s1": 1.0, "s2": 1.0, "s3": 2.0, "s4": 1.0}

    def tokens_for(name):
        count = long_tokens if name in longs else 10
        return [hash_token(name, index) for index in range(count)]

    def hash_token(name, index):
        return (sum(map(ord, name)) * 131 + index * 7) % 5000 + 10

    def events(first, count, gap, stall=None):
        times = [first + index * gap for index in range(count)]
        if stall is not None:
            at, seconds = stall
            times = [time if index < at else time + seconds for index, time in enumerate(times)]
        return times

    def root(kind, done, *, gap, stall=None, captures=1, identity=None, simulated=False, drop=None):
        requests = []
        for name, arrival in {**longs, **shorts}.items():
            if name in longs:
                first = arrival + 3.0
            elif kind == "continuous":
                first = arrival + 1.0
            else:
                first = done[name] - 1.0
            count = len(tokens_for(name))
            entry = {
                "name": name,
                "class": "long" if name in longs else "short",
                "arrival_seconds": arrival,
                "first_event_seconds": first,
                "done_seconds": done[name],
                "content_event_seconds": events(first, count, gap, stall if name == "l1" else None),
                "token_ids": tokens_for(name),
                "finish_reason": "length" if name in longs else "stop",
                "trace_captures": captures,
            }
            requests.append(entry)
        mapping = {
            "schema_version": 1,
            "root_kind": kind,
            "simulated": simulated,
            "identity": {
                "package": "/nix/store/package-qwen38",
                "execution_identity": "/nix/store/identity",
                "runtime_identity": "/nix/store/tt-metal",
                "physical_pcie_device_ids": [0, 1],
                "scheduler_profile_id": policy.scheduler_profile_id if kind == "continuous" else None,
                "scheduler_policy_blake3": "a" * 64 if kind == "continuous" else None,
                **(identity or {}),
            },
            "workload": {
                "workload_id": policy.workload_id,
                "blake3": "b" * 64,
                "requests": [{"name": r["name"], "class": r["class"]} for r in requests],
            },
            "startup": {"device_dram_free_bytes": [9_800_000_000, 9_800_000_000]},
            "solo": [
                {"name": name, "token_ids": tokens_for(name), "wall_seconds": wall}
                for name, wall in solo_walls.items()
                if name != drop
            ],
            "repetitions": [{"requests": copy.deepcopy(requests)} for _ in range(policy.repetitions)],
        }
        return mapping

    baseline_done = {"l1": 50.0, "l2": 50.0, "s1": 63.0, "s2": 63.0, "s3": 63.0, "s4": 63.0}
    candidate_done = {"l1": 58.0, "l2": 57.0, "s1": 6.0, "s2": 10.0, "s3": 17.0, "s4": 18.0}
    baseline = root("group", baseline_done, gap=0.045, captures=2)
    candidate = root("continuous", candidate_done, gap=0.05, stall=(100, 3.5), captures=1)

    def judge(candidate_mapping, baseline_mapping=baseline):
        return admission.evaluate(
            policy,
            admission.root_from_mapping(baseline_mapping, expected_kind="group"),
            admission.root_from_mapping(candidate_mapping, expected_kind="continuous"),
        )

    result = judge(candidate)
    require(result.admitted, f"a complete, better candidate was rejected: {result.reasons}")
    gate_names = {gate.name for gate in result.gates}
    require(not any("capture" in name for name in gate_names), "trace captures were gated")
    receipt = build_receipt(
        policy_mapping=policy_mapping,
        baseline_mapping=baseline,
        candidate_mapping=candidate,
        policy_blake3="c" * 64,
        baseline_blake3="d" * 64,
        candidate_blake3="e" * 64,
    )
    require(
        receipt["admission"]["admitted"]
        and receipt["admission"]["admitted_profile"]["scheduler_profile_id"] == policy.scheduler_profile_id
        and receipt["simulated"] is False,
        "the admitted receipt does not name the measured profile",
    )

    def rejected(mapping, reason, message, baseline_mapping=baseline):
        verdict = judge(mapping, baseline_mapping)
        require(not verdict.admitted and reason in verdict.reasons, f"{message}: {verdict.reasons}")

    # Fewer captures cannot admit a candidate whose short requests got no faster.
    rejected(
        root("continuous", {**baseline_done, "l1": 50.0, "l2": 50.0}, gap=0.05, captures=0),
        "gate-failed:short_median_wall_ratio",
        "fewer captures admitted a candidate without a latency benefit",
    )
    # Every matched pair must pass: one good pass cannot carry a bad one.
    mixed = root("continuous", candidate_done, gap=0.05, stall=(100, 3.5))
    mixed["repetitions"][1] = root("continuous", baseline_done, gap=0.05)["repetitions"][1]
    rejected(mixed, "gate-failed:short_median_wall_ratio", "a lucky pass admitted an unlucky one")
    # A request left far behind its solo time fails the starvation guard.
    rejected(
        root("continuous", {**candidate_done, "s1": 200.0}, gap=0.05, stall=(100, 3.5)),
        "gate-failed:slowdown_ratio",
        "a starved request was admitted",
    )
    # Higher delivered throughput because a request never finished.
    starved = root("continuous", candidate_done, gap=0.05, stall=(100, 3.5))
    starved["repetitions"][0]["requests"][4]["finish_reason"] = None
    starved_result = judge(starved)
    throughput_gate = next(g for g in starved_result.gates if g.name == "delivered_throughput_ratio")
    require(
        throughput_gate.passed
        and not starved_result.admitted
        and "gate-failed:all_completed" in starved_result.reasons,
        f"an unfinished request hid behind higher throughput: {starved_result.reasons}",
    )
    # Better aggregate numbers but a declared tail stall beyond its bound.
    rejected(
        root("continuous", candidate_done, gap=0.05, stall=(100, 8.0)),
        "gate-failed:inter_token_gap_seconds",
        "a stall beyond the declared tail bound was admitted",
    )
    # A candidate row that no longer matches its solo reference.
    lost = root("continuous", candidate_done, gap=0.05, stall=(100, 3.5))
    lost["repetitions"][1]["requests"][2]["token_ids"][3] += 1
    rejected(lost, "gate-failed:exact_candidate_parity", "a parity mismatch was admitted")
    # Identity drift, simulation, and a different workload.
    rejected(
        root("continuous", candidate_done, gap=0.05, identity={"package": "/nix/store/other"}),
        "identity-differs:package",
        "roots from different packages were compared",
    )
    rejected(
        root("continuous", candidate_done, gap=0.05, simulated=True),
        "simulated-root",
        "a simulated root was admitted",
    )
    other_workload = root("continuous", candidate_done, gap=0.05)
    other_workload["workload"]["blake3"] = "f" * 64
    rejected(other_workload, "workload-differs", "different workloads were compared")
    single_pass = root("continuous", candidate_done, gap=0.05)
    single_pass["repetitions"] = single_pass["repetitions"][:1]
    rejected(single_pass, "repetition-count:continuous", "one measured pass was accepted")
    # Incomplete roots fail with a specific evidence error, and the receipt keeps them.
    for broken, fragment in (
        (root("continuous", candidate_done, gap=0.05, drop="s3"), "solo references cover"),
        ({**candidate, "simulated": None}, "simulated"),
    ):
        try:
            admission.root_from_mapping(broken, expected_kind="continuous")
        except admission.AdmissionEvidenceError as error:
            require(fragment in str(error), f"unspecific evidence error: {error}")
        else:
            require(False, f"an incomplete root was accepted ({fragment})")
    missing_pass = copy.deepcopy(candidate)
    del missing_pass["repetitions"][0]["requests"][1]
    receipt = build_receipt(
        policy_mapping=policy_mapping,
        baseline_mapping=baseline,
        candidate_mapping=missing_pass,
        policy_blake3="c" * 64,
        baseline_blake3="d" * 64,
        candidate_blake3="e" * 64,
    )
    require(
        receipt["admission"]["admitted"] is False
        and "every declared request" in (receipt["admission"]["evidence_error"] or ""),
        f"an incomplete root disappeared from the receipt: {receipt['admission']}",
    )


# r[verify tenstorrent.qwen38_p150x2.hybrid_prefix_identity]
def probe_prefix_cache_policy():
    """Hybrid prefix keys bind tokens, identity, and tenant; the scheduler donates, hits,
    pins, evicts, and drops entries without ever lending a live request's blocks."""
    import copy
    import hashlib

    from qwen36_service import continuous_slot_core as core
    from qwen36_service import prefix_cache_core as prefix
    from qwen36_service.kv_pool_core import KvPoolLayout

    def digest(data):
        return hashlib.blake2b(data, digest_size=32).digest()

    with Path(__file__).with_name("prefix-cache-policy.json").open(encoding="utf-8") as handle:
        mapping = json.load(handle)
    policy = prefix.PrefixCachePolicy.from_mapping(mapping)
    require(policy.tenant == "local" and policy.state_slots >= 1, "the packaged policy changed shape")
    for change, reason in (
        ({"extra": 1}, "an undeclared policy field was accepted"),
        ({"minimum_prefix_tokens": 1500}, "a minimum prefix off the chunk boundary was accepted"),
        ({"state_slots": 0}, "a cache without state buffers was accepted"),
        ({"tenant": ""}, "an empty tenant scope was accepted"),
    ):
        try:
            prefix.PrefixCachePolicy.from_mapping({**mapping, **change})
        except prefix.PrefixCachePolicyError:
            pass
        else:
            require(False, reason)

    # Keys: exact token ids under one identity and tenant, prefixes strictly inside the prompt.
    tokens = list(range(5000))
    base = prefix.prefix_digests(tokens, identity="model-a", tenant="local", blake3_digest=digest)
    require(len(base) == 4, f"a 5,000-token prompt should have 4 chunk prefixes, got {len(base)}")
    require(
        prefix.prefix_digests(tokens[:4097], identity="model-a", tenant="local", blake3_digest=digest)
        == base,
        "the same prefixes of a longer prompt changed digest",
    )
    require(
        prefix.prefix_digests(tokens[:1024], identity="model-a", tenant="local", blake3_digest=digest) == (),
        "a prompt no longer than one chunk got a prefix digest",
    )
    drifted = tokens[:]
    drifted[1500] += 1
    drift = prefix.prefix_digests(drifted, identity="model-a", tenant="local", blake3_digest=digest)
    require(drift[0] == base[0] and all(a != b for a, b in zip(drift[1:], base[1:])),
            "a changed token id did not change every digest from its chunk on")
    for identity, tenant, reason in (
        ("model-b", "local", "another model or precision matched"),
        ("model-a", "other", "another tenant matched"),
    ):
        other = prefix.prefix_digests(tokens, identity=identity, tenant=tenant, blake3_digest=digest)
        require(not set(other) & set(base), reason)

    # Entries must name KV and state; a KV-only or mis-sized entry is invalid.
    good = prefix.PrefixEntry(1, "a" * 64, 2048, ((16, 32),), 0, True, 0, 0.0)
    good.validate(64, policy.state_slots)
    for broken, reason in (
        (copy.replace(good, block_ranges=()), "an entry without KV blocks was accepted"),
        (copy.replace(good, block_ranges=((16, 31),)), "an entry whose blocks miss part of its prefix was accepted"),
        (copy.replace(good, state_index=policy.state_slots), "an entry without a state buffer was accepted"),
        (copy.replace(good, prefix_tokens=2000), "an entry off the chunk boundary was accepted"),
    ):
        try:
            broken.validate(64, policy.state_slots)
        except prefix.PrefixCachePolicyError:
            pass
        else:
            require(False, reason)

    layout = KvPoolLayout(pool_blocks=4096, row_blocks=1024, kv_block_size=64)
    kv = core.KvBlockRange(layout.reserved_prefix_blocks, layout.scratch_block, layout.row_blocks, 64)
    with Path(__file__).with_name("continuous-slot-policy.json").open(encoding="utf-8") as handle:
        slot_policy = core.ContinuousSlotPolicy.from_mapping(json.load(handle))

    def request(name, prompt, *, max_tokens=64, identity="model-a", tenant="local", arrival=0.0):
        return core.SlotRequest(
            request_id=name,
            prompt_tokens=len(prompt),
            max_tokens=max_tokens,
            route="step",
            blocks=layout.request_blocks(prompt_tokens=len(prompt), generation_tokens=max_tokens),
            arrival_seconds=arrival,
            deadline_seconds=1000.0,
            prefix_digests=prefix.prefix_digests(prompt, identity=identity, tenant=tenant, blake3_digest=digest),
        )

    def boundary(schedule, now, *, use=policy, available=True):
        return core.plan_boundary(
            schedule, slot_policy, kv, now=now, prefix_policy=use, prefix_available=available
        )

    def run_to_completion(schedule, effects, now):
        for effect in effects:
            if isinstance(effect, core.PrefillSlot):
                schedule = core.observe_prefill(
                    schedule, slot=effect.slot, generation=effect.generation, token_id=5, stopped=True
                )
        return boundary(schedule, now)

    def prefills(effects):
        return [effect for effect in effects if isinstance(effect, core.PrefillSlot)]

    def entry_blocks(schedule):
        return prefix.cached_blocks(schedule.prefix_entries)

    # Turn 1 of a conversation donates its longest prefix; retiring keeps those blocks.
    turn1 = list(range(100, 100 + 5000))
    schedule = core.initial_schedule(slot_policy)
    schedule, _ = core.submit(schedule, slot_policy, kv, request("t1", turn1))
    schedule, effects = boundary(schedule, 1.0)
    (p1,) = prefills(effects)
    require(p1.save_at == 4096 and p1.save_state is not None and p1.resume_state is None,
            f"turn 1 did not plan its donation: {p1}")
    require(not schedule.prefix_entries[0].ready, "a donation was ready before its prefill ran")
    schedule, effects = run_to_completion(schedule, effects, 2.0)
    (entry1,) = schedule.prefix_entries
    require(entry1.ready and entry1.prefix_tokens == 4096 and entry1.pins == 0, f"turn 1's entry: {entry1}")
    require(entry_blocks(schedule) == frozenset(range(p1.block_start, p1.block_start + 64)),
            "the entry did not keep exactly the donor's first 64 blocks")

    # Turn 2 extends turn 1: it resumes from the entry, owns blocks only past the prefix,
    # and donates a longer chained entry.
    turn2 = turn1 + list(range(9000, 9000 + 3000))
    schedule, _ = core.submit(schedule, slot_policy, kv, request("t2", turn2))
    schedule, effects = boundary(schedule, 3.0)
    (p2,) = prefills(effects)
    require(
        p2.hit_entry == entry1.entry_id
        and p2.prefix_tokens == 4096
        and p2.resume_state == entry1.state_index
        and p2.prefix_ranges == entry1.block_ranges,
        f"turn 2 did not resume from turn 1's entry: {p2}",
    )
    full = layout.request_blocks(prompt_tokens=len(turn2), generation_tokens=64)
    require(p2.block_count == full - 64, "a hit still reserved KV for the cached prefix")
    require(
        not set(range(p2.block_start, p2.block_start + p2.block_count))
        & prefix.cached_blocks([entry1]),
        "a live request's own run overlaps the cached prefix it reads",
    )
    require(p2.save_at == 7168 and p2.save_state == entry1.state_index, f"turn 2's donation: {p2}")
    (chained,) = schedule.prefix_entries
    require(chained.entry_id == p2.donation_entry, "turn 2's entry did not supersede turn 1's")
    require(chained.block_ranges == entry1.block_ranges + ((p2.block_start, 48),),
            f"the superseding entry does not reuse the prefix blocks: {chained.block_ranges}")
    require(chained.pins == 1, "the superseding entry did not take over its reader's pin")
    schedule, effects = run_to_completion(schedule, effects, 4.0)
    require(all(entry.pins == 0 for entry in schedule.prefix_entries), "retirement left an entry pinned")
    # Turn 3 resumes from the longest cached prefix, the chained one.
    turn3 = turn2 + list(range(30000, 30000 + 1500))
    turn3_schedule, _ = core.submit(schedule, slot_policy, kv, request("t3", turn3))
    turn3_schedule, effects = boundary(turn3_schedule, 4.5)
    require(prefills(effects)[0].hit_entry == chained.entry_id and prefills(effects)[0].prefix_tokens == 7168,
            f"turn 3 did not resume from the longest prefix: {prefills(effects)[0]}")

    # Another tenant or identity never hits.
    for name, identity, tenant in (("other-tenant", "model-a", "other"), ("other-model", "model-b", "local")):
        probe_schedule, _ = core.submit(schedule, slot_policy, kv, request(name, turn2, identity=identity, tenant=tenant))
        probe_schedule, effects = boundary(probe_schedule, 5.0)
        require(prefills(effects)[0].hit_entry is None, f"{name} hit another scope's entry")

    # No prefix plans while the device cannot restore or save.
    unavailable, _ = core.submit(schedule, slot_policy, kv, request("t3", turn2 + [1] * 2000))
    unavailable, effects = boundary(unavailable, 5.0, available=False)
    (p3,) = prefills(effects)
    require(p3.hit_entry is None and p3.save_state is None, "a prefix was planned while unavailable")

    # State buffers run out: the least recently used unpinned entry is evicted.
    small = copy.replace(policy, state_slots=1)
    schedule = core.initial_schedule(slot_policy)
    for index, now in ((0, 1.0), (1, 3.0)):
        prompt = list(range(20000 * (index + 1), 20000 * (index + 1) + 3000))
        schedule, _ = core.submit(schedule, slot_policy, kv, request(f"s{index}", prompt))
        schedule, effects = boundary(schedule, now, use=small)
        schedule, _ = run_to_completion(schedule, effects, now + 1)
    require(len(schedule.prefix_entries) == 1 and schedule.prefix_entries[0].prefix_tokens == 2048,
            "a donation beyond the state buffers did not evict the older entry")
    require(entry_blocks(schedule) == frozenset(expand for expand in prefix.expand_blocks(schedule.prefix_entries[0].block_ranges)),
            "an evicted entry kept its blocks")

    # The KV budget evicts, and a pinned entry is never evicted.
    tight = copy.replace(policy, maximum_cached_blocks=40)
    schedule = core.initial_schedule(slot_policy)
    first = list(range(3000))
    schedule, _ = core.submit(schedule, slot_policy, kv, request("b0", first))
    schedule, effects = boundary(schedule, 1.0, use=tight)
    schedule, _ = run_to_completion(schedule, effects, 2.0)
    reader = first + list(range(70000, 70000 + 2000))
    schedule, _ = core.submit(schedule, slot_policy, kv, request("reader", reader, max_tokens=500))
    schedule, effects = boundary(schedule, 3.0, use=tight)
    (hit,) = prefills(effects)
    require(hit.hit_entry is not None and hit.save_state is None,
            "a donation over the block budget was planned while the only victim was pinned")
    schedule = core.observe_prefill(schedule, slot=hit.slot, generation=hit.generation, token_id=5, stopped=False)
    before = schedule.prefix_entries
    schedule, _ = core.submit(schedule, slot_policy, kv, request("other", list(range(90000, 93000))))
    schedule, effects = boundary(schedule, 4.0, use=tight)
    require(any(entry.entry_id == hit.hit_entry for entry in schedule.prefix_entries),
            "a pinned entry was evicted")
    require(schedule.prefix_entries == before, "an over-budget donation displaced a pinned entry")

    # A live request takes an unused entry's blocks back when nothing else fits.
    small_kv = core.KvBlockRange(16, 16 + layout.row_blocks + 20, layout.row_blocks, 64)

    def small_boundary(schedule, now):
        return core.plan_boundary(
            schedule, slot_policy, small_kv, now=now, prefix_policy=policy, prefix_available=True
        )

    crowded = core.initial_schedule(slot_policy)
    crowded, _ = core.submit(crowded, slot_policy, small_kv, request("seed", list(range(3000))))
    crowded, effects = small_boundary(crowded, 1.0)
    for effect in prefills(effects):
        crowded = core.observe_prefill(
            crowded, slot=effect.slot, generation=effect.generation, token_id=5, stopped=True
        )
    crowded, _ = small_boundary(crowded, 2.0)
    held = entry_blocks(crowded)
    (seed_entry,) = crowded.prefix_entries
    require(len(held) == 32, f"the seed entry should hold 32 blocks, held {len(held)}")
    long_max = layout.row_blocks * 64 - 60000
    crowded, _ = core.submit(crowded, slot_policy, small_kv, request("big", [7] * 60000, max_tokens=long_max))
    crowded, effects = small_boundary(crowded, 3.0)
    (admitted,) = prefills(effects)
    run = set(range(admitted.block_start, admitted.block_start + admitted.block_count))
    require(admitted.block_count == layout.row_blocks, "the full-length request was not admitted whole")
    require(
        all(entry.entry_id != seed_entry.entry_id for entry in crowded.prefix_entries) and run & held,
        "an unused entry kept blocks a live request needed",
    )

    # Reads and failures drop entries; a pinned entry survives invalidation.
    schedule2 = core.invalidate_prefix_cache(schedule)
    require(all(entry.pins for entry in schedule2.prefix_entries), "invalidation kept an unpinned entry")
    failed, _ = core.quarantine_group(schedule)
    require(failed.prefix_entries == (), "a device failure kept cached prefixes")


# r[verify tenstorrent.qwen38_p150x2.hybrid_prefix_state]
def probe_prefix_cache_restore():
    """Hybrid restore: the adapter composes shared prefix blocks with the request's own run
    and plans resume/save; a multi-turn conversation served from cached prefixes decodes
    exactly what cold prefill decodes, including across eviction."""
    import hashlib
    import threading
    import time

    import torch
    from qwen36_service import batch_decode_adapter as adapter_module
    from qwen36_service import continuous_slot_core as core
    from qwen36_service.batch_decode_adapter import (
        BatchDecodeAdapterError,
        BatchedTracedDecodeAdapter,
        SlotRefillResult,
    )
    from qwen36_service.continuous_slot_service import ContinuousSlotCoordinator
    from qwen36_service.core import CompletionRequest, Endpoint
    from qwen36_service.kv_pool_core import KvPoolError, KvPoolLayout
    from qwen36_service.prefix_cache_core import PrefixCachePolicy, expand_blocks

    # Part 1: the real adapter's row composition and plans, on fake TTNN.
    plans = []

    class FakeModel:
        def __init__(self):
            self.layers = [SimpleNamespace(is_full_attention=False, attention=SimpleNamespace(_decode_fusion_mode="step"))]
            self.buckets = (1024,)
            self.rows = []

        def masked_prefill_traces_stale(self):
            return False

        def masked_prefill_trace_buckets(self):
            return self.buckets

        def release_masked_prefill_traces(self):
            self.buckets = ()

        def prefix_cache_state_count(self):
            return 2

        def prefill_paged_slots(self, tokens, page_table, slots, valid_lens, first_token_callback, masked_traces, prefixes=None):
            self.rows.append(page_table.clone())
            plans.append(prefixes)
            logits = torch.zeros((1, 1, 64))
            logits[0, 0, 7] = 1.0
            first_token_callback([logits])
            return [logits]

    layout = KvPoolLayout(pool_blocks=256, row_blocks=128, kv_block_size=64)
    model = FakeModel()
    adapter = object.__new__(BatchedTracedDecodeAdapter)
    adapter.model = model
    adapter.profile = SimpleNamespace(maximum_sequence_length=8192, kv_block_size=64)
    adapter.maximum_width = 4
    adapter.mesh = SimpleNamespace(num_program_cache_entries=lambda: 1)
    adapter.vocabulary_size = 64
    adapter.blocks_per_user = layout.row_blocks
    adapter.kv_pool = layout
    adapter.page_table = torch.zeros((4, layout.row_blocks), dtype=torch.int32)
    adapter._traces = {}
    adapter._pending_trace_plan = None
    adapter._prior_reset_complete = True
    adapter._closed = False
    adapter._slot_width = None
    adapter._slot_runs = {}
    adapter._slot_inputs_stale = False
    adapter.slot_trace_captures = 0
    adapter.prefix_states = 2
    adapter.reset_all = lambda: None
    original_ttnn = adapter_module.ttnn
    adapter_module.ttnn = SimpleNamespace(synchronize_device=lambda _mesh: None, release_trace=lambda *_: None, deallocate=lambda _t: None)
    scratch = layout.scratch_block

    def prompt(length):
        return torch.ones((1, length), dtype=torch.int64)

    def rejects(reason, **fields):
        before = (adapter.page_table.clone(), dict(adapter._slot_runs), len(plans))
        try:
            adapter.refill_slot(**fields)
        except BatchDecodeAdapterError:
            pass
        else:
            require(False, reason)
        require(
            torch.equal(before[0], adapter.page_table) and before[1] == adapter._slot_runs and before[2] == len(plans),
            f"a rejected prefix refill changed adapter state: {reason}",
        )

    try:
        adapter.begin_slots(width=4, route="step")
        require(adapter.prefix_cache_available(), "the fake adapter did not offer the prefix cache")
        adapter.refill_slot(slot=0, token_ids=prompt(3000), block_start=16, block_count=48, save_state=0, save_at=2048)
        require(plans[-1] == [{"save": (0, 2048)}], f"the donation plan did not reach the model: {plans[-1]}")
        adapter.refill_slot(
            slot=1,
            token_ids=prompt(5000),
            block_start=64,
            block_count=48,
            prefix_ranges=((16, 32),),
            prefix_tokens=2048,
            resume_state=0,
            save_state=1,
            save_at=4096,
        )
        require(plans[-1] == [{"resume": (0, 2048), "save": (1, 4096)}], f"the resume plan: {plans[-1]}")
        row = model.rows[-1][0].tolist()
        require(row[:32] == list(range(16, 48)) and row[32:80] == list(range(64, 112)) and set(row[80:]) == {scratch},
                "the page-table row is not prefix blocks, then the own run, then scratch")
        # A second reader shares the same prefix blocks.
        adapter.refill_slot(slot=2, token_ids=prompt(3000), block_start=112, block_count=16,
                            prefix_ranges=((16, 32),), prefix_tokens=2048, resume_state=0)
        require(plans[-1] == [{"resume": (0, 2048)}], "a shared prefix was refused")
        rejects("an own run overlapping its prefix was accepted", slot=3, token_ids=prompt(3000), block_start=40,
                block_count=16, prefix_ranges=((16, 32),), prefix_tokens=2048, resume_state=0)
        rejects("prefix blocks that do not cover the prefix were accepted", slot=3, token_ids=prompt(3000),
                block_start=128, block_count=16, prefix_ranges=((16, 31),), prefix_tokens=2048, resume_state=0)
        rejects("a resume without a prefix was accepted", slot=3, token_ids=prompt(3000), block_start=128,
                block_count=48, resume_state=0)
        rejects("a prefix covering the whole prompt was accepted", slot=3, token_ids=prompt(2048), block_start=128,
                block_count=16, prefix_ranges=((16, 32),), prefix_tokens=2048, resume_state=0)
        rejects("a save point inside the prefix was accepted", slot=3, token_ids=prompt(3000), block_start=128,
                block_count=16, prefix_ranges=((16, 32),), prefix_tokens=2048, resume_state=0, save_state=1, save_at=2048)
        rejects("a run too short for the rest of the prompt was accepted", slot=3, token_ids=prompt(5000),
                block_start=128, block_count=8, prefix_ranges=((16, 32),), prefix_tokens=2048, resume_state=0)
        model.buckets = ()
        rejects("a prefix plan ran without the masked traces", slot=3, token_ids=prompt(3000), block_start=128,
                block_count=16, prefix_ranges=((16, 32),), prefix_tokens=2048, resume_state=0)
        model.buckets = (1024,)
        adapter.refill_slot(slot=3, token_ids=prompt(100), block_start=128, block_count=16)
        require(plans[-1] is None, "a cold refill passed a prefix plan")
    finally:
        adapter_module.ttnn = original_ttnn

    # Part 2: a conversation through the shell on a fake device that checks every restore.
    def token_after(sequence):
        return 10 + int(hashlib.blake2b(bytes(str(sequence[-64:]) + str(len(sequence)), "utf-8"), digest_size=4).hexdigest(), 16) % 1000

    class FakePrefixDevice:
        """Row-independent fake device with a KV store and saved states it verifies."""

        maximum_width = 16

        def __init__(self, states):
            self.kv_pool = KvPoolLayout(pool_blocks=4096, row_blocks=1024, kv_block_size=64)
            self.fusion_mode = "step"
            self.prefix_states = states
            self.slot_trace_captures = 0
            self.kv = {}
            self.saved = {}
            self.owned = {}
            self.width = None
            self.restores = 0
            self.lock = threading.Lock()

        def prefix_cache_available(self):
            return self.prefix_states > 0

        def request_blocks(self, *, prompt_tokens, generation_tokens):
            try:
                return self.kv_pool.request_blocks(prompt_tokens=prompt_tokens, generation_tokens=generation_tokens)
            except KvPoolError as error:
                raise BatchDecodeAdapterError(str(error)) from error

        def begin_slots(self, *, width, route):
            self.width = width
            self.owned = {}

        def refill_slot(self, *, slot, token_ids, block_start, block_count, prefix_ranges=(), prefix_tokens=0,
                        resume_state=None, save_state=None, save_at=0):
            sequence = tuple(int(token) for token in token_ids[0].tolist())
            row = expand_blocks(prefix_ranges) + list(range(block_start, block_start + block_count))
            for other in self.owned.values():
                if set(row[len(expand_blocks(prefix_ranges)):]) & set(other["own"]):
                    raise BatchDecodeAdapterError("own KV runs overlap")
            if resume_state is not None:
                if self.saved.get(resume_state) != sequence[:prefix_tokens]:
                    raise BatchDecodeAdapterError("the restored state belongs to another prefix")
                for index in range(prefix_tokens // 64):
                    if self.kv.get(row[index]) != sequence[index * 64 : (index + 1) * 64]:
                        raise BatchDecodeAdapterError("a cached prefix block was overwritten")
                self.restores += 1
            extent = -(-len(sequence) // 1024) * 1024
            for index in range(prefix_tokens // 64, extent // 64):
                self.kv[row[index]] = sequence[index * 64 : (index + 1) * 64]
            if save_state is not None:
                self.saved[save_state] = sequence[:save_at]
            first = token_after(sequence)
            self.owned[slot] = {"sequence": list(sequence) + [first], "row": row,
                                "own": list(range(block_start, block_start + block_count))}
            return SlotRefillResult(first, 0.002, 0.001, True, False, False)

        def retire_slot(self, *, slot):
            self.owned.pop(slot)

        def abandon_slots(self):
            self.width = None
            self.owned = {}

        def decode_slots_greedy(self, *, token_ids, positions, steps):
            def stream():
                for _ in range(steps):
                    tokens = []
                    for row in range(self.width):
                        state = self.owned.get(row)
                        if state is None:
                            tokens.append(0)
                            continue
                        position = len(state["sequence"]) - 1
                        self.kv[state["row"][position // 64]] = ("decoded", row, position)
                        token = token_after(state["sequence"])
                        state["sequence"].append(token)
                        tokens.append(token)
                    yield tuple(tokens)

            return stream()

    class FakeTokenizer:
        @staticmethod
        def decode(tokens, skip_special_tokens=True):
            return " ".join(str(token) for token in tokens)

    with Path(__file__).with_name("continuous-slot-policy.json").open(encoding="utf-8") as handle:
        slot_policy = core.ContinuousSlotPolicy.from_mapping(json.load(handle))
    with Path(__file__).with_name("prefix-cache-policy.json").open(encoding="utf-8") as handle:
        prefix_mapping = json.load(handle)

    def serve(conversations, *, states):
        device = FakePrefixDevice(states)
        prefix_policy = (
            PrefixCachePolicy.from_mapping({**prefix_mapping, "state_slots": states}) if states else None
        )
        coordinator = ContinuousSlotCoordinator(
            slot_policy=slot_policy,
            prefix_policy=prefix_policy,
            prefix_identity="fake-identity" if states else "",
            blake3_digest=(lambda data: hashlib.blake2b(data, digest_size=32).digest()) if states else None,
            adapter=device,
            tokenizer=FakeTokenizer(),
            model=SimpleNamespace(args=SimpleNamespace(vocab_size=2048)),
            profile=SimpleNamespace(profile_id="decode-test"),
            model_alias="model-test",
            model_identity="model-identity-test",
            package_identity="package-test",
            runtime_identity="runtime-test",
            maximum_sequence_length=65536,
            physical_pcie_device_ids=(0, 1),
            emitter=lambda *_args, **_fields: None,
            stop_token_ids=frozenset(),
        )
        with coordinator._condition:
            coordinator._accepting_requests = True
        history = {name: list(start) for name, start in conversations["starts"].items()}
        results = []
        try:
            for name, extra in conversations["turns"]:
                sequence = history[name] + extra
                started = time.monotonic()
                response = coordinator.generate(
                    request=CompletionRequest(endpoint=Endpoint.COMPLETIONS, model="m", max_tokens=12, prompt="x", messages=()),
                    request_id=f"{name}-{len(results)}",
                    token_ids=torch.tensor([sequence], dtype=torch.int64),
                    prompt_token_count=len(sequence),
                    started=started,
                    deadline=started + 60,
                )
                evidence = response["x_onix_decode_evidence"]
                history[name] = sequence + evidence["generated_token_ids"]
                results.append((name, evidence["generated_token_ids"], evidence["prefix_hit_tokens"], evidence["prefix_saved_tokens"]))
        finally:
            coordinator.close()
        return results, device

    conversation = {
        "starts": {"a": list(range(100, 3600)), "b": list(range(5000, 7600))},
        "turns": [("a", []), ("a", list(range(20000, 21800))), ("a", list(range(22000, 23500)))],
    }
    cold, _ = serve(conversation, states=0)
    warm, device = serve(conversation, states=4)
    require([r[1] for r in warm] == [r[1] for r in cold], "cached prefixes changed a conversation's tokens")
    require([r[2] for r in warm] == [0, 3072, 5120], f"the turns did not resume from their prefixes: {warm}")
    require([r[3] for r in warm] == [3072, 5120, 6144], f"the turns did not save their prefixes: {warm}")
    require(device.restores == 2, "the device did not restore two prefixes")
    require(all(r[2] == 0 and r[3] == 0 for r in cold), "the cold path used the cache")

    interleaved = {
        "starts": conversation["starts"],
        "turns": [("a", []), ("b", []), ("a", list(range(20000, 21800))), ("b", list(range(30000, 31200)))],
    }
    cold, _ = serve(interleaved, states=0)
    evicting, device = serve(interleaved, states=1)
    require([r[1] for r in evicting] == [r[1] for r in cold], "eviction changed a conversation's tokens")
    require([r[2] for r in evicting] == [0, 0, 0, 0], f"an evicted prefix was resumed: {evicting}")
    roomy, device = serve(interleaved, states=2)
    require([r[1] for r in roomy] == [r[1] for r in cold], "two cached conversations changed tokens")
    require([r[2] for r in roomy] == [0, 0, 3072, 2048], f"interleaved turns did not resume: {roomy}")


# r[verify tenstorrent.qwen38_p150x2.hybrid_prefix_admission]
def probe_prefix_cache_admission():
    """Prefix-cache admission: faster resumed turns with exact parity admit; a parity loss,
    no benefit, costly misses, a memory overrun, a cache in the cold root, or a missing
    turn each reject with its own reason."""
    import copy

    from qwen36_service import prefix_cache_admission_core as admission
    from qwen36_service.prefix_cache_receipt import build_receipt

    with Path(__file__).with_name("prefix-cache-admission-policy.json").open(encoding="utf-8") as handle:
        policy_mapping = json.load(handle)
    policy = admission.PrefixAdmissionPolicy.from_mapping(policy_mapping)
    broken = copy.deepcopy(policy_mapping)
    broken["gates"]["require_exact_parity"] = False
    try:
        admission.PrefixAdmissionPolicy.from_mapping(broken)
    except admission.PrefixAdmissionPolicyError:
        pass
    else:
        require(False, "an admission without exact parity was accepted")

    def root(kind, conversations, turns, *, ttft, wall, hits=True, dram=9_900_000_000, profile=None, tokens=None):
        requests = []
        for turn in range(turns):
            for conversation in range(conversations):
                hit = kind == "cached" and hits and turn > 0
                requests.append(
                    {
                        "conversation": conversation,
                        "turn": turn,
                        "token_ids": (tokens or {}).get((conversation, turn), [conversation, turn, 7]),
                        "ttft_seconds": ttft(turn, hit),
                        "wall_seconds": wall(turn, hit),
                        "prefix_hit_tokens": 3072 if hit else 0,
                    }
                )
        return {
            "schema_version": 1,
            "root_kind": kind,
            "simulated": False,
            "identity": {
                "package": "/nix/store/p", "execution_identity": "/nix/store/e", "runtime_identity": "/nix/store/r",
                "physical_pcie_device_ids": [0, 1],
                "prefix_cache_profile_id": profile if profile is not None else (policy.prefix_cache_profile_id if kind == "cached" else None),
                "prefix_policy_blake3": "a" * 64 if kind == "cached" else None,
            },
            "workload": {"workload_id": "w", "blake3": "b" * 64, "conversations": conversations, "turns": turns},
            "startup": {"device_dram_free_bytes": [dram]},
            "requests": requests,
        }

    def cold_timing(turn, hit):
        return 3.0 + turn

    def fast_ttft(turn, hit):
        return 1.0 if hit else 3.0 + turn

    def fast_wall(turn, hit):
        return 3.0 if hit else 5.0 + turn

    def cold_wall(turn, hit):
        return 5.0 + turn

    def pairs(**cached_options):
        result = []
        for conversations, turns in policy.workloads:
            evicting = conversations > 4
            cold = root("cold", conversations, turns, ttft=cold_timing, wall=cold_wall)
            options = {"ttft": fast_ttft, "wall": fast_wall, "hits": not evicting, **cached_options}
            result.append((cold, root("cached", conversations, turns, **options)))
        return result

    def judge(pair_mappings):
        return admission.evaluate(
            policy,
            [
                (admission.root_from_mapping(cold, expected_kind="cold"), admission.root_from_mapping(cached, expected_kind="cached"))
                for cold, cached in pair_mappings
            ],
        )

    result = judge(pairs())
    require(result.admitted and result.hit_turns == 9 and result.miss_turns == 13,
            f"a faster exact cache was rejected: {result.reasons} ({result.hit_turns} hits, {result.miss_turns} misses)")
    receipt = build_receipt(
        policy_mapping=policy_mapping,
        pair_mappings=pairs(),
        policy_blake3="c" * 64,
        root_blake3s=[("d" * 64, "e" * 64)] * len(policy.workloads),
    )
    require(receipt["admission"]["admitted"] and receipt["admission"]["admitted_profile"]["prefix_cache_profile_id"]
            == policy.prefix_cache_profile_id, "the receipt does not name the measured cache profile")

    def rejected(pair_mappings, reason, message):
        verdict = judge(pair_mappings)
        require(not verdict.admitted and reason in verdict.reasons, f"{message}: {verdict.reasons}")

    lost = pairs()
    lost[0][1]["requests"][5]["token_ids"] = [9, 9, 9]
    rejected(lost, "gate-failed:exact_parity", "a changed continuation was admitted")
    rejected(pairs(ttft=cold_timing, wall=cold_wall), "gate-failed:hit_median_ttft_ratio",
             "resumed turns without a speedup were admitted")
    rejected(pairs(wall=lambda turn, hit: 3.0 if hit else 8.0 + turn), "gate-failed:miss_wall_ratio",
             "misses much slower than cold prefill were admitted")
    rejected(pairs(dram=9_900_000_000 - 600 * 2**20), "gate-failed:startup_dram_drop_bytes",
             "a memory overrun was admitted")
    no_hits = pairs(hits=False)
    try:
        judge(no_hits)
    except admission.PrefixAdmissionEvidenceError as error:
        require("no turn resumed" in str(error), f"unspecific error: {error}")
    else:
        require(False, "a cache that never resumed was evaluated")
    cold_with_cache = pairs()
    cold_with_cache[0][0]["identity"]["prefix_cache_profile_id"] = policy.prefix_cache_profile_id
    rejected(cold_with_cache, "cold-root-used-a-cache", "a cold root with a cache was accepted")
    missing = pairs()
    del missing[1][1]["requests"][3]
    receipt = build_receipt(
        policy_mapping=policy_mapping,
        pair_mappings=missing,
        policy_blake3="c" * 64,
        root_blake3s=[("d" * 64, "e" * 64)] * len(policy.workloads),
    )
    require(receipt["admission"]["admitted"] is False and "every declared turn" in (receipt["admission"]["evidence_error"] or ""),
            f"a missing turn disappeared from the receipt: {receipt['admission']}")


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

    from unittest import mock

    import qwen36_service.batch_decode_adapter as batch_decode_adapter_module
    import torch

    resident_trace = SimpleNamespace(device_inputs=("tokens", "positions", "rope", "page-table"))
    adapter.vocabulary_size = test_vocabulary_size
    adapter.profile = SimpleNamespace(maximum_sequence_length=test_sequence_length)
    adapter._traces = {mixed_user_count: resident_trace}
    adapter._pending_trace_plan = reuse_plan
    adapter._binding_matches = lambda trace: trace is resident_trace
    # The current group's rows, as prefill_users leaves them before trace preparation.
    adapter.page_table = torch.arange(MAXIMUM_ACTIVE_BATCH_WIDTH * 4, dtype=torch.int32).reshape(
        MAXIMUM_ACTIVE_BATCH_WIDTH, 4
    )
    adapter.model = SimpleNamespace(
        prepare_decode_inputs_host=lambda tokens, positions, page_table: (
            tokens,
            positions,
            "rope",
            page_table,
        )
    )
    restaged = []
    with mock.patch.object(
        batch_decode_adapter_module,
        "copy_host_to_device",
        lambda host, device_tensors: restaged.append((tuple(host), tuple(device_tensors))),
    ):
        reused_preparation = adapter.prepare_width(
            token_ids=tuple(user.token_id for user in mixed_users),
            positions=tuple(user.position for user in mixed_users),
        )
    require(
        len(restaged) == 1
        and restaged[0][1] == ("page-table",)
        and torch.equal(restaged[0][0][0], adapter.page_table[:mixed_user_count]),
        "a reused decode trace kept the previous group's page table",
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

    # r[verify tenstorrent.qwen38_p150x2.batch_b16_admission]
    b16_profile_path = Path(__file__).with_name("batch-admission-b16-profile.json")
    with b16_profile_path.open(encoding="utf-8") as b16_profile_file:
        b16_profile_mapping = json.load(b16_profile_file)
    b16_profile = BatchAdmissionProfile.from_mapping(b16_profile_mapping)
    require(b16_profile.candidate_width == 16, "B=16 profile width changed")
    require(
        len(b16_profile.prompts) == 16,
        "B=16 profile does not carry sixteen prompts",
    )
    require(
        b16_profile.minimum_aggregate_decode_speedup_ratio == 6.0
        and b16_profile.maximum_ttft_ratio == 9.0
        and b16_profile.minimum_per_user_decode_retention_ratio == 0.75
        and b16_profile.maximum_tail_latency_ratio == 1.5,
        "B=16 profile bounds drifted from the shared policy",
    )
    b16_admission_mode, b16_admission_eligible = b16_profile.measurement(
        enabled=True,
        width=16,
    )
    require(
        b16_admission_mode == ADMISSION_MEASUREMENT_MODE and b16_admission_eligible,
        "explicit B=16 measurement did not become eligible",
    )
    b16_single_mode, b16_single_eligible = b16_profile.measurement(
        enabled=True,
        width=1,
    )
    require(
        b16_single_mode == ADMISSION_MEASUREMENT_MODE and b16_single_eligible,
        "B=16 single-stream width lost eligibility",
    )
    for invalid_b16_width in (4, 8, 17):
        try:
            b16_profile.measurement(enabled=True, width=invalid_b16_width)
        except BatchAdmissionPolicyError as error:
            require(
                "outside the profile" in str(error),
                f"wrong B=16 width diagnostic: {error}",
            )
        else:
            require(False, f"invalid B=16 width was accepted: {invalid_b16_width!r}")

    invalid_b16_mappings = []
    b16_unsupported_width = json.loads(json.dumps(b16_profile_mapping))
    b16_unsupported_width["workload"]["candidate_width"] = 32
    invalid_b16_mappings.append((b16_unsupported_width, "workload widths"))
    b16_admitted_without_receipt = json.loads(json.dumps(b16_profile_mapping))
    b16_admitted_without_receipt["physically_admitted"] = True
    b16_admitted_without_receipt["identities"]["receipt_blake3"] = "not-measured"
    invalid_b16_mappings.append((b16_admitted_without_receipt, "no measured receipt"))
    for invalid_b16_mapping, diagnostic in invalid_b16_mappings:
        try:
            BatchAdmissionProfile.from_mapping(invalid_b16_mapping)
        except BatchAdmissionPolicyError as error:
            require(
                diagnostic in str(error),
                f"wrong B=16 profile diagnostic: {error}",
            )
        else:
            require(False, f"invalid B=16 profile was accepted: {diagnostic}")

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

        def prepare_masked_prefill_traces(self, warm_token_id, routes):
            return ()

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


# A compact chat template in the model's format (chat_template.jinja): tools in the system
# turn, history calls through `arguments|items`, tool results as user turns, and the
# thinking switch on the generation prompt. The build sandbox has no model directory.
FIXTURE_CHAT_TEMPLATE = (
    "{%- if tools %}{{- '<|im_start|>system\\n<tools>' }}"
    "{%- for tool in tools %}{{- '\\n' + tool | tojson }}{%- endfor %}"
    "{{- '\\n</tools><|im_end|>\\n' }}{%- endif %}"
    "{%- for message in messages %}"
    "{%- if message.content == 'template error' %}"
    "{{- raise_exception('the fixture rejects this message') }}{%- endif %}"
    "{%- if message.role == 'assistant' %}{{- '<|im_start|>assistant\\n' }}"
    "{%- if message.reasoning_content is string %}"
    "{{- '<think>\\n' + message.reasoning_content + '\\n</think>\\n\\n' }}{%- endif %}"
    "{{- message.content }}"
    "{%- for call in message.tool_calls or [] %}"
    "{{- '<tool_call>\\n<function=' + call.function.name + '>\\n' }}"
    "{%- for name, value in call.function.arguments|items %}"
    "{{- '<parameter=' + name + '>\\n' + (value if value is string else value | tojson)"
    " + '\\n</parameter>\\n' }}{%- endfor %}"
    "{{- '</function>\\n</tool_call>' }}{%- endfor %}"
    "{{- '<|im_end|>\\n' }}"
    "{%- elif message.role == 'tool' %}"
    "{{- '<|im_start|>user\\n<tool_response>\\n' + message.content + '\\n</tool_response><|im_end|>\\n' }}"
    "{%- else %}{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- endif %}{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n<think>\\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is false %}{{- '\\n</think>\\n\\n' }}"
    "{%- endif %}{%- endif %}"
)

READ_TOOL = {
    "function": {
        "description": "Read a file.",
        "name": "read",
        "parameters": {
            "properties": {
                "limit": {"type": "integer"},
                "path": {"type": "string"},
                "state": {"anyOf": [{"type": "object"}, {"type": "null"}]},
            },
            "required": ["path"],
            "type": "object",
        },
    },
    "type": "function",
}
TOOL_CALL_OUTPUT = (
    "reasoning\n</think>\n\n<tool_call>\n<function=read>\n<parameter=path>\n/etc/hosts\n"
    "</parameter>\n</function>\n</tool_call>"
)


class FixtureTokenizer:
    """`apply_chat_template` through transformers' own Jinja environment and the fixture."""

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False, tokenize=False, **kwargs):
        from transformers.utils.chat_template_utils import render_jinja_template

        assert not tokenize
        rendered, _ = render_jinja_template(
            [messages],
            tools=tools,
            chat_template=FIXTURE_CHAT_TEMPLATE,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
        return rendered[0]


def probe_service_contract():
    """Positive and negative tests for the pure HTTP request policy."""
    from qwen36_service.core import (
        Endpoint,
        RequestValidationError,
        admit_prompt,
        build_completion_response,
        parse_completion_request,
        parse_endpoint,
        template_arguments,
        template_messages,
    )

    admitted_model = "Qwen3.8-27B"
    maximum_generation_tokens = 64
    requested_generation_tokens = 4
    maximum_sequence_length = 2048

    def parse(payload, endpoint=Endpoint.CHAT_COMPLETIONS):
        return parse_completion_request(payload, endpoint, admitted_model, maximum_generation_tokens)

    require(
        parse_endpoint("/v1/completions") is Endpoint.COMPLETIONS,
        "completion route did not resolve",
    )
    require(
        parse_endpoint("/v1/chat/completions") is Endpoint.CHAT_COMPLETIONS,
        "chat-completion route did not resolve",
    )

    completion = parse(
        {
            "max_tokens": requested_generation_tokens,
            "model": admitted_model,
            "prompt": "The capital of France is",
            "temperature": 0,
        },
        Endpoint.COMPLETIONS,
    )
    require(
        completion.prompt == "The capital of France is", "completion prompt changed"
    )
    require(
        completion.max_tokens == requested_generation_tokens,
        "completion max_tokens changed",
    )
    admitted_completion = admit_prompt(completion, completion.prompt, 5, maximum_sequence_length)
    text_response = build_completion_response(
        request=admitted_completion,
        request_id="cmpl-text",
        created=0,
        generated_text=" Paris</think>\n\n<tool_call>",
        prompt_token_count=5,
        generated_token_count=4,
        finish_reason="stop",
    )
    require(
        text_response["choices"][0]["text"] == " Paris</think>\n\n<tool_call>"
        and admitted_completion.output_plan is None,
        "plain completions stopped returning raw text",
    )

    chat = parse(
        {
            "max_tokens": requested_generation_tokens,
            "messages": [{"content": "Name France's capital.", "role": "user"}],
            "model": admitted_model,
        }
    )
    require(len(chat.messages) == 1, "valid chat request lost its message")
    prompt_token_count = 5
    generated_token_count = 3

    def chat_response(request, prompt, text, finish_reason):
        return build_completion_response(
            request=admit_prompt(request, prompt, prompt_token_count, maximum_sequence_length),
            request_id="cmpl-test",
            created=0,
            generated_text=text,
            prompt_token_count=prompt_token_count,
            generated_token_count=generated_token_count,
            finish_reason=finish_reason,
        )

    response = chat_response(chat, "<|im_start|>assistant\n<think>\n\n</think>\n\n", "Paris", "stop")
    require(
        response["choices"][0]["message"] == {"content": "Paris", "role": "assistant"}
        and response["choices"][0]["finish_reason"] == "stop",
        f"a no-think chat response changed: {response['choices'][0]}",
    )
    require(
        response["usage"]["total_tokens"] == prompt_token_count + generated_token_count,
        "chat response token total changed",
    )
    response = chat_response(
        chat, "<|im_start|>assistant\n<think>\n", "The capital.\n</think>\n\nParis", "stop"
    )
    require(
        response["choices"][0]["message"]
        == {"content": "Paris", "reasoning_content": "The capital.", "role": "assistant"},
        f"reasoning was not split from content: {response['choices'][0]['message']}",
    )

    # Acceptance: a complete call ended by the stop token, and the same output cut by the budget.
    tool_chat = parse(
        {"messages": [{"content": "Show /etc/hosts.", "role": "user"}], "tools": [READ_TOOL]}
    )
    thinking_prompt = "<|im_start|>assistant\n<think>\n"
    choice = chat_response(tool_chat, thinking_prompt, TOOL_CALL_OUTPUT, "stop")["choices"][0]
    message = choice["message"]
    calls = message.get("tool_calls", [])
    require(
        choice["finish_reason"] == "tool_calls"
        and message["content"] is None
        and message["reasoning_content"] == "reasoning"
        and len(calls) == 1
        and calls[0]["type"] == "function"
        and calls[0]["id"].startswith("call_")
        and calls[0]["function"] == {"arguments": '{"path": "/etc/hosts"}', "name": "read"},
        f"a complete tool call was not reported: {choice}",
    )
    choice = chat_response(tool_chat, thinking_prompt, TOOL_CALL_OUTPUT, "length")["choices"][0]
    require(choice["finish_reason"] == "length", "a call cut by the budget was reported as tool_calls")
    cut = TOOL_CALL_OUTPUT[: TOOL_CALL_OUTPUT.index("hosts")]
    choice = chat_response(tool_chat, thinking_prompt, cut, "length")["choices"][0]
    require(
        choice["finish_reason"] == "length"
        and json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]) == {"path": "/etc/"},
        f"a truncated call lost its valid JSON or its length finish: {choice}",
    )
    choice = chat_response(tool_chat, thinking_prompt, cut, "stop")["choices"][0]
    require(choice["finish_reason"] == "stop", "a call left open at the stop token was reported as tool_calls")

    # max_completion_tokens is an alias; without either, generation fills the sequence up to the cap.
    require(
        parse({"max_completion_tokens": 7, "prompt": "ok"}, Endpoint.COMPLETIONS).max_tokens == 7
        and parse({"max_completion_tokens": 7, "max_tokens": 7, "prompt": "ok"}, Endpoint.COMPLETIONS).max_tokens == 7,
        "max_completion_tokens was not read as max_tokens",
    )
    defaulted = parse({"messages": [{"content": "hi", "role": "user"}]})
    require(
        defaulted.max_tokens == maximum_generation_tokens and not defaulted.max_tokens_explicit,
        "a request without a limit did not default to the service cap",
    )
    require(
        admit_prompt(defaulted, "p", 10, maximum_sequence_length).max_tokens == maximum_generation_tokens
        and admit_prompt(defaulted, "p", maximum_sequence_length - 5, maximum_sequence_length).max_tokens == 5
        and admit_prompt(chat, "p", maximum_sequence_length - 4, maximum_sequence_length).max_tokens == 4,
        "the default limit did not fit the room the prompt leaves",
    )
    for request, prompt_tokens in ((defaulted, maximum_sequence_length), (chat, maximum_sequence_length - 3)):
        try:
            admit_prompt(request, "p", prompt_tokens, maximum_sequence_length)
        except RequestValidationError as error:
            require("maximum context length is 2048 tokens" in str(error), f"wrong sequence-limit diagnostic: {error}")
        else:
            require(False, "a request over the sequence limit was admitted")

    # Thinking controls: top level wins, omp's efforts map onto the template's, other kwargs stay out.
    thinking = parse(
        {
            "chat_template_kwargs": {
                "enable_thinking": True,
                "preserve_thinking": True,
                "reasoning_effort": "low",
                "unrelated": "x",
            },
            "enable_thinking": False,
            "messages": [{"content": "hi", "role": "user"}],
            "reasoning_effort": "high",
        }
    )
    require(
        template_arguments(thinking)
        == {"enable_thinking": False, "preserve_thinking": True, "reasoning_effort": "xhigh"},
        f"thinking controls resolved wrongly: {template_arguments(thinking)}",
    )
    require(template_arguments(chat) == {}, "an unset thinking control reached the template")
    for effort, mapped in (("minimal", "low"), ("medium", "medium"), ("max", "xhigh")):
        request = parse({"messages": [{"content": "hi", "role": "user"}], "reasoning_effort": effort})
        require(request.reasoning_effort == mapped, f"reasoning effort {effort} mapped to {request.reasoning_effort}")

    # Tool selection.
    other_tool = {"function": {"name": "write"}, "type": "function"}
    base = {"messages": [{"content": "hi", "role": "user"}], "tools": [READ_TOOL, other_tool]}
    require(parse(base).tools == (READ_TOOL, other_tool), "auto did not offer every tool")
    require(parse({**base, "tool_choice": "none"}).tools == (), "tool_choice none still offered tools")
    require(parse({**base, "tool_choice": "required"}).tools == (READ_TOOL, other_tool), "required dropped tools")
    require(
        parse({**base, "tool_choice": {"function": {"name": "write"}, "type": "function"}}).tools == (other_tool,),
        "a named tool_choice did not narrow the tools",
    )
    require(not parse({**base, "parallel_tool_calls": False}).parallel_tool_calls, "parallel_tool_calls was ignored")

    # History: string arguments become the template's mapping; empty assistant content and tool results pass.
    history = parse(
        {
            "messages": [
                {"content": "You are helpful.", "role": "system"},
                {"content": "Show /etc/hosts.", "role": "user"},
                {
                    "content": "",
                    "reasoning_content": "I should read it.",
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {"arguments": '{"path": "/etc/hosts", "limit": 3}', "name": "read"},
                            "id": "call_1",
                            "type": "function",
                        },
                        {"function": {"arguments": "", "name": "read"}, "id": "call_2", "type": "function"},
                    ],
                },
                {"content": "127.0.0.1 localhost", "role": "tool", "tool_call_id": "call_1"},
                {"content": "", "role": "tool", "tool_call_id": "call_2"},
                {"content": None, "role": "assistant"},
                {"content": [{"text": "Thanks.", "type": "text"}], "role": "user"},
            ],
            "tools": [],
        }
    )
    rendered = template_messages(history)
    require(
        rendered[2]
        == {
            "content": "",
            "reasoning_content": "I should read it.",
            "role": "assistant",
            "tool_calls": [
                {"function": {"arguments": {"limit": 3, "path": "/etc/hosts"}, "name": "read"}, "type": "function"},
                {"function": {"arguments": {}, "name": "read"}, "type": "function"},
            ],
        }
        and rendered[3] == {"content": "127.0.0.1 localhost", "role": "tool"}
        and rendered[4] == {"content": "", "role": "tool"}
        and rendered[5] == {"content": "", "role": "assistant"}
        and history.tools == (),
        f"history did not convert for the template: {rendered}",
    )

    # Streaming requests parse, and text content parts join as the chat template joins them.
    streamed = parse(
        {
            "messages": [
                {"content": [{"text": "Name ", "type": "text"}, {"text": "France's capital.", "type": "text"}], "role": "user"}
            ],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    )
    require(streamed.stream and streamed.include_usage, "a streaming request lost its stream options")
    require(streamed.messages[0].content == "Name France's capital.", "text parts did not join")
    require(not chat.stream and not chat.include_usage, "a plain request was marked streaming")

    user = {"content": "hi", "role": "user"}
    assistant_call = {
        "content": "",
        "role": "assistant",
        "tool_calls": [{"function": {"arguments": "{}", "name": "read"}, "type": "function"}],
    }

    def call_history(arguments):
        return {
            "messages": [
                user,
                {"content": "", "role": "assistant", "tool_calls": [{"function": {"arguments": arguments, "name": "read"}}]},
            ]
        }

    rejected = (
        ([], Endpoint.COMPLETIONS, "JSON object"),
        ({"prompt": ""}, Endpoint.COMPLETIONS, "non-empty"),
        ({"prompt": "ok", "max_tokens": 0}, Endpoint.COMPLETIONS, "between"),
        (
            {"prompt": "ok", "max_tokens": maximum_generation_tokens + 1},
            Endpoint.COMPLETIONS,
            "between",
        ),
        (
            {"prompt": "ok", "max_completion_tokens": maximum_generation_tokens + 1},
            Endpoint.COMPLETIONS,
            "max_completion_tokens must be between",
        ),
        ({"prompt": "ok", "max_tokens": 4, "max_completion_tokens": 5}, Endpoint.COMPLETIONS, "differ"),
        ({"prompt": "ok", "stream": "yes"}, Endpoint.COMPLETIONS, "stream must be a boolean"),
        ({"prompt": "ok", "stream_options": {"include_usage": True}}, Endpoint.COMPLETIONS, "requires stream"),
        (
            {"prompt": "ok", "stream": True, "stream_options": {"include_usage": 1}},
            Endpoint.COMPLETIONS,
            "include_usage must be a boolean",
        ),
        ({"prompt": "ok", "temperature": 1}, Endpoint.COMPLETIONS, "greedy"),
        ({"prompt": "ok", "n": 2}, Endpoint.COMPLETIONS, "n must be 1"),
        ({"prompt": "ok", "model": "wrong"}, Endpoint.COMPLETIONS, "model must be"),
        (
            {"messages": [{"content": "answer", "role": "assistant"}]},
            Endpoint.CHAT_COMPLETIONS,
            "user message",
        ),
        (
            {"messages": [{"content": "<tool_response>\nx\n</tool_response>", "role": "user"}]},
            Endpoint.CHAT_COMPLETIONS,
            "user message",
        ),
        (
            {"messages": [{"content": [{"type": "text"}], "role": "user"}]},
            Endpoint.CHAT_COMPLETIONS,
            "non-empty text",
        ),
        (
            {"messages": [{"content": [{"image_url": {"url": "x"}, "type": "image_url"}], "role": "user"}]},
            Endpoint.CHAT_COMPLETIONS,
            "must be a text part",
        ),
        ({"messages": [user, {"content": "late", "role": "system"}]}, Endpoint.CHAT_COMPLETIONS, "must come first"),
        ({"messages": [{"content": "x", "role": "developer"}, user]}, Endpoint.CHAT_COMPLETIONS, "role is not supported"),
        ({"messages": [user, {"content": "x", "role": "tool"}]}, Endpoint.CHAT_COMPLETIONS, "must follow an assistant"),
        ({"messages": [user, assistant_call, {"content": 5, "role": "tool"}]}, Endpoint.CHAT_COMPLETIONS, "must be text"),
        (
            {"messages": [user, assistant_call, {"content": [{"image_url": {}, "type": "image_url"}], "role": "tool"}]},
            Endpoint.CHAT_COMPLETIONS,
            "must be a text part",
        ),
        (call_history("{not json"), Endpoint.CHAT_COMPLETIONS, "arguments must be a JSON object"),
        (call_history("[1, 2]"), Endpoint.CHAT_COMPLETIONS, "arguments must be a JSON object"),
        (
            {"messages": [user, {"content": "", "role": "assistant", "tool_calls": [{"function": {"arguments": "{}"}}]}]},
            Endpoint.CHAT_COMPLETIONS,
            "name must be a non-empty string",
        ),
        (
            {"messages": [user, {"content": "", "reasoning_content": 3, "role": "assistant"}]},
            Endpoint.CHAT_COMPLETIONS,
            "reasoning_content must be a string",
        ),
        ({"messages": [user], "tools": {"read": READ_TOOL}}, Endpoint.CHAT_COMPLETIONS, "tools must be an array"),
        ({"messages": [user], "tools": [{"type": "function", "function": {}}]}, Endpoint.CHAT_COMPLETIONS, "name must be"),
        ({"messages": [user], "tool_choice": "required"}, Endpoint.CHAT_COMPLETIONS, "needs tools"),
        (
            {"messages": [user], "tool_choice": {"function": {"name": "write"}, "type": "function"}, "tools": [READ_TOOL]},
            Endpoint.CHAT_COMPLETIONS,
            "not in tools",
        ),
        ({"messages": [user], "tool_choice": "sometimes"}, Endpoint.CHAT_COMPLETIONS, "tool_choice must be"),
        ({"messages": [user], "parallel_tool_calls": "no"}, Endpoint.CHAT_COMPLETIONS, "parallel_tool_calls must be"),
        ({"messages": [user], "reasoning_effort": "extreme"}, Endpoint.CHAT_COMPLETIONS, "reasoning_effort must be"),
        (
            {"messages": [user], "chat_template_kwargs": {"enable_thinking": "yes"}},
            Endpoint.CHAT_COMPLETIONS,
            "enable_thinking must be a boolean",
        ),
        ({"messages": [user], "chat_template_kwargs": []}, Endpoint.CHAT_COMPLETIONS, "chat_template_kwargs must be"),
    )
    for payload, endpoint, diagnostic in rejected:
        try:
            parse(payload, endpoint)
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


def probe_chat_output_parser():
    """Reasoning, content and tool calls from Qwen output, streamed or whole."""
    import itertools

    from qwen36_service.chat_output_core import (
        ChatOutputParser,
        ChatOutputPlan,
        chat_output_plan,
        coerce_argument,
        collect_output,
        event_deltas,
        parse_chat_output,
        schema_types,
    )

    def call_ids():
        counter = itertools.count()
        return lambda: f"call_{next(counter)}"

    def parse(text, plan):
        return parse_chat_output(text, plan, call_ids())

    def chunked(text, plan, size):
        parser = ChatOutputParser(plan, call_ids())
        events = []
        for start in range(0, len(text), size):
            events += parser.push(text[start : start + size])
        events += parser.finish()
        return collect_output(events, parser.completed_call_count, parser.open_call), events

    thinking = chat_output_plan("<|im_start|>assistant\n<think>\n", [READ_TOOL], True)
    no_think = chat_output_plan("<|im_start|>assistant\n<think>\n\n</think>\n\n", [READ_TOOL], True)
    require(thinking.starts_in_reasoning and not no_think.starts_in_reasoning, "the prompt did not seed the thinking state")

    output = parse(TOOL_CALL_OUTPUT, thinking)
    require(
        output.reasoning == "reasoning"
        and output.content == ""
        and [(call.name, call.arguments) for call in output.tool_calls] == [("read", '{"path": "/etc/hosts"}')]
        and output.completed_call_count == 1
        and not output.open_call,
        f"a complete call parsed wrongly: {output}",
    )

    multiple = (
        "Let me look.\n\n<tool_call>\n<function=read>\n<parameter=path>\na\n\nb\n</parameter>\n"
        "<parameter=limit>\n12\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=write>\n"
        "</function>\n</tool_call>\nDone."
    )
    output = parse(multiple, no_think)
    require(
        output.reasoning == ""
        and output.content == "Let me look.\n\nDone."
        and [(call.call_id, call.name, json.loads(call.arguments)) for call in output.tool_calls]
        == [("call_0", "read", {"limit": 12, "path": "a\n\nb"}), ("call_1", "write", {})]
        and output.completed_call_count == 2,
        f"multiple calls or the text around them parsed wrongly: {output}",
    )
    single = ChatOutputPlan(
        starts_in_reasoning=False,
        parse_tool_calls=True,
        parallel_tool_calls=False,
        argument_types=no_think.argument_types,
    )
    output = parse(multiple, single)
    require(
        [call.name for call in output.tool_calls] == ["read"] and output.content == "Let me look.\n\nDone.",
        f"parallel_tool_calls false kept more than the first call: {output}",
    )

    truncated = parse(TOOL_CALL_OUTPUT[: TOOL_CALL_OUTPUT.index("<parameter=path>") + 20], thinking)
    require(
        truncated.open_call
        and truncated.completed_call_count == 0
        and json.loads(truncated.tool_calls[0].arguments) == {"path": "/et"},
        f"a truncated call was not closed and marked open: {truncated}",
    )
    buffered_cut = parse("<tool_call>\n<function=read>\n<parameter=limit>\n1", no_think)
    require(
        buffered_cut.open_call and json.loads(buffered_cut.tool_calls[0].arguments) == {"limit": 1},
        f"a truncated typed value did not close: {buffered_cut}",
    )

    unclosed = parse("still thinking <tool_call>\n<function=read>", thinking)
    require(
        unclosed.reasoning == "still thinking <tool_call>\n<function=read>"
        and not unclosed.tool_calls
        and unclosed.content == "",
        "output without </think> was not all reasoning",
    )
    require(
        parse("Hi.\n</think>\n\n", chat_output_plan("x", [], True)).content == "Hi.\n</think>",
        "a no-think output lost text",
    )
    hidden = parse("Use <tool_call>\n<function=read>\n</function>\n</tool_call>", chat_output_plan("x", [], True))
    require(
        hidden.content == "Use <tool_call>\n<function=read>\n</function>\n</tool_call>" and not hidden.tool_calls,
        "a call was parsed with no tools offered",
    )
    malformed = parse('<tool_call>\n{"name": "read"}\n</tool_call>', no_think)
    require(
        malformed.content == '<tool_call>\n{"name": "read"}\n</tool_call>' and not malformed.tool_calls,
        "a call block without a function header was dropped",
    )

    # Argument types from the schema: strings stay raw, typed and anyOf/oneOf values are JSON.
    require(schema_types({"type": "string"}) == frozenset({"string"}), "a string schema lost its type")
    require(
        schema_types({"anyOf": [{"type": "object"}, {"type": ["null", "integer"]}]})
        == frozenset({"object", "null", "integer"}),
        "anyOf member types were not collected",
    )
    require(schema_types({"oneOf": [{"type": "string"}, {}]}) is None, "an untyped oneOf member was typed")
    require(schema_types({"enum": ["a", "b"]}) == frozenset({"string"}), "enum values were not typed")
    string_only = frozenset({"string"})
    coercions = (
        ("12", string_only, "12"),
        ("12", frozenset({"integer"}), 12),
        ("12", frozenset({"number"}), 12),
        ("1.5", frozenset({"integer"}), "1.5"),
        ('{"k": [1]}', frozenset({"object", "null"}), {"k": [1]}),
        ("null", frozenset({"object", "null"}), None),
        ("12", frozenset({"integer", "string"}), 12),
        ('"quoted"', frozenset({"integer", "string"}), '"quoted"'),
        ("true", None, True),
        ("plain words", None, "plain words"),
        ("NaN", None, "NaN"),
        ("1e999", frozenset({"number"}), "1e999"),
        ("{broken", frozenset({"object"}), "{broken"),
    )
    for raw, admitted, expected in coercions:
        value = coerce_argument(raw, admitted)
        require(
            value == expected and type(value) is type(expected),
            f"{raw!r} with {admitted} coerced to {value!r}",
        )
    typed = parse(
        "<tool_call>\n<function=read>\n<parameter=path>\n12\n</parameter>\n<parameter=state>\n"
        '{"open": true}\n</parameter>\n<parameter=extra>\n[1, 2]\n</parameter>\n</function>\n</tool_call>',
        no_think,
    )
    require(
        typed.tool_calls[0].arguments == '{"path": "12", "state": {"open": true}, "extra": [1, 2]}',
        f"typed arguments did not follow the schema: {typed.tool_calls[0].arguments}",
    )

    # Tags split anywhere: character-by-character and small chunks equal the one-shot parse.
    samples = (
        (TOOL_CALL_OUTPUT, thinking),
        (multiple, no_think),
        (multiple, single),
        ("Plan <tool_c\n</think>\n\nA < B </thinking> <tool_callx>\n\n", thinking),
        ('<tool_call>\n<function=read>\n<parameter=path>\nx\n</param\n</parameter>\n</function>', no_think),
        ("a</think>b<tool_call>\n<function=read>\n<parameter=path>\n/et", thinking),
        ("r</think>\n\n<tool_call>\n<function= read >\ngarbage<parameter=limit>7</parameter></tool_call>tail", thinking),
        ("\n\n  indented\n\n", no_think),
    )
    for text, plan in samples:
        whole = parse(text, plan)
        for size in (1, 2, 3, 7):
            pieces, _events = chunked(text, plan, size)
            require(pieces == whole, f"chunks of {size} parsed {text!r} differently: {pieces} != {whole}")

    _output, events = chunked(TOOL_CALL_OUTPUT, thinking, 1)
    deltas = event_deltas(events)
    reasoning = [delta for delta in deltas if "reasoning_content" in delta]
    calls = deltas[len(reasoning) :]
    require(
        "".join(delta["reasoning_content"] for delta in reasoning) == "reasoning"
        and calls[0]
        == {
            "tool_calls": [
                {"function": {"arguments": "", "name": "read"}, "id": "call_0", "index": 0, "type": "function"}
            ]
        }
        and all(set(delta) == {"tool_calls"} and set(delta["tool_calls"][0]) == {"function", "index"} for delta in calls[1:])
        and "".join(delta["tool_calls"][0]["function"]["arguments"] for delta in calls[1:]) == '{"path": "/etc/hosts"}',
        f"stream deltas have the wrong shape: {deltas}",
    )
    require(len(calls) > 3, "a string argument was not streamed as it arrived")


def probe_chat_template_rendering():
    """Chat requests render through the template with tools; template errors are request errors."""
    from qwen36_service.core import (
        ChatMessage,
        CompletionRequest,
        Endpoint,
        RequestValidationError,
        ToolCall,
        admit_prompt,
        parse_completion_request,
    )
    from qwen36_service.server import render_prompt

    tokenizer = FixtureTokenizer()

    def parse(payload):
        return parse_completion_request(payload, Endpoint.CHAT_COMPLETIONS, "Qwen3.8-27B", 64)

    request = parse(
        {
            "messages": [
                {"content": "Show /etc/hosts.", "role": "user"},
                {
                    "content": "",
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {"arguments": '{"path": "/etc/hosts", "limit": 3}', "name": "read"},
                            "id": "call_1",
                            "type": "function",
                        }
                    ],
                },
                {"content": "127.0.0.1 localhost", "role": "tool", "tool_call_id": "call_1"},
            ],
            "tools": [READ_TOOL],
        }
    )
    prompt = render_prompt(request, tokenizer)
    require(
        "<parameter=path>\n/etc/hosts\n</parameter>\n<parameter=limit>\n3\n</parameter>" in prompt
        and "<tool_response>\n127.0.0.1 localhost\n</tool_response>" in prompt
        and json.dumps(READ_TOOL, ensure_ascii=False) in prompt,
        f"history or tools did not render: {prompt!r}",
    )
    require(
        prompt.endswith("<think>\n") and admit_prompt(request, prompt, 10, 2048).output_plan.starts_in_reasoning,
        "the default prompt did not open reasoning",
    )
    quiet = parse({"enable_thinking": False, "messages": [{"content": "hi", "role": "user"}], "tools": [READ_TOOL]})
    quiet_prompt = render_prompt(quiet, tokenizer)
    plan = admit_prompt(quiet, quiet_prompt, 10, 2048).output_plan
    require(
        quiet_prompt.endswith("</think>\n\n") and not plan.starts_in_reasoning and plan.parse_tool_calls,
        "enable_thinking false did not close the reasoning section",
    )
    hidden = parse({"messages": [{"content": "hi", "role": "user"}], "tool_choice": "none", "tools": [READ_TOOL]})
    hidden_prompt = render_prompt(hidden, tokenizer)
    require(
        "<tools>" not in hidden_prompt and not admit_prompt(hidden, hidden_prompt, 10, 2048).output_plan.parse_tool_calls,
        "tool_choice none still offered tools",
    )

    failing = (
        parse({"messages": [{"content": "template error", "role": "user"}]}),
        # A string reaching `arguments|items` raises TypeError inside the template.
        CompletionRequest(
            endpoint=Endpoint.CHAT_COMPLETIONS,
            model="Qwen3.8-27B",
            max_tokens=4,
            prompt=None,
            messages=(
                ChatMessage(role="user", content="hi"),
                ChatMessage(role="assistant", content="", tool_calls=(ToolCall(name="read", arguments='{"a": 1}'),)),
            ),
        ),
    )
    for request in failing:
        try:
            render_prompt(request, tokenizer)
        except RequestValidationError as error:
            require("chat template rejected" in str(error), f"wrong template diagnostic: {error}")
        else:
            require(False, "a template failure was not a request error")


def probe_streaming():
    """Server-sent events: whole characters, parsed deltas, one id, usage, and JSON errors."""
    import http.client
    import threading
    from http.server import ThreadingHTTPServer

    from qwen36_service.core import (
        admit_prompt,
        build_completion_response,
        finish_generation,
    )
    from qwen36_service.server import PreparedGeneration, make_handler, render_prompt
    from qwen36_service.stream_core import IncrementalDetokenizer

    # "€" is split across tokens 3 and 4; token 9 is the stop token.
    pieces = {1: b"Hel", 2: b"lo ", 3: b"\xe2\x82", 4: b"\xac", 5: b"!", 9: b""}
    stop_token_ids = frozenset({9})
    decoded_lengths = []

    def decode(token_ids, skip_special_tokens=True):
        decoded_lengths.append(len(token_ids))
        return b"".join(pieces[token] for token in token_ids).decode("utf-8", errors="replace")

    detokenizer = IncrementalDetokenizer(decode, stop_token_ids)
    deltas = [detokenizer.push(token) for token in (1, 2, 3, 4, 9, 5)]
    require(
        deltas == ["Hel", "lo ", "", "€", "", ""],
        f"incremental text split a character or passed the stop token: {deltas}",
    )
    require(detokenizer.stopped, "the stop token did not end the stream")
    require(
        detokenizer.finish([1, 2, 3, 4]) == "" and not detokenizer.diverged,
        "a complete stream sent a tail",
    )
    late = IncrementalDetokenizer(decode, stop_token_ids)
    late.push(1)
    require(late.finish([1, 2, 5]) == "lo !" and not late.diverged, "the final tokens did not complete the stream")
    require(late.finish([2]) == "" and late.diverged, "diverging final tokens were not flagged")
    cut = IncrementalDetokenizer(decode, stop_token_ids)
    cut.push(3)
    require(cut.finish([3]) == "\ufffd", "the final flush held back a partial character")
    decoded_lengths.clear()
    long = IncrementalDetokenizer(decode, stop_token_ids)
    long_text = "".join(long.push(token) for token in [1, 2, 3, 4, 5] * 400)
    require(long_text == "Hello €!" * 400, "a long stream lost text")
    require(max(decoded_lengths) <= 3, f"streaming decoded a growing window: {max(decoded_lengths)} tokens")

    # The model's chat output as small text pieces; tags split across pieces.
    script_pieces = [TOOL_CALL_OUTPUT[index : index + 3] for index in range(0, len(TOOL_CALL_OUTPUT), 3)]
    for index, piece in enumerate(script_pieces):
        pieces[100 + index] = piece.encode("utf-8")
    tool_tokens = [100 + index for index in range(len(script_pieces))]
    cut_tokens = tool_tokens[: len(tool_tokens) - 12]
    scripts = {
        "call": tool_tokens + [9],
        "cut": cut_tokens,
        "hi": [1, 2, 3, 4, 9, 5],
        "late": tool_tokens + [9],
    }

    class FakeEngine:
        model_alias = "Qwen3.8-27B"
        sampled_serving_enabled = False
        structured_reads_available = False
        stop_token_ids = frozenset({9})
        tokenizer = SimpleNamespace(decode=decode)

        def prepare(self, request):
            prompt = render_prompt(request, FixtureTokenizer())
            prompt_tokens = 2048 if request.messages and request.messages[-1].content == "too long" else 3
            return PreparedGeneration(
                request=admit_prompt(request, prompt, prompt_tokens, 2048),
                token_ids=None,
                prompt_token_count=prompt_tokens,
            )

        def generate(self, request, token_sink=None, request_id=None):
            return self.generate_prepared(self.prepare(request), token_sink, request_id)

        def generate_prepared(self, prepared, token_sink=None, request_id=None):
            request = prepared.request
            name = request.messages[-1].content if request.messages else "hi"
            tokens = scripts[name]
            # The MTP and eager routes send no tokens early.
            if token_sink is not None and name != "late":
                for token in tokens:
                    token_sink(token)
            kept, finish_reason = finish_generation(tokens, self.stop_token_ids)
            response = build_completion_response(
                request=request,
                request_id=request_id or "cmpl-plain",
                created=0,
                generated_text=decode(kept),
                prompt_token_count=prepared.prompt_token_count,
                generated_token_count=len(kept),
                finish_reason=finish_reason,
            )
            response["x_onix_decode_evidence"] = {"generated_token_ids": kept}
            return response

    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(FakeEngine(), 64))
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def post(body, path="/v1/chat/completions"):
        payload = json.dumps(body)
        connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=30)
        connection.request(
            "POST",
            path,
            body=payload,
            headers={"Content-Type": "application/json", "Content-Length": str(len(payload))},
        )
        response = connection.getresponse()
        return response.status, response.getheader("Content-Type"), response.read().decode("utf-8")

    def stream(body, path="/v1/chat/completions"):
        status, content_type, text = post({**body, "stream": True, "stream_options": {"include_usage": True}}, path)
        require(
            status == 200 and content_type == "text/event-stream",
            f"a stream answered {status} {content_type}: {text[:200]}",
        )
        events = [line[len("data: ") :] for line in text.split("\n") if line.startswith("data: ")]
        require(events[-1] == "[DONE]", "the stream did not end with [DONE]")
        chunks = [json.loads(event) for event in events[:-1]]
        require(len({chunk["id"] for chunk in chunks}) == 1, "stream chunks carry different ids")
        require(
            chunks[-1]["choices"] == [] and "completion_tokens" in chunks[-1]["usage"],
            "the stream lacks its usage chunk",
        )
        finals = [chunk for chunk in chunks if chunk["choices"] and chunk["choices"][0]["finish_reason"]]
        require(len(finals) == 1 and finals[0] is chunks[-2], "the stream lacks exactly one finish chunk")
        return chunks, finals[0]["choices"][0]["finish_reason"]

    def joined(chunks):
        """The complete message the client assembles from the deltas."""
        message = {"content": "", "reasoning_content": "", "tool_calls": {}}
        for chunk in chunks:
            if not chunk["choices"]:
                continue
            delta = chunk["choices"][0]["delta"]
            message["content"] += delta.get("content", "")
            message["reasoning_content"] += delta.get("reasoning_content", "")
            for call in delta.get("tool_calls", []):
                entry = message["tool_calls"].setdefault(call["index"], {"arguments": "", "name": None, "id": None})
                entry["id"] = call.get("id", entry["id"])
                entry["name"] = call["function"].get("name", entry["name"])
                entry["arguments"] += call["function"]["arguments"]
        return message

    def no_think(content):
        return {"enable_thinking": False, "messages": [{"content": content, "role": "user"}]}

    tool_request = {"messages": [{"content": "call", "role": "user"}], "tools": [READ_TOOL]}
    try:
        chunks, finish_reason = stream(no_think("hi"))
        require(
            chunks[0]["choices"][0]["delta"] == {"content": "", "role": "assistant"},
            "the stream did not open with the assistant role",
        )
        message = joined(chunks)
        require(message["content"] == "Hello €" and finish_reason == "stop", f"the streamed text differs: {message}")

        for name in ("call", "late"):
            chunks, finish_reason = stream({**tool_request, "messages": [{"content": name, "role": "user"}]})
            message = joined(chunks)
            call = message["tool_calls"].get(0, {})
            require(
                finish_reason == "tool_calls"
                and message["content"] == ""
                and message["reasoning_content"] == "reasoning"
                and list(message["tool_calls"]) == [0]
                and call["name"] == "read"
                and call["id"].startswith("call_")
                and call["arguments"] == '{"path": "/etc/hosts"}',
                f"the {name} stream did not carry the tool call: {finish_reason} {message}",
            )
            require(
                all(
                    set(chunk["choices"][0]["delta"]) <= {"content", "role", "reasoning_content", "tool_calls"}
                    for chunk in chunks
                    if chunk["choices"]
                ),
                "a stream delta carried an unexpected field",
            )
        streamed_arguments = [
            chunk
            for chunk in stream(tool_request)[0]
            if chunk["choices"] and "tool_calls" in chunk["choices"][0]["delta"]
        ]
        require(len(streamed_arguments) > 2, "the call arguments were not streamed as they arrived")

        status, _content_type, body = post(tool_request)
        choice = json.loads(body)["choices"][0]
        require(
            status == 200
            and choice["finish_reason"] == "tool_calls"
            and choice["message"]["content"] is None
            and choice["message"]["reasoning_content"] == "reasoning"
            and choice["message"]["tool_calls"][0]["function"] == {"arguments": '{"path": "/etc/hosts"}', "name": "read"},
            f"the complete tool-call response differs: {choice}",
        )

        cut_request = {**tool_request, "messages": [{"content": "cut", "role": "user"}]}
        chunks, finish_reason = stream(cut_request)
        cut_arguments = joined(chunks)["tool_calls"][0]["arguments"]
        status, _content_type, body = post(cut_request)
        cut_choice = json.loads(body)["choices"][0]
        require(
            finish_reason == "length"
            and cut_choice["finish_reason"] == "length"
            and json.loads(cut_arguments) == json.loads(cut_choice["message"]["tool_calls"][0]["function"]["arguments"]),
            f"a call cut by the budget did not finish with length: {finish_reason} {cut_arguments!r} {cut_choice}",
        )

        # Plain completions keep raw text: no reasoning split, no call parsing.
        chunks, finish_reason = stream({"prompt": "x", "max_tokens": 8}, "/v1/completions")
        require(
            "".join(chunk["choices"][0]["text"] for chunk in chunks if chunk["choices"]) == "Hello €"
            and finish_reason == "stop",
            "a completion stream changed its text",
        )

        for body, diagnostic in (
            (no_think("too long"), "maximum context length is 2048 tokens"),
            (no_think("template error"), "chat template rejected"),
            (
                {"messages": [{"content": "hi", "role": "user"}, {"content": "x", "role": "tool"}]},
                "must follow an assistant",
            ),
        ):
            for streaming in (True, False):
                status, _content_type, answer = post({**body, "stream": streaming})
                require(
                    status == 400 and diagnostic in json.loads(answer)["error"]["message"],
                    f"a rejected request (stream={streaming}) did not answer a JSON 400: {status} {answer}",
                )
        status, _content_type, body = post(no_think("hi"))
        require(
            status == 200 and json.loads(body)["choices"][0]["message"]["content"] == "Hello €",
            "a plain request stopped answering JSON",
        )
    finally:
        server.shutdown()
        server.server_close()


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


# r[verify tenstorrent.qwen38_p150x2.gdn_decode_step]
def probe_gdn_decode_step_policy():
    """Positive and negative tests for the fused decode step mode and its geometry bound."""
    from models.demos.blackhole.qwen36.tt.gdn import (
        decode_fusion_candidate as candidate,
    )
    from models.demos.blackhole.qwen36.tt.gdn.decode_step import GdnDecodeGeometry

    require(
        candidate.selected_mode({candidate.ENV_GATE: " STEP "}) == candidate.MODE_STEP,
        "step mode was not normalized",
    )
    require(
        candidate.uses_fused_step(candidate.MODE_STEP),
        "step mode did not select the fused decode step",
    )
    for mode in sorted(candidate.MODES - {candidate.MODE_STEP}):
        require(
            not candidate.uses_fused_step(mode),
            f"{mode} mode selected the fused decode step",
        )
    require(
        not candidate.uses_fused_norm(candidate.MODE_STEP)
        and not candidate.uses_fused_state_update(candidate.MODE_STEP)
        and not candidate.uses_fused_recurrence(candidate.MODE_STEP),
        "step mode enabled a composite fusion",
    )

    # Qwen3.8-27B at TP=2: 8 key heads, 24 value heads, 128-wide heads.
    tp2 = GdnDecodeGeometry(
        num_k_heads=8, num_v_heads=24, head_k_dim=128, head_v_dim=128, max_batch=1
    )
    tp2.validate()
    require(tp2.qkv_dim == 5120, "TP2 q/k/v width changed")
    require(tp2.qkvzab_dim == 8240, "TP2 [qkv|z|a|b] width changed")
    require(tp2.prep_tiles == 258, "TP2 prep tile count changed")
    GdnDecodeGeometry(
        num_k_heads=8, num_v_heads=24, head_k_dim=128, head_v_dim=128, max_batch=32
    ).validate()
    for label, geometry in (
        ("zero batch", GdnDecodeGeometry(8, 24, 128, 128, 0)),
        ("batch above one tile row", GdnDecodeGeometry(8, 24, 128, 128, 33)),
        ("64-wide key head", GdnDecodeGeometry(8, 24, 64, 128, 1)),
        ("64-wide value head", GdnDecodeGeometry(8, 24, 128, 64, 1)),
        (
            "value heads not a multiple of key heads",
            GdnDecodeGeometry(8, 20, 128, 128, 1),
        ),
    ):
        try:
            geometry.validate()
        except ValueError:
            continue
        raise AssertionError(
            f"fused decode step accepted an unsupported geometry: {label}"
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
        ROUTE_FUSED_STEP,
        ROUTE_SAMPLED,
        ROUTES,
        RouterPolicyError,
        classify_temperature,
        deterministic_route_for_mode,
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

    # A service started on the fused decode step keeps greedy traffic there.
    require(
        deterministic_route_for_mode(ROUTE_FUSED_STEP) == ROUTE_FUSED_STEP,
        "a step-mode service must keep greedy requests on the fused step",
    )
    for mode in (ROUTE_DETERMINISTIC, ROUTE_SAMPLED, "unknown"):
        require(
            deterministic_route_for_mode(mode) == ROUTE_DETERMINISTIC,
            f"startup mode {mode!r} must keep the exact-parity greedy route",
        )
    require(
        classify_temperature(0.0, ROUTE_FUSED_STEP) == ROUTE_FUSED_STEP
        and classify_temperature(None, ROUTE_FUSED_STEP) == ROUTE_FUSED_STEP
        and classify_temperature(0.6, ROUTE_FUSED_STEP) == ROUTE_SAMPLED,
        "the fused step must replace only the deterministic route",
    )
    step_greedy = resolve_batch_route(
        (0.0,),
        sampled_admitted=True,
        current_mode=ROUTE_FUSED_STEP,
        deterministic_route=ROUTE_FUSED_STEP,
    )
    require(
        step_greedy.route == ROUTE_FUSED_STEP and not step_greedy.switch_required,
        "a greedy batch on the resident fused step must not switch",
    )
    step_mixed = resolve_batch_route(
        (0.6, 0.0),
        sampled_admitted=True,
        current_mode=ROUTE_SAMPLED,
        deterministic_route=ROUTE_FUSED_STEP,
    )
    require(
        step_mixed.route == ROUTE_FUSED_STEP and step_mixed.switch_required,
        "a mixed batch must return to the fused step",
    )
    for bad_route in (ROUTE_SAMPLED, "state"):
        try:
            route_request(
                0.0, current_mode=ROUTE_FUSED_STEP, deterministic_route=bad_route
            )
        except RouterPolicyError:
            continue
        raise AssertionError(f"{bad_route!r} was accepted as a deterministic route")

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
    per_user_scratch_bytes = 8 * 1024 * 1024  # one unit = one byte in these fixtures
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
        capacity_units=four_users_scratch_bytes,
        units_per_user=per_user_scratch_bytes,
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
        capacity_units=eight_users_scratch_bytes,
        units_per_user=per_user_scratch_bytes,
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
        capacity_units=eight_users_scratch_bytes,
        units_per_user=per_user_scratch_bytes,
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
        capacity_units=four_users_scratch_bytes,
        units_per_user=per_user_scratch_bytes,
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
        capacity_units=four_users_scratch_bytes,
        units_per_user=per_user_scratch_bytes,
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
        capacity_units=four_users_scratch_bytes,
        units_per_user=per_user_scratch_bytes,
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
        capacity_units=eight_users_scratch_bytes,
        units_per_user=per_user_scratch_bytes,
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
        capacity_units=four_users_scratch_bytes,
        units_per_user=per_user_scratch_bytes,
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
        capacity_units=four_users_scratch_bytes,
        units_per_user=per_user_scratch_bytes,
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
        capacity_units=per_user_scratch_bytes,
        units_per_user=per_user_scratch_bytes,
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
            capacity_units=four_users_scratch_bytes,
            units_per_user=per_user_scratch_bytes,
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


# r[verify tenstorrent.qwen38_p150x2.grouped_prefill_state]
def probe_grouped_prefill_routing():
    """The adapter consults the planner and fails closed on every refusal."""
    import json
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace

    from qwen36_service.batch_decode_adapter import (
        CHUNKED_VALIDATED_INSTANCES_PER_CORE,
        GRID_UNAVAILABLE_REASON,
        GROUPED_PREFILL_ENV,
        NON_CHUNKED_CAPABILITY_SOURCE,
        PACKAGE_CAPABILITY_SOURCE,
        BatchedTracedDecodeAdapter,
        grouped_prefill_enabled,
        load_runtime_capability_record,
    )

    grid = SimpleNamespace(x=11, y=10)
    chunked_record = SimpleNamespace(
        gdn_scan_chunked_instances=True, source=PACKAGE_CAPABILITY_SOURCE
    )
    one_instance_record = SimpleNamespace(
        gdn_scan_chunked_instances=False, source=PACKAGE_CAPABILITY_SOURCE
    )

    def stub(record, mesh=SimpleNamespace(compute_with_storage_grid_size=lambda: grid), capacity=8):
        value_heads = 24
        layer = SimpleNamespace(is_full_attention=False, attention=SimpleNamespace(Nv=value_heads))
        full = SimpleNamespace(is_full_attention=True, attention=SimpleNamespace(Nv=0))

        class RouteStub(BatchedTracedDecodeAdapter):
            """The real route planner bound to stub device and model surfaces."""

            def __init__(self):
                self._capability_record = record
                self.mesh = mesh
                self.maximum_width = capacity
                self.model = SimpleNamespace(
                    args=SimpleNamespace(gdn_chunk_size=128), layers=[full, layer]
                )

        return RouteStub()

    plan = BatchedTracedDecodeAdapter.plan_prefill_route
    eight_lengths = (128,) * 8

    # The gate defaults off, so the served path keeps the admitted identity.
    require(not grouped_prefill_enabled({}), "grouped prefill defaulted on")
    for value in ("1", "true", "yes", "on", "TRUE"):
        require(
            grouped_prefill_enabled({GROUPED_PREFILL_ENV: value}),
            f"grouped prefill did not accept {value}",
        )
    for value in ("", "0", "off", "maybe"):
        require(
            not grouped_prefill_enabled({GROUPED_PREFILL_ENV: value}),
            f"grouped prefill accepted {value}",
        )
    grouped, size, reason = plan(stub(chunked_record), 8, eight_lengths, grouped_enabled=False)
    require((not grouped) and reason == "grouped-prefill-disabled", "the gate did not hold")

    # Chunked runtime: one group of eight at the validated per-core chunk.
    grouped, size, reason = plan(stub(chunked_record), 8, eight_lengths, grouped_enabled=True)
    require(grouped and size == 8, f"chunked route planned {grouped}/{size}/{reason}")
    require(reason == "grouped-eligible", f"wrong grouped reason: {reason}")

    # A one-instance runtime derives a tighter bound and splits the users into
    # contiguous groups, which install in order.
    split_grouped, split_size, tight_reason = plan(
        stub(one_instance_record), 8, eight_lengths, grouped_enabled=True
    )
    require(
        split_grouped and split_size == 4 and tight_reason == "grouped-eligible",
        f"the tighter split bound was not grouped by four: {split_grouped}/{split_size}/{tight_reason}",
    )

    # An unreadable grid refuses grouping rather than guessing a core count.
    broken_mesh = SimpleNamespace(
        compute_with_storage_grid_size=lambda: (_ for _ in ()).throw(RuntimeError("no grid"))
    )
    grouped, size, reason = plan(
        stub(chunked_record, mesh=broken_mesh), 8, eight_lengths, grouped_enabled=True
    )
    require(
        (not grouped) and reason == GRID_UNAVAILABLE_REASON,
        f"an unreadable grid did not refuse grouping: {grouped}/{reason}",
    )

    # A prompt beyond the single-pass bucket is a per-user candidate.
    grouped, size, reason = plan(
        stub(chunked_record), 8, (1024,) * 8, grouped_enabled=True
    )
    require(
        (not grouped) and reason == "bucket-above-single-pass",
        f"an oversized bucket was grouped: {grouped}/{reason}",
    )

    # A group narrower than the decode buffer installs into its leading rows.
    narrower, narrow_size, narrow_reason = plan(
        stub(chunked_record, capacity=16), 8, eight_lengths, grouped_enabled=True
    )
    require(
        narrower and narrow_size == 8,
        f"a width below the decode capacity was refused: {narrower}/{narrow_reason}",
    )
    # An uneven split still partitions contiguously: six users by four is 4 + 2.
    uneven, uneven_size, uneven_reason = plan(
        stub(one_instance_record, capacity=16), 6, (128,) * 6, grouped_enabled=True
    )
    require(
        uneven and uneven_size == 4,
        f"an uneven contiguous split was refused: {uneven}/{uneven_size}/{uneven_reason}",
    )
    # One user is the per-user route.
    single, _, single_reason = plan(
        stub(chunked_record, capacity=16), 1, (128,), grouped_enabled=True
    )
    require(
        (not single) and single_reason == "single-user",
        f"a single user was grouped: {single}/{single_reason}",
    )

    # The capability record fails closed when absent, malformed, or mis-versioned.
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        require(
            load_runtime_capability_record(root).source == NON_CHUNKED_CAPABILITY_SOURCE,
            "a missing capability record was not refused",
        )
        share = root / "share" / "qwen36"
        share.mkdir(parents=True)
        record_path = share / "runtime-capability.json"
        record_path.write_text("{not json", encoding="utf-8")
        require(
            not load_runtime_capability_record(root).gdn_scan_chunked_instances,
            "malformed capability JSON claimed a chunked scan",
        )
        record_path.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
        require(
            not load_runtime_capability_record(root).gdn_scan_chunked_instances,
            "a mis-versioned capability record claimed a chunked scan",
        )
        record_path.write_text(
            json.dumps({"schema_version": 1, "gdn_scan_chunked_instances": True}),
            encoding="utf-8",
        )
        loaded = load_runtime_capability_record(root)
        require(
            loaded.gdn_scan_chunked_instances and loaded.source == PACKAGE_CAPABILITY_SOURCE,
            "a valid capability record was not read",
        )
        record_path.write_text(
            json.dumps({"schema_version": 1, "gdn_scan_chunked_instances": "yes"}),
            encoding="utf-8",
        )
        require(
            not load_runtime_capability_record(root).gdn_scan_chunked_instances,
            "a non-boolean capability flag was accepted",
        )

    require(
        CHUNKED_VALIDATED_INSTANCES_PER_CORE == 2,
        "the validated chunked instance width changed without new evidence",
    )


def probe_end_of_sequence():
    """Generation ends at the first stop token, and the response says so."""
    import json
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace

    from qwen36_service.core import (
        FINISH_LENGTH,
        FINISH_STOP,
        finish_generation,
        load_stop_token_ids,
    )

    stops = frozenset({7, 9})
    require(finish_generation([1, 2, 7, 3, 9], stops) == ([1, 2], FINISH_STOP), "stop token not truncated")
    require(finish_generation([7], stops) == ([], FINISH_STOP), "a leading stop token was kept")
    require(finish_generation([1, 2, 3], stops) == ([1, 2, 3], FINISH_LENGTH), "an unstopped run was cut")
    require(finish_generation([1, 7], frozenset()) == ([1, 7], FINISH_LENGTH), "an empty stop set stopped")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "generation_config.json").write_text(json.dumps({"eos_token_id": [248046, 248044]}))
        require(
            load_stop_token_ids(root, SimpleNamespace(eos_token_id=248046)) == frozenset({248046, 248044}),
            "generation_config stop ids were not read",
        )
        (root / "generation_config.json").write_text(json.dumps({"eos_token_id": 5}))
        require(
            load_stop_token_ids(root, SimpleNamespace(eos_token_id=6)) == frozenset({5, 6}),
            "a scalar stop id or the tokenizer's was dropped",
        )
        (root / "generation_config.json").unlink()
        require(
            load_stop_token_ids(root, SimpleNamespace(eos_token_id=None)) == frozenset(),
            "stop ids appeared without a source",
        )


def probe_masked_prefill_traces():
    """Batch widths, trace gating, staleness, and the device-data masks."""
    import threading
    from collections import deque
    from types import SimpleNamespace
    from unittest import mock

    import torch

    from models.demos.blackhole.qwen36.tt.model import Qwen36Model
    from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_seq import seq_valid_mask
    from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet import conv_valid_selector
    from qwen36_service.batch_decode_adapter import (
        TRACED_PREFILL_ENV,
        BatchedTracedDecodeAdapter,
        traced_prefill_enabled,
    )
    from qwen36_service.batch_service import BatchGenerationCoordinator
    from qwen36_service.sampled_mode_router_core import ROUTE_FUSED_STEP, ROUTE_SAMPLED

    def adapter_on(grid_x, grid_y):
        grid = SimpleNamespace(x=grid_x, y=grid_y)
        return SimpleNamespace(mesh=SimpleNamespace(compute_with_storage_grid_size=lambda: grid))

    supported = BatchedTracedDecodeAdapter.decode_width_supported
    p150 = adapter_on(11, 10)
    require(
        [w for w in range(1, 17) if not supported(p150, w)] == [13],
        "an 11x10 grid must shard every width to 16 except 13",
    )
    require(not supported(adapter_on(8, 8), 11), "a prime width above an 8x8 grid was accepted")

    # The sampled-fusion kernel places one (user, local value head) instance per core,
    # or two on a runtime that chunks instances: 4 or 9 users of 24 heads on 110 cores.
    def sampled_adapter(chunked):
        adapter = SimpleNamespace(
            decode_width_supported=lambda w: supported(p150, w),
            _compute_cores=lambda: 110,
            _local_value_heads=lambda: 24,
            _capability_record=SimpleNamespace(gdn_scan_chunked_instances=chunked),
        )
        adapter.sampled_width_supported = lambda w: BatchedTracedDecodeAdapter.sampled_width_supported(
            adapter, w
        )
        return adapter

    one_per_core = sampled_adapter(False)
    one_per_core.group_capacity_blocks = 16
    require(
        [w for w in range(1, 17) if one_per_core.sampled_width_supported(w)] == [1, 2, 3, 4],
        "the sampled kernel must fit exactly 4 users of 24 heads on 110 cores",
    )
    require(
        [w for w in range(1, 17) if sampled_adapter(True).sampled_width_supported(w)] == list(range(1, 10)),
        "two instances per core must fit exactly 9 users of 24 heads on 110 cores",
    )

    # The batch worker returns the newest requests until the width is shardable, and an
    # all-sampled batch until the sampled kernel fits it.
    def take(temperatures, sampled_admission=False):
        pending = [
            SimpleNamespace(request=SimpleNamespace(max_tokens=8, temperature=temperature), index=i)
            for i, temperature in enumerate(temperatures)
        ]
        worker = SimpleNamespace(
            _condition=threading.Condition(),
            _queue=deque(pending),
            _closing=False,
            adapter=one_per_core,
            sampled_admission_enabled=sampled_admission,
            deterministic_route=ROUTE_FUSED_STEP,
            _request_blocks=lambda _pending: 1,
        )
        batch = BatchGenerationCoordinator._take_batch(worker)
        return [item.index for item in batch], [item.index for item in worker._queue]

    require(take([0.0] * 13) == (list(range(12)), [12]), "a 13-request batch was not trimmed to 12")
    require(take([0.0] * 11) == (list(range(11)), []), "an 11-request batch was trimmed")
    require(take([0.0] * 16) == (list(range(16)), []), "a full batch was trimmed")
    require(
        take([0.7] * 6, sampled_admission=True) == ([0, 1, 2, 3], [4, 5]),
        "an all-sampled batch passed the sampled kernel's capacity",
    )
    require(
        take([0.7] * 5 + [0.0], sampled_admission=True) == (list(range(6)), []),
        "a mixed batch, which decodes deterministically, was trimmed to the sampled capacity",
    )
    require(take([0.7] * 6) == (list(range(6)), []), "a batch without sampled admission was trimmed")

    # Trace preparation warms the sampled route only at the widths its kernel fits.
    warmed = []
    preparing = SimpleNamespace(
        maximum_width=16,
        page_table=torch.zeros(16, 4, dtype=torch.int32),
        _startup_page_table=torch.zeros(16, 4, dtype=torch.int32),
        fusion_mode=ROUTE_FUSED_STEP,
        model=SimpleNamespace(
            READ_BATCH_WIDTH=4,
            warm_serving_programs=lambda page_table: None,
            capture_masked_prefill_traces=lambda row, read_page_table=None: None,
            masked_prefill_trace_buckets=lambda: (128,),
        ),
        _require_open=lambda: None,
        reset_all=lambda: None,
        _release_traces=lambda: None,
        prepare_width=lambda token_ids, positions: None,
        decode_width_supported=one_per_core.decode_width_supported,
        sampled_width_supported=one_per_core.sampled_width_supported,
        prefix_states=0,
    )

    def switch_fusion_mode(mode):
        preparing.fusion_mode = mode

    def replay(token_ids, positions):
        warmed.append((preparing.fusion_mode, len(token_ids)))

    preparing.switch_fusion_mode = switch_fusion_mode
    preparing.decode_traced = replay
    preparing.decode_traced_sampled = replay
    with mock.patch.dict(os.environ, {TRACED_PREFILL_ENV: "1"}):
        buckets = BatchedTracedDecodeAdapter.prepare_masked_prefill_traces(
            preparing, 7, (ROUTE_FUSED_STEP, ROUTE_SAMPLED)
        )
    require(buckets == (128,), "trace preparation did not report the captured buckets")
    require(
        [w for route, w in warmed if route == ROUTE_FUSED_STEP] == [w for w in range(1, 17) if w != 13],
        "the deterministic warm-up skipped a decodable width",
    )
    require(
        [w for route, w in warmed if route == ROUTE_SAMPLED] == [1, 2, 3, 4],
        "the sampled warm-up ran a width its kernel cannot fit",
    )
    require(preparing.fusion_mode == ROUTE_FUSED_STEP, "trace preparation left the sampled route selected")

    for value, expected in (("1", True), ("on", True), ("0", False), ("", False), ("maybe", False)):
        require(
            traced_prefill_enabled({TRACED_PREFILL_ENV: value}) is expected,
            f"traced prefill gate value {value!r} did not select {expected}",
        )
    require(not traced_prefill_enabled({}), "traced prefill defaulted on without its variable")

    # Stale traces are released: any program compiled after the capture may own trace scratch.
    def stale(buckets, entries_now, entries_at_capture=10):
        model = SimpleNamespace(
            masked_prefill_trace_buckets=lambda: buckets,
            mesh_device=SimpleNamespace(num_program_cache_entries=lambda: entries_now),
            _masked_prefill_program_entries=entries_at_capture,
        )
        return Qwen36Model.masked_prefill_traces_stale(model)

    def plan(buckets, length):
        return Qwen36Model.masked_prefill_trace_for(
            SimpleNamespace(masked_prefill_trace_buckets=lambda: buckets), length
        )

    full = (128, 256, 512, 1024)
    require(plan(full, 10) == [(128, 0, 10)], "a short prompt did not take the smallest bucket")
    require(plan(full, 1024) == [(1024, 0, 1024)], "a full chunk was split")
    require(
        plan(full, 1025) == [(1024, 0, 1024), (128, 1024, 1)],
        "a long prompt did not continue in the smallest tail bucket",
    )
    require(
        plan(full, 2047) == [(1024, 0, 1024), (1024, 1024, 1023)],
        "the longest prompt did not chunk like eager prefill",
    )
    require(plan((128, 256), 300) is None, "a prompt beyond every bucket was planned")
    require(plan((), 10) is None, "a plan was made without traces")

    require(not stale((), 99), "missing traces reported stale")
    require(not stale((128, 256), 10), "unchanged traces reported stale")
    require(stale((128, 256), 11), "a post-capture compile did not mark the traces stale")

    # The device-data masks select exactly the real tokens' conv window and scan rows.
    kernel, length, valid = 4, 16, 7
    padded = torch.arange(kernel - 1 + length, dtype=torch.float32)
    window = conv_valid_selector(1, kernel, kernel - 1 + length, valid)[0] @ padded
    require(
        window.tolist() == padded[valid : valid + kernel - 1].tolist(),
        "the conv selector does not pick the last K-1 real inputs",
    )
    lens = [3, 16]
    mask = seq_valid_mask(4, length, lens)[:, :, 0]
    require(
        mask.sum(dim=1).tolist() == [3.0, 3.0, 16.0, 16.0],
        "the scan mask does not keep each user's real rows",
    )


# r[verify tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency.valid]
def probe_gdn_occupancy_plan():
    """Positive and negative tests for the pure decode-occupancy planner.

    The planner must exploit idle cores by splitting a head's state and must refuse
    to split once the grid is looping, because there the instance count already
    exceeds the core count and per-core work cannot fall.
    """
    from qwen36_service.gdn_occupancy_core import (
        GdnOccupancyPolicyError,
        OccupancyPlan,
        REASON_NO_SPLIT_IMPROVES_COST,
        REASON_SPLIT_REFUSED_LOOPING_GRID,
        REASON_SPLIT_SELECTED,
        candidate_splits,
        instance_coordinates,
        plan_occupancy,
        validate_plan,
    )

    tp2_heads = 24
    compute_cores = 110
    value_tiles = 4
    fixed_share = 0.25
    floor = 0.5

    def plan(batch, share=fixed_share, occupancy=floor, tiles=value_tiles):
        return plan_occupancy(
            heads=tp2_heads,
            batch=batch,
            value_tiles=tiles,
            cores=compute_cores,
            fixed_cost_share=share,
            minimum_occupancy=occupancy,
        )

    # Idle grid: B=1 uses 24 of 110 cores, so splitting fills the grid and lowers
    # the estimated per-core cost.
    idle = plan(1)
    validate_plan(idle)
    require(idle.split == value_tiles, f"B=1 chose split {idle.split}")
    require(idle.reason == REASON_SPLIT_SELECTED, f"wrong B=1 reason: {idle.reason}")
    require(idle.instances == tp2_heads * value_tiles, "B=1 instance count is wrong")
    require(idle.instances_per_core == 1, "B=1 should keep one instance per core")
    require(idle.active_cores == 96, f"B=1 used {idle.active_cores} cores, expected 96")
    require(
        abs(idle.occupancy - 96 / compute_cores) < 1e-12,
        f"B=1 occupancy is {idle.occupancy}",
    )
    require(
        idle.estimated_relative_cost < 1.0,
        "splitting at B=1 did not lower the estimated cost",
    )

    # Near-full grid: B=4 leaves 14 idle cores and splitting would double the
    # instance count, so the planner keeps the unsplit launch.
    near_full = plan(4)
    validate_plan(near_full)
    require(near_full.split == 1, f"B=4 chose split {near_full.split}")
    require(
        near_full.reason == REASON_NO_SPLIT_IMPROVES_COST,
        f"wrong B=4 reason: {near_full.reason}",
    )

    # Looping grid: B=8 and B=16 already run two and four instances per core, and
    # a split multiplies instances without reducing any core's work.
    for batch, per_core in ((8, 2), (16, 4)):
        looping = plan(batch)
        validate_plan(looping)
        require(looping.split == 1, f"B={batch} split the state on a looping grid")
        require(
            looping.reason == REASON_SPLIT_REFUSED_LOOPING_GRID,
            f"wrong B={batch} reason: {looping.reason}",
        )
        require(
            looping.instances_per_core == per_core,
            f"B={batch} has {looping.instances_per_core} instances per core",
        )
        require(
            looping.estimated_relative_cost == float(per_core),
            f"B={batch} cost is {looping.estimated_relative_cost}",
        )

    # A fixed cost share of zero makes splitting purely beneficial on an idle grid;
    # an occupancy floor above what a split can reach keeps the unsplit launch.
    free_split = plan(1, share=0.0)
    require(free_split.split == value_tiles, "a zero fixed share did not split fully")
    unreachable = plan(1, occupancy=0.99)
    require(
        unreachable.split == 1 and unreachable.reason == REASON_NO_SPLIT_IMPROVES_COST,
        f"an unreachable floor still split: {unreachable.split}/{unreachable.reason}",
    )

    # Candidate splits divide the value width exactly.
    require(candidate_splits(4) == (1, 2, 4), f"candidate splits: {candidate_splits(4)}")
    require(candidate_splits(3) == (1, 3), f"candidate splits: {candidate_splits(3)}")
    require(instance_coordinates(0, heads=2, split=2) == (0, 0, 0), "flat 0 maps wrong")
    require(instance_coordinates(5, heads=2, split=2) == (1, 0, 1), "flat 5 maps wrong")

    # Malformed plans must not survive validation: a split that does not divide
    # the value width, a wrong instance count, a repeated instance, a missing one,
    # a non-contiguous chunk, an over-long chunk, and mismatched occupancy or cost.
    good = idle
    malformed = {
        "split does not divide": OccupancyPlan(
            **{**good.__dict__, "split": 3}
        ),
        "wrong instance count": OccupancyPlan(
            **{**good.__dict__, "instances": good.instances - 1}
        ),
        "missing instance": OccupancyPlan(
            **{
                **good.__dict__,
                "per_core_instances": (tuple(range(0, 95)),),
                "active_cores": 1,
                "occupancy": 1 / good.cores,
            }
        ),
        "repeated instance": OccupancyPlan(
            **{**good.__dict__, "per_core_instances": ((0, 0), (1,))}
        ),
        "non-contiguous chunk": OccupancyPlan(
            **{**good.__dict__, "per_core_instances": ((0,), (2,))}
        ),
        "over-long chunk": OccupancyPlan(
            **{**good.__dict__, "per_core_instances": ((0, 1),)}
        ),
        "occupancy mismatch": OccupancyPlan(
            **{**good.__dict__, "occupancy": good.occupancy / 2}
        ),
        "cost mismatch": OccupancyPlan(
            **{**good.__dict__, "estimated_relative_cost": good.estimated_relative_cost * 2}
        ),
    }
    for name, broken in malformed.items():
        try:
            validate_plan(broken)
        except GdnOccupancyPolicyError:
            pass
        else:
            raise AssertionError(f"malformed occupancy plan was accepted: {name}")

    # Rejected inputs: non-positive geometry, a share outside [0, 1), a floor
    # outside (0, 1], and non-numeric or boolean inputs.
    rejected = (
        dict(heads=0, batch=1, value_tiles=4, cores=110, fixed_cost_share=0.25),
        dict(heads=1, batch=0, value_tiles=4, cores=110, fixed_cost_share=0.25),
        dict(heads=1, batch=1, value_tiles=0, cores=110, fixed_cost_share=0.25),
        dict(heads=1, batch=1, value_tiles=4, cores=0, fixed_cost_share=0.25),
        dict(heads=1, batch=1, value_tiles=4, cores=110, fixed_cost_share=1.0),
        dict(heads=1, batch=1, value_tiles=4, cores=110, fixed_cost_share=-0.1),
        dict(heads=1, batch=1, value_tiles=4, cores=110, fixed_cost_share="half"),
        dict(
            heads=1,
            batch=1,
            value_tiles=4,
            cores=110,
            fixed_cost_share=0.25,
            minimum_occupancy=0.0,
        ),
        dict(
            heads=1,
            batch=1,
            value_tiles=4,
            cores=110,
            fixed_cost_share=0.25,
            minimum_occupancy=1.5,
        ),
        dict(heads=True, batch=1, value_tiles=4, cores=110, fixed_cost_share=0.25),
    )
    for arguments in rejected:
        try:
            plan_occupancy(**arguments)
        except GdnOccupancyPolicyError:
            pass
        else:
            raise AssertionError(f"invalid occupancy request was accepted: {arguments}")

    for bad in (0.5, -1, "x"):
        try:
            instance_coordinates(bad, heads=24, split=4)
        except GdnOccupancyPolicyError:
            pass
        else:
            raise AssertionError(f"invalid flat index was accepted: {bad!r}")


# r[verify tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency.plan]
def probe_gdn_occupancy_policy_binding():
    """The occupancy policy is identity-bound and cannot drive a split unmeasured.

    The planner needs the measured share of a single instance's cost that does not
    divide with the value width. Until that is measured the policy records
    ``not-measured`` and the planner must refuse it, so no split can be selected
    from an unmeasured cost model.
    """
    from qwen36_service.gdn_occupancy_core import (
        GdnOccupancyPolicyError,
        plan_occupancy,
    )

    policy_path = Path(__file__).with_name("gdn-occupancy-policy.json")
    with policy_path.open(encoding="utf-8") as policy_file:
        policy = json.load(policy_file)
    require(policy.get("schema_version") == 1, "occupancy policy schema changed")
    require(
        policy.get("profile_id") == "qwen38-p150x2-gdn-occupancy-v1",
        "occupancy policy identity changed",
    )
    require(
        policy.get("value_tiles") == 4,
        f"occupancy policy value_tiles is {policy.get('value_tiles')}",
    )
    require(
        isinstance(policy.get("minimum_occupancy"), (int, float)),
        "occupancy policy minimum_occupancy is not a number",
    )
    share = policy.get("fixed_cost_share")
    require(
        share == "not-measured",
        f"the occupancy policy carries an unmeasured share as {share!r}",
    )

    # Fail closed: the planner must refuse the policy's own placeholder, so the
    # shipped policy cannot select a split before the share is measured.
    try:
        plan_occupancy(
            heads=24,
            batch=1,
            value_tiles=int(policy["value_tiles"]),
            cores=110,
            fixed_cost_share=share,
            minimum_occupancy=float(policy["minimum_occupancy"]),
        )
    except GdnOccupancyPolicyError:
        pass
    else:
        raise AssertionError("an unmeasured occupancy policy selected a split")

    # The same call with a measured share plans a split, so the refusal above is
    # the placeholder and not the geometry.
    measured = plan_occupancy(
        heads=24,
        batch=1,
        value_tiles=int(policy["value_tiles"]),
        cores=110,
        fixed_cost_share=0.25,
        minimum_occupancy=float(policy["minimum_occupancy"]),
    )
    require(measured.split > 1, "a measured share did not plan a split")


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


def probe_structured_reads():
    """Structured reads: slots, read rows, deduplicated draws, refinement, stages, limits."""
    import re
    import threading
    import time
    from collections import deque
    from types import SimpleNamespace

    import torch
    from models.demos.blackhole.qwen36.tt.model import Qwen36Model
    from qwen36_service import structured_read_core as core
    from qwen36_service.batch_service import BatchGenerationCoordinator, _PendingRead

    vocabulary = {}

    def encode(text):
        # Word-level: a leading space joins its word; punctuation and newlines stand alone.
        return [
            vocabulary.setdefault(piece, len(vocabulary))
            for piece in re.findall(r" ?\w+|\n| ?\?|[^\w\s]| ", text)
        ]

    labels = {piece: encode(piece)[0] for piece in (" yes", " no", " A", " B", " C", " 1", " 2", " 3")}
    width = 64 + len(vocabulary) + 256
    reads = []
    preferred = {}
    strength = {}

    def read(requests):
        reads.append(requests)
        results = []
        for request in requests:
            logits = torch.zeros(len(request.positions), width)
            noise = 2.0 if any(token >= 1000 for token in request.token_ids) else 0.0
            for row in range(len(request.positions)):
                # Row i prefers its configured label (" yes" by default) with its configured
                # strength; noise in the prefill shifts the logits, so noise draws differ.
                logits[row, preferred.get(row, labels[" yes"])] = strength.get(row, 6.0) + noise
            results.append(core.DeviceRead(logits=logits, seconds=0.25, route="fake"))
        return tuple(results)

    def ports(**overrides):
        values = dict(
            encode=encode,
            decode=lambda ids: " ".join(str(i) for i in ids),
            chat_prompt=lambda system, user, thinking: encode(system + "\n" + user) + ([7] if thinking else []),
            read=read,
            think=lambda prompt, budget: [5, 6, 999, 4],
            noise_upper=2000,
            thought_close_id=999,
            thought_close_ids=(11, 12),
            maximum_sequence_length=2048,
        )
        values.update(overrides)
        return core.ReadPorts(**values)

    def schema_of(questions, **extensions):
        return core.parse_jev_request(
            {"state": {"ticket": "down"}, "questions": questions, **extensions}, maximum_think_tokens=64
        )

    noul = {"type": "noul", "instructions": "Urgent?"}
    choice = {"type": "choice", "instructions": "Route", "criteria": {"billing": None, "outage": "down"}}
    score = {"type": "score", "instructions": "Tone", "criteria": ["calm", "annoyed", "furious"]}

    # Every label changes one token at one slot per question, in template order.
    schema, _state, _seed = schema_of({"urgent": noul, "bucket": choice, "tone": score})
    template = core.resolve_template(schema.questions, encode, "", schema.answer_format)
    require(
        [slot.label_ids for slot in template.slots]
        == [(labels[" yes"], labels[" no"]), (labels[" A"], labels[" B"]), (labels[" 1"], labels[" 2"], labels[" 3"])],
        "template slots do not carry each question's label ids",
    )
    require(
        [template.tokens[slot.position] for slot in template.slots] == [labels[" yes"], labels[" A"], labels[" 1"]],
        "template slots do not point at the first label",
    )
    wide = dict(choice, criteria={"billing": None, "outage": None, "long label": None})
    try:
        core.resolve_template(
            schema_of({"bucket": wide})[0].questions,
            lambda text: encode(text.replace(" C", " C C")),
            "",
            core.FORMAT_LINES,
        )
        require(False, "a two-token label resolved to a slot")
    except core.SchemaError:
        pass

    # A read stops before the last slot's token and reads the row before each slot.
    request = core.build_read([90, 91], template, [1001, 1002, 1003])
    require(
        request.positions == tuple(2 + slot.position - 1 for slot in template.slots),
        "read rows are not the rows before each slot",
    )
    require(len(request.token_ids) == 2 + template.slots[-1].position, "a read kept tokens no row can see")
    require(
        request.token_ids[2 + template.slots[0].position] == 1001
        and request.token_ids[2 + template.slots[1].position] == 1002,
        "noise fills did not replace the slot tokens",
    )

    # Label probabilities are the softmax over the labels' exact log-probabilities.
    logits = torch.full((width,), -10.0)
    logits[labels[" yes"]], logits[labels[" no"]], logits[3] = 2.0, 1.0, 3.0
    slot_read = core.summarize_slot(logits, (labels[" yes"], labels[" no"]))
    require(
        abs(slot_read.probabilities[0] - 1 / (1 + math.exp(-1.0))) < 1e-5,
        "label probabilities are not the label softmax",
    )
    require(not slot_read.argmax_is_label, "a non-label argmax was reported as a label")
    require(0.0 < slot_read.label_mass < 1.0, "label mass left the unit interval")

    # One question: every noise draw is the same prefill, so the device runs it once.
    reads.clear()
    schema, state, seed = schema_of({"urgent": noul}, samples=4)
    result, _rows = core.decide(ports(), schema, state, seed)
    timing = result["diagnostics"]["timing"]
    require(
        timing["reads"] == 4 and timing["device_reads"] == 1 and sum(len(r) for r in reads) == 1,
        "identical draws reached the device more than once",
    )
    require(result["answers"]["urgent"]["label"] == "yes", "the favoured label did not win")
    require(result["answers"]["urgent"]["stderr"] == 0.0, "identical draws reported a spread")

    # Auto policy: a confident first read stops; an uncertain one reads auto_max draws.
    reads.clear()
    schema, state, seed = schema_of({"urgent": noul, "tone": score}, auto_threshold=100.0)
    result, _rows = core.decide(ports(), schema, state, seed)
    require(result["diagnostics"]["samples"]["n"] == 1, "a confident first read was extended")
    reads.clear()
    schema, state, seed = schema_of({"urgent": noul, "tone": score}, auto_threshold=0.0, auto_max=3)
    result, _rows = core.decide(ports(), schema, state, seed)
    require(
        result["diagnostics"]["samples"]["n"] == 3 and result["diagnostics"]["samples"]["policy"]["extended"],
        "an uncertain first read was not extended to auto_max",
    )
    # Only the first question is uncertain: its row sees no noise, so no draw is added.
    reads.clear()
    strength.update({0: 1.0, 1: 40.0})
    schema, state, seed = schema_of({"urgent": noul, "tone": score}, auto_threshold=0.1, steps=1)
    result, _rows = core.decide(ports(), schema, state, seed)
    strength.clear()
    first = result["diagnostics"]["samples"]["policy"]["first_read_entropy"]
    require(first["urgent"] > 0.1 > first["tone"], "the gate fixture lost its entropy split")
    require(
        result["diagnostics"]["samples"]["n"] == 1 and len(reads) == 1,
        "an uncertain first slot added draws that cannot change it",
    )
    # A one-question read costs one prefill across the auto extension and every step.
    reads.clear()
    schema, state, seed = schema_of({"urgent": noul}, auto_threshold=0.0, steps=5)
    result, _rows = core.decide(ports(), schema, state, seed)
    require(
        result["diagnostics"]["timing"]["device_reads"] == 1 and len(reads) == 1,
        "a one-question decision read the device more than once",
    )
    # By default the first draw refines to its fixed point, and later draws read once, under
    # sampled earlier answers: here the refined " yes" reuses its prefill and " no" adds one.
    reads.clear()
    strength.update({0: 0.0})
    schema, state, seed = schema_of({"urgent": noul, "tone": score}, samples=4)
    core.decide(ports(), schema, state, seed)
    strength.clear()
    require(
        [len(batch) for batch in reads] == [1, 1, 1],
        "the default did not refine the first draw and read later draws once",
    )

    # Steps refine to the label chain and stop at the fixed point.
    reads.clear()
    schema, state, seed = schema_of({"urgent": noul, "tone": score}, samples=1, steps=5)
    core.decide(ports(), schema, state, seed)
    require(len(reads) == 2, "refinement did not stop at its fixed point")
    tmpl = core.resolve_template(schema.questions, encode, "", schema.answer_format)
    second = reads[1][0]
    require(
        all(second.token_ids[second.positions[i] + 1] == slot.label_ids[0] for i, slot in enumerate(tmpl.slots[:-1])),
        "a refinement step did not fill slots with the previous argmax",
    )
    first = reads[0][0]
    require(
        all(first.token_ids[first.positions[i] + 1] == slot.placeholder for i, slot in enumerate(tmpl.slots[:-1]))
        and all(slot.placeholder == encode(" ?")[0] for slot in tmpl.slots),
        "the first draw did not fill its slots with the placeholder",
    )

    # Later draws fill earlier slots with labels sampled from the first draw, never noise,
    # and draws that sample the same visible labels share one prefill.
    reads.clear()
    strength.update({0: 0.0})
    schema, state, seed = schema_of({"urgent": noul, "tone": score}, auto_threshold=0.0, auto_max=8)
    result, _rows = core.decide(ports(), schema, state, seed)
    strength.clear()
    later = [request for batch in reads[1:] for request in batch]
    tmpl = core.resolve_template(schema.questions, encode, "", schema.answer_format)
    slot_row = later[0].positions[0] + 1 if later else 0
    require(
        result["diagnostics"]["samples"]["n"] == 8
        and later
        and all(request.token_ids[slot_row] in tmpl.slots[0].label_ids for request in later),
        "a later draw did not fill its slot with a sampled label",
    )
    require(
        result["diagnostics"]["timing"]["device_reads"] <= 1 + len(tmpl.slots[0].label_ids),
        "draws with the same sampled labels did not share a prefill",
    )

    # A digit label the tokenizer keeps apart from its space takes the bare placeholder token.
    def digits_apart(text):
        return [
            vocabulary.setdefault(piece, len(vocabulary))
            for piece in re.findall(r" ?[A-Za-z_]+|\d| ?\?|\n|[^\w\s]| ", text)
        ]

    apart = core.resolve_template(schema.questions, digits_apart, "", "lines")
    require(
        apart.slots[0].placeholder == digits_apart(" ?")[0] and apart.slots[1].placeholder == digits_apart("?")[0],
        "a digit slot did not take the bare placeholder",
    )

    # A slot whose placeholder is not one token falls back to noise.
    split = core.resolve_template(schema.questions, lambda text: encode(text.replace("?", "? ?")), "", "lines")
    require(all(slot.placeholder is None for slot in split.slots), "a split placeholder was accepted")
    require(
        all(0 <= fill < 2000 for fill in core.first_fills(split, 7, 2000)),
        "a slot without a placeholder was not filled with noise",
    )

    # Stages: a dependent question reads after its dependency, with that answer in the prompt;
    # a failed ask_if skips the question.
    reads.clear()
    schema, state, seed = schema_of(
        {
            "urgent": noul,
            "tone": dict(score, depends_on=["urgent"]),
            "bucket": dict(choice, ask_if={"urgent": ["no"]}),
        },
        samples=1,
    )
    result, _rows = core.decide(ports(), schema, state, seed)
    require(result["diagnostics"]["stages"] == [["urgent"], ["tone"]], "stages ignored depends_on")
    require(result["answers"]["bucket"] is None, "a failed ask_if still asked its question")
    first_prefill, second_prefill = reads[0][0].token_ids, reads[1][0].token_ids
    answered = encode("urgent: yes")
    require(
        any(second_prefill[i : i + len(answered)] == tuple(answered) for i in range(len(second_prefill))),
        "a later stage did not continue the earlier answer",
    )
    require(second_prefill[: len(first_prefill) - 2] == first_prefill[: len(first_prefill) - 2],
            "a later stage changed the shared prompt")

    # A thought ends at the model's close tag and is followed by the canonical close.
    reads.clear()
    schema, state, seed = schema_of({"urgent": noul}, samples=1, think=8)
    result, _rows = core.decide(ports(), schema, state, seed)
    require(result["diagnostics"]["thought"]["tokens"] == 2 and result["diagnostics"]["thought"]["closed"],
            "the thought was not cut at the close tag")
    require(reads[0][0].token_ids[-6:-2] == (5, 6, 11, 12), "the read prefix did not close the thought")

    # One read holds at most 31 slots.
    many = {f"q{i}": noul for i in range(40)}
    schema, _state, _seed = schema_of(many)
    groups = core.chunk_groups(schema, schema.questions, encode)
    require([len(g) for g in groups] == [31, 9], "a read was planned past the logit tile")
    require(schema.answer_format == core.FORMAT_INDEXED, "a large schema kept the line format")

    # A read longer than the service holds is refused before the device.
    reads.clear()
    schema, state, seed = schema_of({"urgent": noul})
    try:
        core.decide(ports(maximum_sequence_length=8), schema, state, seed)
        require(False, "an over-long read was sent to the device")
    except core.SchemaError:
        require(not reads, "an over-long read reached the device")

    for body, reason in (
        ({"state": "s", "questions": {}}, "empty questions"),
        ({"state": "s", "questions": {"u": noul}, "images": ["data:image/png;base64,AA"]}, "images"),
        ({"state": "s", "questions": {"u": noul}, "think": 65}, "think past the generation budget"),
        ({"state": "s", "questions": {"u": dict(choice, criteria={"a": None})}}, "one alternative"),
        ({"state": "s", "questions": {"u": dict(noul, depends_on=["v"])}}, "unknown dependency"),
        ({"state": "s", "questions": {"u": dict(noul, depends_on=["v"]), "v": dict(noul, depends_on=["u"])}},
         "dependency cycle"),
        ({"state": "s", "questions": {"u": noul, "v": dict(noul, depends_on=["u"])}, "ask": ["v"]},
         "ask without its dependency"),
        ({"questions": {"u": noul}}, "missing state"),
        ({"state": "s", "questions": {"u": noul}, "samples": 0}, "zero samples"),
    ):
        try:
            core.parse_jev_request(body, maximum_think_tokens=64)
            require(False, f"{reason} was accepted")
        except core.SchemaError:
            pass

    # Jev shapes: noul probability, choice with confidence, 0-indexed score legend.
    schema, state, seed = schema_of({"urgent": noul, "bucket": choice, "tone": score}, samples=1)
    preferred.update({0: labels[" no"]})
    response = core.jev_response(schema, *core.decide(ports(), schema, state, seed), "Qwen3.8-27B")
    preferred.clear()
    answers = response["answers"]
    require(set(answers["urgent"]) == {"type", "noul"} and answers["urgent"]["noul"] < 0.5,
            "the noul answer is not the yes probability")
    require(set(answers["tone"]["legend"]) == {"0", "1", "2"}, "the score legend is not 0-indexed")
    require(answers["bucket"]["choice"] in ("billing", "outage"), "the choice answer is not an option name")

    # The worker runs a read job on its own and never folds reads into a generation batch.
    def take(items):
        worker = SimpleNamespace(
            _condition=threading.Condition(),
            _queue=deque(items),
            _closing=False,
            adapter=SimpleNamespace(decode_width_supported=lambda w: True, group_capacity_blocks=16),
            sampled_admission_enabled=False,
            _request_blocks=lambda _pending: 1,
        )
        return BatchGenerationCoordinator._take_batch(worker), list(worker._queue)

    job = _PendingRead(requests=((torch.zeros(1, 4), (1,)),), deadline=0.0)
    other = _PendingRead(requests=((torch.zeros(1, 4), (2,)),), deadline=0.0)
    generation = [SimpleNamespace(request=SimpleNamespace(max_tokens=8)) for _ in range(2)]
    taken, left = take([job, generation[0], other, generation[1]])
    require(taken == (job, other) and left == generation, "queued read jobs were not merged apart from generation")
    taken, left = take([generation[0], job, generation[1]])
    require(taken == (generation[0], generation[1]) and left == [job], "a generation batch absorbed a read job")

    # A merged turn fails only its expired jobs; the others read in one call.
    calls = []

    def read_rows_batch(flat):
        calls.append(flat)
        return tuple(SimpleNamespace(masked_traces_released=False, elapsed_seconds=0.1, route="fake") for _ in flat)

    worker = SimpleNamespace(adapter=SimpleNamespace(read_rows_batch=read_rows_batch), emitter=lambda *a, **k: None)
    expired = _PendingRead(requests=((torch.zeros(1, 4), (1,)),), deadline=time.monotonic() - 1.0)
    fresh = _PendingRead(
        requests=((torch.zeros(1, 4), (2,)), (torch.zeros(1, 4), (3,))), deadline=time.monotonic() + 3600.0
    )
    BatchGenerationCoordinator._run_reads(worker, (expired, fresh))
    require(
        expired.error is not None and expired.results is None,
        "an expired read job was run",
    )
    require(
        fresh.error is None and len(fresh.results) == 2 and [len(flat) for flat in calls] == [2],
        "an expired read job failed the jobs merged with it",
    )

    # Batched replays: shortest first, up to the width and the logit rows, and only when a
    # group meets its bucket's minimum; long reads and small groups go alone.
    def batches(lengths, rows):
        return core.plan_read_batches(
            lengths=lengths, row_counts=rows, width=4, minimum_by_bucket={128: 2, 256: 3}, maximum_rows=32
        )

    require(batches([100] * 6, [3] * 6) == ((0, 1, 2, 3), (4, 5)), "reads were not packed to the batch width")
    require(batches([100, 300, 100], [3, 3, 3]) == ((1,), (0, 2)), "a read past the batched buckets was batched")
    require(batches([100] * 3, [20, 20, 5]) == ((0,), (1, 2)), "a batch exceeded the logit rows")
    require(batches([200] * 2, [3] * 2) == ((0,), (1,)), "a group below its bucket's minimum was batched")
    require(batches([200] * 3, [3] * 3) == ((0, 1, 2),), "a group at its bucket's minimum read alone")
    require(batches([200, 100, 200, 100], [3] * 4) == ((1, 3, 0, 2),), "reads were not grouped shortest first")
    require(batches([100], [40]) == ((0,),), "a read wider than the logit tile was batched")

    # A saved read prefix matches only its own row and tokens, and any prefill that writes one
    # of its KV blocks, or passes a page table the host cannot inspect, drops it.
    def saved(blocks=range(480, 496)):
        row = torch.arange(480, 512, dtype=torch.int32).reshape(1, 32)
        tokens = torch.arange(1024, dtype=torch.int32)
        return SimpleNamespace(_read_prefix=(row, tokens, frozenset(blocks))), row, tokens

    model, row, tokens = saved()
    require(Qwen36Model._read_prefix_matches(model, row, tokens), "a saved prefix did not match itself")
    changed = tokens.clone()
    changed[1023] += 1
    require(not Qwen36Model._read_prefix_matches(model, row, changed), "a prefix with other tokens matched")
    require(not Qwen36Model._read_prefix_matches(model, row - 32, tokens), "a prefix on another row matched")
    Qwen36Model._invalidate_read_prefix(model, torch.arange(0, 480, dtype=torch.int32).reshape(15, 32))
    require(model._read_prefix is not None, "generation below full width dropped the saved prefix")
    Qwen36Model._invalidate_read_prefix(model, torch.arange(0, 512, dtype=torch.int32).reshape(16, 32))
    require(model._read_prefix is None, "a full-width prefill kept a prefix whose KV it overwrote")
    model, row, tokens = saved()
    Qwen36Model._invalidate_read_prefix(model, SimpleNamespace(shape=(1, 32)))
    require(model._read_prefix is None, "a device page table kept the saved prefix")

    # The read selector keeps row 0 for the generation logit and stages reads in rows 1..
    selector = Qwen36Model._logit_selector(128, 20, (3, 7))
    require(
        tuple(selector.shape) == (1, 1, 32, 128)
        and selector[0, 0, 0, 19] == 1.0
        and selector[0, 0, 1, 3] == 1.0
        and selector[0, 0, 2, 7] == 1.0
        and float(selector.sum()) == 3.0,
        "the read selector does not stage row 0 and the read rows",
    )
    for bad_rows in (tuple(range(32)), (20,)):
        try:
            Qwen36Model._logit_selector(128, 20, bad_rows)
            require(False, f"read rows {bad_rows[:3]} were staged")
        except AssertionError:
            pass



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
    probe("GDN fused decode step policy", probe_gdn_decode_step_policy)
    probe("TP=2 / P300 gate invariants", probe_p150x2_gate)
    probe("traced decode pure core", probe_traced_decode_contract)
    probe("batch decode scheduler", probe_batch_decode_scheduler)
    probe("pooled KV rows", probe_kv_pool)
    probe("continuous slot policy", probe_continuous_slot_policy)
    probe("continuous slot worker", probe_continuous_slot_worker)
    probe("continuous slot adapter", probe_continuous_slot_adapter)
    probe("continuous slot admission", probe_continuous_slot_admission)
    probe("prefix cache policy", probe_prefix_cache_policy)
    probe("prefix cache restore", probe_prefix_cache_restore)
    probe("prefix cache admission", probe_prefix_cache_admission)
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
    probe("chat output parser", probe_chat_output_parser)
    probe("chat template rendering", probe_chat_template_rendering)
    probe("streaming responses", probe_streaming)
    probe("kernel distribution admission", probe_kernel_distribution_admission)
    probe("kernel distribution probe builder", probe_kernel_distribution_probe)
    probe("sampled-mode router", probe_sampled_mode_router)
    probe("sampled decode", probe_sampled_decode)
    probe("MTP serving round policy", probe_mtp_serving_core)
    probe("GDN grid instance cap assignment", probe_gdn_grid_instance_cap)
    probe("GDN grid in-range mapping unchanged", probe_gdn_grid_mapping_unchanged)
    probe("grouped prefill eligibility", probe_grouped_prefill_eligibility)
    probe("grouped prefill routing", probe_grouped_prefill_routing)
    probe("masked prefill traces", probe_masked_prefill_traces)
    probe("end of sequence", probe_end_of_sequence)
    probe("structured reads", probe_structured_reads)
    probe("GDN decode occupancy plan", probe_gdn_occupancy_plan)
    probe("GDN occupancy policy binding", probe_gdn_occupancy_policy_binding)

    if FAILURES:
        print("qwen36 device-free check: FAIL")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("qwen36 device-free check: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
