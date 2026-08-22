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
    probe("TP=2 / P300 gate invariants", probe_p150x2_gate)
    probe("traced decode pure core", probe_traced_decode_contract)
    probe("prefill program profile", probe_prefill_program_profile)
    probe("MTP target-verification pure core", probe_mtp_target_verification)
    probe("MTP drafter pure core", probe_mtp_drafter)
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
