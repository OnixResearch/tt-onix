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
    stale_binding = TraceBinding(
        **{
            **binding.__dict__,
            "buffer_generation": binding.buffer_generation + 1,
        }
    )
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
    require(admitted.admitted, f"passing decode evidence was rejected: {admitted.reasons}")
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
        torch.equal(
            bounded_table, adapter.page_table[:, :rounded_required_blocks]
        ),
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
