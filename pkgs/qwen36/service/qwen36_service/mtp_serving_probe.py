"""Bounded, non-admitting physical comparison of the serving route."""

import statistics
import time

from .core import CompletionRequest, Endpoint
from .server import DECODE_MODE_EAGER, QwenEngine

GENERATION_TOKENS = 64
MAXIMUM_PROMPTS = 5


def compare_serving(
    *,
    model,
    tokenizer,
    decode_adapter,
    drafter,
    model_path,
    decode_profile,
    identity,
    model_alias,
):
    """Compare plain traced decode with real greedy serving requests."""
    if drafter.trace_profile is None:
        raise ValueError("serving comparison requires the MTP trace profile")
    engine = QwenEngine(
        model,
        tokenizer,
        model_alias,
        decode_profile.maximum_sequence_length,
        model_path=model_path,
        decode_mode=DECODE_MODE_EAGER,
        decode_profile=decode_profile,
        runtime_identity=identity.runtime_identity,
        package_identity=identity.package_identity,
        model_identity=identity.model_identity,
        enable_prefill_profile=False,
        enable_host_timeline_profile=False,
        mtp_serving_enabled=True,
        mtp_drafter_profile=drafter.profile,
        mtp_trace_profile=drafter.trace_profile,
    )
    engine._decode_adapter = decode_adapter
    engine._mtp_drafter = drafter
    observations = []
    for prompt in drafter.profile.prompts[:MAXIMUM_PROMPTS]:
        tokens = tokenizer(prompt, return_tensors="pt").input_ids
        decode_adapter.release_trace()
        decode_adapter.reset_for_request()
        started = time.monotonic()
        control, metrics = engine._run_persistent_generation(
            tokens,
            GENERATION_TOKENS,
            started + decode_profile.request_timeout_seconds,
        )
        if metrics["decode_path"] != "traced":
            raise RuntimeError("serving control did not use traced plain decode")
        plain_seconds = statistics.median(metrics["token_latency_s"])
        engine._mtp_plain_step_seconds = plain_seconds
        request = CompletionRequest(
            endpoint=Endpoint.COMPLETIONS,
            model=model_alias,
            max_tokens=GENERATION_TOKENS,
            prompt=prompt,
            messages=(),
            temperature=0.0,
        )
        response = engine.generate(request)
        candidate = response["x_onix_decode_evidence"]
        served = candidate["generated_token_ids"]
        parity = served == control
        first_divergence = None
        if not parity:
            first_divergence = next(
                (
                    index
                    for index, (left, right) in enumerate(zip(control, served))
                    if left != right
                ),
                min(len(control), len(served)),
            )
        observations.append({
            "prompt": prompt,
            "control_token_ids": control,
            "candidate": candidate,
            "token_parity": parity,
            "plain_step_seconds": plain_seconds,
            "first_divergence_index": first_divergence,
            "minimum_verification_margin": candidate.get("minimum_verification_margin"),
        })
        if not parity:
            raise RuntimeError(
                "speculative serving token mismatch at index "
                f"{first_divergence}: control {control[first_divergence]} "
                f"served {served[first_divergence]}; classify against the "
                "minimum verification top-two margin before blaming the round"
            )
        if candidate["mtp_rounds"] < 1:
            raise RuntimeError("serving comparison did not execute speculation")
    return {
        "simulated": False,
        "admission_eligible": False,
        "generation_tokens": GENERATION_TOKENS,
        "observations": observations,
        "non_claims": [
            "This comparison does not enable production speculation.",
            "Host fixtures do not establish physical state equality.",
        ],
    }
