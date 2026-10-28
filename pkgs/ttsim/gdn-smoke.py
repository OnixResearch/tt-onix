#!/usr/bin/env python3
"""Check the fused gated-delta recurrence against a NumPy reference."""

import argparse
import json
from pathlib import Path

import numpy as np

DEVICE_ID = 0
TILE_EDGE = 32
HEAD_TILES = 4
HEAD_DIM = HEAD_TILES * TILE_EDGE
HEAD_COUNT = 2
RECURRENT_STEPS = 3
ROW_AXIS = -2
COLUMN_AXIS = -1
ABSOLUTE_TOLERANCE = 1e-4
RELATIVE_TOLERANCE = 1e-3
VALUE_SCALE = 0.1
STATE_SCALE = 0.01
BETA = 0.7
LOG_DECAY = -0.2
HALF = 0.5
DIRECTION_OFFSET = 0.3
ORACLE_VALUE = 0.25
CORRUPTION = 1.0
PRECISION_REGRESSION_DELTA = 2.2e-4
FLOAT_DTYPE = np.float32
VECTOR_SHAPE = (HEAD_COUNT, 1, HEAD_DIM)
STATE_SHAPE = (HEAD_COUNT, HEAD_DIM, HEAD_DIM)
SCALAR_SHAPE = (HEAD_COUNT, 1, 1)
WORKLOAD_NAME = "gdn-fp32-reference-state-carry"


def configure(head_count, head_tiles, recurrent_steps):
    """Override the default smoke shape for a model-representative replay."""
    global HEAD_COUNT, HEAD_TILES, HEAD_DIM, RECURRENT_STEPS
    global VECTOR_SHAPE, STATE_SHAPE, SCALAR_SHAPE
    if head_count <= 0 or head_tiles <= 0 or recurrent_steps <= 0:
        raise ValueError("head count, head tiles, and steps must be positive")
    HEAD_COUNT = head_count
    HEAD_TILES = head_tiles
    HEAD_DIM = HEAD_TILES * TILE_EDGE
    RECURRENT_STEPS = recurrent_steps
    VECTOR_SHAPE = (HEAD_COUNT, 1, HEAD_DIM)
    STATE_SHAPE = (HEAD_COUNT, HEAD_DIM, HEAD_DIM)
    SCALAR_SHAPE = (HEAD_COUNT, 1, 1)


def reference_step(q, k, v, beta, g, state):
    """Return the output and next state without changes to the inputs."""
    expected_shapes = (
        VECTOR_SHAPE,
        VECTOR_SHAPE,
        VECTOR_SHAPE,
        SCALAR_SHAPE,
        SCALAR_SHAPE,
        STATE_SHAPE,
    )
    for value, expected in zip((q, k, v, beta, g, state), expected_shapes, strict=True):
        if value.shape != expected or value.dtype != FLOAT_DTYPE:
            raise ValueError(
                f"expected float32 {expected}, got {value.dtype} {value.shape}"
            )
        if not np.isfinite(value).all():
            raise ValueError("reference inputs must be finite")
    decayed = np.exp(g) * state
    delta = beta * (v - k @ decayed)
    next_state = decayed + np.swapaxes(k, ROW_AXIS, COLUMN_AXIS) @ delta
    return q @ next_state, next_state


def assert_close(actual, expected):
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise AssertionError("the result has an invalid shape or non-finite values")
    np.testing.assert_allclose(
        actual, expected, atol=ABSOLUTE_TOLERANCE, rtol=RELATIVE_TOLERANCE
    )


def inputs_for_step(step):
    index = np.arange(np.prod(VECTOR_SHAPE), dtype=FLOAT_DTYPE).reshape(VECTOR_SHAPE)
    q = np.sin(index + step + 1)
    k = np.cos(index + step + DIRECTION_OFFSET)
    q /= np.linalg.norm(q, axis=COLUMN_AXIS, keepdims=True) * np.sqrt(HEAD_DIM)
    k /= np.linalg.norm(k, axis=COLUMN_AXIS, keepdims=True)
    v = (VALUE_SCALE * np.sin(index + step)).astype(FLOAT_DTYPE)
    beta = np.full(SCALAR_SHAPE, BETA, dtype=FLOAT_DTYPE)
    g = np.full(SCALAR_SHAPE, LOG_DECAY, dtype=FLOAT_DTYPE)
    return q.astype(FLOAT_DTYPE), k.astype(FLOAT_DTYPE), v, beta, g


def host_checks():
    basis = np.zeros(VECTOR_SHAPE, dtype=FLOAT_DTYPE)
    basis[..., 0] = 1
    values = np.full(VECTOR_SHAPE, ORACLE_VALUE, dtype=FLOAT_DTYPE)
    ones = np.ones(SCALAR_SHAPE, dtype=FLOAT_DTYPE)
    zero_g = np.zeros(SCALAR_SHAPE, dtype=FLOAT_DTYPE)
    state = np.zeros(STATE_SHAPE, dtype=FLOAT_DTYPE)
    output, next_state = reference_step(basis, basis, values, ones, zero_g, state)
    assert_close(output, values)
    expected_state = np.zeros_like(state)
    expected_state[:, 0, :] = ORACLE_VALUE
    assert_close(next_state, expected_state)
    assert_close(state, np.zeros_like(state))

    decay = np.full(SCALAR_SHAPE, np.log(HALF), dtype=FLOAT_DTYPE)
    output, retained = reference_step(
        basis, basis, values, np.zeros_like(ones), decay, next_state
    )
    assert_close(retained, HALF * next_state)
    assert_close(output, HALF * values)

    invalid_state = state.copy()
    invalid_state.flat[0] = np.nan
    for invalid in (state[..., :-1], invalid_state):
        try:
            reference_step(basis, basis, values, ones, zero_g, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("the reference admitted an invalid state")
    precision_values = np.full(VECTOR_SHAPE, STATE_SCALE, dtype=FLOAT_DTYPE)
    assert_close(precision_values + ABSOLUTE_TOLERANCE * HALF, precision_values)
    corrupt_results = (
        (values + CORRUPTION, values),
        (precision_values + PRECISION_REGRESSION_DELTA, precision_values),
    )
    for actual, expected in corrupt_results:
        try:
            assert_close(actual, expected)
        except AssertionError:
            pass
        else:
            raise AssertionError("the numerical check admitted a corrupted result")


def device_checks(validation_only=False):
    import ttnn

    def to_device(value, dtype=ttnn.float32):
        return ttnn.Tensor(value, dtype, device, ttnn.TILE_LAYOUT)

    def to_host(value):
        return np.asarray(value.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_numpy())

    device = ttnn.open_device(device_id=DEVICE_ID)
    try:
        device.enable_program_cache()
        initial = (
            STATE_SCALE * np.sin(np.arange(np.prod(STATE_SHAPE), dtype=FLOAT_DTYPE))
        ).reshape(STATE_SHAPE)
        expected_state = initial.copy()
        state = to_device(initial)
        errors = []
        for step in range(0 if validation_only else RECURRENT_STEPS):
            host_inputs = inputs_for_step(step)
            expected_output, expected_state = reference_step(
                *host_inputs, expected_state
            )
            device_inputs = tuple(to_device(value) for value in host_inputs)
            cache_entries_before = device.num_program_cache_entries()
            # First use fresh output storage. Later calls exercise in-place state carry
            # and the cache-hit path with newly allocated input buffers.
            output, next_state = ttnn.transformer.gated_delta_decode(
                *device_inputs,
                state,
                output_state=None if step == 0 else state,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.synchronize_device(device)
            if step > 1 and device.num_program_cache_entries() != cache_entries_before:
                raise AssertionError(
                    "the recurrent step did not reuse the cached program"
                )
            actual_output = to_host(output)
            actual_state = to_host(next_state)
            assert_close(actual_output, expected_output)
            assert_close(actual_state, expected_state)
            if step == 0:
                assert_close(to_host(state), initial)
            else:
                assert_close(to_host(state), expected_state)
            errors.append(float(np.max(np.abs(actual_state - expected_state))))
            state = next_state

        host_inputs = inputs_for_step(RECURRENT_STEPS)
        device_inputs = tuple(to_device(value) for value in host_inputs)
        q_host, _, v_host, _, _ = host_inputs
        q, k, v, beta, g = device_inputs
        bad_q = to_device(q_host, ttnn.bfloat16)
        bad_v = to_device(v_host[..., :TILE_EDGE].copy())
        invalid_cases = (
            ((bad_q, k, v, beta, g), "float32"),
            ((q, k, bad_v, beta, g), "val_dim"),
        )
        for arguments, diagnostic in invalid_cases:
            try:
                ttnn.transformer.gated_delta_decode(
                    *arguments, state, output_state=state
                )
            except (RuntimeError, ValueError) as error:
                if diagnostic not in str(error):
                    raise AssertionError(f"unexpected rejection: {error}") from error
            else:
                raise AssertionError(
                    f"the device operation admitted invalid {diagnostic}"
                )
            assert_close(to_host(state), expected_state)

        valid_config = {
            "math_fidelity": ttnn.MathFidelity.HiFi4,
            "math_approx_mode": False,
            "fp32_dest_acc_en": True,
            "packer_l1_acc": False,
            "dst_full_sync_en": False,
        }
        invalid_configs = (
            {"math_fidelity": ttnn.MathFidelity.HiFi2},
            {"math_approx_mode": True},
            {"fp32_dest_acc_en": False},
            {"packer_l1_acc": True},
            {"dst_full_sync_en": True},
        )
        for override in invalid_configs:
            config = ttnn.WormholeComputeKernelConfig(**(valid_config | override))
            try:
                ttnn.transformer.gated_delta_decode(
                    *device_inputs,
                    state,
                    output_state=state,
                    compute_kernel_config=config,
                )
            except (RuntimeError, ValueError) as error:
                if "requires HiFi4" not in str(error):
                    raise AssertionError(
                        f"unexpected config rejection: {error}"
                    ) from error
            else:
                raise AssertionError(
                    f"the operation admitted invalid config: {override}"
                )
            assert_close(to_host(state), expected_state)

        if validation_only:
            return {
                "workload": "gdn-invalid-input-rejection",
                "status": "passed",
                "negative_cases": len(invalid_cases) + len(invalid_configs),
                "non_claims": [
                    "numerical correctness",
                    "recurrent state carry",
                    "cache reuse",
                ],
            }

        expected_output, expected_state = reference_step(*host_inputs, expected_state)
        output, state = ttnn.transformer.gated_delta_decode(
            *device_inputs,
            state,
            output_state=state,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(**valid_config),
        )
        assert_close(to_host(output), expected_output)
        assert_close(to_host(state), expected_state)
        return {
            "workload": WORKLOAD_NAME,
            "status": "passed",
            "head_count": HEAD_COUNT,
            "head_dim": HEAD_DIM,
            "recurrent_steps": RECURRENT_STEPS,
            "negative_cases": len(invalid_cases) + len(invalid_configs),
            "post_rejection_retry": "passed",
            "cache_reuse_checked": True,
            "state_max_abs_errors": errors,
        }
    finally:
        ttnn.close_device(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--host-only", action="store_true")
    mode.add_argument("--validation-only", action="store_true")
    parser.add_argument("--result", type=Path)
    parser.add_argument("--head-count", type=int, default=HEAD_COUNT)
    parser.add_argument("--head-tiles", type=int, default=HEAD_TILES)
    parser.add_argument("--steps", type=int, default=RECURRENT_STEPS)
    parser.add_argument("--workload-name", default="gdn-fp32-reference-state-carry")
    arguments = parser.parse_args()
    global WORKLOAD_NAME
    WORKLOAD_NAME = arguments.workload_name
    configure(arguments.head_count, arguments.head_tiles, arguments.steps)
    host_checks()
    if arguments.host_only:
        print("GDN reference positive and negative checks passed")
    else:
        result = json.dumps(device_checks(arguments.validation_only), sort_keys=True)
        if arguments.result is not None:
            arguments.result.write_text(result + "\n")
        print(result, flush=True)


if __name__ == "__main__":
    main()
