"""TTNN adapter for isolated batched Qwen decode on P150x2."""

# r[impl tenstorrent.qwen38_p150x2.batched_concurrency]

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import ttnn
from models.demos.blackhole.qwen36.tt.generator_interface import unpack_rope
from models.demos.blackhole.qwen36.tt.model import masked_prefill_plan
from models.tt_transformers.tt.common import copy_host_to_device

from .batch_decode_core import (
    MAXIMUM_ACTIVE_BATCH_WIDTH,
    MINIMUM_ACTIVE_BATCH_WIDTH,
    TRACE_REUSE_REASON_BINDING,
    TracePrefillPlan,
    plan_same_width_trace_prefill,
    plan_slot_group_trace,
)
from .decode_adapter import TRACE_PAGE_TABLE_ALIGNMENT_BLOCKS
from .decode_core import DecodeProfile
from .grouped_prefill_core import (
    REASON_ELIGIBLE,
    GroupedPrefillPolicyError,
    RuntimeCapability,
    plan_grouped_prefill,
)
from .kv_pool_core import (
    READ_BATCH_MAX_BUCKET_TOKENS,
    READ_BATCH_MIN_BUCKET_TOKENS,
    KvPoolError,
    KvPoolLayout,
)
from .sampled_decode_core import ShardCandidates
from .sampled_mode_router_core import ROUTE_SAMPLED, ROUTES
from .structured_read_core import plan_read_batches

MAX_REDUCTION_COLUMNS = 32
TILE_EDGE_ELEMENTS = 32
PADDED_LOGIT_SENTINEL = -1.0e30
TRACE_COMMAND_QUEUE_ID = 0
TRACE_BLOCKING_REPLAY = False
GREEDY_COMBINE_DEVICE_COUNT = 2
# The token input is the first decode input; a greedy stream feeds it back on device.
GREEDY_FED_BACK_INPUT_COUNT = 1
# Replays kept queued ahead of the host's readback in a greedy stream.
GREEDY_STREAM_DEPTH = 1
MUTABLE_INPUT_COUNT = 3
EXPECTED_DECODE_INPUT_COUNT = 4
MINIMUM_POSITION = 0

# Grouped prefill stays opt-in until its own matched screen admits it, so the
# default served path keeps the execution identity its receipts describe.
GROUPED_PREFILL_ENV = "QWEN35_GROUPED_PREFILL"
TRACED_PREFILL_ENV = "QWEN36_TRACED_PREFILL"
GROUPED_PREFILL_ENABLED_VALUES = frozenset({"1", "true", "yes", "on"})
RUNTIME_CAPABILITY_RELATIVE_PATH = ("share", "qwen36", "runtime-capability.json")
CAPABILITY_SCHEMA_VERSION = 1
# Instances per core validated on silicon for the chunked scan: 192 instances
# over 110 cores at chunk 2, bitwise equal to the serialized reference.
CHUNKED_VALIDATED_INSTANCES_PER_CORE = 2
ONE_INSTANCE_PER_CORE = 1
NON_CHUNKED_CAPABILITY_SOURCE = "fallback-one-instance-per-core"
PACKAGE_CAPABILITY_SOURCE = "package-runtime-capability"
GRID_UNAVAILABLE_REASON = "compute-grid-unavailable"
NON_POSITIONAL_DESTINATIONS_REASON = "non-positional-state-destinations"
TRACE_PARKED_REASON = "decode-trace-parked"
MASKED_TRACE_REASON = "masked-prefill-trace"


class BatchDecodeAdapterError(RuntimeError):
    """The TTNN batch adapter cannot preserve user-state isolation."""


class TraceBindingDriftError(BatchDecodeAdapterError):
    """A resident decode trace no longer binds the buffers it was captured with."""


@dataclass(frozen=True)
class BatchPrefillResult:
    first_token_ids: tuple[int, ...]
    elapsed_seconds: float
    first_token_ready_seconds: float
    page_table_blocks: int
    state_install_seconds: float
    trace_reuse_planned: bool = False
    trace_transition_reason: str | None = None
    prefill_route: str = "per-user"
    prefill_route_reason: str | None = None
    masked_traces_released: bool = False


@dataclass(frozen=True)
class TracePreparationResult:
    capture_seconds: float
    capture_count: int
    reused: bool
    resident_trace_count: int
    binding_stable: bool
    transition_reason: str | None


@dataclass(frozen=True)
class SlotRefillResult:
    """One request prefilled into its decode slot beside the other slots' live state."""

    first_token_id: int
    elapsed_seconds: float
    first_token_ready_seconds: float
    masked_traces_used: bool
    masked_traces_released: bool
    # The prefill compiled a program or dropped the masked traces, so the resident
    # decode trace was released instead of replayed; the next window recaptures.
    decode_trace_released: bool


@dataclass(frozen=True)
class StructuredReadRows:
    """Host next-token logits after each requested position of one structured read."""

    logits: torch.Tensor
    elapsed_seconds: float
    route: str
    masked_traces_released: bool
    # Physical KV blocks this replay may have written, including padded bucket rows.
    kv_written_blocks: frozenset[int]


# The decode slot a structured read's prefill installs into. Reads never decode, and the
# next generation batch resets every slot before its own prefill.
STRUCTURED_READ_SLOT = 0
READ_ROUTE_MASKED_TRACE = "masked-trace"
READ_ROUTE_EAGER = "eager-masked-body"
READ_ROUTE_BATCHED_TRACE = "batched-read-trace"
# A traced read that restored its first 1,024 tokens' state from an earlier read.
READ_ROUTE_PREFIX_REUSE = "masked-trace-prefix-reuse"
# Fewest sequences a batched replay must carry to beat reading them alone, per bucket. A
# 4-wide replay costs the same for any fill. P150x2, 2026-09-25: a 128-token replay took
# 0.42 s against 0.22 s per single read, and a 256-token replay 0.68 s against 0.25 s.
READ_BATCH_MINIMUM_BY_BUCKET = {128: 2, 256: 3}


@dataclass
class _BatchTrace:
    width: int
    trace_id: Any
    device_inputs: tuple[Any, ...]
    logits: Any
    indices: Any
    values: Any
    binding: tuple[str, ...]
    capture_seconds: float


# Per-shard candidate depth of the sampled trace tail. Matches the model's
# recommended sampling top_k; the host merge keeps the best of both shards.
SAMPLED_CANDIDATE_COUNT = 20


@dataclass(frozen=True)
class RuntimeCapabilityRecord:
    """What the installed package says about its own linked runtime."""

    gdn_scan_chunked_instances: bool
    source: str


def traced_prefill_enabled(env: Any = None) -> bool:
    """Return whether the caller opted into masked prefill traces."""
    if env is None:
        env = os.environ
    return (
        env.get(TRACED_PREFILL_ENV, "").strip().lower()
        in GROUPED_PREFILL_ENABLED_VALUES
    )


def grouped_prefill_enabled(env: Any = None) -> bool:
    """Return whether the caller opted into grouped prefill."""
    if env is None:
        env = os.environ
    value = env.get(GROUPED_PREFILL_ENV, "")
    if not isinstance(value, str):
        return False
    return value.strip().lower() in GROUPED_PREFILL_ENABLED_VALUES


def load_runtime_capability_record(
    package_root: Path | None = None,
) -> RuntimeCapabilityRecord:
    """Read the package's own runtime capability record, failing closed.

    A missing or malformed record yields the one-instance-per-core reading, so
    an unreadable record can only make the planner more conservative.
    """
    if package_root is None:
        package_root = Path(__file__).resolve().parents[4]
    path = package_root.joinpath(*RUNTIME_CAPABILITY_RELATIVE_PATH)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return RuntimeCapabilityRecord(False, NON_CHUNKED_CAPABILITY_SOURCE)
    if (
        not isinstance(record, dict)
        or record.get("schema_version") != CAPABILITY_SCHEMA_VERSION
    ):
        return RuntimeCapabilityRecord(False, NON_CHUNKED_CAPABILITY_SOURCE)
    chunked = record.get("gdn_scan_chunked_instances")
    if not isinstance(chunked, bool):
        return RuntimeCapabilityRecord(False, NON_CHUNKED_CAPABILITY_SOURCE)
    return RuntimeCapabilityRecord(chunked, PACKAGE_CAPABILITY_SOURCE)


@dataclass(frozen=True)
class _GdnSnapshot:
    layer: Any
    recurrent: Any
    convolution: tuple[Any, ...]


@dataclass(frozen=True)
class _GdnZeroSources:
    layer: Any
    recurrent: Any
    convolution: Any
    convolution_carry: Any


class BatchedTracedDecodeAdapter:
    """Own fixed-capacity user state and exact-width decode traces."""

    def __init__(
        self,
        *,
        model: Any,
        profile: DecodeProfile,
        maximum_width: int = MAXIMUM_ACTIVE_BATCH_WIDTH,
        prefix_states: int = 0,
    ) -> None:
        if model.num_devices <= MINIMUM_ACTIVE_BATCH_WIDTH:
            raise BatchDecodeAdapterError(
                "batched TP decode requires more than one device"
            )
        if (
            not MINIMUM_ACTIVE_BATCH_WIDTH
            <= maximum_width
            <= MAXIMUM_ACTIVE_BATCH_WIDTH
        ):
            raise BatchDecodeAdapterError("batch capacity is outside policy")
        if int(model.args.max_batch_size) < maximum_width:
            raise BatchDecodeAdapterError(
                "model batch capacity is smaller than adapter capacity"
            )
        if model.args.vocab_size % model.num_devices != 0:
            raise BatchDecodeAdapterError(
                "vocabulary size is not divisible by the device count"
            )
        self.model = model
        self.profile = profile
        self.maximum_width = maximum_width
        # Saved carried-state buffers for the hybrid prefix cache, allocated with the masked
        # prefill traces; 0 disables the cache.
        self.prefix_states = prefix_states
        self.mesh = model.mesh_device
        self.vocabulary_size = int(model.args.vocab_size)
        self.device_count = int(model.num_devices)
        self.vocabulary_per_shard = self.vocabulary_size // self.device_count
        self.blocks_per_user = self._aligned_blocks_per_user()
        try:
            self.kv_pool = KvPoolLayout(
                pool_blocks=profile.kv_pool_blocks,
                row_blocks=self.blocks_per_user,
                kv_block_size=profile.kv_block_size,
            )
            if int(model.READ_BATCH_WIDTH) != self.kv_pool.batch_read_width or tuple(
                model.READ_BATCH_BUCKETS
            ) != (READ_BATCH_MIN_BUCKET_TOKENS, READ_BATCH_MAX_BUCKET_TOKENS):
                raise KvPoolError(
                    "model batched read traces do not match reserved KV lanes"
                )
            startup_rows = self.kv_pool.startup_rows(width=maximum_width)
        except KvPoolError as error:
            raise BatchDecodeAdapterError(str(error)) from error
        # Warm-up and single-read capture retain their original rows. Batched reads
        # capture against four dedicated rows outside the generation block range.
        self._startup_page_table = torch.tensor(startup_rows, dtype=torch.int32)
        self._read_page_table = self._startup_page_table[-1:].clone()
        self._batch_read_page_table = torch.tensor(
            self.kv_pool.batch_read_rows(), dtype=torch.int32
        )
        self.page_table = self._startup_page_table.clone()
        self._mesh_reader = ttnn.ConcatMeshToTensor(self.mesh, dim=0)
        self._traces: dict[int, _BatchTrace] = {}
        self._pending_trace_plan: TracePrefillPlan | None = None
        self._prior_reset_complete = True
        self._closed = False
        # Continuous slots (begin_slots): the trace width, each owned slot's KV run, and
        # whether the resident trace's inputs must be restaged before the next replay.
        self._slot_width: int | None = None
        self._slot_runs: dict[int, tuple[int, int]] = {}
        self._slot_inputs_stale = False
        # Decode traces captured for slot groups since startup, for served evidence.
        self.slot_trace_captures = 0
        self._capability_record = load_runtime_capability_record()
        self._allocate_persistent_state()
        self._gdn_zero_sources = self._capture_gdn_zero_sources()

    def _aligned_blocks_per_user(self) -> int:
        block_count = (
            self.profile.maximum_sequence_length + self.profile.kv_block_size - 1
        ) // self.profile.kv_block_size
        return (
            (block_count + TRACE_PAGE_TABLE_ALIGNMENT_BLOCKS - 1)
            // TRACE_PAGE_TABLE_ALIGNMENT_BLOCKS
        ) * TRACE_PAGE_TABLE_ALIGNMENT_BLOCKS

    def _allocate_persistent_state(self) -> None:
        cache_shape = [
            self.kv_pool.pool_blocks,
            int(self.model.args.n_local_kv_heads),
            self.profile.kv_block_size,
            int(self.model.args.head_dim),
        ]
        self.model.allocate_kv_caches(
            cache_shape,
            ttnn.bfloat16,
            batch_size=self.maximum_width,
        )

    def request_blocks(self, *, prompt_tokens: int, generation_tokens: int) -> int:
        """KV blocks one request's row takes from the group pool."""
        try:
            return self.kv_pool.request_blocks(
                prompt_tokens=prompt_tokens, generation_tokens=generation_tokens
            )
        except KvPoolError as error:
            raise BatchDecodeAdapterError(str(error)) from error

    @property
    def group_capacity_blocks(self) -> int:
        """KV blocks one generation group may take across all of its rows."""
        return self.kv_pool.group_capacity_blocks

    def _capture_gdn_zero_sources(self) -> tuple[_GdnZeroSources, ...]:
        sources = []
        for layer in self.model.layers:
            if layer.is_full_attention:
                continue
            attention = layer.attention
            if (
                attention.rec_state is None
                or attention.conv_states is None
                or attention.conv_carry is None
                or attention._zero_rec is None
                or attention._zero_conv0 is None
                or attention._zero_conv_carry is None
            ):
                raise BatchDecodeAdapterError("GDN zero state is not allocated")
            if tuple(attention._zero_rec.shape) != tuple(attention.rec_state.shape):
                raise BatchDecodeAdapterError(
                    "GDN recurrent zero shape differs from decode state"
                )
            if tuple(attention._zero_conv0.shape) != tuple(
                attention.conv_states[0].shape
            ):
                raise BatchDecodeAdapterError(
                    "GDN convolution zero shape differs from decode state"
                )
            sources.append(
                _GdnZeroSources(
                    layer=attention,
                    recurrent=attention._zero_rec,
                    convolution=attention._zero_conv0,
                    convolution_carry=attention._zero_conv_carry,
                )
            )
        if not sources:
            raise BatchDecodeAdapterError("model has no GDN zero state")
        return tuple(sources)

    def prepare_width(
        self,
        *,
        token_ids: tuple[int, ...],
        positions: tuple[int, ...],
    ) -> TracePreparationResult:
        """Reuse one safe trace or capture one exact-width replacement."""
        width = len(token_ids)
        self._require_width(width)
        self._require_open()
        if len(positions) != width:
            raise BatchDecodeAdapterError("trace token and position widths differ")
        if any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or not 0 <= token < self.vocabulary_size
            for token in token_ids
        ):
            raise BatchDecodeAdapterError("trace token is outside the vocabulary")
        if any(
            isinstance(position, bool)
            or not isinstance(position, int)
            or not MINIMUM_POSITION <= position < self.profile.maximum_sequence_length
            for position in positions
        ):
            raise BatchDecodeAdapterError("trace position is outside the profile bound")
        pending_plan = self._pending_trace_plan
        self._pending_trace_plan = None
        transition_reason = pending_plan.reason if pending_plan is not None else None
        if pending_plan is not None and pending_plan.reuse_after_prefill:
            resident_trace = self._traces.get(width)
            if resident_trace is not None and self._binding_matches(resident_trace):
                # The resident trace still holds the previous group's rows.
                host_inputs = self.model.prepare_decode_inputs_host(
                    torch.tensor(token_ids, dtype=torch.int32).reshape(
                        width,
                        MINIMUM_ACTIVE_BATCH_WIDTH,
                    ),
                    torch.tensor(positions, dtype=torch.int32),
                    page_table=self.page_table[:width].contiguous(),
                )
                copy_host_to_device(
                    host_inputs[MUTABLE_INPUT_COUNT:],
                    device_tensors=resident_trace.device_inputs[MUTABLE_INPUT_COUNT:],
                )
                return TracePreparationResult(
                    capture_seconds=0.0,
                    capture_count=0,
                    reused=True,
                    resident_trace_count=len(self._traces),
                    binding_stable=True,
                    transition_reason=None,
                )
            transition_reason = TRACE_REUSE_REASON_BINDING
            self._release_traces()
        elif self._traces:
            self._release_traces()

        tokens = torch.tensor(token_ids, dtype=torch.int32).reshape(
            width,
            MINIMUM_ACTIVE_BATCH_WIDTH,
        )
        position_tensor = torch.tensor(positions, dtype=torch.int32)
        host_inputs = self.model.prepare_decode_inputs_host(
            tokens,
            position_tensor,
            page_table=self.page_table[:width].contiguous(),
        )
        device_inputs = tuple(copy_host_to_device(host_inputs, mesh_device=self.mesh))
        if len(device_inputs) != EXPECTED_DECODE_INPUT_COUNT:
            raise BatchDecodeAdapterError("batch trace inputs are incomplete")

        trace_id = None
        logits = None
        indices = None
        values = None
        snapshots = self._clone_gdn_state()
        capture_started = time.monotonic()
        try:
            warm_logits = self._forward_sharded(device_inputs)
            warm_indices, warm_values = self._reduce_for_mode(warm_logits, width)
            self._feed_back_winners(warm_indices, warm_values, device_inputs)
            ttnn.synchronize_device(self.mesh)
            for tensor in (warm_indices, warm_values, warm_logits):
                if tensor is not None:
                    ttnn.deallocate(tensor)
            self._restore_gdn_state(snapshots)
            ttnn.synchronize_device(self.mesh)

            trace_id = ttnn.begin_trace_capture(
                self.mesh,
                cq_id=TRACE_COMMAND_QUEUE_ID,
            )
            logits = self._forward_sharded(device_inputs)
            indices, values = self._reduce_for_mode(logits, width)
            self._feed_back_winners(indices, values, device_inputs)
            ttnn.end_trace_capture(
                self.mesh,
                trace_id,
                cq_id=TRACE_COMMAND_QUEUE_ID,
            )
            ttnn.synchronize_device(self.mesh)
            capture_seconds = time.monotonic() - capture_started
            if (
                not math.isfinite(capture_seconds)
                or capture_seconds <= 0
                or capture_seconds > self.profile.trace_capture_timeout_seconds
            ):
                raise BatchDecodeAdapterError("batch trace capture exceeded its bound")
            trace = _BatchTrace(
                width=width,
                trace_id=trace_id,
                device_inputs=device_inputs,
                logits=logits,
                indices=indices,
                values=values,
                binding=self._observe_binding(device_inputs, logits),
                capture_seconds=capture_seconds,
            )
            self._restore_gdn_state(snapshots)
            ttnn.synchronize_device(self.mesh)
            self._traces[width] = trace
            return TracePreparationResult(
                capture_seconds=capture_seconds,
                capture_count=1,
                reused=False,
                resident_trace_count=len(self._traces),
                binding_stable=True,
                transition_reason=transition_reason,
            )
        except (AssertionError, OSError, RuntimeError, ValueError) as error:
            self._restore_gdn_state(snapshots)
            ttnn.synchronize_device(self.mesh)
            if trace_id is not None:
                try:
                    ttnn.release_trace(self.mesh, trace_id)
                except (AssertionError, OSError, RuntimeError, ValueError):
                    pass
            for tensor in (indices, values, logits):
                if tensor is not None:
                    ttnn.deallocate(tensor)
            for tensor in device_inputs:
                if tensor is not None:
                    ttnn.deallocate(tensor)
            raise BatchDecodeAdapterError("batch trace capture failed") from error
        finally:
            self._deallocate_gdn_state(snapshots)

    def _compute_cores(self) -> int | None:
        """Measured compute-core count of one die, or None when unreadable."""
        grid = None
        try:
            grid = self.mesh.compute_with_storage_grid_size()
        except Exception:  # noqa: BLE001 - any API shape difference is unmeasurable
            submeshes = getattr(self.mesh, "get_devices", None)
            if callable(submeshes):
                devices = submeshes()
                if devices:
                    try:
                        grid = devices[0].compute_with_storage_grid_size()
                    except Exception:  # noqa: BLE001
                        grid = None
        if grid is None:
            return None
        cores = int(getattr(grid, "x", 0)) * int(getattr(grid, "y", 0))
        return cores if cores > 0 else None

    def _local_value_heads(self) -> int:
        for layer in self.model.layers:
            if getattr(layer, "is_full_attention", False):
                continue
            attention = getattr(layer, "attention", None)
            for candidate in (
                getattr(attention, "Nv", None),
                getattr(attention, "num_value_heads", None),
                getattr(layer, "Nv", None),
            ):
                if isinstance(candidate, int) and candidate > 0:
                    return candidate
        raise BatchDecodeAdapterError("model exposes no local value-head count")

    def plan_prefill_route(
        self,
        width: int,
        valid_lengths: tuple[int, ...],
        *,
        grouped_enabled: bool | None = None,
    ) -> tuple[bool, int, str]:
        """Decide the prefill route with explicit, positional destinations.

        Returns ``(grouped, group_size, reason)``. The grouped path installs
        state by row, so it is used only when the destinations are exactly the
        user rows in order; any other destination mapping falls back to the
        per-user route that installs per slot.
        """
        if grouped_enabled is None:
            grouped_enabled = grouped_prefill_enabled()
        if not grouped_enabled:
            return False, ONE_INSTANCE_PER_CORE, "grouped-prefill-disabled"
        cores = self._compute_cores()
        if cores is None:
            return False, ONE_INSTANCE_PER_CORE, GRID_UNAVAILABLE_REASON
        slots = tuple(range(width))
        record = self._capability_record
        capability = RuntimeCapability(
            chunked_scan_instances=record.gdn_scan_chunked_instances,
            compute_cores=cores,
            instances_per_core=(
                CHUNKED_VALIDATED_INSTANCES_PER_CORE
                if record.gdn_scan_chunked_instances
                else ONE_INSTANCE_PER_CORE
            ),
        )
        bucket_chunk = int(self.model.args.gdn_chunk_size)
        bucket = max(
            bucket_chunk,
            ((max(valid_lengths) + bucket_chunk - 1) // bucket_chunk) * bucket_chunk,
        )
        try:
            plan = plan_grouped_prefill(
                user_count=width,
                user_lengths=tuple(valid_lengths),
                slots=slots,
                bucket=bucket,
                local_value_heads=self._local_value_heads(),
                gdn_bucket_chunk=bucket_chunk,
                capability=capability,
                capacity_units=self.maximum_width,
                units_per_user=MINIMUM_ACTIVE_BATCH_WIDTH,
            )
        except GroupedPrefillPolicyError:
            return False, ONE_INSTANCE_PER_CORE, NON_POSITIONAL_DESTINATIONS_REASON
        if not plan.grouped:
            return False, ONE_INSTANCE_PER_CORE, plan.reason
        if plan.destinations != slots:
            return False, ONE_INSTANCE_PER_CORE, NON_POSITIONAL_DESTINATIONS_REASON
        # The planner's groups are contiguous runs of its group size, the model's own
        # partition. The model assembles them in order into rows 0..width-1 and
        # zero-fills the rest of the decode buffer, so partial widths and several
        # groups install positionally.
        return True, len(plan.groups[0]), REASON_ELIGIBLE

    def prefill_users(
        self,
        token_ids: tuple[torch.Tensor, ...],
        *,
        generation_tokens: int,
        trace_reuse_enabled: bool = False,
    ) -> BatchPrefillResult:
        """Prefill isolated state after a pure one-trace transition decision."""
        width = len(token_ids)
        self._require_width(width)
        self._require_open()
        trace_plan = plan_same_width_trace_prefill(
            requested_width=width,
            resident_widths=tuple(sorted(self._traces)),
            reuse_enabled=trace_reuse_enabled,
            prior_reset_complete=self._prior_reset_complete,
            binding_stable=self._single_trace_binding_stable(),
        )
        if not trace_plan.reuse_after_prefill:
            self._release_traces()
        self._pending_trace_plan = trace_plan
        self._close_slot_group()
        self.reset_all()
        self._prior_reset_complete = False
        required_blocks = []
        valid_lengths = []
        for user_tokens in token_ids:
            if not isinstance(user_tokens, torch.Tensor) or user_tokens.ndim != 2:
                raise BatchDecodeAdapterError("prefill tokens must be rank-two tensors")
            if user_tokens.shape[0] != MINIMUM_ACTIVE_BATCH_WIDTH:
                raise BatchDecodeAdapterError("prefill tokens must contain one user")
            prompt_tokens = int(user_tokens.shape[1])
            required_blocks.append(
                self.request_blocks(
                    prompt_tokens=prompt_tokens,
                    generation_tokens=generation_tokens,
                )
            )
            valid_lengths.append(prompt_tokens)
        page_table_blocks = max(required_blocks)
        try:
            rows = self.kv_pool.assign_group(required_blocks, width=self.maximum_width)
        except KvPoolError as error:
            raise BatchDecodeAdapterError(str(error)) from error
        self.page_table = torch.tensor(rows, dtype=torch.int32)
        # Full-width rows keep one program per prefill shape; a request-bounded width
        # would compile page-table programs per width, and a first-time compile releases
        # the masked prefill traces. Entries past each row's run point at the scratch block.
        prefill_page_table = self.page_table[:width].contiguous()
        started = time.monotonic()
        observed_first_tokens: tuple[int, ...] | None = None
        first_token_ready_seconds: float | None = None

        def observe_first_tokens(host_logits: list[torch.Tensor]) -> None:
            nonlocal observed_first_tokens, first_token_ready_seconds
            if (
                observed_first_tokens is not None
                or first_token_ready_seconds is not None
            ):
                raise BatchDecodeAdapterError(
                    "prefill reported first tokens more than once"
                )
            if len(host_logits) != width:
                raise BatchDecodeAdapterError(
                    "prefill output count differs from active width"
                )
            first_token_ids = []
            for user_logits in host_logits:
                rows = user_logits.float().reshape(-1, self.vocabulary_size)
                if rows.shape[0] != MINIMUM_ACTIVE_BATCH_WIDTH:
                    raise BatchDecodeAdapterError(
                        "prefill logits have the wrong row count"
                    )
                first_token_ids.append(int(torch.argmax(rows[0]).item()))
            observed_first_tokens = tuple(first_token_ids)
            first_token_ready_seconds = time.monotonic() - started

        grouping_enabled = grouped_prefill_enabled()
        # Masked prefill traces. A replay overwrites every device buffer allocated after
        # its capture that is still alive, so:
        # - a program compiled since the capture may own such a buffer; the traces are then
        #   released for the process, never replayed (the model also checks after the
        #   eager users of a mixed batch);
        # - a parked decode trace (trace reuse) keeps eager prefill.
        masked_traces_released = False
        if self.model.masked_prefill_traces_stale():
            self.model.release_masked_prefill_traces()
            masked_traces_released = True
        traced_buckets = self.model.masked_prefill_trace_buckets()
        use_masked_traces = bool(traced_buckets) and not trace_plan.reuse_after_prefill
        if use_masked_traces and any(
            self.model.masked_prefill_trace_for(length) is not None
            for length in valid_lengths
        ):
            # One replay per user beats a grouped eager forward at every width.
            grouped, group_size, route_reason = (
                False,
                ONE_INSTANCE_PER_CORE,
                MASKED_TRACE_REASON,
            )
        elif trace_plan.reuse_after_prefill:
            # A parked decode trace must not meet a first-time grouped program
            # compile, so trace reuse keeps the per-user programs.
            grouped, group_size, route_reason = (
                False,
                ONE_INSTANCE_PER_CORE,
                TRACE_PARKED_REASON,
            )
        else:
            grouped, group_size, route_reason = self.plan_prefill_route(
                width, tuple(valid_lengths), grouped_enabled=grouping_enabled
            )
        prefill_route = "grouped" if grouped else "per-user"
        if grouped:
            # The grouped path returns host logits for every user, so readiness
            # is observed once it returns, and state installs by row == slot.
            host_logits = self.model.prefill_paged_grouped(
                list(token_ids),
                prefill_page_table,
                valid_lens=valid_lengths,
                group_size=group_size,
            )
            if len(host_logits) != width:
                raise BatchDecodeAdapterError(
                    "grouped prefill output count differs from active width"
                )
            # The grouped path returns replicated device logits, so read them back
            # the same way the model does for its own per-user logits.
            host_logits = [
                ttnn
                .to_torch(logit, mesh_composer=self._mesh_reader)
                .reshape(1, 1, -1)[:, :, : self.vocabulary_size]
                .clone()
                for logit in host_logits
            ]
            observe_first_tokens(host_logits)
            ttnn.synchronize_device(self.mesh)
            elapsed_seconds = time.monotonic() - started
            if observed_first_tokens is None or first_token_ready_seconds is None:
                raise BatchDecodeAdapterError(
                    "grouped prefill did not report first-token readiness"
                )
            state_install_seconds = elapsed_seconds - first_token_ready_seconds
            if state_install_seconds < 0:
                raise BatchDecodeAdapterError("prefill phase timing regressed")
            return BatchPrefillResult(
                first_token_ids=observed_first_tokens,
                elapsed_seconds=elapsed_seconds,
                first_token_ready_seconds=first_token_ready_seconds,
                page_table_blocks=page_table_blocks,
                state_install_seconds=state_install_seconds,
                trace_reuse_planned=trace_plan.reuse_after_prefill,
                trace_transition_reason=trace_plan.reason,
                prefill_route=prefill_route,
                prefill_route_reason=route_reason,
                masked_traces_released=masked_traces_released,
            )
        traces_before = bool(self.model.masked_prefill_trace_buckets())
        logits = self.model.prefill_paged_slots(
            list(token_ids),
            prefill_page_table,
            list(range(width)),
            valid_lens=valid_lengths,
            first_token_callback=observe_first_tokens,
            masked_traces=use_masked_traces,
        )
        # A mixed batch's eager users can compile a program, after which the model
        # releases the traces before any replay.
        if traces_before and not self.model.masked_prefill_trace_buckets():
            masked_traces_released = True
        ttnn.synchronize_device(self.mesh)
        elapsed_seconds = time.monotonic() - started
        if len(logits) != width:
            raise BatchDecodeAdapterError(
                "prefill output count differs from active width"
            )
        if observed_first_tokens is None or first_token_ready_seconds is None:
            raise BatchDecodeAdapterError(
                "prefill did not report first-token readiness"
            )
        state_install_seconds = elapsed_seconds - first_token_ready_seconds
        if state_install_seconds < 0:
            raise BatchDecodeAdapterError("prefill phase timing regressed")
        return BatchPrefillResult(
            first_token_ids=observed_first_tokens,
            elapsed_seconds=elapsed_seconds,
            first_token_ready_seconds=first_token_ready_seconds,
            page_table_blocks=page_table_blocks,
            state_install_seconds=state_install_seconds,
            trace_reuse_planned=trace_plan.reuse_after_prefill,
            trace_transition_reason=trace_plan.reason,
            prefill_route=prefill_route,
            prefill_route_reason=route_reason,
            masked_traces_released=masked_traces_released,
        )

    # r[impl tenstorrent.qwen38_p150x2.continuous_slot_effects]
    # Continuous slots. One decode trace of `width` rows stays resident while requests
    # come and go. The allocation contract:
    # - Every buffer a slot owns (GDN rows, the KV pool, the trace's inputs and outputs)
    #   is allocated before the trace is captured. A refill must not compile a program or
    #   drop the masked prefill traces; if it does, the trace is released, not replayed.
    # - A refill writes only its own slot: prefill_paged_slots installs the new user's
    #   GDN state into that row, and its KV lands in the run the scheduler assigned.
    # - An inactive row (never filled, or retired) keeps decoding with token 0 at small
    #   positions; its page-table row points only at the scratch block, so its KV writes
    #   never reach a block a live or future request owns, and a refill overwrites its
    #   GDN row. Rows decode independently, so it cannot change another row's tokens.
    # - A masked prefill replay may overwrite buffers the decode trace allocated after the
    #   masked capture, so after any refill or retirement every decode input, including
    #   the page table, is restaged from the host before the next replay.

    def _scratch_row(self) -> torch.Tensor:
        return torch.full(
            (self.blocks_per_user,), self.kv_pool.scratch_block, dtype=torch.int32
        )

    def _require_slot_group(self) -> int:
        self._require_open()
        if self._slot_width is None:
            raise BatchDecodeAdapterError("no continuous slot group is open")
        return self._slot_width

    def _require_slot_index(self, slot: int, width: int) -> None:
        if isinstance(slot, bool) or not isinstance(slot, int) or not 0 <= slot < width:
            raise BatchDecodeAdapterError("slot is outside the slot group")

    # r[impl tenstorrent.qwen38_p150x2.resident_slot_trace]
    def begin_slots(self, *, width: int, route: str) -> TracePrefillPlan:
        """Open an empty slot group: route selected, state zeroed, every row inactive.

        The one resident decode trace of this width and route stays resident when its
        binding is stable and the masked prefill traces can serve the group's refills, so the
        group's first decode step restages its inputs instead of capturing a trace."""
        self._require_width(width)
        self._require_open()
        if route != self.fusion_mode:
            self.switch_fusion_mode(route)
        plan = plan_slot_group_trace(
            width=width,
            resident_widths=tuple(self._traces),
            binding_stable=all(
                self._binding_matches(trace) for trace in self._traces.values()
            ),
            masked_prefill_ready=bool(self.model.masked_prefill_trace_buckets())
            and not self.model.masked_prefill_traces_stale(),
        )
        if not plan.reuse_after_prefill:
            self._release_traces()
        self.reset_all()
        self.page_table = self._scratch_row().repeat(self.maximum_width, 1)
        self._slot_width = width
        self._slot_runs = {}
        self._slot_inputs_stale = True
        return plan

    def prefix_cache_available(self) -> bool:
        """Whether a refill can restore and save hybrid prefix state now.

        Prefix restore replays the masked prefill traces from a chunk boundary; eager prefill
        would rewrite the shared prefix blocks from position 0, so no traces, no cache."""
        return (
            self.prefix_states > 0
            and bool(self.model.masked_prefill_trace_buckets())
            and not self.model.masked_prefill_traces_stale()
            and self.model.prefix_cache_state_count() == self.prefix_states
        )

    # r[impl tenstorrent.qwen38_p150x2.hybrid_prefix_state]
    def refill_slot(
        self,
        *,
        slot: int,
        token_ids: torch.Tensor,
        block_start: int,
        block_count: int,
        prefix_ranges: tuple[tuple[int, int], ...] = (),
        prefix_tokens: int = 0,
        resume_state: int | None = None,
        save_state: int | None = None,
        save_at: int = 0,
    ) -> SlotRefillResult:
        """Prefill one request into an unowned slot while the other slots stay live.

        With a prefix hit, the page-table row starts with the cached entry's KV blocks
        (`prefix_ranges`, covering tokens [0, prefix_tokens)), the saved carried state
        `resume_state` is restored, and only the rest of the prompt is prefilled; the
        request's own run starts at block_start. `save_state` saves the carried state after
        the chunk ending at `save_at` for a new cache entry."""
        width = self._require_slot_group()
        self._require_slot_index(slot, width)
        if slot in self._slot_runs:
            raise BatchDecodeAdapterError(f"slot {slot} is already owned")
        if not isinstance(token_ids, torch.Tensor) or token_ids.ndim != 2:
            raise BatchDecodeAdapterError("prefill tokens must be a rank-two tensor")
        if token_ids.shape[0] != MINIMUM_ACTIVE_BATCH_WIDTH or token_ids.shape[1] < 1:
            raise BatchDecodeAdapterError("prefill tokens must contain one prompt")
        prompt_tokens = int(token_ids.shape[1])
        prefix_blocks = [
            block
            for start, count in prefix_ranges
            for block in range(start, start + count)
        ]
        if len(prefix_blocks) * self.profile.kv_block_size != prefix_tokens:
            raise BatchDecodeAdapterError(
                "the prefix blocks must cover exactly the prefix tokens"
            )
        if (resume_state is None) != (prefix_tokens == 0):
            raise BatchDecodeAdapterError(
                "a prefix needs a saved state to resume from, and only then"
            )
        if prefix_tokens >= prompt_tokens:
            raise BatchDecodeAdapterError(
                "a prefix hit must leave prompt tokens to prefill"
            )
        if save_state is not None and not prefix_tokens < save_at < prompt_tokens:
            raise BatchDecodeAdapterError(
                "a save point must lie past the prefix inside the prompt"
            )
        needed = self.request_blocks(prompt_tokens=prompt_tokens, generation_tokens=1)
        if block_count + len(prefix_blocks) < needed:
            raise BatchDecodeAdapterError(
                "the KV run does not cover the prompt's prefill"
            )
        block_end = block_start + block_count
        if (
            block_start < self.kv_pool.reserved_prefix_blocks
            or block_end > self.kv_pool.generation_end_block
            or block_count + len(prefix_blocks) > self.blocks_per_user
            or any(
                not self.kv_pool.reserved_prefix_blocks
                <= block
                < self.kv_pool.generation_end_block
                for block in prefix_blocks
            )
        ):
            raise BatchDecodeAdapterError(
                "the KV run leaves the generation block range"
            )
        if any(block_start <= block < block_end for block in prefix_blocks):
            raise BatchDecodeAdapterError("the KV run overlaps its own prefix blocks")
        for owner, (start, count) in self._slot_runs.items():
            if block_start < start + count and start < block_end:
                raise BatchDecodeAdapterError(f"the KV run overlaps slot {owner}")
        trace = self._traces.get(width)
        if trace is not None:
            self._require_binding(trace)
        masked_traces_released = False
        if self.model.masked_prefill_traces_stale():
            self.model.release_masked_prefill_traces()
            masked_traces_released = True
        use_masked_traces = bool(self.model.masked_prefill_trace_buckets())
        plan: dict[str, tuple[int, int]] = {}
        if resume_state is not None:
            plan["resume"] = (resume_state, prefix_tokens)
        if save_state is not None:
            plan["save"] = (save_state, save_at)
        if plan and not self.prefix_cache_available():
            raise BatchDecodeAdapterError(
                "prefix restore and save need the masked prefill traces"
            )

        row = self._scratch_row()
        own = torch.arange(block_start, block_end, dtype=torch.int32)
        row[: len(prefix_blocks)] = torch.tensor(prefix_blocks, dtype=torch.int32)
        row[len(prefix_blocks) : len(prefix_blocks) + block_count] = own
        self.page_table[slot] = row
        self._slot_runs[slot] = (block_start, block_count)
        self._slot_inputs_stale = True
        self._prior_reset_complete = False

        programs_before = self.mesh.num_program_cache_entries()
        started = time.monotonic()
        first: list[tuple[int, float]] = []

        def observe_first_token(host_logits: list[torch.Tensor]) -> None:
            if first or len(host_logits) != 1:
                raise BatchDecodeAdapterError(
                    "a slot prefill reported the wrong first tokens"
                )
            logits = host_logits[0].float().reshape(-1, self.vocabulary_size)
            first.append((
                int(torch.argmax(logits[0]).item()),
                time.monotonic() - started,
            ))

        self.model.prefill_paged_slots(
            [token_ids],
            self.page_table[slot : slot + 1].contiguous(),
            [slot],
            valid_lens=[prompt_tokens],
            first_token_callback=observe_first_token,
            masked_traces=use_masked_traces,
            **({"prefixes": [plan]} if plan else {}),
        )
        ttnn.synchronize_device(self.mesh)
        elapsed_seconds = time.monotonic() - started
        if not first:
            raise BatchDecodeAdapterError(
                "a slot prefill did not report its first token"
            )
        if use_masked_traces and not self.model.masked_prefill_trace_buckets():
            masked_traces_released = True
        decode_trace_released = False
        if trace is not None:
            self._require_binding(trace)
            if (
                masked_traces_released
                or self.mesh.num_program_cache_entries() != programs_before
            ):
                self._release_traces()
                decode_trace_released = True
        first_token_id, first_token_ready_seconds = first[0]
        return SlotRefillResult(
            first_token_id=first_token_id,
            elapsed_seconds=elapsed_seconds,
            first_token_ready_seconds=first_token_ready_seconds,
            masked_traces_used=use_masked_traces,
            masked_traces_released=masked_traces_released,
            decode_trace_released=decode_trace_released,
        )

    def retire_slot(self, *, slot: int) -> None:
        """Return a slot's KV run: its row decodes into the scratch block from now on."""
        width = self._require_slot_group()
        self._require_slot_index(slot, width)
        if self._slot_runs.pop(slot, None) is None:
            raise BatchDecodeAdapterError(f"slot {slot} is not owned")
        self.page_table[slot] = self._scratch_row()
        self._slot_inputs_stale = True

    def _close_slot_group(self) -> None:
        self._slot_width = None
        self._slot_runs = {}
        self._slot_inputs_stale = False

    def abandon_slots(self) -> None:
        """Drop the slot group after a failure: no trace survives, every row is zeroed."""
        self._require_open()
        self._close_slot_group()
        self._release_traces()
        self.reset_all()

    def _slot_trace(
        self, token_ids: tuple[int, ...], positions: tuple[int, ...]
    ) -> None:
        """Capture the slot trace, or restage every input of the resident one if stale."""
        width = self._require_slot_group()
        if len(token_ids) != width or len(positions) != width:
            raise BatchDecodeAdapterError("slot decode inputs do not cover every row")
        trace = self._traces.get(width)
        if trace is None:
            self._pending_trace_plan = None
            self.prepare_width(token_ids=token_ids, positions=positions)
            self.slot_trace_captures += 1
            self._slot_inputs_stale = False
            return
        self._require_binding(trace)
        if self._slot_inputs_stale:
            host_inputs = self.model.prepare_decode_inputs_host(
                torch.tensor(token_ids, dtype=torch.int32).reshape(
                    width, MINIMUM_ACTIVE_BATCH_WIDTH
                ),
                torch.tensor(positions, dtype=torch.int32),
                page_table=self.page_table[:width].contiguous(),
            )
            copy_host_to_device(host_inputs, device_tensors=trace.device_inputs)
            self._slot_inputs_stale = False

    def decode_slots_greedy(
        self,
        *,
        token_ids: tuple[int, ...],
        positions: tuple[int, ...],
        steps: int,
    ) -> Iterator[tuple[int, ...]]:
        """A greedy stream over every slot row; inactive rows pass token 0 at position 0."""
        self._slot_trace(token_ids, positions)
        return self.decode_greedy_stream(
            token_ids=token_ids, positions=positions, steps=steps
        )

    def decode_slots_sampled(
        self,
        *,
        token_ids: tuple[int, ...],
        positions: tuple[int, ...],
    ) -> tuple[tuple[ShardCandidates, ...], ...]:
        """One sampled step over every slot row; inactive rows pass token 0 at position 0."""
        self._slot_trace(token_ids, positions)
        return self.decode_traced_sampled(token_ids=token_ids, positions=positions)

    def _single_read_written_blocks(self, length: int, traced: bool) -> frozenset[int]:
        buckets = (
            self.model.masked_prefill_trace_buckets()
            if traced
            else self.model.MASKED_PREFILL_TRACE_BUCKETS
        )
        plan = masked_prefill_plan(length, buckets)
        if plan is None:
            raise BatchDecodeAdapterError("no masked prefill plan holds this read")
        bucket, start, _length = plan[-1]
        blocks = (start + bucket) // self.kv_pool.kv_block_size
        return frozenset(self._read_page_table[0, :blocks].tolist())

    def _batch_read_written_blocks(self, bucket: int) -> frozenset[int]:
        # A batched replay fills every captured row, including unused user rows.
        blocks = bucket // self.kv_pool.kv_block_size
        return frozenset(self._batch_read_page_table[:, :blocks].flatten().tolist())

    def read_rows(
        self,
        token_ids: torch.Tensor,
        positions: tuple[int, ...],
    ) -> StructuredReadRows:
        """Prefill one sequence and return the next-token logits after each position.

        The read runs the masked prefill segments with the positions staged into the spare
        rows of each segment's logit tile, so it costs one generation prefill and no decode.
        It writes KV through the fixed read row. Generation groups are never assigned the
        blocks under its first 1,024 tokens, so a later read sharing them replays only the
        rest.
        """
        length, ordered = self._validated_read(token_ids, positions)
        masked_traces_released = self._prepare_read()
        traced = bool(self.model.masked_prefill_trace_buckets())
        started = time.monotonic()
        logits, reused_tokens = self.model.prefill_masked_read(
            token_ids.to(torch.int32),
            self._read_page_table,
            length,
            STRUCTURED_READ_SLOT,
            ordered,
        )
        elapsed_seconds = time.monotonic() - started
        if tuple(logits.shape) != (len(ordered), self.vocabulary_size):
            raise BatchDecodeAdapterError(
                "structured read returned the wrong logit shape"
            )
        if not traced:
            route = READ_ROUTE_EAGER
        elif reused_tokens:
            route = READ_ROUTE_PREFIX_REUSE
        else:
            route = READ_ROUTE_MASKED_TRACE
        return StructuredReadRows(
            logits=logits,
            elapsed_seconds=elapsed_seconds,
            route=route,
            masked_traces_released=masked_traces_released,
            kv_written_blocks=self._single_read_written_blocks(length, traced),
        )

    def read_rows_batch(
        self,
        requests: tuple[tuple[torch.Tensor, tuple[int, ...]], ...],
        *,
        protected_blocks: frozenset[int] = frozenset(),
    ) -> tuple[StructuredReadRows, ...]:
        """read_rows for several sequences, sharing batched replays where they fit.

        Sequences within the largest batched-read bucket share one replay, up to its width
        and logit rows; the rest read alone. When a captured batch row would overwrite
        protected generation KV, use the disjoint single-read row instead. Each result
        reports its share of its replay."""
        validated = [
            self._validated_read(token_ids, positions)
            for token_ids, positions in requests
        ]
        masked_traces_released = self._prepare_read()
        captured = self.model.masked_read_batch_buckets()
        groups = plan_read_batches(
            lengths=tuple(length for length, _ordered in validated),
            row_counts=tuple(len(ordered) for _length, ordered in validated),
            width=int(self.model.READ_BATCH_WIDTH),
            minimum_by_bucket={
                bucket: minimum
                for bucket, minimum in READ_BATCH_MINIMUM_BY_BUCKET.items()
                if bucket in captured
                and (
                    not protected_blocks
                    or self._batch_read_written_blocks(bucket).isdisjoint(
                        protected_blocks
                    )
                )
            },
            maximum_rows=int(self.model.MASKED_LOGIT_ROWS),
        )
        results: list[StructuredReadRows | None] = [None] * len(requests)
        for group in groups:
            if len(group) == 1:
                index = group[0]
                single = self.read_rows(*requests[index])
                results[index] = StructuredReadRows(
                    logits=single.logits,
                    elapsed_seconds=single.elapsed_seconds,
                    route=single.route,
                    masked_traces_released=masked_traces_released
                    or single.masked_traces_released,
                    kv_written_blocks=single.kv_written_blocks,
                )
                continue
            bucket = next(
                bucket
                for bucket in captured
                if bucket >= max(validated[index][0] for index in group)
            )
            written = self._batch_read_written_blocks(bucket)
            started = time.monotonic()
            rows = self.model.prefill_masked_read_batch(
                [requests[index][0].to(torch.int32) for index in group],
                [validated[index][1] for index in group],
            )
            share = (time.monotonic() - started) / len(group)
            for index, logits in zip(group, rows, strict=True):
                if tuple(logits.shape) != (
                    len(validated[index][1]),
                    self.vocabulary_size,
                ):
                    raise BatchDecodeAdapterError(
                        "batched read returned the wrong logit shape"
                    )
                results[index] = StructuredReadRows(
                    logits=logits,
                    elapsed_seconds=share,
                    route=READ_ROUTE_BATCHED_TRACE,
                    masked_traces_released=masked_traces_released,
                    kv_written_blocks=written,
                )
        if any(result is None for result in results):
            raise BatchDecodeAdapterError("a structured read was not planned")
        return tuple(result for result in results if result is not None)

    def _validated_read(
        self, token_ids: torch.Tensor, positions: tuple[int, ...]
    ) -> tuple[int, tuple[int, ...]]:
        self._require_open()
        if not isinstance(token_ids, torch.Tensor) or token_ids.ndim != 2:
            raise BatchDecodeAdapterError("read tokens must be a rank-two tensor")
        if token_ids.shape[0] != MINIMUM_ACTIVE_BATCH_WIDTH:
            raise BatchDecodeAdapterError("read tokens must contain one sequence")
        length = int(token_ids.shape[1])
        if not 1 <= length <= self.profile.maximum_sequence_length:
            raise BatchDecodeAdapterError(
                "read length is outside the admitted sequence length"
            )
        ordered = tuple(int(position) for position in positions)
        if not ordered or ordered != tuple(sorted(set(ordered))):
            raise BatchDecodeAdapterError(
                "read positions must be ascending and distinct"
            )
        if ordered[0] < MINIMUM_POSITION or ordered[-1] >= length:
            raise BatchDecodeAdapterError("read positions must lie inside the sequence")
        return length, ordered

    def _prepare_read(self) -> bool:
        """Release what a read replay cannot share memory with; True when stale traces went."""
        # A decode trace captured after the masked prefill traces owns memory their replays
        # use as scratch, so it cannot stay resident across a read, as for prefill_users.
        self._release_traces()
        # A read installs into slot 0, so no slot group survives it.
        self._close_slot_group()
        self._prior_reset_complete = False
        if self.model.masked_prefill_traces_stale():
            self.model.release_masked_prefill_traces()
            return True
        return False

    def decode_traced(
        self,
        *,
        token_ids: tuple[int, ...],
        positions: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Advance every active-prefix user by exactly one traced decode step."""
        width = len(token_ids)
        self._require_width(width)
        self._require_open()
        if len(positions) != width:
            raise BatchDecodeAdapterError("token and position widths differ")
        trace = self._traces.get(width)
        if trace is None:
            raise BatchDecodeAdapterError("batch width has no prepared trace")
        self._require_binding(trace)
        if any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or not 0 <= token < self.vocabulary_size
            for token in token_ids
        ):
            raise BatchDecodeAdapterError("batch token is outside the vocabulary")
        if any(
            isinstance(position, bool)
            or not isinstance(position, int)
            or not MINIMUM_POSITION <= position < self.profile.maximum_sequence_length
            for position in positions
        ):
            raise BatchDecodeAdapterError("batch position is outside the profile bound")

        host_inputs = self.model.prepare_decode_inputs_host(
            torch.tensor(token_ids, dtype=torch.int32).reshape(
                width,
                MINIMUM_ACTIVE_BATCH_WIDTH,
            ),
            torch.tensor(positions, dtype=torch.int32),
            page_table=None,
        )
        copy_host_to_device(
            host_inputs[:MUTABLE_INPUT_COUNT],
            device_tensors=trace.device_inputs[:MUTABLE_INPUT_COUNT],
        )
        ttnn.execute_trace(
            self.mesh,
            trace.trace_id,
            cq_id=TRACE_COMMAND_QUEUE_ID,
            blocking=TRACE_BLOCKING_REPLAY,
        )
        # No device synchronize: the readback below is enqueued on the trace's
        # command queue, so it completes only after the replay does.
        return self._read_global_winners(trace.indices, trace.values, width)

    def decode_greedy_stream(
        self,
        *,
        token_ids: tuple[int, ...],
        positions: tuple[int, ...],
        steps: int,
    ) -> Iterator[tuple[int, ...]]:
        """Yield `steps` greedy decode steps, keeping the next step queued on the device.

        Each replay writes its tokens into the next replay's token input, so only the
        first step stages tokens from the host; every step stages its positions and RoPE
        rows. The host reads step k while step k + 1 runs. Closing the generator early
        waits for the queued step, whose tokens are dropped.
        """
        width = len(token_ids)
        self._require_width(width)
        self._require_open()
        if steps < MINIMUM_ACTIVE_BATCH_WIDTH:
            raise BatchDecodeAdapterError("a greedy stream needs at least one step")
        if len(positions) != width:
            raise BatchDecodeAdapterError("token and position widths differ")
        trace = self._traces.get(width)
        if trace is None:
            raise BatchDecodeAdapterError("batch width has no prepared trace")
        if trace.values is not None:
            raise BatchDecodeAdapterError("a greedy stream needs the greedy trace tail")
        self._require_binding(trace)
        if any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or not 0 <= token < self.vocabulary_size
            for token in token_ids
        ):
            raise BatchDecodeAdapterError("batch token is outside the vocabulary")
        last_position = max(positions) + steps - 1
        if (
            min(positions) < MINIMUM_POSITION
            or last_position >= self.profile.maximum_sequence_length
        ):
            raise BatchDecodeAdapterError("batch position is outside the profile bound")

        winner_replica = ttnn.get_device_tensors(trace.indices)[0]
        pending: list[tuple[Any, Any]] = []
        try:
            for step in range(steps):
                host_inputs = self.model.prepare_decode_inputs_host(
                    torch.tensor(token_ids, dtype=torch.int32).reshape(
                        width, MINIMUM_ACTIVE_BATCH_WIDTH
                    ),
                    torch.tensor(
                        [position + step for position in positions], dtype=torch.int32
                    ),
                    page_table=None,
                )
                first_input = 0 if step == 0 else GREEDY_FED_BACK_INPUT_COUNT
                copy_host_to_device(
                    host_inputs[first_input:MUTABLE_INPUT_COUNT],
                    device_tensors=trace.device_inputs[first_input:MUTABLE_INPUT_COUNT],
                )
                ttnn.execute_trace(
                    self.mesh,
                    trace.trace_id,
                    cq_id=TRACE_COMMAND_QUEUE_ID,
                    blocking=TRACE_BLOCKING_REPLAY,
                )
                # Queued behind this replay and ahead of the next one on the same queue.
                host = ttnn.from_device(
                    winner_replica, blocking=False, queue_id=TRACE_COMMAND_QUEUE_ID
                )
                pending.append((
                    host,
                    ttnn.record_event(self.mesh, cq_id=TRACE_COMMAND_QUEUE_ID),
                ))
                if len(pending) > GREEDY_STREAM_DEPTH:
                    yield self._settle_winners(pending.pop(0), width)
            while pending:
                yield self._settle_winners(pending.pop(0), width)
        finally:
            for _host, event in pending:
                ttnn.event_synchronize(event)

    def _settle_winners(self, read: tuple[Any, Any], width: int) -> tuple[int, ...]:
        host, event = read
        ttnn.event_synchronize(event)
        return self._winner_tuple(ttnn.to_torch(host), width)

    def _reduce_for_mode(self, sharded_logits: Any, width: int) -> tuple[Any, Any]:
        """Reduce the traced tail for the resident fusion mode.

        The exact-parity path keeps the argmax winner tail. The sampled
        path replaces it with a per-shard top-K reduction so the host can
        draw from a real candidate distribution; the two tails never mix
        because every mode switch releases all traces first.
        """
        if self.fusion_mode == ROUTE_SAMPLED:
            return self._reduce_shards_topk(sharded_logits, width)
        indices, values = self._reduce_shards(sharded_logits, width)
        return self._combine_shard_winners(indices, values, width), None

    def _combine_shard_winners(self, indices: Any, values: Any, width: int) -> Any:
        """Global greedy token per user, replicated on every device as [width, 1] uint32.

        All-gathers each shard's (max, local index) and keeps device 0 on ties, exactly as
        the host combine did with torch.argmax over devices. Token ids stay below 2**24, so
        the float32 arithmetic is exact.
        """
        if self.device_count != GREEDY_COMBINE_DEVICE_COUNT:
            raise BatchDecodeAdapterError("greedy winner combine supports two shards")
        from models.tt_transformers.tt.ccl import tt_all_gather

        index = ttnn.typecast(
            ttnn.to_layout(ttnn.reshape(indices, (1, 1, 1, width)), ttnn.TILE_LAYOUT),
            ttnn.float32,
        )
        value = ttnn.typecast(ttnn.reshape(values, (1, 1, 1, width)), ttnn.float32)
        ttnn.deallocate(indices)
        ttnn.deallocate(values)
        packed = ttnn.concat([value, index], dim=2)
        ttnn.deallocate(index)
        ttnn.deallocate(value)
        gathered = tt_all_gather(
            packed,
            self.mesh,
            self.model.tt_ccl,
            cluster_axis=None,
            dim=1,
            dtype=ttnn.float32,
            topology=self.model.args.ccl_topology(),
        )
        value0 = ttnn.slice(gathered, (0, 0, 0, 0), (1, 1, 1, width))
        index0 = ttnn.slice(gathered, (0, 0, 1, 0), (1, 1, 2, width))
        value1 = ttnn.slice(gathered, (0, 1, 0, 0), (1, 2, 1, width))
        index1 = ttnn.slice(gathered, (0, 1, 1, 0), (1, 2, 2, width))
        ttnn.deallocate(gathered)
        take_second = ttnn.gt(value1, value0)
        offset = ttnn.sub(ttnn.add(index1, float(self.vocabulary_per_shard)), index0)
        token = ttnn.add(index0, ttnn.multiply(take_second, offset))
        for tensor in (value0, index0, value1, index1, take_second, offset):
            ttnn.deallocate(tensor)
        winners = ttnn.to_layout(
            ttnn.typecast(token, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT
        )
        ttnn.deallocate(token)
        return ttnn.reshape(winners, (width, MINIMUM_ACTIVE_BATCH_WIDTH))

    @staticmethod
    def _feed_back_winners(
        winners: Any, values: Any, device_inputs: tuple[Any, ...]
    ) -> None:
        """Greedy tails write their tokens into the next step's token input."""
        if values is None:
            ttnn.copy(winners, device_inputs[0])

    def _reduce_shards_topk(self, sharded_logits: Any, width: int) -> tuple[Any, Any]:
        """Per-shard top-K candidate reduction for the sampled trace tail.

        topk requires tile layout (FillPad asserts on row-major input), so
        this reduces the sharded logits directly; the readback reshapes the
        concatenated per-device results afterwards.
        """
        topk_values, topk_indices = ttnn.topk(
            sharded_logits,
            SAMPLED_CANDIDATE_COUNT,
            dim=-1,
        )
        return topk_indices, topk_values

    def _read_global_candidates(
        self,
        indices: Any,
        values: Any,
        width: int,
    ) -> tuple[tuple[ShardCandidates, ...], ...]:
        """Read per-user per-shard candidates with global token ids."""
        if values is None:
            raise BatchDecodeAdapterError("sampled readback needs a candidate tail")
        host_indices = ttnn.to_torch(
            indices,
            mesh_composer=self._mesh_reader,
            cq_id=TRACE_COMMAND_QUEUE_ID,
        ).reshape(self.device_count, width, SAMPLED_CANDIDATE_COUNT)
        host_values = ttnn.to_torch(
            values,
            mesh_composer=self._mesh_reader,
            cq_id=TRACE_COMMAND_QUEUE_ID,
        ).reshape(self.device_count, width, SAMPLED_CANDIDATE_COUNT)
        rows: list[tuple[ShardCandidates, ...]] = []
        for user in range(width):
            shards = []
            for device in range(self.device_count):
                shard_tokens = tuple(
                    int(
                        device * self.vocabulary_per_shard
                        + int(host_indices[device, user, rank].item())
                    )
                    for rank in range(SAMPLED_CANDIDATE_COUNT)
                )
                shard_values = tuple(
                    float(host_values[device, user, rank].item())
                    for rank in range(SAMPLED_CANDIDATE_COUNT)
                )
                shards.append(
                    ShardCandidates(values=shard_values, token_ids=shard_tokens)
                )
            rows.append(tuple(shards))
        return tuple(rows)

    def decode_traced_sampled(
        self,
        *,
        token_ids: tuple[int, ...],
        positions: tuple[int, ...],
    ) -> tuple[tuple[ShardCandidates, ...], ...]:
        """One traced decode step returning per-user sampling candidates.

        Replays the sampled-tail trace and reads back each shard's top-K
        candidates with global token ids. The coordinator owns the merge,
        the distribution, and the seeded draw.
        """
        if self.fusion_mode != ROUTE_SAMPLED:
            raise BatchDecodeAdapterError(
                "sampled decode requires the sampled fusion mode"
            )
        width = len(token_ids)
        self._require_width(width)
        self._require_open()
        if len(positions) != width:
            raise BatchDecodeAdapterError("token and position widths differ")
        trace = self._traces.get(width)
        if trace is None:
            raise BatchDecodeAdapterError("batch width has no prepared trace")
        self._require_binding(trace)
        if any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or not 0 <= token < self.vocabulary_size
            for token in token_ids
        ):
            raise BatchDecodeAdapterError("batch token is outside the vocabulary")
        if any(
            isinstance(position, bool)
            or not isinstance(position, int)
            or not MINIMUM_POSITION <= position < self.profile.maximum_sequence_length
            for position in positions
        ):
            raise BatchDecodeAdapterError("batch position is outside the profile bound")

        host_inputs = self.model.prepare_decode_inputs_host(
            torch.tensor(token_ids, dtype=torch.int32).reshape(
                width,
                MINIMUM_ACTIVE_BATCH_WIDTH,
            ),
            torch.tensor(positions, dtype=torch.int32),
            page_table=None,
        )
        copy_host_to_device(
            host_inputs[:MUTABLE_INPUT_COUNT],
            device_tensors=trace.device_inputs[:MUTABLE_INPUT_COUNT],
        )
        ttnn.execute_trace(
            self.mesh,
            trace.trace_id,
            cq_id=TRACE_COMMAND_QUEUE_ID,
            blocking=TRACE_BLOCKING_REPLAY,
        )
        # No device synchronize: the readback below is enqueued on the trace's
        # command queue, so it completes only after the replay does.
        return self._read_global_candidates(trace.indices, trace.values, width)

    def switch_fusion_mode(self, mode: str) -> None:
        """Switch the batch decode path between fusion modes.

        Releases every resident batch trace, then flips the per-layer fusion
        attribute the GDN decode forward reads at call time. The next
        prepare_width call recaptures in the new mode. Physical evidence
        (dual-trace probe, 2026-08-28): a capture while another trace is
        resident corrupts the new trace, so releases must precede the next
        capture; release-then-recapture is exact in both directions.
        """
        if mode not in ROUTES:
            raise BatchDecodeAdapterError(
                f"fusion mode must be one of {ROUTES}, got {mode!r}"
            )
        self._require_open()
        self._release_traces()
        for layer in self.model.layers:
            if not layer.is_full_attention:
                layer.attention._decode_fusion_mode = mode

    @property
    def fusion_mode(self) -> str:
        """The fusion mode the GDN layers currently use at forward time."""
        self._require_open()
        modes = {
            getattr(layer.attention, "_decode_fusion_mode", None)
            for layer in self.model.layers
            if not layer.is_full_attention
        }
        if len(modes) != 1 or None in modes:
            raise BatchDecodeAdapterError("GDN layers disagree on the fusion mode")
        return next(iter(modes))

    def reset_all(self) -> None:
        """Zero all persistent GDN rows without using the B=1 prefill scratch zeros."""
        self._require_open()
        for sources in self._gdn_zero_sources:
            layer = sources.layer
            if layer.rec_state is None or layer.conv_states is None:
                raise BatchDecodeAdapterError("GDN reset target is not allocated")
            if tuple(layer.rec_state.shape) != tuple(sources.recurrent.shape):
                raise BatchDecodeAdapterError("GDN recurrent reset shape changed")
            ttnn.copy(sources.recurrent, layer.rec_state)
            for state in layer.conv_states:
                if tuple(state.shape) != tuple(sources.convolution.shape):
                    raise BatchDecodeAdapterError("GDN convolution reset shape changed")
                ttnn.copy(sources.convolution, state)
            if layer.conv_carry is not None:
                ttnn.copy(sources.convolution_carry, layer.conv_carry)
        ttnn.synchronize_device(self.mesh)
        self._prior_reset_complete = True

    def _clone_gdn_state(self) -> tuple[_GdnSnapshot, ...]:
        snapshots = []
        for layer in self.model.layers:
            if layer.is_full_attention:
                continue
            attention = layer.attention
            if attention.rec_state is None or attention.conv_states is None:
                raise BatchDecodeAdapterError("GDN state is not allocated")
            snapshots.append(
                _GdnSnapshot(
                    layer=attention,
                    recurrent=ttnn.clone(attention.rec_state),
                    convolution=tuple(
                        ttnn.clone(state) for state in attention.conv_states
                    ),
                )
            )
        if not snapshots:
            raise BatchDecodeAdapterError("model has no GDN state")
        return tuple(snapshots)

    @staticmethod
    def _restore_gdn_state(snapshots: tuple[_GdnSnapshot, ...]) -> None:
        for snapshot in snapshots:
            ttnn.copy(snapshot.recurrent, snapshot.layer.rec_state)
            for source, target in zip(
                snapshot.convolution,
                snapshot.layer.conv_states,
                strict=True,
            ):
                ttnn.copy(source, target)

    @staticmethod
    def _deallocate_gdn_state(snapshots: tuple[_GdnSnapshot, ...]) -> None:
        for snapshot in snapshots:
            ttnn.deallocate(snapshot.recurrent)
            for state in snapshot.convolution:
                ttnn.deallocate(state)

    def _forward_sharded(self, device_inputs: tuple[Any, ...]) -> Any:
        token_buffer, position_buffer, rope_buffer, page_table_buffer = device_inputs
        cos, sin = unpack_rope(rope_buffer)
        return self.model._forward_decode(
            token_buffer,
            cos,
            sin,
            position_buffer,
            page_table_buffer,
            sharded_lm_head=True,
        )

    def _reduce_shards(self, sharded_logits: Any, width: int) -> tuple[Any, Any]:
        logits_row_major = ttnn.to_layout(sharded_logits, ttnn.ROW_MAJOR_LAYOUT)
        indices = ttnn.argmax(logits_row_major, dim=-1, keepdim=False)
        ttnn.deallocate(logits_row_major)
        padded_rows = (
            (
                (self.vocabulary_per_shard + MAX_REDUCTION_COLUMNS - 1)
                // MAX_REDUCTION_COLUMNS
                + TILE_EDGE_ELEMENTS
                - 1
            )
            // TILE_EDGE_ELEMENTS
        ) * TILE_EDGE_ELEMENTS
        padded_elements = padded_rows * MAX_REDUCTION_COLUMNS
        padded = ttnn.pad(
            sharded_logits,
            [
                (0, 0),
                (0, 0),
                (0, 0),
                (0, padded_elements - self.vocabulary_per_shard),
            ],
            value=PADDED_LOGIT_SENTINEL,
        )
        grid = ttnn.reshape(
            padded,
            (MINIMUM_ACTIVE_BATCH_WIDTH, width, padded_rows, MAX_REDUCTION_COLUMNS),
        )
        partial = ttnn.max(grid, dim=-1)
        partial_row = ttnn.reshape(
            partial,
            (
                MINIMUM_ACTIVE_BATCH_WIDTH,
                MINIMUM_ACTIVE_BATCH_WIDTH,
                width,
                padded_rows,
            ),
        )
        values = ttnn.max(partial_row, dim=-1)
        ttnn.deallocate(padded)
        ttnn.deallocate(grid)
        ttnn.deallocate(partial)
        ttnn.deallocate(partial_row)
        return indices, values

    def _read_global_winners(
        self,
        winners: Any,
        values: Any,
        width: int,
    ) -> tuple[int, ...]:
        """Read the combined greedy tokens from one replica."""
        if values is not None:
            raise BatchDecodeAdapterError(
                "greedy readback needs the combined winner tail"
            )
        host = ttnn.to_torch(
            ttnn.get_device_tensors(winners)[0],
            cq_id=TRACE_COMMAND_QUEUE_ID,
        )
        return self._winner_tuple(host, width)

    def _winner_tuple(self, host: Any, width: int) -> tuple[int, ...]:
        tokens = host.reshape(-1)
        if tokens.numel() != width:
            raise BatchDecodeAdapterError(
                "batch winner count differs from active width"
            )
        result = tuple(int(token) for token in tokens.tolist())
        if any(not 0 <= token < self.vocabulary_size for token in result):
            raise BatchDecodeAdapterError("batch winner is outside the vocabulary")
        return result

    def _observe_binding(
        self,
        device_inputs: tuple[Any, ...],
        logits: Any,
    ) -> tuple[str, ...]:
        gdn_state = tuple(
            (layer.attention.rec_state, tuple(layer.attention.conv_states))
            for layer in self.model.layers
            if not layer.is_full_attention
        )
        return (
            *(_tensor_identity(tensor) for tensor in device_inputs),
            _nested_identity(self.model._paged_kv_caches),
            _nested_identity(gdn_state),
            _tensor_identity(logits),
        )

    def _binding_matches(self, trace: _BatchTrace) -> bool:
        try:
            observed = self._observe_binding(trace.device_inputs, trace.logits)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False
        return observed == trace.binding

    def _single_trace_binding_stable(self) -> bool:
        if len(self._traces) != 1:
            return False
        return self._binding_matches(next(iter(self._traces.values())))

    def _require_binding(self, trace: _BatchTrace) -> None:
        if not self._binding_matches(trace):
            raise TraceBindingDriftError("batch trace binding changed")

    def decode_width_supported(self, width: int) -> bool:
        """Whether decode can shard `width` users one per core as a grid rectangle."""
        grid = self.mesh.compute_with_storage_grid_size()
        return any(
            width % columns == 0 and width // columns <= grid.y
            for columns in range(1, min(width, grid.x) + 1)
        )

    def sampled_width_supported(self, width: int) -> bool:
        """Whether the sampled-fusion decode kernel fits `width` users.

        The fused gated_delta_decode kernel places one (user, local value head)
        instance per core, or CHUNKED_VALIDATED_INSTANCES_PER_CORE on a runtime that
        chunks instances. Past that it stops with "num_heads exceeds the available
        compute grid": 4 users x 24 heads fit the 110-core P150, 5 do not.
        """
        cores = self._compute_cores()
        if cores is None or not self.decode_width_supported(width):
            return False
        per_core = (
            CHUNKED_VALIDATED_INSTANCES_PER_CORE
            if self._capability_record.gdn_scan_chunked_instances
            else ONE_INSTANCE_PER_CORE
        )
        return width * self._local_value_heads() <= cores * per_core

    def prepare_masked_prefill_traces(
        self, warm_token_id: int, routes: tuple[str, ...]
    ) -> tuple[int, ...]:
        """Compile every program serving can reach, then capture the masked prefill traces.

        A replay overwrites device buffers allocated after the capture, and a first-time
        program can allocate buffers that live for the process, so the capture comes after
        decode at every selectable width and route, the eager prefill shapes beyond the
        traced buckets, and every decode-slot install. Returns the captured buckets; empty
        when the traces are disabled or the runtime cannot capture them.
        """
        if not traced_prefill_enabled():
            return ()
        self._require_open()
        initial_route = self.fusion_mode
        token = int(warm_token_id)
        # Warm-up groups leave their own rows behind; the programs and the batched-read
        # trace capture on the startup layout, whose rows are all full and disjoint.
        self.page_table = self._startup_page_table.clone()
        try:
            for route in routes:
                if route != self.fusion_mode:
                    self.switch_fusion_mode(route)
                replay = (
                    self.decode_traced_sampled
                    if route == ROUTE_SAMPLED
                    else self.decode_traced
                )
                for width in range(MINIMUM_ACTIVE_BATCH_WIDTH, self.maximum_width + 1):
                    if not self.decode_width_supported(width):
                        continue
                    if route == ROUTE_SAMPLED and not self.sampled_width_supported(
                        width
                    ):
                        continue
                    self.reset_all()
                    self.prepare_width(
                        token_ids=(token,) * width,
                        positions=(MINIMUM_ACTIVE_BATCH_WIDTH,) * width,
                    )
                    replay(
                        token_ids=(token,) * width,
                        positions=(MINIMUM_ACTIVE_BATCH_WIDTH,) * width,
                    )
                    self._release_traces()
        finally:
            if self.fusion_mode != initial_route:
                self.switch_fusion_mode(initial_route)
        self._release_traces()
        self.reset_all()
        self.model.warm_serving_programs(self.page_table)
        self.reset_all()
        self.model.capture_masked_prefill_traces(
            self.page_table[:1].contiguous(),
            read_page_table=self._batch_read_page_table,
            **({"prefix_states": self.prefix_states} if self.prefix_states else {}),
        )
        return self.model.masked_prefill_trace_buckets()

    def _require_width(self, width: int) -> None:
        if isinstance(width, bool) or not isinstance(width, int):
            raise BatchDecodeAdapterError("active width must be an integer")
        if not MINIMUM_ACTIVE_BATCH_WIDTH <= width <= self.maximum_width:
            raise BatchDecodeAdapterError("active width is outside adapter capacity")

    def _require_open(self) -> None:
        if self._closed:
            raise BatchDecodeAdapterError("batch decode adapter is closed")

    def _release_traces(self) -> None:
        for trace in self._traces.values():
            ttnn.release_trace(self.mesh, trace.trace_id)
            for tensor in (trace.indices, trace.values, trace.logits):
                if tensor is not None:
                    ttnn.deallocate(tensor)
            for tensor in trace.device_inputs:
                if tensor is not None:
                    ttnn.deallocate(tensor)
        self._traces.clear()
        self._pending_trace_plan = None

    def close(self) -> None:
        if self._closed:
            return
        self._release_traces()
        self.model.release_masked_prefill_traces()
        self._closed = True


def _tensor_identity(value: Any) -> str:
    # TT-Metal 0.77 shard views share one MeshBuffer. Read its address once
    # without constructing temporary Tensor views for every binding check.
    # Keep the shard count and exact legacy representation in the identity.
    if type(value) is ttnn.Tensor:
        try:
            address = str(int(value.buffer_address()))
            addresses = [address] * len(value.device_coords())
            return f"{type(value).__name__}:device:[{','.join(addresses)}]"
        except (AttributeError, RuntimeError, TypeError, ValueError):
            # Host, unallocated, and older-API tensors retain the old path.
            pass
    try:
        device_tensors = tuple(ttnn.get_device_tensors(value))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        device_tensors = (value,)
    addresses = []
    for tensor in device_tensors:
        try:
            addresses.append(str(int(tensor.buffer_address())))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return f"{type(value).__name__}:host:{id(value)}"
    return f"{type(value).__name__}:device:[{','.join(addresses)}]"


def _nested_identity(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_nested_identity(item) for item in value) + "]"
    return _tensor_identity(value)
