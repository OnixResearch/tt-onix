"""Pure policy and evidence evaluation for Qwen decode component replay."""

# r[impl tenstorrent.qwen38_p150x2.decode_component_attribution]

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

EMBEDDING_COMPONENT = "embedding"
GDN_MIXER_COMPONENT = "gdn_mixer"
MLP_COMPONENT = "mlp"
FULL_ATTENTION_MIXER_COMPONENT = "full_attention_mixer"
LM_HEAD_ARGMAX_COMPONENT = "lm_head_argmax"
LINEAR_ATTENTION_TYPE = "linear_attention"
FULL_ATTENTION_TYPE = "full_attention"
COMPONENT_ORDER = (
    EMBEDDING_COMPONENT,
    GDN_MIXER_COMPONENT,
    MLP_COMPONENT,
    FULL_ATTENTION_MIXER_COMPONENT,
    LM_HEAD_ARGMAX_COMPONENT,
)
REPRESENTATIVE_LAYER_COUNT = 3
REVISION_LENGTH = 40
BLAKE3_LENGTH = 64
MINIMUM_RATIO = 1.0


class DecodeComponentPolicyError(ValueError):
    """The component replay policy or evidence is invalid."""


@dataclass(frozen=True)
class DecodeComponentProfile:
    profile_id: str
    execution_path: str
    samples_per_group: int
    warmup_replays: int
    maximum_spread_ratio: float
    trace_capture_timeout_seconds: int
    minimum_closure_ratio: float
    maximum_closure_ratio: float
    minimum_weighted_share: float
    minimum_dominance_ratio: float
    authority_blake3: str
    checkpoint_revision: str
    model_source_revision: str
    tokenizer_revision: str
    expected_pcie_device_ids: tuple[int, ...]
    mesh_shape: tuple[int, int]
    expected_cpu_affinity: tuple[int, ...]
    trace_region_size_bytes: int
    maximum_sequence_length: int
    kv_block_size: int
    layer_count: int
    gdn_layer_count: int
    full_attention_layer_count: int
    mlp_layer_count: int
    gdn_representatives: tuple[int, ...]
    full_attention_representatives: tuple[int, ...]
    mlp_representatives: tuple[int, ...]
    prompt: str

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]) -> DecodeComponentProfile:
        root = _require_mapping(source, "profile")
        if _require_bool(root.get("diagnostic_only"), "diagnostic_only") is not True:
            raise DecodeComponentPolicyError(
                "component profile must remain diagnostic-only"
            )
        closure = _require_mapping(root.get("closure"), "closure")
        selection = _require_mapping(root.get("selection"), "selection")
        identities = _require_mapping(root.get("identities"), "identities")
        topology = _require_mapping(root.get("topology"), "topology")
        architecture = _require_mapping(root.get("architecture"), "architecture")
        representatives = _require_mapping(
            root.get("representatives"), "representatives"
        )

        minimum_closure_ratio = _require_positive_float(
            closure.get("minimum_ratio"), "closure.minimum_ratio"
        )
        maximum_closure_ratio = _require_positive_float(
            closure.get("maximum_ratio"), "closure.maximum_ratio"
        )
        if minimum_closure_ratio > maximum_closure_ratio:
            raise DecodeComponentPolicyError("component closure bounds are reversed")

        minimum_weighted_share = _require_positive_float(
            selection.get("minimum_weighted_share"),
            "selection.minimum_weighted_share",
        )
        if minimum_weighted_share > 1.0:
            raise DecodeComponentPolicyError("minimum weighted share exceeds one")
        minimum_dominance_ratio = _require_positive_float(
            selection.get("minimum_dominance_ratio"),
            "selection.minimum_dominance_ratio",
        )
        if minimum_dominance_ratio < MINIMUM_RATIO:
            raise DecodeComponentPolicyError("minimum dominance ratio is less than one")

        minimum_cpu = _require_nonnegative_int(
            topology.get("minimum_cpu"), "topology.minimum_cpu"
        )
        maximum_cpu = _require_nonnegative_int(
            topology.get("maximum_cpu"), "topology.maximum_cpu"
        )
        if minimum_cpu > maximum_cpu:
            raise DecodeComponentPolicyError("CPU affinity bounds are reversed")

        layer_count = _require_positive_int(
            architecture.get("layer_count"), "architecture.layer_count"
        )
        gdn_layer_count = _require_positive_int(
            architecture.get("gdn_layer_count"), "architecture.gdn_layer_count"
        )
        full_attention_layer_count = _require_positive_int(
            architecture.get("full_attention_layer_count"),
            "architecture.full_attention_layer_count",
        )
        mlp_layer_count = _require_positive_int(
            architecture.get("mlp_layer_count"), "architecture.mlp_layer_count"
        )
        if gdn_layer_count + full_attention_layer_count != layer_count:
            raise DecodeComponentPolicyError(
                "mixer counts do not equal the layer count"
            )
        if mlp_layer_count != layer_count:
            raise DecodeComponentPolicyError("MLP count does not equal the layer count")

        gdn_representatives = _require_layer_indices(
            representatives.get("gdn"), "representatives.gdn", layer_count
        )
        full_attention_representatives = _require_layer_indices(
            representatives.get("full_attention"),
            "representatives.full_attention",
            layer_count,
        )
        mlp_representatives = _require_layer_indices(
            representatives.get("mlp"), "representatives.mlp", layer_count
        )
        if set(gdn_representatives) & set(full_attention_representatives):
            raise DecodeComponentPolicyError(
                "GDN and full-attention representatives overlap"
            )

        maximum_spread_ratio = _require_positive_float(
            root.get("maximum_spread_ratio"), "maximum_spread_ratio"
        )
        if maximum_spread_ratio < MINIMUM_RATIO:
            raise DecodeComponentPolicyError("maximum spread ratio is less than one")

        expected_pcie_device_ids = _require_int_tuple(
            topology.get("expected_pcie_device_ids"),
            "topology.expected_pcie_device_ids",
            positive=False,
        )
        if not expected_pcie_device_ids or len(set(expected_pcie_device_ids)) != len(
            expected_pcie_device_ids
        ):
            raise DecodeComponentPolicyError(
                "physical device IDs must be non-empty and unique"
            )

        return cls(
            profile_id=_require_nonempty_string(root.get("profile_id"), "profile_id"),
            execution_path=_require_nonempty_string(
                root.get("execution_path"), "execution_path"
            ),
            samples_per_group=_require_positive_int(
                root.get("samples_per_group"), "samples_per_group"
            ),
            warmup_replays=_require_positive_int(
                root.get("warmup_replays"), "warmup_replays"
            ),
            maximum_spread_ratio=maximum_spread_ratio,
            trace_capture_timeout_seconds=_require_positive_int(
                root.get("trace_capture_timeout_seconds"),
                "trace_capture_timeout_seconds",
            ),
            minimum_closure_ratio=minimum_closure_ratio,
            maximum_closure_ratio=maximum_closure_ratio,
            minimum_weighted_share=minimum_weighted_share,
            minimum_dominance_ratio=minimum_dominance_ratio,
            authority_blake3=_require_hex_string(
                identities.get("authority_blake3"),
                "identities.authority_blake3",
                BLAKE3_LENGTH,
            ),
            checkpoint_revision=_require_hex_string(
                identities.get("checkpoint_revision"),
                "identities.checkpoint_revision",
                REVISION_LENGTH,
            ),
            model_source_revision=_require_hex_string(
                identities.get("model_source_revision"),
                "identities.model_source_revision",
                REVISION_LENGTH,
            ),
            tokenizer_revision=_require_hex_string(
                identities.get("tokenizer_revision"),
                "identities.tokenizer_revision",
                REVISION_LENGTH,
            ),
            expected_pcie_device_ids=expected_pcie_device_ids,
            mesh_shape=(
                _require_positive_int(topology.get("mesh_rows"), "topology.mesh_rows"),
                _require_positive_int(
                    topology.get("mesh_columns"), "topology.mesh_columns"
                ),
            ),
            expected_cpu_affinity=tuple(range(minimum_cpu, maximum_cpu + 1)),
            trace_region_size_bytes=_require_positive_int(
                topology.get("trace_region_size_bytes"),
                "topology.trace_region_size_bytes",
            ),
            maximum_sequence_length=_require_positive_int(
                topology.get("maximum_sequence_length"),
                "topology.maximum_sequence_length",
            ),
            kv_block_size=_require_positive_int(
                topology.get("kv_block_size"), "topology.kv_block_size"
            ),
            layer_count=layer_count,
            gdn_layer_count=gdn_layer_count,
            full_attention_layer_count=full_attention_layer_count,
            mlp_layer_count=mlp_layer_count,
            gdn_representatives=gdn_representatives,
            full_attention_representatives=full_attention_representatives,
            mlp_representatives=mlp_representatives,
            prompt=_require_nonempty_string(root.get("prompt"), "prompt"),
        )


@dataclass(frozen=True)
class DecodeComponentIdentity:
    runtime_identity: str
    package_identity: str
    model_identity: str
    authority_blake3: str
    checkpoint_revision: str
    model_source_revision: str
    tokenizer_revision: str


@dataclass(frozen=True)
class LayerTypeObservation:
    layer_index: int
    mixer_type: str


@dataclass(frozen=True)
class ComponentReplaySample:
    component: str
    layer_index: int | None
    sample_index: int
    seconds: float


@dataclass(frozen=True)
class FullReplaySample:
    sample_index: int
    device_seconds: float
    token_id: int


@dataclass(frozen=True)
class HostBoundarySample:
    sample_index: int
    input_update_seconds: float
    winner_readback_seconds: float


@dataclass(frozen=True)
class DecodeComponentEvidence:
    identity: DecodeComponentIdentity
    physical_pcie_device_ids: tuple[int, ...]
    mesh_shape: tuple[int, int]
    host_cpu_affinity: tuple[int, ...]
    layer_types: tuple[LayerTypeObservation, ...]
    component_samples: tuple[ComponentReplaySample, ...]
    full_samples: tuple[FullReplaySample, ...]
    host_samples: tuple[HostBoundarySample, ...]
    cleanup_complete: bool


@dataclass(frozen=True)
class ReplayGroupSummary:
    component: str
    layer_index: int | None
    sample_count: int
    median_seconds: float
    spread_ratio: float

    def to_mapping(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "layer_index": self.layer_index,
            "sample_count": self.sample_count,
            "median_seconds": self.median_seconds,
            "spread_ratio": self.spread_ratio,
        }


@dataclass(frozen=True)
class WeightedComponentSummary:
    component: str
    representative_median_seconds: float
    multiplicity: int
    weighted_seconds: float
    weighted_share: float

    def to_mapping(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "representative_median_seconds": self.representative_median_seconds,
            "multiplicity": self.multiplicity,
            "weighted_seconds": self.weighted_seconds,
            "weighted_share": self.weighted_share,
        }


@dataclass(frozen=True)
class DecodeComponentDecision:
    attribution_valid: bool
    selected_component: str | None
    reasons: tuple[str, ...]
    group_summaries: tuple[ReplayGroupSummary, ...]
    weighted_components: tuple[WeightedComponentSummary, ...]
    modeled_full_device_seconds: float | None
    measured_full_device_seconds: float | None
    closure_ratio: float | None
    full_spread_ratio: float | None
    median_input_update_seconds: float | None
    input_update_spread_ratio: float | None
    median_winner_readback_seconds: float | None
    winner_readback_spread_ratio: float | None
    repeated_token_id: int | None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "attribution_valid": self.attribution_valid,
            "selected_component": self.selected_component,
            "reasons": list(self.reasons),
            "group_summaries": [
                summary.to_mapping() for summary in self.group_summaries
            ],
            "weighted_components": [
                summary.to_mapping() for summary in self.weighted_components
            ],
            "modeled_full_device_seconds": self.modeled_full_device_seconds,
            "measured_full_device_seconds": self.measured_full_device_seconds,
            "closure_ratio": self.closure_ratio,
            "full_spread_ratio": self.full_spread_ratio,
            "median_input_update_seconds": self.median_input_update_seconds,
            "input_update_spread_ratio": self.input_update_spread_ratio,
            "median_winner_readback_seconds": self.median_winner_readback_seconds,
            "winner_readback_spread_ratio": self.winner_readback_spread_ratio,
            "repeated_token_id": self.repeated_token_id,
            "verdict": "validated" if self.attribution_valid else "rejected",
        }


def evaluate_decode_components(
    profile: DecodeComponentProfile,
    evidence: DecodeComponentEvidence,
) -> DecodeComponentDecision:
    reasons: list[str] = []
    _validate_identity(profile, evidence.identity, reasons)
    if evidence.physical_pcie_device_ids != profile.expected_pcie_device_ids:
        _append_once(reasons, "physical-device-identity")
    if evidence.mesh_shape != profile.mesh_shape:
        _append_once(reasons, "mesh-shape")
    if evidence.host_cpu_affinity != profile.expected_cpu_affinity:
        _append_once(reasons, "host-cpu-affinity")
    if not evidence.cleanup_complete:
        _append_once(reasons, "cleanup-incomplete")

    _validate_layer_types(profile, evidence.layer_types, reasons)
    expected_groups = _expected_component_groups(profile)
    grouped: dict[tuple[str, int | None], list[ComponentReplaySample]] = {
        key: [] for key in expected_groups
    }
    seen_component_samples: set[tuple[str, int | None, int]] = set()
    for sample in evidence.component_samples:
        group_key = (sample.component, sample.layer_index)
        sample_key = (sample.component, sample.layer_index, sample.sample_index)
        if group_key not in grouped:
            _append_once(reasons, "unexpected-component-group")
            continue
        if sample_key in seen_component_samples:
            _append_once(reasons, "duplicate-component-sample")
            continue
        seen_component_samples.add(sample_key)
        if not 1 <= sample.sample_index <= profile.samples_per_group:
            _append_once(reasons, "component-sample-index")
        if not _valid_duration(sample.seconds):
            _append_once(reasons, "component-duration")
        grouped[group_key].append(sample)

    group_summaries: list[ReplayGroupSummary] = []
    for component, layer_index in expected_groups:
        samples = grouped[(component, layer_index)]
        expected_indices = set(range(1, profile.samples_per_group + 1))
        actual_indices = {sample.sample_index for sample in samples}
        if (
            len(samples) != profile.samples_per_group
            or actual_indices != expected_indices
        ):
            _append_once(reasons, f"{_group_label(component, layer_index)}-samples")
            continue
        durations = [sample.seconds for sample in samples]
        if any(not _valid_duration(duration) for duration in durations):
            continue
        spread_ratio = max(durations) / min(durations)
        if spread_ratio > profile.maximum_spread_ratio:
            _append_once(reasons, f"{_group_label(component, layer_index)}-instability")
        group_summaries.append(
            ReplayGroupSummary(
                component=component,
                layer_index=layer_index,
                sample_count=len(samples),
                median_seconds=statistics.median(durations),
                spread_ratio=spread_ratio,
            )
        )

    full_median, full_spread, repeated_token = _summarize_full(
        profile, evidence.full_samples, reasons
    )
    update_median, update_spread, readback_median, readback_spread = _summarize_host(
        profile, evidence.host_samples, reasons
    )

    weighted_components: tuple[WeightedComponentSummary, ...] = ()
    modeled_full: float | None = None
    closure_ratio: float | None = None
    selected_component: str | None = None
    representative_medians = _representative_component_medians(profile, group_summaries)
    if representative_medians is not None and full_median is not None:
        multiplicities = {
            EMBEDDING_COMPONENT: 1,
            GDN_MIXER_COMPONENT: profile.gdn_layer_count,
            MLP_COMPONENT: profile.mlp_layer_count,
            FULL_ATTENTION_MIXER_COMPONENT: profile.full_attention_layer_count,
            LM_HEAD_ARGMAX_COMPONENT: 1,
        }
        weighted_values = {
            component: representative_medians[component] * multiplicities[component]
            for component in COMPONENT_ORDER
        }
        modeled_full = sum(weighted_values.values())
        closure_ratio = modeled_full / full_median
        if (
            not profile.minimum_closure_ratio
            <= closure_ratio
            <= profile.maximum_closure_ratio
        ):
            _append_once(reasons, "additive-model-closure")
        weighted_components = tuple(
            WeightedComponentSummary(
                component=component,
                representative_median_seconds=representative_medians[component],
                multiplicity=multiplicities[component],
                weighted_seconds=weighted_values[component],
                weighted_share=weighted_values[component] / modeled_full,
            )
            for component in COMPONENT_ORDER
        )
        ranked = sorted(
            weighted_components,
            key=lambda summary: (-summary.weighted_seconds, summary.component),
        )
        top = ranked[0]
        second = ranked[1]
        if top.weighted_share < profile.minimum_weighted_share:
            _append_once(reasons, "minimum-weighted-share")
        if (
            top.weighted_seconds / second.weighted_seconds
            < profile.minimum_dominance_ratio
        ):
            _append_once(reasons, "minimum-dominance-ratio")
        if not reasons:
            selected_component = top.component

    return DecodeComponentDecision(
        attribution_valid=not reasons,
        selected_component=selected_component,
        reasons=tuple(reasons),
        group_summaries=tuple(group_summaries),
        weighted_components=weighted_components,
        modeled_full_device_seconds=modeled_full,
        measured_full_device_seconds=full_median,
        closure_ratio=closure_ratio,
        full_spread_ratio=full_spread,
        median_input_update_seconds=update_median,
        input_update_spread_ratio=update_spread,
        median_winner_readback_seconds=readback_median,
        winner_readback_spread_ratio=readback_spread,
        repeated_token_id=repeated_token,
    )


def _validate_identity(
    profile: DecodeComponentProfile,
    identity: DecodeComponentIdentity,
    reasons: list[str],
) -> None:
    for label, value in (
        ("runtime", identity.runtime_identity),
        ("package", identity.package_identity),
        ("model", identity.model_identity),
    ):
        if not isinstance(value, str) or not value:
            _append_once(reasons, f"{label}-identity")
    if identity.authority_blake3 != profile.authority_blake3:
        _append_once(reasons, "authority-identity")
    if identity.checkpoint_revision != profile.checkpoint_revision:
        _append_once(reasons, "checkpoint-identity")
    if identity.model_source_revision != profile.model_source_revision:
        _append_once(reasons, "model-source-identity")
    if identity.tokenizer_revision != profile.tokenizer_revision:
        _append_once(reasons, "tokenizer-identity")


def _validate_layer_types(
    profile: DecodeComponentProfile,
    observations: Sequence[LayerTypeObservation],
    reasons: list[str],
) -> None:
    observed: dict[int, str] = {}
    for observation in observations:
        if observation.layer_index in observed:
            _append_once(reasons, "duplicate-layer-type")
            continue
        observed[observation.layer_index] = observation.mixer_type
    expected_indices = set(profile.gdn_representatives)
    expected_indices.update(profile.full_attention_representatives)
    expected_indices.update(profile.mlp_representatives)
    if set(observed) != expected_indices:
        _append_once(reasons, "representative-layer-set")
    for layer_index in profile.gdn_representatives:
        if observed.get(layer_index) != LINEAR_ATTENTION_TYPE:
            _append_once(reasons, f"gdn-layer-{layer_index}-type")
    for layer_index in profile.full_attention_representatives:
        if observed.get(layer_index) != FULL_ATTENTION_TYPE:
            _append_once(reasons, f"full-attention-layer-{layer_index}-type")
    for layer_index in profile.mlp_representatives:
        if observed.get(layer_index) not in (
            LINEAR_ATTENTION_TYPE,
            FULL_ATTENTION_TYPE,
        ):
            _append_once(reasons, f"mlp-layer-{layer_index}-type")


def _expected_component_groups(
    profile: DecodeComponentProfile,
) -> tuple[tuple[str, int | None], ...]:
    groups: list[tuple[str, int | None]] = [(EMBEDDING_COMPONENT, None)]
    groups.extend((GDN_MIXER_COMPONENT, index) for index in profile.gdn_representatives)
    groups.extend((MLP_COMPONENT, index) for index in profile.mlp_representatives)
    groups.extend(
        (FULL_ATTENTION_MIXER_COMPONENT, index)
        for index in profile.full_attention_representatives
    )
    groups.append((LM_HEAD_ARGMAX_COMPONENT, None))
    return tuple(groups)


def _representative_component_medians(
    profile: DecodeComponentProfile,
    summaries: Sequence[ReplayGroupSummary],
) -> dict[str, float] | None:
    by_group = {
        (summary.component, summary.layer_index): summary.median_seconds
        for summary in summaries
    }
    expected_groups = _expected_component_groups(profile)
    if any(group not in by_group for group in expected_groups):
        return None
    return {
        EMBEDDING_COMPONENT: by_group[(EMBEDDING_COMPONENT, None)],
        GDN_MIXER_COMPONENT: statistics.median(
            by_group[(GDN_MIXER_COMPONENT, index)]
            for index in profile.gdn_representatives
        ),
        MLP_COMPONENT: statistics.median(
            by_group[(MLP_COMPONENT, index)] for index in profile.mlp_representatives
        ),
        FULL_ATTENTION_MIXER_COMPONENT: statistics.median(
            by_group[(FULL_ATTENTION_MIXER_COMPONENT, index)]
            for index in profile.full_attention_representatives
        ),
        LM_HEAD_ARGMAX_COMPONENT: by_group[(LM_HEAD_ARGMAX_COMPONENT, None)],
    }


def _summarize_full(
    profile: DecodeComponentProfile,
    samples: Sequence[FullReplaySample],
    reasons: list[str],
) -> tuple[float | None, float | None, int | None]:
    if len(samples) != profile.samples_per_group:
        _append_once(reasons, "full-replay-samples")
        return None, None, None
    expected_indices = set(range(1, profile.samples_per_group + 1))
    actual_indices = {sample.sample_index for sample in samples}
    if len(actual_indices) != len(samples) or actual_indices != expected_indices:
        _append_once(reasons, "full-replay-sample-index")
    durations = [sample.device_seconds for sample in samples]
    if any(not _valid_duration(duration) for duration in durations):
        _append_once(reasons, "full-replay-duration")
        return None, None, None
    token_ids = {sample.token_id for sample in samples}
    if len(token_ids) != 1 or next(iter(token_ids)) < 0:
        _append_once(reasons, "full-replay-token")
        repeated_token = None
    else:
        repeated_token = next(iter(token_ids))
    spread = max(durations) / min(durations)
    if spread > profile.maximum_spread_ratio:
        _append_once(reasons, "full-replay-instability")
    return statistics.median(durations), spread, repeated_token


def _summarize_host(
    profile: DecodeComponentProfile,
    samples: Sequence[HostBoundarySample],
    reasons: list[str],
) -> tuple[float | None, float | None, float | None, float | None]:
    if len(samples) != profile.samples_per_group:
        _append_once(reasons, "host-boundary-samples")
        return None, None, None, None
    expected_indices = set(range(1, profile.samples_per_group + 1))
    actual_indices = {sample.sample_index for sample in samples}
    if len(actual_indices) != len(samples) or actual_indices != expected_indices:
        _append_once(reasons, "host-boundary-sample-index")
    updates = [sample.input_update_seconds for sample in samples]
    readbacks = [sample.winner_readback_seconds for sample in samples]
    if any(not _valid_duration(duration) for duration in updates + readbacks):
        _append_once(reasons, "host-boundary-duration")
        return None, None, None, None
    update_spread = max(updates) / min(updates)
    readback_spread = max(readbacks) / min(readbacks)
    if update_spread > profile.maximum_spread_ratio:
        _append_once(reasons, "host-input-update-instability")
    if readback_spread > profile.maximum_spread_ratio:
        _append_once(reasons, "host-winner-readback-instability")
    return (
        statistics.median(updates),
        update_spread,
        statistics.median(readbacks),
        readback_spread,
    )


def _group_label(component: str, layer_index: int | None) -> str:
    return component if layer_index is None else f"{component}-layer-{layer_index}"


def _valid_duration(value: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _append_once(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DecodeComponentPolicyError(f"{label} must be an object")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise DecodeComponentPolicyError(f"{label} must be a boolean")
    return value


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise DecodeComponentPolicyError(f"{label} must be a non-empty string")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise DecodeComponentPolicyError(f"{label} must be a positive integer")
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise DecodeComponentPolicyError(f"{label} must be a non-negative integer")
    return value


def _require_positive_float(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise DecodeComponentPolicyError(f"{label} must be a positive number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise DecodeComponentPolicyError(f"{label} must be a finite positive number")
    return converted


def _require_hex_string(value: Any, label: str, length: int) -> str:
    text = _require_nonempty_string(value, label)
    if len(text) != length or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise DecodeComponentPolicyError(
            f"{label} must be {length} lowercase hexadecimal characters"
        )
    return text


def _require_int_tuple(value: Any, label: str, *, positive: bool) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise DecodeComponentPolicyError(f"{label} must be an array")
    validator = _require_positive_int if positive else _require_nonnegative_int
    return tuple(
        validator(item, f"{label}[{index}]") for index, item in enumerate(value)
    )


def _require_layer_indices(value: Any, label: str, layer_count: int) -> tuple[int, ...]:
    indices = _require_int_tuple(value, label, positive=False)
    if len(indices) != REPRESENTATIVE_LAYER_COUNT:
        raise DecodeComponentPolicyError(
            f"{label} must contain exactly {REPRESENTATIVE_LAYER_COUNT} indices"
        )
    if len(set(indices)) != len(indices):
        raise DecodeComponentPolicyError(f"{label} contains duplicate indices")
    if any(index >= layer_count for index in indices):
        raise DecodeComponentPolicyError(f"{label} contains an out-of-range index")
    return indices
