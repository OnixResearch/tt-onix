#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json_fwd.hpp>

namespace rwkv7::persistent_decode {

constexpr std::size_t kBindingKindCount = 9U;
constexpr std::uint32_t kPersistentTelemetrySchemaVersion = 1U;

enum class BindingKind : std::uint8_t {
  Token,
  State,
  Slot,
  Weight,
  Program,
  Runtime,
  Checkpoint,
  Output,
  Profile,
};

enum class RequestPhase : std::uint8_t {
  Pending,
  Admitted,
  DeviceActive,
  CompletionPending,
  CancellationPending,
  TimeoutPending,
  FailurePending,
  Zeroing,
  Completed,
  Cancelled,
  TimedOut,
  Failed,
};

enum class TerminalReason : std::uint8_t {
  None,
  Completed,
  Cancelled,
  TimedOut,
  Failed,
};

enum class EventKind : std::uint8_t {
  Admit,
  Start,
  StepCompleted,
  Complete,
  Cancel,
  Timeout,
  Fail,
  Acknowledge,
  StateZeroed,
};

enum class ExecutionMode : std::uint8_t {
  NormalDispatch,
  PersistentTrace,
};

enum class FallbackReason : std::uint8_t {
  None,
  BindingMismatch,
  TraceUnavailable,
  ParityEvidenceMissing,
  SimulatorEvidenceMissing,
  ProfilerEvidenceMissing,
  StableGainMissing,
  ProfileNotAdmitted,
  WindowNotAdmitted,
  CanaryInterlockBlocked,
};

enum class CanaryInterlockState : std::uint8_t {
  NotPresent,
  Armed,
  Retain,
  Hold,
  Disabled,
  Invalid,
};

inline constexpr std::uint64_t kCanaryMaximumRequestExecutions = 104U;
inline constexpr std::uint64_t kCanaryMaximumDurationSeconds = 600U;
inline constexpr double kCanaryCandidateTrafficFraction = 0.5;

enum class TelemetryParityStatus : std::uint8_t {
  NotApplicable,
  Passed,
};

enum class TelemetryCleanupStatus : std::uint8_t {
  NotApplicable,
  Complete,
};

struct Limits {
  std::uint32_t maximum_decode_steps;
  std::uint64_t maximum_elapsed_milliseconds;
  std::uint32_t queue_depth;
  std::uint32_t cancellation_poll_interval_steps;
  std::uint32_t output_capacity_tokens;
  std::vector<std::uint32_t> admitted_decode_windows;
};

struct AdmissionRequirements {
  bool require_exact_parity;
  bool require_simulator_smoke;
  bool require_zero_per_token_host_submissions;
  bool require_stable_physical_gain;
};

struct PrecisionRequirements {
  bool fp32_recurrence_vectors;
  bool bf16_readout_boundary;
  std::string recurrence_math_fidelity;
  std::string readout_math_fidelity;
  std::string correction_owner;
};

struct CandidateLineage {
  std::string predecessor_profile_blake3;
  std::string diagnostic_receipt_blake3;
  PrecisionRequirements precision;
};

struct Profile {
  std::uint32_t schema_version;
  std::string profile_id;
  std::string model_profile_blake3;
  std::string mechanism;
  bool physically_admitted;
  Limits limits;
  AdmissionRequirements admission;
  std::optional<CandidateLineage> candidate_lineage;
};

struct BindingSet {
  std::array<std::string, kBindingKindCount> identities;
};

struct ValidationResult {
  bool valid;
  std::optional<BindingKind> binding;
  std::string reason;
};

struct WindowLimitsResult {
  bool accepted;
  Limits limits;
  std::string error;
};

struct RequestState {
  RequestPhase phase;
  TerminalReason terminal_reason;
  std::uint32_t slot;
  std::uint32_t completed_steps;
  std::uint64_t elapsed_milliseconds;
  std::uint32_t output_tokens;
  std::string failure_code;
};

struct Event {
  EventKind kind;
  std::uint32_t completed_steps;
  std::uint64_t elapsed_milliseconds;
  std::uint32_t output_tokens;
  std::string failure_code;
};

struct TransitionResult {
  bool accepted;
  RequestState state;
  std::string error;
};

struct CapabilityEvidence {
  bool window_admitted;
  bool canary_interlock_allows_persistent;
  bool trace_available;
  bool exact_parity;
  bool simulator_smoke;
  bool zero_per_token_host_submissions;
  bool stable_physical_gain;
};

struct CanaryInterlockEvidence {
  bool present;
  CanaryInterlockState state;
  std::string reason;
  std::string expected_file_blake3;
  std::string actual_file_blake3;
  std::string expected_canary_binding_blake3;
  std::string actual_canary_binding_blake3;
  std::string profile_id;
  std::string profile_blake3;
  std::uint64_t maximum_request_executions;
  std::uint64_t maximum_duration_seconds;
  double candidate_traffic_fraction;
};

struct CanaryInterlockDecision {
  bool valid;
  bool allows_persistent;
  CanaryInterlockState state;
  std::string reason;
};

struct ExecutionDecision {
  ExecutionMode mode;
  FallbackReason fallback_reason;
  std::string detail;
};

struct TerminalOutcomeCounts {
  std::uint32_t completed;
  std::uint32_t cancelled;
  std::uint32_t timed_out;
  std::uint32_t failed;
};

struct PersistentTelemetryInput {
  bool window_admitted;
  bool workload_eligible;
  bool bindings_validated;
  bool profile_physically_admitted;
  bool candidate_requested;
  std::optional<std::uint32_t> selected_window_steps;
  ExecutionDecision decision;
  std::span<const RequestState> lifecycle_states;
  bool trace_replayed;
  bool exact_generated_token_parity;
  bool exact_fp32_recurrent_state_parity;
  double trace_setup_seconds;
  double trace_replay_seconds;
  double cleanup_seconds;
};

struct PersistentTelemetry {
  std::uint32_t schema_version;
  bool window_admitted;
  bool workload_eligible;
  bool bindings_validated;
  bool profile_physically_admitted;
  bool candidate_requested;
  std::optional<std::uint32_t> selected_window_steps;
  ExecutionMode selected_mode;
  FallbackReason fallback_reason;
  TerminalOutcomeCounts terminal_outcomes;
  TelemetryParityStatus generated_token_parity;
  TelemetryParityStatus fp32_recurrent_state_parity;
  TelemetryCleanupStatus cleanup_status;
  double trace_setup_seconds;
  double trace_replay_seconds;
  double cleanup_seconds;
};

[[nodiscard]] std::string_view binding_kind_name(BindingKind kind);
[[nodiscard]] std::string_view request_phase_name(RequestPhase phase);
[[nodiscard]] std::string_view terminal_reason_name(TerminalReason reason);
[[nodiscard]] std::string_view execution_mode_name(ExecutionMode mode);
[[nodiscard]] std::string_view fallback_reason_name(FallbackReason reason);
[[nodiscard]] std::string_view
canary_interlock_state_name(CanaryInterlockState state);
[[nodiscard]] std::string_view
telemetry_parity_status_name(TelemetryParityStatus status);
[[nodiscard]] std::string_view
telemetry_cleanup_status_name(TelemetryCleanupStatus status);

[[nodiscard]] Profile parse_profile(const nlohmann::json &document);
[[nodiscard]] ValidationResult validate_profile(const Profile &profile);
[[nodiscard]] ValidationResult
validate_precision_binding(const Profile &profile,
                           bool fp32_recurrence_vectors_enabled,
                           bool bf16_readout_boundary_enabled);
[[nodiscard]] WindowLimitsResult
select_window_limits(const Profile &profile, std::uint32_t decode_steps);

[[nodiscard]] std::string blake3_hex(std::string_view bytes);
[[nodiscard]] std::string derive_binding_identity(
    BindingKind kind, std::span<const std::string_view> components);
[[nodiscard]] ValidationResult validate_bindings(const BindingSet &expected,
                                                 const BindingSet &actual);
[[nodiscard]] ValidationResult
validate_active_slots(std::span<const std::uint32_t> slots,
                      std::uint32_t queue_depth);
[[nodiscard]] CanaryInterlockDecision validate_canary_interlock(
    const CanaryInterlockEvidence &evidence, std::string_view expected_profile_id,
    std::string_view expected_profile_blake3);

[[nodiscard]] RequestState initial_state(std::uint32_t slot);
[[nodiscard]] TransitionResult transition(const RequestState &state,
                                          const Event &event,
                                          const Profile &profile);
[[nodiscard]] bool slot_releasable(const RequestState &state);
[[nodiscard]] bool terminal(const RequestState &state);

[[nodiscard]] ExecutionDecision
select_execution_mode(const Profile &profile,
                      const ValidationResult &binding_validation,
                      const CapabilityEvidence &evidence);

[[nodiscard]] PersistentTelemetry
summarize_persistent_telemetry(const PersistentTelemetryInput &input);
[[nodiscard]] ValidationResult
validate_persistent_telemetry(const PersistentTelemetry &telemetry);

[[nodiscard]] std::string derive_receipt_identity(
    std::string_view profile_identity, const BindingSet &bindings,
    const RequestState &state);

} // namespace rwkv7::persistent_decode
