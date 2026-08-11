#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include <nlohmann/json_fwd.hpp>

namespace rwkv7::persistent_decode {

constexpr std::size_t kBindingKindCount = 9U;

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
};

struct Limits {
  std::uint32_t maximum_decode_steps;
  std::uint64_t maximum_elapsed_milliseconds;
  std::uint32_t queue_depth;
  std::uint32_t cancellation_poll_interval_steps;
  std::uint32_t output_capacity_tokens;
};

struct AdmissionRequirements {
  bool require_exact_parity;
  bool require_simulator_smoke;
  bool require_zero_per_token_host_submissions;
  bool require_stable_physical_gain;
};

struct Profile {
  std::uint32_t schema_version;
  std::string profile_id;
  std::string model_profile_blake3;
  std::string mechanism;
  bool physically_admitted;
  Limits limits;
  AdmissionRequirements admission;
};

struct BindingSet {
  std::array<std::string, kBindingKindCount> identities;
};

struct ValidationResult {
  bool valid;
  std::optional<BindingKind> binding;
  std::string reason;
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
  bool trace_available;
  bool exact_parity;
  bool simulator_smoke;
  bool zero_per_token_host_submissions;
  bool stable_physical_gain;
};

struct ExecutionDecision {
  ExecutionMode mode;
  FallbackReason fallback_reason;
  std::string detail;
};

[[nodiscard]] std::string_view binding_kind_name(BindingKind kind);
[[nodiscard]] std::string_view request_phase_name(RequestPhase phase);
[[nodiscard]] std::string_view execution_mode_name(ExecutionMode mode);
[[nodiscard]] std::string_view fallback_reason_name(FallbackReason reason);

[[nodiscard]] Profile parse_profile(const nlohmann::json &document);
[[nodiscard]] ValidationResult validate_profile(const Profile &profile);

[[nodiscard]] std::string blake3_hex(std::string_view bytes);
[[nodiscard]] std::string derive_binding_identity(
    BindingKind kind, std::span<const std::string_view> components);
[[nodiscard]] ValidationResult validate_bindings(const BindingSet &expected,
                                                 const BindingSet &actual);
[[nodiscard]] ValidationResult
validate_active_slots(std::span<const std::uint32_t> slots,
                      std::uint32_t queue_depth);

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

[[nodiscard]] std::string derive_receipt_identity(
    std::string_view profile_identity, const BindingSet &bindings,
    const RequestState &state);

} // namespace rwkv7::persistent_decode
