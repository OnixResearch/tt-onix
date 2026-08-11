#include "persistent_decode.hpp"

#include <blake3.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rwkv7::persistent_decode {
namespace {

constexpr std::uint32_t kProfileSchemaVersion = 1U;
constexpr std::size_t kProfileFieldCount = 7U;
constexpr std::size_t kLimitFieldCount = 5U;
constexpr std::size_t kAdmissionFieldCount = 4U;
constexpr std::size_t kBlake3DigestBytes = 32U;
constexpr std::size_t kHexCharactersPerByte = 2U;
constexpr std::size_t kBlake3HexCharacters =
    kBlake3DigestBytes * kHexCharactersPerByte;
constexpr std::uint8_t kLowNibbleMask = 0x0fU;
constexpr std::uint32_t kBitsPerNibble = 4U;
constexpr std::uint32_t kBitsPerByte = 8U;
constexpr std::size_t kHexDigitCount = 16U;
constexpr std::string_view kBindingDomain =
    "rwkv7-persistent-decode-binding-v1";
constexpr std::string_view kReceiptDomain =
    "rwkv7-persistent-decode-receipt-v1";
constexpr std::string_view kPersistentMechanism =
    "bounded_metal_trace_chunk";

bool valid_blake3_hex(std::string_view value) {
  if (value.size() != kBlake3HexCharacters) {
    return false;
  }
  return std::ranges::all_of(value, [](char character) {
    return (character >= '0' && character <= '9') ||
           (character >= 'a' && character <= 'f');
  });
}

std::array<std::uint8_t, sizeof(std::uint64_t)>
encode_u64(std::uint64_t value) {
  std::array<std::uint8_t, sizeof(std::uint64_t)> bytes{};
  for (std::size_t index = 0U; index < bytes.size(); ++index) {
    const auto shift = static_cast<unsigned int>(index * kBitsPerByte);
    bytes[index] = static_cast<std::uint8_t>(value >> shift);
  }
  return bytes;
}

void hash_frame(blake3_hasher &hasher, std::string_view value) {
  const auto length = encode_u64(static_cast<std::uint64_t>(value.size()));
  blake3_hasher_update(&hasher, length.data(), length.size());
  blake3_hasher_update(&hasher, value.data(), value.size());
}

std::string finalize_hex(blake3_hasher &hasher) {
  constexpr std::array<char, kHexDigitCount> digits = {
      '0', '1', '2', '3', '4', '5', '6', '7',
      '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
  std::array<std::uint8_t, kBlake3DigestBytes> digest{};
  blake3_hasher_finalize(&hasher, digest.data(), digest.size());
  std::string output;
  output.reserve(kBlake3HexCharacters);
  for (const std::uint8_t byte : digest) {
    output.push_back(digits[byte >> kBitsPerNibble]);
    output.push_back(digits[byte & kLowNibbleMask]);
  }
  return output;
}

ValidationResult valid_result() {
  return ValidationResult{true, std::nullopt, {}};
}

ValidationResult invalid_result(std::string reason) {
  return ValidationResult{false, std::nullopt, std::move(reason)};
}

RequestPhase pending_phase(TerminalReason reason) {
  switch (reason) {
  case TerminalReason::Completed:
    return RequestPhase::CompletionPending;
  case TerminalReason::Cancelled:
    return RequestPhase::CancellationPending;
  case TerminalReason::TimedOut:
    return RequestPhase::TimeoutPending;
  case TerminalReason::Failed:
    return RequestPhase::FailurePending;
  case TerminalReason::None:
    break;
  }
  return RequestPhase::FailurePending;
}

RequestPhase terminal_phase(TerminalReason reason) {
  switch (reason) {
  case TerminalReason::Completed:
    return RequestPhase::Completed;
  case TerminalReason::Cancelled:
    return RequestPhase::Cancelled;
  case TerminalReason::TimedOut:
    return RequestPhase::TimedOut;
  case TerminalReason::Failed:
    return RequestPhase::Failed;
  case TerminalReason::None:
    break;
  }
  return RequestPhase::Failed;
}

RequestState begin_exit(RequestState next, TerminalReason reason,
                        std::string failure_code = {}) {
  next.phase = pending_phase(reason);
  next.terminal_reason = reason;
  next.failure_code = std::move(failure_code);
  return next;
}

TransitionResult reject_transition(const RequestState &state,
                                   std::string error) {
  return TransitionResult{false, state, std::move(error)};
}

TransitionResult accept_transition(RequestState state) {
  return TransitionResult{true, std::move(state), {}};
}

std::string unsigned_text(std::uint64_t value) {
  std::array<char, std::numeric_limits<std::uint64_t>::digits10 + 2U> buffer{};
  const auto [end, error] =
      std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
  if (error != std::errc{}) {
    throw std::runtime_error("failed to encode persistent decode identity value");
  }
  return std::string(buffer.data(), end);
}

} // namespace

std::string_view binding_kind_name(BindingKind kind) {
  switch (kind) {
  case BindingKind::Token:
    return "token";
  case BindingKind::State:
    return "state";
  case BindingKind::Slot:
    return "slot";
  case BindingKind::Weight:
    return "weight";
  case BindingKind::Program:
    return "program";
  case BindingKind::Runtime:
    return "runtime";
  case BindingKind::Checkpoint:
    return "checkpoint";
  case BindingKind::Output:
    return "output";
  case BindingKind::Profile:
    return "profile";
  }
  return "invalid";
}

std::string_view request_phase_name(RequestPhase phase) {
  switch (phase) {
  case RequestPhase::Pending:
    return "pending";
  case RequestPhase::Admitted:
    return "admitted";
  case RequestPhase::DeviceActive:
    return "device_active";
  case RequestPhase::CompletionPending:
    return "completion_pending";
  case RequestPhase::CancellationPending:
    return "cancellation_pending";
  case RequestPhase::TimeoutPending:
    return "timeout_pending";
  case RequestPhase::FailurePending:
    return "failure_pending";
  case RequestPhase::Zeroing:
    return "zeroing";
  case RequestPhase::Completed:
    return "completed";
  case RequestPhase::Cancelled:
    return "cancelled";
  case RequestPhase::TimedOut:
    return "timed_out";
  case RequestPhase::Failed:
    return "failed";
  }
  return "invalid";
}

std::string_view execution_mode_name(ExecutionMode mode) {
  switch (mode) {
  case ExecutionMode::NormalDispatch:
    return "normal_dispatch";
  case ExecutionMode::PersistentTrace:
    return "persistent_trace";
  }
  return "invalid";
}

std::string_view fallback_reason_name(FallbackReason reason) {
  switch (reason) {
  case FallbackReason::None:
    return "none";
  case FallbackReason::BindingMismatch:
    return "binding_mismatch";
  case FallbackReason::TraceUnavailable:
    return "trace_unavailable";
  case FallbackReason::ParityEvidenceMissing:
    return "parity_evidence_missing";
  case FallbackReason::SimulatorEvidenceMissing:
    return "simulator_evidence_missing";
  case FallbackReason::ProfilerEvidenceMissing:
    return "profiler_evidence_missing";
  case FallbackReason::StableGainMissing:
    return "stable_gain_missing";
  case FallbackReason::ProfileNotAdmitted:
    return "profile_not_admitted";
  }
  return "invalid";
}

Profile parse_profile(const nlohmann::json &document) {
  if (!document.is_object() || document.size() != kProfileFieldCount) {
    throw std::invalid_argument(
        "persistent decode profile must contain exactly seven fields");
  }
  const nlohmann::json &limits = document.at("limits");
  const nlohmann::json &admission = document.at("admission");
  if (!limits.is_object() || limits.size() != kLimitFieldCount) {
    throw std::invalid_argument(
        "persistent decode limits must contain exactly five fields");
  }
  if (!admission.is_object() || admission.size() != kAdmissionFieldCount) {
    throw std::invalid_argument(
        "persistent decode admission must contain exactly four fields");
  }
  Profile profile{
      document.at("schema_version").get<std::uint32_t>(),
      document.at("profile_id").get<std::string>(),
      document.at("model_profile_blake3").get<std::string>(),
      document.at("mechanism").get<std::string>(),
      document.at("physically_admitted").get<bool>(),
      Limits{limits.at("maximum_decode_steps").get<std::uint32_t>(),
             limits.at("maximum_elapsed_milliseconds").get<std::uint64_t>(),
             limits.at("queue_depth").get<std::uint32_t>(),
             limits.at("cancellation_poll_interval_steps")
                 .get<std::uint32_t>(),
             limits.at("output_capacity_tokens").get<std::uint32_t>()},
      AdmissionRequirements{
          admission.at("require_exact_parity").get<bool>(),
          admission.at("require_simulator_smoke").get<bool>(),
          admission.at("require_zero_per_token_host_submissions").get<bool>(),
          admission.at("require_stable_physical_gain").get<bool>()}};
  const ValidationResult validation = validate_profile(profile);
  if (!validation.valid) {
    throw std::invalid_argument(validation.reason);
  }
  return profile;
}

ValidationResult validate_profile(const Profile &profile) {
  if (profile.schema_version != kProfileSchemaVersion) {
    return invalid_result("persistent decode profile schema mismatch");
  }
  if (profile.profile_id.empty()) {
    return invalid_result("persistent decode profile ID must not be empty");
  }
  if (!valid_blake3_hex(profile.model_profile_blake3)) {
    return invalid_result("persistent decode model profile identity is invalid");
  }
  if (profile.mechanism != kPersistentMechanism) {
    return invalid_result("persistent decode mechanism is unsupported");
  }
  if (profile.limits.maximum_decode_steps == 0U ||
      profile.limits.maximum_elapsed_milliseconds == 0U ||
      profile.limits.queue_depth == 0U ||
      profile.limits.cancellation_poll_interval_steps == 0U ||
      profile.limits.output_capacity_tokens == 0U) {
    return invalid_result("persistent decode limits must be positive");
  }
  if (profile.limits.cancellation_poll_interval_steps <= 1U) {
    return invalid_result(
        "persistent decode cancellation polling must cover multiple steps");
  }
  if (profile.limits.cancellation_poll_interval_steps >
          profile.limits.maximum_decode_steps ||
      profile.limits.maximum_decode_steps %
              profile.limits.cancellation_poll_interval_steps !=
          0U) {
    return invalid_result(
        "persistent decode step bound must contain whole polling intervals");
  }
  if (profile.limits.output_capacity_tokens <
      profile.limits.maximum_decode_steps) {
    return invalid_result(
        "persistent decode output capacity is below the step bound");
  }
  if (!profile.admission.require_exact_parity ||
      !profile.admission.require_simulator_smoke ||
      !profile.admission.require_zero_per_token_host_submissions ||
      !profile.admission.require_stable_physical_gain) {
    return invalid_result(
        "persistent decode admission requirements must remain fail closed");
  }
  return valid_result();
}

std::string blake3_hex(std::string_view bytes) {
  blake3_hasher hasher;
  blake3_hasher_init(&hasher);
  blake3_hasher_update(&hasher, bytes.data(), bytes.size());
  return finalize_hex(hasher);
}

std::string derive_binding_identity(
    BindingKind kind, std::span<const std::string_view> components) {
  if (components.empty()) {
    throw std::invalid_argument(
        "persistent decode binding identity requires a component");
  }
  blake3_hasher hasher;
  blake3_hasher_init(&hasher);
  hash_frame(hasher, kBindingDomain);
  hash_frame(hasher, binding_kind_name(kind));
  const auto count = encode_u64(static_cast<std::uint64_t>(components.size()));
  blake3_hasher_update(&hasher, count.data(), count.size());
  for (const std::string_view component : components) {
    hash_frame(hasher, component);
  }
  return finalize_hex(hasher);
}

ValidationResult validate_bindings(const BindingSet &expected,
                                   const BindingSet &actual) {
  for (std::size_t index = 0U; index < kBindingKindCount; ++index) {
    const BindingKind kind = static_cast<BindingKind>(index);
    if (!valid_blake3_hex(expected.identities[index]) ||
        !valid_blake3_hex(actual.identities[index])) {
      return ValidationResult{false, kind,
                              std::string(binding_kind_name(kind)) +
                                  " binding identity is malformed"};
    }
    if (expected.identities[index] != actual.identities[index]) {
      return ValidationResult{false, kind,
                              std::string(binding_kind_name(kind)) +
                                  " binding identity changed"};
    }
  }
  return valid_result();
}

ValidationResult validate_active_slots(std::span<const std::uint32_t> slots,
                                       std::uint32_t queue_depth) {
  if (queue_depth == 0U) {
    return invalid_result("persistent decode queue depth must be positive");
  }
  if (slots.empty()) {
    return invalid_result("persistent decode admission requires an active slot");
  }
  if (slots.size() > static_cast<std::size_t>(queue_depth)) {
    return invalid_result("persistent decode active slots exceed queue depth");
  }
  std::vector<bool> used(queue_depth, false);
  for (const std::uint32_t slot : slots) {
    if (slot >= queue_depth) {
      return invalid_result("persistent decode slot is outside queue depth");
    }
    if (used[slot]) {
      return invalid_result("persistent decode active slots contain a duplicate");
    }
    used[slot] = true;
  }
  return valid_result();
}

RequestState initial_state(std::uint32_t slot) {
  return RequestState{RequestPhase::Pending, TerminalReason::None, slot, 0U,
                      0U, 0U, {}};
}

TransitionResult transition(const RequestState &state, const Event &event,
                            const Profile &profile) {
  const ValidationResult profile_validation = validate_profile(profile);
  if (!profile_validation.valid) {
    return reject_transition(state, profile_validation.reason);
  }
  RequestState next = state;
  switch (event.kind) {
  case EventKind::Admit:
    if (state.phase != RequestPhase::Pending) {
      return reject_transition(state,
                               "only a pending request can be admitted");
    }
    if (state.slot >= profile.limits.queue_depth) {
      return reject_transition(state,
                               "request slot is outside the profile queue");
    }
    next.phase = RequestPhase::Admitted;
    return accept_transition(std::move(next));
  case EventKind::Start:
    if (state.phase != RequestPhase::Admitted) {
      return reject_transition(state,
                               "only an admitted request can start");
    }
    next.phase = RequestPhase::DeviceActive;
    return accept_transition(std::move(next));
  case EventKind::StepCompleted:
    if (state.phase != RequestPhase::DeviceActive) {
      return reject_transition(state,
                               "only an active request can complete a step");
    }
    if (event.completed_steps != state.completed_steps + 1U) {
      return reject_transition(state,
                               "persistent decode step sequence is not exact");
    }
    if (event.elapsed_milliseconds < state.elapsed_milliseconds ||
        event.output_tokens < state.output_tokens) {
      return reject_transition(
          state, "persistent decode progress counters decreased");
    }
    next.completed_steps = event.completed_steps;
    next.elapsed_milliseconds = event.elapsed_milliseconds;
    next.output_tokens = event.output_tokens;
    if (next.output_tokens > profile.limits.output_capacity_tokens) {
      return accept_transition(begin_exit(std::move(next),
                                          TerminalReason::Failed,
                                          "output_overflow"));
    }
    if (next.completed_steps > profile.limits.maximum_decode_steps) {
      return accept_transition(begin_exit(std::move(next),
                                          TerminalReason::Failed,
                                          "step_overflow"));
    }
    if (next.elapsed_milliseconds >=
        profile.limits.maximum_elapsed_milliseconds) {
      return accept_transition(
          begin_exit(std::move(next), TerminalReason::TimedOut));
    }
    if (next.completed_steps == profile.limits.maximum_decode_steps ||
        next.output_tokens == profile.limits.output_capacity_tokens) {
      return accept_transition(
          begin_exit(std::move(next), TerminalReason::Completed));
    }
    return accept_transition(std::move(next));
  case EventKind::Complete:
    if (state.phase != RequestPhase::DeviceActive) {
      return reject_transition(state,
                               "only an active request can complete");
    }
    return accept_transition(
        begin_exit(std::move(next), TerminalReason::Completed));
  case EventKind::Cancel:
    if (state.phase != RequestPhase::Admitted &&
        state.phase != RequestPhase::DeviceActive) {
      return reject_transition(state,
                               "only an admitted or active request can cancel");
    }
    return accept_transition(
        begin_exit(std::move(next), TerminalReason::Cancelled));
  case EventKind::Timeout:
    if (state.phase != RequestPhase::Admitted &&
        state.phase != RequestPhase::DeviceActive) {
      return reject_transition(state,
                               "only an admitted or active request can time out");
    }
    if (event.elapsed_milliseconds < state.elapsed_milliseconds) {
      return reject_transition(state, "timeout elapsed time decreased");
    }
    next.elapsed_milliseconds = event.elapsed_milliseconds;
    return accept_transition(
        begin_exit(std::move(next), TerminalReason::TimedOut));
  case EventKind::Fail:
    if (state.phase != RequestPhase::Admitted &&
        state.phase != RequestPhase::DeviceActive) {
      return reject_transition(state,
                               "only an admitted or active request can fail");
    }
    if (event.failure_code.empty()) {
      return reject_transition(state, "failure event requires an error code");
    }
    return accept_transition(begin_exit(std::move(next),
                                        TerminalReason::Failed,
                                        event.failure_code));
  case EventKind::Acknowledge:
    if (state.phase != RequestPhase::CompletionPending &&
        state.phase != RequestPhase::CancellationPending &&
        state.phase != RequestPhase::TimeoutPending &&
        state.phase != RequestPhase::FailurePending) {
      return reject_transition(
          state, "only a terminal handshake can be acknowledged");
    }
    next.phase = RequestPhase::Zeroing;
    return accept_transition(std::move(next));
  case EventKind::StateZeroed:
    if (state.phase != RequestPhase::Zeroing ||
        state.terminal_reason == TerminalReason::None) {
      return reject_transition(state,
                               "state can be zeroed only after acknowledgement");
    }
    next.phase = terminal_phase(state.terminal_reason);
    return accept_transition(std::move(next));
  }
  return reject_transition(state, "persistent decode event is invalid");
}

bool terminal(const RequestState &state) {
  return state.phase == RequestPhase::Completed ||
         state.phase == RequestPhase::Cancelled ||
         state.phase == RequestPhase::TimedOut ||
         state.phase == RequestPhase::Failed;
}

bool slot_releasable(const RequestState &state) {
  return terminal(state);
}

ExecutionDecision
select_execution_mode(const Profile &profile,
                      const ValidationResult &binding_validation,
                      const CapabilityEvidence &evidence) {
  if (!binding_validation.valid) {
    return ExecutionDecision{ExecutionMode::NormalDispatch,
                             FallbackReason::BindingMismatch,
                             binding_validation.reason};
  }
  if (!profile.physically_admitted) {
    return ExecutionDecision{ExecutionMode::NormalDispatch,
                             FallbackReason::ProfileNotAdmitted,
                             "the persistent candidate is not physically admitted"};
  }
  if (!evidence.trace_available) {
    return ExecutionDecision{ExecutionMode::NormalDispatch,
                             FallbackReason::TraceUnavailable,
                             "Metal Trace is unavailable"};
  }
  if (!evidence.exact_parity) {
    return ExecutionDecision{ExecutionMode::NormalDispatch,
                             FallbackReason::ParityEvidenceMissing,
                             "exact token and FP32 state parity is unproven"};
  }
  if (!evidence.simulator_smoke) {
    return ExecutionDecision{ExecutionMode::NormalDispatch,
                             FallbackReason::SimulatorEvidenceMissing,
                             "the simulator smoke receipt is missing"};
  }
  if (!evidence.zero_per_token_host_submissions) {
    return ExecutionDecision{
        ExecutionMode::NormalDispatch,
        FallbackReason::ProfilerEvidenceMissing,
        "profiler evidence does not prove zero per-token host submissions"};
  }
  if (!evidence.stable_physical_gain) {
    return ExecutionDecision{ExecutionMode::NormalDispatch,
                             FallbackReason::StableGainMissing,
                             "stable physical throughput gain is unproven"};
  }
  return ExecutionDecision{ExecutionMode::PersistentTrace,
                           FallbackReason::None, {}};
}

std::string derive_receipt_identity(std::string_view profile_identity,
                                    const BindingSet &bindings,
                                    const RequestState &state) {
  if (!valid_blake3_hex(profile_identity)) {
    throw std::invalid_argument(
        "persistent decode receipt profile identity is malformed");
  }
  blake3_hasher hasher;
  blake3_hasher_init(&hasher);
  hash_frame(hasher, kReceiptDomain);
  hash_frame(hasher, profile_identity);
  for (std::size_t index = 0U; index < kBindingKindCount; ++index) {
    if (!valid_blake3_hex(bindings.identities[index])) {
      throw std::invalid_argument(
          "persistent decode receipt binding identity is malformed");
    }
    hash_frame(hasher, binding_kind_name(static_cast<BindingKind>(index)));
    hash_frame(hasher, bindings.identities[index]);
  }
  hash_frame(hasher, request_phase_name(state.phase));
  const std::string slot = unsigned_text(state.slot);
  const std::string steps = unsigned_text(state.completed_steps);
  const std::string elapsed = unsigned_text(state.elapsed_milliseconds);
  const std::string outputs = unsigned_text(state.output_tokens);
  hash_frame(hasher, slot);
  hash_frame(hasher, steps);
  hash_frame(hasher, elapsed);
  hash_frame(hasher, outputs);
  hash_frame(hasher, state.failure_code);
  return finalize_hex(hasher);
}

} // namespace rwkv7::persistent_decode
