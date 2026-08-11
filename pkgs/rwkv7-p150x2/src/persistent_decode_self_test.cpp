#include "persistent_decode_self_test.hpp"

#include <array>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace rwkv7::persistent_decode {
namespace {

constexpr std::uint32_t kFirstSlot = 0U;
constexpr std::uint32_t kSecondSlot = 1U;
constexpr std::uint64_t kNominalElapsedMilliseconds = 1U;
constexpr std::size_t kBindingComponentCount = 2U;
constexpr std::size_t kConcurrentSlotCount = 2U;
constexpr std::string_view kFailureCode = "self_test_failure";

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

Event simple_event(EventKind kind) {
  return Event{kind, 0U, 0U, 0U, {}};
}

RequestState apply(const RequestState &state, const Event &event,
                   const Profile &profile) {
  const TransitionResult result = transition(state, event, profile);
  require(result.accepted, result.error);
  return result.state;
}

RequestState start_request(std::uint32_t slot, const Profile &profile) {
  RequestState state = initial_state(slot);
  state = apply(state, simple_event(EventKind::Admit), profile);
  state = apply(state, simple_event(EventKind::Start), profile);
  return state;
}

RequestState acknowledge_and_zero(RequestState state, const Profile &profile) {
  state = apply(state, simple_event(EventKind::Acknowledge), profile);
  require(state.phase == RequestPhase::Zeroing,
          "persistent decode acknowledgement did not start zeroing");
  require(!slot_releasable(state),
          "persistent decode released a slot before state zeroing");
  return apply(state, simple_event(EventKind::StateZeroed), profile);
}

BindingSet sample_bindings(const Profile &profile) {
  BindingSet bindings{};
  for (std::size_t index = 0U; index < kBindingKindCount; ++index) {
    const BindingKind kind = static_cast<BindingKind>(index);
    const std::array<std::string_view, kBindingComponentCount> components = {
        profile.profile_id, binding_kind_name(kind)};
    bindings.identities[index] = derive_binding_identity(kind, components);
  }
  return bindings;
}

void test_normal_completion(const Profile &profile) {
  RequestState state = start_request(kFirstSlot, profile);
  const Event progress{EventKind::StepCompleted, 1U,
                       kNominalElapsedMilliseconds, 1U, {}};
  state = apply(state, progress, profile);
  require(state.phase == RequestPhase::DeviceActive,
          "persistent decode left the active phase too early");
  state = apply(state, simple_event(EventKind::Complete), profile);
  require(state.phase == RequestPhase::CompletionPending,
          "persistent decode did not enter the completion handshake");
  require(!slot_releasable(state),
          "persistent decode released a slot before completion acknowledgement");
  state = acknowledge_and_zero(std::move(state), profile);
  require(state.phase == RequestPhase::Completed && slot_releasable(state),
          "persistent decode did not complete deterministic cleanup");
}

void test_maximum_length_completion(const Profile &profile) {
  RequestState state = start_request(kFirstSlot, profile);
  for (std::uint32_t step = 1U;
       step <= profile.limits.maximum_decode_steps; ++step) {
    const Event progress{EventKind::StepCompleted, step,
                         static_cast<std::uint64_t>(step), step, {}};
    state = apply(state, progress, profile);
  }
  require(state.phase == RequestPhase::CompletionPending,
          "maximum-length decode did not request completion acknowledgement");
  state = acknowledge_and_zero(std::move(state), profile);
  require(state.phase == RequestPhase::Completed,
          "maximum-length decode did not complete");
}

void test_cancel_timeout_and_failure(const Profile &profile) {
  RequestState cancelled = start_request(kFirstSlot, profile);
  cancelled = apply(cancelled, simple_event(EventKind::Cancel), profile);
  require(cancelled.phase == RequestPhase::CancellationPending &&
              !slot_releasable(cancelled),
          "cancellation bypassed its acknowledgement handshake");
  cancelled = acknowledge_and_zero(std::move(cancelled), profile);
  require(cancelled.phase == RequestPhase::Cancelled,
          "cancellation did not reach its terminal phase");

  RequestState timed_out = start_request(kFirstSlot, profile);
  Event timeout = simple_event(EventKind::Timeout);
  timeout.elapsed_milliseconds =
      profile.limits.maximum_elapsed_milliseconds;
  timed_out = apply(timed_out, timeout, profile);
  require(timed_out.phase == RequestPhase::TimeoutPending,
          "timeout did not enter its acknowledgement handshake");
  timed_out = acknowledge_and_zero(std::move(timed_out), profile);
  require(timed_out.phase == RequestPhase::TimedOut,
          "timeout did not reach its terminal phase");

  RequestState failed = start_request(kFirstSlot, profile);
  Event failure = simple_event(EventKind::Fail);
  failure.failure_code = kFailureCode;
  failed = apply(failed, failure, profile);
  require(failed.phase == RequestPhase::FailurePending,
          "failure did not enter its acknowledgement handshake");
  failed = acknowledge_and_zero(std::move(failed), profile);
  require(failed.phase == RequestPhase::Failed &&
              failed.failure_code == kFailureCode,
          "failure did not retain its deterministic error code");
}

void test_invalid_transitions(const Profile &profile) {
  const RequestState pending = initial_state(kFirstSlot);
  const Event premature_step{EventKind::StepCompleted, 1U,
                             kNominalElapsedMilliseconds, 1U, {}};
  require(!transition(pending, premature_step, profile).accepted,
          "persistent decode accepted a step before admission");

  const RequestState active = start_request(kFirstSlot, profile);
  require(!transition(active, simple_event(EventKind::Acknowledge), profile)
               .accepted,
          "persistent decode accepted an acknowledgement before exit");
  require(!transition(active, simple_event(EventKind::StateZeroed), profile)
               .accepted,
          "persistent decode accepted zeroing before acknowledgement");

  const Event skipped_step{EventKind::StepCompleted, 2U,
                           kNominalElapsedMilliseconds, 1U, {}};
  require(!transition(active, skipped_step, profile).accepted,
          "persistent decode accepted a nonsequential step");

  Event empty_failure = simple_event(EventKind::Fail);
  require(!transition(active, empty_failure, profile).accepted,
          "persistent decode accepted a failure without an error code");
}

void test_output_overflow(const Profile &profile) {
  RequestState state = start_request(kFirstSlot, profile);
  const Event overflow{EventKind::StepCompleted, 1U,
                       kNominalElapsedMilliseconds,
                       profile.limits.output_capacity_tokens + 1U, {}};
  state = apply(state, overflow, profile);
  require(state.phase == RequestPhase::FailurePending &&
              state.failure_code == "output_overflow",
          "persistent decode did not fail closed on output overflow");
  state = acknowledge_and_zero(std::move(state), profile);
  require(state.phase == RequestPhase::Failed,
          "output overflow did not finish deterministic cleanup");
}

void test_slot_admission_and_recycling(const Profile &profile) {
  require(profile.limits.queue_depth > kSecondSlot,
          "persistent decode self-test profile needs two queue slots");
  const std::array<std::uint32_t, kConcurrentSlotCount> distinct = {
      kFirstSlot, kSecondSlot};
  const std::array<std::uint32_t, kConcurrentSlotCount> duplicate = {
      kFirstSlot, kFirstSlot};
  const std::array<std::uint32_t, 1U> outside = {profile.limits.queue_depth};
  require(validate_active_slots(distinct, profile.limits.queue_depth).valid,
          "persistent decode rejected distinct concurrent slots");
  require(!validate_active_slots(duplicate, profile.limits.queue_depth).valid,
          "persistent decode accepted duplicate active slots");
  require(!validate_active_slots(outside, profile.limits.queue_depth).valid,
          "persistent decode accepted a slot outside the queue");

  RequestState first = start_request(kFirstSlot, profile);
  RequestState second = start_request(kSecondSlot, profile);
  first = apply(first, simple_event(EventKind::Cancel), profile);
  require(!slot_releasable(first) && !slot_releasable(second),
          "persistent decode released a live concurrent slot");
  first = acknowledge_and_zero(std::move(first), profile);
  require(slot_releasable(first) && !slot_releasable(second),
          "persistent decode did not isolate concurrent slot cleanup");
  RequestState recycled = start_request(kFirstSlot, profile);
  require(recycled.phase == RequestPhase::DeviceActive,
          "persistent decode did not recycle a zeroed slot");
}

void test_binding_drift_and_receipts(const Profile &profile,
                                     std::string_view profile_bytes) {
  const BindingSet expected = sample_bindings(profile);
  require(validate_bindings(expected, expected).valid,
          "persistent decode rejected stable bindings");
  for (std::size_t index = 0U; index < kBindingKindCount; ++index) {
    BindingSet drifted = expected;
    const BindingKind kind = static_cast<BindingKind>(index);
    const std::array<std::string_view, kBindingComponentCount> components = {
        profile.profile_id, "drifted"};
    drifted.identities[index] = derive_binding_identity(kind, components);
    const ValidationResult validation = validate_bindings(expected, drifted);
    require(!validation.valid && validation.binding == kind,
            "persistent decode failed to identify binding drift");
  }

  RequestState completed = start_request(kFirstSlot, profile);
  completed = apply(completed, simple_event(EventKind::Complete), profile);
  completed = acknowledge_and_zero(std::move(completed), profile);
  const std::string profile_identity = blake3_hex(profile_bytes);
  const std::string first =
      derive_receipt_identity(profile_identity, expected, completed);
  const std::string second =
      derive_receipt_identity(profile_identity, expected, completed);
  require(first == second,
          "persistent decode receipt identity is not deterministic");
}

void test_fail_closed_mode_selection(const Profile &profile) {
  const BindingSet bindings = sample_bindings(profile);
  const ValidationResult stable = validate_bindings(bindings, bindings);
  const CapabilityEvidence complete{true, true, true, true, true};
  Profile admitted = profile;
  admitted.physically_admitted = true;
  require(select_execution_mode(admitted, stable, complete).mode ==
              ExecutionMode::PersistentTrace,
          "persistent decode rejected complete admission evidence");

  const CapabilityEvidence no_trace{false, true, true, true, true};
  require(select_execution_mode(admitted, stable, no_trace).fallback_reason ==
              FallbackReason::TraceUnavailable,
          "persistent decode did not reject a missing trace mechanism");
  const CapabilityEvidence no_parity{true, false, true, true, true};
  require(select_execution_mode(admitted, stable, no_parity).fallback_reason ==
              FallbackReason::ParityEvidenceMissing,
          "persistent decode did not reject missing parity evidence");
  const CapabilityEvidence no_simulator{true, true, false, true, true};
  require(select_execution_mode(admitted, stable, no_simulator)
              .fallback_reason == FallbackReason::SimulatorEvidenceMissing,
          "persistent decode did not reject missing simulator evidence");
  const CapabilityEvidence no_profiler{true, true, true, false, true};
  require(select_execution_mode(admitted, stable, no_profiler).fallback_reason ==
              FallbackReason::ProfilerEvidenceMissing,
          "persistent decode did not reject missing profiler evidence");
  const CapabilityEvidence no_gain{true, true, true, true, false};
  require(select_execution_mode(admitted, stable, no_gain).fallback_reason ==
              FallbackReason::StableGainMissing,
          "persistent decode did not reject missing physical gain");
  Profile not_admitted = profile;
  not_admitted.physically_admitted = false;
  require(select_execution_mode(not_admitted, stable, complete).fallback_reason ==
              FallbackReason::ProfileNotAdmitted,
          "persistent decode did not preserve normal dispatch before admission");

  BindingSet drifted = bindings;
  const std::array<std::string_view, kBindingComponentCount> components = {
      profile.profile_id, "mode-drift"};
  drifted.identities[0U] = derive_binding_identity(BindingKind::Token, components);
  const ValidationResult mismatch = validate_bindings(bindings, drifted);
  require(select_execution_mode(admitted, mismatch, complete).fallback_reason ==
              FallbackReason::BindingMismatch,
          "persistent decode did not fail closed on binding drift");
}

void test_invalid_profile(const Profile &profile) {
  Profile invalid = profile;
  invalid.limits.maximum_decode_steps = 0U;
  require(!validate_profile(invalid).valid,
          "persistent decode accepted an unbounded profile");
  invalid = profile;
  invalid.admission.require_exact_parity = false;
  require(!validate_profile(invalid).valid,
          "persistent decode accepted an open admission profile");
}

} // namespace

void run_persistent_decode_self_test(const Profile &profile,
                                     std::string_view profile_bytes) {
  require(validate_profile(profile).valid,
          "installed persistent decode profile is invalid");
  require(!profile_bytes.empty(),
          "installed persistent decode profile is empty");
  test_normal_completion(profile);
  test_maximum_length_completion(profile);
  test_cancel_timeout_and_failure(profile);
  test_invalid_transitions(profile);
  test_output_overflow(profile);
  test_slot_admission_and_recycling(profile);
  test_binding_drift_and_receipts(profile, profile_bytes);
  test_fail_closed_mode_selection(profile);
  test_invalid_profile(profile);
}

} // namespace rwkv7::persistent_decode
