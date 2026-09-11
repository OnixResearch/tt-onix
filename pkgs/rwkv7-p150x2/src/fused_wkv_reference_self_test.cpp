#include "fused_wkv_reference_self_test.hpp"

#include "fused_wkv_reference.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace rwkv7::fused_wkv_reference {
namespace {

constexpr std::size_t kHeadCount = 1U;
constexpr std::size_t kHeadSize = 2U;
constexpr float kEpsilon = 0.001F;
constexpr std::string_view kDiagnosticRuntimeName =
    "rwkv7-p150x2-window16-readout-diagnostic-runtime";

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

Inputs valid_inputs() {
  return Inputs{
      kHeadCount,
      kHeadSize,
      {0.5F, -0.25F},
      {0.75F, 0.5F},
      {0.125F, -0.375F},
      {0.25F, -0.5F},
      {0.625F, -0.125F},
      {-0.25F, 0.375F},
      {0.5F, -0.75F},
      {1.0F, 0.5F},
      {0.125F, -0.25F},
      {0.75F, 0.25F},
      {0.125F, -0.25F, 0.375F, 0.5F},
      kEpsilon,
  };
}

void expect_failure(Inputs inputs, std::string_view expected_message) {
  try {
    static_cast<void>(evaluate(inputs));
  } catch (const std::exception &error) {
    require(std::string_view(error.what()).find(expected_message) !=
                std::string_view::npos,
            "fused WKV reference returned the wrong negative diagnostic");
    return;
  }
  throw std::runtime_error(
      "fused WKV reference accepted invalid self-test input");
}

} // namespace

void run_fused_wkv_reference_self_tests() {
  const Result first = evaluate(valid_inputs());
  const Result second = evaluate(valid_inputs());
  require(first.next_state == second.next_state &&
              first.corrected_rows_bfloat16 ==
                  second.corrected_rows_bfloat16,
          "fused WKV reference is not deterministic");
  require(first.next_state.size() == kHeadCount * kHeadSize * kHeadSize,
          "fused WKV reference changed state geometry");
  require(first.corrected_rows_bfloat16.size() == kHeadCount * kHeadSize,
          "fused WKV reference changed output geometry");
  for (const float value : first.next_state) {
    require(std::isfinite(value),
            "fused WKV reference produced non-finite state");
  }

  const DiagnosticBuildDecision valid_build = validate_diagnostic_build(kDiagnosticRuntimeName, true, true);
  require(valid_build.valid,
          "diagnostic runtime binding rejected the expected package");
  require(
      !validate_diagnostic_build("rwkv7-p150x2-runtime", true, true).valid,
      "diagnostic runtime binding accepted production");
  require(
      !validate_diagnostic_build(kDiagnosticRuntimeName, false, true).valid,
      "diagnostic runtime binding accepted an FP32 readout");
  require(
      !validate_diagnostic_build(kDiagnosticRuntimeName, true, false).valid,
      "diagnostic runtime binding accepted BF16 recurrence vectors");

  Inputs malformed = valid_inputs();
  malformed.state.pop_back();
  expect_failure(malformed, "state has the wrong element count");
  Inputs non_finite = valid_inputs();
  non_finite.a[0] = std::bit_cast<float>(0x7fc00000U);
  expect_failure(non_finite, "a contains a non-finite value");
}

} // namespace rwkv7::fused_wkv_reference
