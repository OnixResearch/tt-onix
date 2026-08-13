#include "fused_wkv_reference.hpp"

#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace rwkv7::fused_wkv_reference {
namespace {

constexpr std::uint32_t kBfloat16Shift = 16U;
constexpr std::uint32_t kBfloat16RoundBias = 0x7fffU;
constexpr std::string_view kDiagnosticRuntimeName =
    "rwkv7-p150x2-window16-readout-diagnostic-runtime";

[[noreturn]] void fail(std::string_view message) {
  throw std::runtime_error(std::string(message));
}

void require(bool condition, std::string_view message) {
  if (!condition) {
    fail(message);
  }
}

void validate_vector(std::string_view name, const std::vector<float> &values,
                     std::size_t expected_size) {
  require(values.size() == expected_size,
          std::string(name) + " has the wrong element count");
  for (const float value : values) {
    require(std::isfinite(value),
            std::string(name) + " contains a non-finite value");
  }
}

std::uint16_t float_to_bfloat16_bits(float value) {
  require(std::isfinite(value), "cannot convert a non-finite BF16 value");
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t retained_lsb = (bits >> kBfloat16Shift) & 1U;
  const std::uint32_t rounded = bits + kBfloat16RoundBias + retained_lsb;
  const auto result = static_cast<std::uint16_t>(rounded >> kBfloat16Shift);
  const std::uint32_t restored_bits =
      static_cast<std::uint32_t>(result) << kBfloat16Shift;
  require(std::isfinite(std::bit_cast<float>(restored_bits)),
          "BF16 conversion overflowed");
  return result;
}

float bfloat16_bits_to_float(std::uint16_t bits) {
  const std::uint32_t restored =
      static_cast<std::uint32_t>(bits) << kBfloat16Shift;
  return std::bit_cast<float>(restored);
}

} // namespace

DiagnosticBuildDecision validate_diagnostic_build(
    std::string_view runtime_name, bool bfloat16_readout_boundary) {
  if (runtime_name != kDiagnosticRuntimeName) {
    return DiagnosticBuildDecision{false,
                                   "unexpected_diagnostic_runtime_name"};
  }
  if (!bfloat16_readout_boundary) {
    return DiagnosticBuildDecision{false,
                                   "bfloat16_readout_boundary_disabled"};
  }
  return DiagnosticBuildDecision{true, "diagnostic_build_bound"};
}

Result evaluate(const Inputs &inputs) {
  require(inputs.head_count > 0U && inputs.head_size > 0U,
          "fused WKV reference geometry must be positive");
  require(std::isfinite(inputs.group_norm_epsilon) &&
              inputs.group_norm_epsilon > 0.0F,
          "group normalization epsilon must be finite and positive");
  require(inputs.head_count <=
              std::numeric_limits<std::size_t>::max() / inputs.head_size,
          "fused WKV reference vector geometry overflows");
  const std::size_t vector_size = inputs.head_count * inputs.head_size;
  require(vector_size <=
              std::numeric_limits<std::size_t>::max() / inputs.head_size,
          "fused WKV reference state geometry overflows");
  const std::size_t state_size = vector_size * inputs.head_size;
  validate_vector("a", inputs.a, vector_size);
  validate_vector("w", inputs.w, vector_size);
  validate_vector("k", inputs.k, vector_size);
  validate_vector("v", inputs.v, vector_size);
  validate_vector("r", inputs.r, vector_size);
  validate_vector("b", inputs.b, vector_size);
  validate_vector("r_k", inputs.r_k, vector_size);
  validate_vector("group_norm_weight", inputs.group_norm_weight,
                  vector_size);
  validate_vector("group_norm_bias", inputs.group_norm_bias, vector_size);
  validate_vector("gate", inputs.gate, vector_size);
  validate_vector("state", inputs.state, state_size);

  const std::size_t width = inputs.head_size;
  Result result{std::vector<float>(state_size, 0.0F), {}};
  std::vector<float> readout(vector_size, 0.0F);
  result.corrected_rows_bfloat16.reserve(vector_size);

  for (std::size_t head = 0U; head < inputs.head_count; ++head) {
    const std::size_t vector_base = head * width;
    const std::size_t state_base = head * width * width;
    for (std::size_t row = 0U; row < width; ++row) {
      float state_dot_a = 0.0F;
      for (std::size_t column = 0U; column < width; ++column) {
        state_dot_a +=
            inputs.a[vector_base + column] *
            inputs.state[state_base + row * width + column];
      }
      for (std::size_t column = 0U; column < width; ++column) {
        result.next_state[state_base + row * width + column] =
            inputs.state[state_base + row * width + column] *
                inputs.w[vector_base + column] +
            state_dot_a * inputs.b[vector_base + column] +
            inputs.v[vector_base + row] * inputs.k[vector_base + column];
      }
      float row_readout = 0.0F;
      for (std::size_t column = 0U; column < width; ++column) {
        row_readout +=
            result.next_state[state_base + row * width + column] *
            inputs.r[vector_base + column];
      }
      readout[vector_base + row] = bfloat16_bits_to_float(
          float_to_bfloat16_bits(row_readout));
    }
  }

  for (std::size_t head = 0U; head < inputs.head_count; ++head) {
    const std::size_t base = head * width;
    float mean = 0.0F;
    float correction_scalar = 0.0F;
    for (std::size_t channel = 0U; channel < width; ++channel) {
      mean += readout[base + channel];
      correction_scalar +=
          inputs.r[base + channel] * inputs.k[base + channel] *
          inputs.r_k[base + channel];
    }
    mean /= static_cast<float>(width);
    float variance = 0.0F;
    for (std::size_t channel = 0U; channel < width; ++channel) {
      const float centered = readout[base + channel] - mean;
      variance += centered * centered;
    }
    variance /= static_cast<float>(width);
    const float inverse_standard_deviation =
        1.0F / std::sqrt(variance + inputs.group_norm_epsilon);
    for (std::size_t channel = 0U; channel < width; ++channel) {
      const float normalized =
          (readout[base + channel] - mean) * inverse_standard_deviation;
      const float affine =
          normalized * inputs.group_norm_weight[base + channel] +
          inputs.group_norm_bias[base + channel];
      const float corrected =
          (affine + correction_scalar * inputs.v[base + channel]) *
          inputs.gate[base + channel];
      result.corrected_rows_bfloat16.push_back(
          float_to_bfloat16_bits(corrected));
    }
  }
  return result;
}

} // namespace rwkv7::fused_wkv_reference
