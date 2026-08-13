#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace rwkv7::fused_wkv_reference {

struct Inputs {
  std::size_t head_count;
  std::size_t head_size;
  std::vector<float> a;
  std::vector<float> w;
  std::vector<float> k;
  std::vector<float> v;
  std::vector<float> r;
  std::vector<float> b;
  std::vector<float> r_k;
  std::vector<float> group_norm_weight;
  std::vector<float> group_norm_bias;
  std::vector<float> gate;
  std::vector<float> state;
  float group_norm_epsilon;
};

struct Result {
  std::vector<float> next_state;
  std::vector<std::uint16_t> corrected_rows_bfloat16;
};

struct DiagnosticBuildDecision {
  bool valid;
  std::string reason;
};

Result evaluate(const Inputs &inputs);

DiagnosticBuildDecision validate_diagnostic_build(
    std::string_view runtime_name, bool bfloat16_readout_boundary, bool fp32_recurrence_vectors);

} // namespace rwkv7::fused_wkv_reference
