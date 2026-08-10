#pragma once

#include <tt-metalium/mesh_device.hpp>
#include <ttnn/tensor/tensor.hpp>

#include <cstdint>
#include <filesystem>
#include <string>

namespace rwkv7::fused_wkv {

inline constexpr std::uint32_t kHeadCount = 12U;
inline constexpr std::uint32_t kHeadSize = 64U;
inline constexpr std::uint32_t kMaximumBatchSize = 16U;

struct Inputs {
  ttnn::Tensor a;
  ttnn::Tensor w;
  ttnn::Tensor k;
  ttnn::Tensor v;
  ttnn::Tensor r;
  ttnn::Tensor b;
  ttnn::Tensor r_k;
  ttnn::Tensor group_norm_weight;
  ttnn::Tensor group_norm_bias;
  ttnn::Tensor gate;
  ttnn::Tensor state;
};

struct Outputs {
  ttnn::Tensor next_state;
  ttnn::Tensor output;
};

struct DispatchDecision {
  bool capability_supported;
  bool physically_admitted;
  bool use_fused_path;
  std::string reason;
};

[[nodiscard]] DispatchDecision
select_dispatch(const Inputs &inputs, bool physically_admitted);

[[nodiscard]] Outputs
run(tt::tt_metal::distributed::MeshDevice &mesh, const Inputs &inputs,
    const std::filesystem::path &kernel_root);

} // namespace rwkv7::fused_wkv
