#include "config.hpp"
#include "fused_wkv_program.hpp"

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <ttnn/distributed/distributed_tensor.hpp>
#include <ttnn/operations/creation/creation.hpp>
#include <ttnn/operations/generic/generic_op.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rwkv7::fused_wkv {
namespace {

using Tensor = ttnn::Tensor;
using MeshDevice = tt::tt_metal::distributed::MeshDevice;
using Buffer = tt::tt_metal::Buffer;
using CBDescriptor = tt::tt_metal::CBDescriptor;
using CBFormatDescriptor = tt::tt_metal::CBFormatDescriptor;
using CoreCoord = tt::tt_metal::CoreCoord;
using CoreRange = tt::tt_metal::CoreRange;
using CoreRangeSet = tt::tt_metal::CoreRangeSet;
using KernelDescriptor = tt::tt_metal::KernelDescriptor;
using ProgramDescriptor = tt::tt_metal::ProgramDescriptor;
using SemaphoreDescriptor = tt::tt_metal::SemaphoreDescriptor;

constexpr std::uint32_t kTileWidth = 32U;
constexpr std::size_t kCircularBufferCount = 64U;
constexpr std::uint32_t kHeadTileCount = kHeadSize / kTileWidth;
constexpr std::uint32_t kStateTileCount =
    kHeadTileCount * kHeadTileCount;
constexpr std::uint32_t kScalarTileCount = 1U;
constexpr std::uint32_t kCorrectionScalarTileCapacity =
    kHeadTileCount + kScalarTileCount;
constexpr std::uint32_t kRecurrenceVectorCount = 6U;
constexpr std::uint32_t kSharedVectorCount = 3U;
constexpr std::uint32_t kPerWorkCorrectionVectorCount = 1U;
constexpr std::uint32_t kVectorCount =
    kRecurrenceVectorCount + kSharedVectorCount +
    kPerWorkCorrectionVectorCount;
constexpr std::size_t kFirstSharedVectorIndex = kRecurrenceVectorCount;
constexpr std::size_t kLastSharedVectorIndex =
    kFirstSharedVectorIndex + kSharedVectorCount - 1U;
constexpr std::uint32_t kCompletionSemaphoreId = 0U;
constexpr std::uint32_t kInitialSemaphoreValue = 0U;
constexpr std::uint32_t kFirstIndex = 0U;
constexpr std::uint32_t kBatchDimension = 0U;
constexpr std::uint32_t kHeadDimension = 1U;
constexpr std::uint32_t kRowDimension = 2U;
constexpr std::uint32_t kColumnDimension = 3U;
constexpr std::uint32_t kTensorRank = 4U;
constexpr std::uint32_t kMeshRow = 0U;

constexpr std::uint8_t kRawACb = 0U;
constexpr std::uint8_t kRawWCb = 1U;
constexpr std::uint8_t kRawKCb = 2U;
constexpr std::uint8_t kRawVCb = 3U;
constexpr std::uint8_t kRawRCb = 4U;
constexpr std::uint8_t kRawBCb = 5U;
constexpr std::uint8_t kStateCb = 6U;
constexpr std::uint8_t kRawRkCb = 7U;
constexpr std::uint8_t kStateDotACb = 8U;
constexpr std::uint8_t kNextStateCb = 9U;
constexpr std::uint8_t kReadoutCb = 10U;
constexpr std::uint8_t kRawGroupNormWeightCb = 11U;
constexpr std::uint8_t kACb = 12U;
constexpr std::uint8_t kWCb = 13U;
constexpr std::uint8_t kKCb = 14U;
constexpr std::uint8_t kVCb = 15U;
constexpr std::uint8_t kOutputCb = 16U;
constexpr std::uint8_t kNextStateOutputCb = 17U;
constexpr std::uint8_t kRCb = 18U;
constexpr std::uint8_t kBCb = 19U;
constexpr std::uint8_t kRawGroupNormBiasCb = 20U;
constexpr std::uint8_t kRawGateCb = 21U;
constexpr std::uint8_t kReadoutRowCb = 22U;
constexpr std::uint8_t kMeanCb = 23U;
constexpr std::uint8_t kCenteredCb = 24U;
constexpr std::uint8_t kSquaredCb = 25U;
constexpr std::uint8_t kVarianceCb = 26U;
constexpr std::uint8_t kInverseStdCb = 27U;
constexpr std::uint8_t kAffineCb = 28U;
constexpr std::uint8_t kCorrectionScalarCb = 29U;
constexpr std::uint8_t kCorrectionVectorCb = 30U;
constexpr std::uint8_t kReductionScalerCb = 31U;
constexpr std::uint8_t kFirstAddLhsCb = 0U;
constexpr std::uint8_t kFirstAddRhsCb = 1U;
constexpr std::uint8_t kBf16ReadoutStateCb = 0U;
constexpr std::uint8_t kBf16ReadoutRCb = 1U;
constexpr std::uint8_t kExactStateCb = 22U;
constexpr std::uint8_t kDecayCb = 23U;
constexpr std::uint8_t kRankUpdateCb = 24U;
constexpr std::uint8_t kOuterUpdateCb = 25U;
constexpr std::uint8_t kFirstAddCb = 26U;

constexpr std::array<std::uint8_t, kVectorCount> kRawVectorCbs = {
    kRawACb,
    kRawWCb,
    kRawKCb,
    kRawVCb,
    kRawRCb,
    kRawBCb,
    kRawRkCb,
    kRawGroupNormWeightCb,
    kRawGroupNormBiasCb,
    kRawGateCb,
};
constexpr std::array<std::uint8_t, kRecurrenceVectorCount>
    kOrientedVectorCbs = {kACb, kWCb, kKCb, kVCb, kRCb, kBCb};
constexpr std::array<bool, kRecurrenceVectorCount> kGenericFp32RecurrenceVector = {true, true, true, true, false, true};

constexpr std::string_view kReaderKernelName = "fused_wkv_reader.cpp";
constexpr std::string_view kWriterKernelName = "fused_wkv_writer.cpp";
constexpr std::string_view kComputeKernelName = "fused_wkv_compute.cpp";
constexpr std::string_view kStateDotAReaderKernelName = "fused_wkv_state_dot_a_reader.cpp";
constexpr std::string_view kStateDotAWriterKernelName = "fused_wkv_state_dot_a_writer.cpp";
constexpr std::string_view kStateDotAComputeKernelName = "fused_wkv_state_dot_a_compute.cpp";
constexpr std::string_view kDecayReaderKernelName = "fused_wkv_decay_reader.cpp";
constexpr std::string_view kDecayWriterKernelName = "fused_wkv_decay_writer.cpp";
constexpr std::string_view kDecayComputeKernelName = "fused_wkv_decay_compute.cpp";
constexpr std::string_view kRankUpdateReaderKernelName = "fused_wkv_rank_update_reader.cpp";
constexpr std::string_view kRankUpdateWriterKernelName = "fused_wkv_rank_update_writer.cpp";
constexpr std::string_view kRankUpdateComputeKernelName = "fused_wkv_rank_update_compute.cpp";
constexpr std::string_view kOuterUpdateReaderKernelName = "fused_wkv_outer_update_reader.cpp";
constexpr std::string_view kOuterUpdateWriterKernelName = "fused_wkv_outer_update_writer.cpp";
constexpr std::string_view kOuterUpdateComputeKernelName = "fused_wkv_outer_update_compute.cpp";
constexpr std::string_view kFirstAddReaderKernelName = "fused_wkv_first_add_reader.cpp";
constexpr std::string_view kFirstAddWriterKernelName = "fused_wkv_first_add_writer.cpp";
constexpr std::string_view kFirstAddComputeKernelName = "fused_wkv_first_add_compute.cpp";
constexpr std::string_view kBf16ReadoutReaderKernelName = "fused_wkv_bf16_readout_reader.cpp";
constexpr std::string_view kBf16ReadoutWriterKernelName = "fused_wkv_bf16_readout_writer.cpp";
constexpr std::string_view kBf16ReadoutComputeKernelName = "fused_wkv_bf16_readout_compute.cpp";
constexpr std::string_view kFp32RecurrenceVectorsDefine = "RWKV7_FUSED_WKV_FP32_RECURRENCE_VECTORS";
constexpr std::string_view kSplitReadoutDefine = "RWKV7_FUSED_WKV_SPLIT_READOUT";
constexpr bool kSplitReadout =
    RWKV7_FUSED_WKV_BF16_READOUT_BOUNDARY != 0 && RWKV7_FUSED_WKV_FP32_RECURRENCE_VECTORS != 0;

[[noreturn]] void fail(std::string_view message) {
  throw std::runtime_error(std::string(message));
}

void require(bool condition, std::string_view message) {
  if (!condition) {
    fail(message);
  }
}

DispatchDecision reject(bool physically_admitted, std::string reason) {
  return DispatchDecision{false, physically_admitted, false,
                          std::move(reason)};
}

bool has_vector_shape(const Tensor &tensor, std::uint32_t batch_size) {
  const auto &shape = tensor.logical_shape();
  return shape.rank() == kTensorRank &&
         shape[kBatchDimension] == batch_size &&
         shape[kHeadDimension] == kHeadCount &&
         shape[kRowDimension] == 1U &&
         shape[kColumnDimension] == kHeadSize;
}

bool has_state_shape(const Tensor &tensor, std::uint32_t batch_size) {
  const auto &shape = tensor.logical_shape();
  return shape.rank() == kTensorRank &&
         shape[kBatchDimension] == batch_size &&
         shape[kHeadDimension] == kHeadCount &&
         shape[kRowDimension] == kHeadSize &&
         shape[kColumnDimension] == kHeadSize;
}

std::array<const Tensor *, kVectorCount> vectors(const Inputs &inputs) {
  return {
      &inputs.a,
      &inputs.w,
      &inputs.k,
      &inputs.v,
      &inputs.r,
      &inputs.b,
      &inputs.r_k,
      &inputs.group_norm_weight,
      &inputs.group_norm_bias,
      &inputs.gate,
  };
}

struct OwnershipPlan {
  std::vector<CoreCoord> cores;
  std::vector<std::vector<std::uint32_t>> work_by_core;
};

OwnershipPlan make_ownership_plan(const CoreCoord &grid,
                                  std::uint32_t batch_size) {
  require(grid.x > 0U && grid.y > 0U,
          "fused WKV requires a non-empty compute grid");
  require(batch_size > 0U && batch_size <= kMaximumBatchSize,
          "fused WKV batch size is outside the supported range");
  require(batch_size <=
              std::numeric_limits<std::uint32_t>::max() / kHeadCount,
          "fused WKV work count overflows");
  const std::size_t work_count =
      static_cast<std::size_t>(batch_size) * kHeadCount;
  require(grid.x <= std::numeric_limits<std::size_t>::max() / grid.y,
          "fused WKV grid size overflows");
  const std::size_t available_cores = grid.x * grid.y;
  const std::size_t active_core_count =
      std::min(work_count, available_cores);
  require(active_core_count > 0U,
          "fused WKV did not select an active compute core");

  OwnershipPlan plan;
  plan.cores.reserve(active_core_count);
  plan.work_by_core.resize(active_core_count);
  for (std::size_t core_index = 0U; core_index < active_core_count;
       ++core_index) {
    plan.cores.push_back(
        CoreCoord{core_index % grid.x, core_index / grid.x});
  }
  for (std::size_t work_item = 0U; work_item < work_count; ++work_item) {
    const std::size_t owner = work_item % active_core_count;
    plan.work_by_core[owner].push_back(
        static_cast<std::uint32_t>(work_item));
  }

  std::vector<bool> seen(work_count, false);
  for (const auto &core_work : plan.work_by_core) {
    require(!core_work.empty(),
            "fused WKV ownership assigned an idle active core");
    for (const std::uint32_t work_item : core_work) {
      require(work_item < work_count,
              "fused WKV ownership contains an invalid work item");
      require(!seen[work_item],
              "fused WKV ownership contains more than one writer");
      seen[work_item] = true;
    }
  }
  require(std::all_of(seen.begin(), seen.end(), [](bool owned) {
            return owned;
          }),
          "fused WKV ownership does not cover every request head");
  return plan;
}

CoreRangeSet make_core_ranges(const std::vector<CoreCoord> &cores) {
  std::set<CoreRange> ranges;
  for (const CoreCoord &core : cores) {
    ranges.insert(CoreRange{core, core});
  }
  return CoreRangeSet(std::move(ranges));
}

CBDescriptor make_cb(std::uint8_t index, tt::DataFormat format,
                     std::uint32_t tile_count,
                     const CoreRangeSet &core_ranges) {
  const std::uint32_t page_size = tt::tile_size(format);
  require(tile_count <= std::numeric_limits<std::uint32_t>::max() / page_size,
          "fused WKV circular-buffer size overflows");
  CBFormatDescriptor format_descriptor{};
  format_descriptor.buffer_index = index;
  format_descriptor.data_format = format;
  format_descriptor.page_size = page_size;
  CBDescriptor descriptor{};
  descriptor.total_size = tile_count * page_size;
  descriptor.core_ranges = core_ranges;
  descriptor.format_descriptors.push_back(std::move(format_descriptor));
  return descriptor;
}

ProgramDescriptor::CBDescriptors
make_cbs(const CoreRangeSet &core_ranges) {
  const tt::DataFormat bf16_format =
      tt::tt_metal::datatype_to_dataformat_converter(
          tt::tt_metal::DataType::BFLOAT16);
  const tt::DataFormat fp32_format =
      tt::tt_metal::datatype_to_dataformat_converter(
          tt::tt_metal::DataType::FLOAT32);

  ProgramDescriptor::CBDescriptors cbs;
  for (const std::uint8_t cb : kRawVectorCbs) {
    cbs.push_back(make_cb(cb, bf16_format, kHeadTileCount, core_ranges));
  }
  cbs.push_back(make_cb(kStateCb, fp32_format, kStateTileCount, core_ranges));
  cbs.push_back(
      make_cb(kStateDotACb, fp32_format, kHeadTileCount, core_ranges));
  cbs.push_back(
      make_cb(kNextStateCb, fp32_format, kStateTileCount, core_ranges));
  // Preserve the known-completing FP32 matmul output. The diagnostic BF16
  // boundary is applied when this column is transposed into a row.
  cbs.push_back(make_cb(kReadoutCb, fp32_format, kHeadTileCount, core_ranges));
  for (std::size_t vector = 0U; vector < kOrientedVectorCbs.size(); ++vector) {
#if RWKV7_FUSED_WKV_FP32_RECURRENCE_VECTORS
      const tt::DataFormat format = kGenericFp32RecurrenceVector[vector] ? fp32_format : bf16_format;
#else
      const tt::DataFormat format = bf16_format;
#endif
      cbs.push_back(make_cb(kOrientedVectorCbs[vector], format, kHeadTileCount, core_ranges));
  }
  cbs.push_back(make_cb(kOutputCb, bf16_format, kHeadTileCount, core_ranges));
  cbs.push_back(
      make_cb(kNextStateOutputCb, fp32_format, kStateTileCount, core_ranges));
#if (RWKV7_FUSED_WKV_FP32_RECURRENCE_VECTORS != 0) && (RWKV7_FUSED_WKV_BF16_READOUT_BOUNDARY != 0)
  for (const std::uint8_t cb : {kExactStateCb, kDecayCb, kRankUpdateCb, kOuterUpdateCb, kFirstAddCb}) {
      cbs.push_back(make_cb(cb, fp32_format, kStateTileCount, core_ranges));
  }
#else
#if RWKV7_FUSED_WKV_BF16_READOUT_BOUNDARY
  // Match generic_wkv_step by packing the FP32 recurrent readout to BF16
  // before group normalization without making BF16 the matmul output CB.
  cbs.push_back(make_cb(kReadoutRowCb, bf16_format, kHeadTileCount, core_ranges));
#else
  cbs.push_back(make_cb(kReadoutRowCb, fp32_format, kHeadTileCount, core_ranges));
#endif
  for (const std::uint8_t cb : {kCenteredCb, kSquaredCb, kAffineCb, kCorrectionVectorCb}) {
      cbs.push_back(make_cb(cb, fp32_format, kHeadTileCount, core_ranges));
  }
  cbs.push_back(make_cb(kCorrectionScalarCb, fp32_format,
                        kCorrectionScalarTileCapacity, core_ranges));
  for (const std::uint8_t cb :
       {kMeanCb, kVarianceCb, kInverseStdCb, kReductionScalerCb}) {
    cbs.push_back(make_cb(cb, fp32_format, kScalarTileCount, core_ranges));
  }
#endif
  return cbs;
}

KernelDescriptor::Defines recurrence_kernel_defines() {
    return {
        {std::string(kFp32RecurrenceVectorsDefine), RWKV7_FUSED_WKV_FP32_RECURRENCE_VECTORS ? "1" : "0"},
        {std::string(kSplitReadoutDefine), kSplitReadout ? "1" : "0"},
    };
}

KernelDescriptor::CompileTimeArgs reader_compile_args(const Tensor &vector,
                                                       const Tensor &state) {
  KernelDescriptor::CompileTimeArgs args;
  tt::tt_metal::TensorAccessorArgs(*vector.buffer()).append_to(args);
  tt::tt_metal::TensorAccessorArgs(*state.buffer()).append_to(args);
  return args;
}

KernelDescriptor::CompileTimeArgs writer_compile_args(const Tensor &output,
                                                       const Tensor &state) {
  KernelDescriptor::CompileTimeArgs args;
  tt::tt_metal::TensorAccessorArgs(*output.buffer()).append_to(args);
  tt::tt_metal::TensorAccessorArgs(*state.buffer()).append_to(args);
  return args;
}

void add_reader_runtime_args(KernelDescriptor &reader,
                             const OwnershipPlan &plan,
                             const std::array<Buffer *, kVectorCount> &vector_buffers,
                             Buffer *state_buffer) {
  for (std::size_t core_index = 0U; core_index < plan.cores.size();
       ++core_index) {
    const auto &work = plan.work_by_core[core_index];
    require(work.size() <= std::numeric_limits<std::uint32_t>::max(),
            "fused WKV reader work count overflows");
    KernelDescriptor::RTArgList args;
    for (Buffer *buffer : vector_buffers) {
      args.push_back(buffer);
    }
    args.push_back(state_buffer);
    args.push_back(RWKV7_FUSED_WKV_FP32_RECURRENCE_VECTORS ? 1U : 0U);
    args.push_back(kSplitReadout ? 1U : 0U);
    args.push_back(static_cast<std::uint32_t>(work.size()));
    args.append(work);
    reader.emplace_runtime_args(plan.cores[core_index], args);
  }
}

void add_writer_runtime_args(KernelDescriptor &writer,
                             const OwnershipPlan &plan, Buffer *output_buffer,
                             Buffer *next_state_buffer) {
  for (std::size_t core_index = 0U; core_index < plan.cores.size();
       ++core_index) {
    const auto &work = plan.work_by_core[core_index];
    require(work.size() <= std::numeric_limits<std::uint32_t>::max(),
            "fused WKV writer work count overflows");
    KernelDescriptor::RTArgList args;
    args.push_back(output_buffer);
    args.push_back(next_state_buffer);
    args.push_back(kCompletionSemaphoreId);
    args.push_back(kSplitReadout ? 1U : 0U);
    args.push_back(static_cast<std::uint32_t>(work.size()));
    args.append(work);
    writer.emplace_runtime_args(plan.cores[core_index], args);
  }
}

void add_compute_runtime_args(KernelDescriptor &compute,
                              const OwnershipPlan &plan) {
  for (std::size_t core_index = 0U; core_index < plan.cores.size();
       ++core_index) {
    const auto &work = plan.work_by_core[core_index];
    require(work.size() <= std::numeric_limits<std::uint32_t>::max(),
            "fused WKV compute work count overflows");
    compute.emplace_runtime_args(
        plan.cores[core_index],
        {static_cast<std::uint32_t>(work.size())});
  }
}

ProgramDescriptor::CBDescriptors make_state_dot_a_cbs(const CoreRangeSet& core_ranges) {
    const tt::DataFormat bf16_format = tt::tt_metal::datatype_to_dataformat_converter(tt::tt_metal::DataType::BFLOAT16);
    const tt::DataFormat fp32_format = tt::tt_metal::datatype_to_dataformat_converter(tt::tt_metal::DataType::FLOAT32);
    ProgramDescriptor::CBDescriptors cbs;
    cbs.push_back(make_cb(kRawACb, bf16_format, kHeadTileCount, core_ranges));
    cbs.push_back(make_cb(kRawWCb, fp32_format, kStateTileCount, core_ranges));
    cbs.push_back(make_cb(kRawKCb, fp32_format, kHeadTileCount, core_ranges));
    cbs.push_back(make_cb(kOutputCb, fp32_format, kHeadTileCount, core_ranges));
    return cbs;
}

ProgramDescriptor::CBDescriptors make_decay_cbs(const CoreRangeSet& core_ranges) {
    const tt::DataFormat bf16_format = tt::tt_metal::datatype_to_dataformat_converter(tt::tt_metal::DataType::BFLOAT16);
    const tt::DataFormat fp32_format = tt::tt_metal::datatype_to_dataformat_converter(tt::tt_metal::DataType::FLOAT32);
    ProgramDescriptor::CBDescriptors cbs;
    cbs.push_back(make_cb(kRawACb, bf16_format, kHeadTileCount, core_ranges));
    cbs.push_back(make_cb(kRawWCb, fp32_format, kStateTileCount, core_ranges));
    cbs.push_back(make_cb(kRawKCb, fp32_format, kHeadTileCount, core_ranges));
    cbs.push_back(make_cb(kOutputCb, fp32_format, kStateTileCount, core_ranges));
    return cbs;
}

ProgramDescriptor::CBDescriptors make_rank_update_cbs(const CoreRangeSet& core_ranges) {
    const tt::DataFormat bf16_format = tt::tt_metal::datatype_to_dataformat_converter(tt::tt_metal::DataType::BFLOAT16);
    const tt::DataFormat fp32_format = tt::tt_metal::datatype_to_dataformat_converter(tt::tt_metal::DataType::FLOAT32);
    ProgramDescriptor::CBDescriptors cbs;
    cbs.push_back(make_cb(kRawACb, bf16_format, kHeadTileCount, core_ranges));
    cbs.push_back(make_cb(kRawWCb, bf16_format, kHeadTileCount, core_ranges));
    cbs.push_back(make_cb(kRawKCb, fp32_format, kStateTileCount, core_ranges));
    cbs.push_back(make_cb(kRawVCb, fp32_format, kHeadTileCount, core_ranges));
    cbs.push_back(make_cb(kRawRCb, fp32_format, kHeadTileCount, core_ranges));
    cbs.push_back(make_cb(kRawBCb, fp32_format, kHeadTileCount, core_ranges));
    cbs.push_back(make_cb(kOutputCb, fp32_format, kStateTileCount, core_ranges));
    return cbs;
}

ProgramDescriptor::CBDescriptors make_outer_update_cbs(const CoreRangeSet& core_ranges) {
    const tt::DataFormat bf16_format = tt::tt_metal::datatype_to_dataformat_converter(tt::tt_metal::DataType::BFLOAT16);
    const tt::DataFormat fp32_format = tt::tt_metal::datatype_to_dataformat_converter(tt::tt_metal::DataType::FLOAT32);
    ProgramDescriptor::CBDescriptors cbs;
    cbs.push_back(make_cb(kRawACb, bf16_format, kHeadTileCount, core_ranges));
    cbs.push_back(make_cb(kRawWCb, bf16_format, kHeadTileCount, core_ranges));
    cbs.push_back(make_cb(kRawKCb, fp32_format, kHeadTileCount, core_ranges));
    cbs.push_back(make_cb(kRawVCb, fp32_format, kHeadTileCount, core_ranges));
    cbs.push_back(make_cb(kOutputCb, fp32_format, kStateTileCount, core_ranges));
    return cbs;
}

ProgramDescriptor::CBDescriptors make_first_add_cbs(const CoreRangeSet& core_ranges) {
    const tt::DataFormat fp32_format = tt::tt_metal::datatype_to_dataformat_converter(tt::tt_metal::DataType::FLOAT32);
    ProgramDescriptor::CBDescriptors cbs;
    cbs.push_back(make_cb(kFirstAddLhsCb, fp32_format, kStateTileCount, core_ranges));
    cbs.push_back(make_cb(kFirstAddRhsCb, fp32_format, kStateTileCount, core_ranges));
    cbs.push_back(make_cb(kOutputCb, fp32_format, kStateTileCount, core_ranges));
    return cbs;
}

ProgramDescriptor::CBDescriptors make_bf16_readout_cbs(const CoreRangeSet& core_ranges) {
    const tt::DataFormat bf16_format = tt::tt_metal::datatype_to_dataformat_converter(tt::tt_metal::DataType::BFLOAT16);
    const tt::DataFormat fp32_format = tt::tt_metal::datatype_to_dataformat_converter(tt::tt_metal::DataType::FLOAT32);
    ProgramDescriptor::CBDescriptors cbs;
    cbs.push_back(make_cb(kBf16ReadoutStateCb, fp32_format, kStateTileCount, core_ranges));
    cbs.push_back(make_cb(kBf16ReadoutRCb, bf16_format, kHeadTileCount, core_ranges));
    cbs.push_back(make_cb(kOutputCb, bf16_format, kHeadTileCount, core_ranges));
    return cbs;
}

ProgramDescriptor make_program_descriptor(
    const std::filesystem::path &kernel_root, const OwnershipPlan &plan,
    const std::array<const Tensor *, kVectorCount> &vector_tensors,
    const Tensor &state, Tensor &output, Tensor &next_state) {
  const CoreRangeSet core_ranges = make_core_ranges(plan.cores);
  KernelDescriptor reader{};
  reader.kernel_source = (kernel_root / kReaderKernelName).string();
  reader.core_ranges = core_ranges;
  reader.compile_time_args =
      reader_compile_args(*vector_tensors[kFirstIndex], state);
  reader.config = tt::tt_metal::ReaderConfigDescriptor{};

  KernelDescriptor writer{};
  writer.kernel_source = (kernel_root / kWriterKernelName).string();
  writer.core_ranges = core_ranges;
  writer.compile_time_args = writer_compile_args(output, next_state);
  writer.config = tt::tt_metal::WriterConfigDescriptor{};

  tt::tt_metal::ComputeConfigDescriptor compute_config{};
  compute_config.math_fidelity = tt::tt_metal::MathFidelity::HiFi4;
  compute_config.fp32_dest_acc_en = true;
  compute_config.dst_full_sync_en = false;
  compute_config.math_approx_mode = false;
  if (kSplitReadout) {
      compute_config.unpack_to_dest_mode.assign(kCircularBufferCount, tt::tt_metal::UnpackToDestMode::Default);
      const std::array<std::uint8_t, 6U> exact_fp32_cbs = {
          kExactStateCb, kWCb, kDecayCb, kRankUpdateCb, kOuterUpdateCb, kFirstAddCb};
      for (const std::uint8_t cb : exact_fp32_cbs) {
          compute_config.unpack_to_dest_mode[cb] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
      }
  }
  KernelDescriptor compute{};
  compute.kernel_source = (kernel_root / kComputeKernelName).string();
  compute.core_ranges = core_ranges;
  compute.defines = recurrence_kernel_defines();
  compute.config = std::move(compute_config);

  std::array<Buffer *, kVectorCount> vector_buffers{};
  for (std::size_t vector = 0U; vector < vector_tensors.size(); ++vector) {
    vector_buffers[vector] = vector_tensors[vector]->buffer();
  }
  add_reader_runtime_args(reader, plan, vector_buffers, state.buffer());
  add_writer_runtime_args(writer, plan, output.buffer(), next_state.buffer());
  add_compute_runtime_args(compute, plan);

  SemaphoreDescriptor semaphore{};
  semaphore.id = kCompletionSemaphoreId;
  semaphore.core_type = tt::CoreType::WORKER;
  semaphore.core_ranges = core_ranges;
  semaphore.initial_value = kInitialSemaphoreValue;

  ProgramDescriptor descriptor{};
  descriptor.kernels.push_back(std::move(reader));
  descriptor.kernels.push_back(std::move(writer));
  descriptor.kernels.push_back(std::move(compute));
  descriptor.semaphores.push_back(std::move(semaphore));
  descriptor.cbs = make_cbs(core_ranges);
  return descriptor;
}

ProgramDescriptor make_state_dot_a_program_descriptor(
    const std::filesystem::path& kernel_root,
    const OwnershipPlan& plan,
    const Tensor& a,
    const Tensor& state,
    Tensor& output) {
    const CoreRangeSet core_ranges = make_core_ranges(plan.cores);

    KernelDescriptor reader{};
    reader.kernel_source = (kernel_root / kStateDotAReaderKernelName).string();
    reader.core_ranges = core_ranges;
    reader.compile_time_args = reader_compile_args(a, state);
    reader.config = tt::tt_metal::ReaderConfigDescriptor{};

    KernelDescriptor writer{};
    writer.kernel_source = (kernel_root / kStateDotAWriterKernelName).string();
    writer.core_ranges = core_ranges;
    tt::tt_metal::TensorAccessorArgs(*output.buffer()).append_to(writer.compile_time_args);
    writer.config = tt::tt_metal::WriterConfigDescriptor{};

    tt::tt_metal::ComputeConfigDescriptor compute_config{};
    compute_config.math_fidelity = tt::tt_metal::MathFidelity::HiFi4;
    compute_config.fp32_dest_acc_en = true;
    compute_config.dst_full_sync_en = false;
    compute_config.math_approx_mode = false;
    KernelDescriptor compute{};
    compute.kernel_source = (kernel_root / kStateDotAComputeKernelName).string();
    compute.core_ranges = core_ranges;
    compute.config = std::move(compute_config);

    for (std::size_t core_index = 0U; core_index < plan.cores.size(); ++core_index) {
        const auto& work = plan.work_by_core[core_index];
        require(
            work.size() <= std::numeric_limits<std::uint32_t>::max(), "state_dot_a diagnostic work count overflows");
        const std::uint32_t work_count = static_cast<std::uint32_t>(work.size());

        KernelDescriptor::RTArgList reader_args;
        reader_args.push_back(a.buffer());
        reader_args.push_back(state.buffer());
        reader_args.push_back(work_count);
        reader_args.append(work);
        reader.emplace_runtime_args(plan.cores[core_index], reader_args);

        KernelDescriptor::RTArgList writer_args;
        writer_args.push_back(output.buffer());
        writer_args.push_back(kCompletionSemaphoreId);
        writer_args.push_back(work_count);
        writer_args.append(work);
        writer.emplace_runtime_args(plan.cores[core_index], writer_args);

        compute.emplace_runtime_args(plan.cores[core_index], {work_count});
    }

    SemaphoreDescriptor semaphore{};
    semaphore.id = kCompletionSemaphoreId;
    semaphore.core_type = tt::CoreType::WORKER;
    semaphore.core_ranges = core_ranges;
    semaphore.initial_value = kInitialSemaphoreValue;

    ProgramDescriptor descriptor{};
    descriptor.kernels.push_back(std::move(reader));
    descriptor.kernels.push_back(std::move(writer));
    descriptor.kernels.push_back(std::move(compute));
    descriptor.semaphores.push_back(std::move(semaphore));
    descriptor.cbs = make_state_dot_a_cbs(core_ranges);
    return descriptor;
}

ProgramDescriptor make_decay_program_descriptor(
    const std::filesystem::path& kernel_root,
    const OwnershipPlan& plan,
    const Tensor& w,
    const Tensor& state,
    Tensor& output) {
    const CoreRangeSet core_ranges = make_core_ranges(plan.cores);

    KernelDescriptor reader{};
    reader.kernel_source = (kernel_root / kDecayReaderKernelName).string();
    reader.core_ranges = core_ranges;
    reader.compile_time_args = reader_compile_args(w, state);
    reader.config = tt::tt_metal::ReaderConfigDescriptor{};

    KernelDescriptor writer{};
    writer.kernel_source = (kernel_root / kDecayWriterKernelName).string();
    writer.core_ranges = core_ranges;
    tt::tt_metal::TensorAccessorArgs(*output.buffer()).append_to(writer.compile_time_args);
    writer.config = tt::tt_metal::WriterConfigDescriptor{};

    tt::tt_metal::ComputeConfigDescriptor compute_config{};
    compute_config.math_fidelity = tt::tt_metal::MathFidelity::HiFi4;
    compute_config.fp32_dest_acc_en = true;
    compute_config.dst_full_sync_en = false;
    compute_config.math_approx_mode = false;
    compute_config.unpack_to_dest_mode.assign(kCircularBufferCount, tt::tt_metal::UnpackToDestMode::Default);
    compute_config.unpack_to_dest_mode[kRawWCb] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    compute_config.unpack_to_dest_mode[kRawKCb] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    KernelDescriptor compute{};
    compute.kernel_source = (kernel_root / kDecayComputeKernelName).string();
    compute.core_ranges = core_ranges;
    compute.config = std::move(compute_config);

    for (std::size_t core_index = 0U; core_index < plan.cores.size(); ++core_index) {
        const auto& work = plan.work_by_core[core_index];
        require(work.size() <= std::numeric_limits<std::uint32_t>::max(), "decay diagnostic work count overflows");
        const std::uint32_t work_count = static_cast<std::uint32_t>(work.size());

        KernelDescriptor::RTArgList reader_args;
        reader_args.push_back(w.buffer());
        reader_args.push_back(state.buffer());
        reader_args.push_back(work_count);
        reader_args.append(work);
        reader.emplace_runtime_args(plan.cores[core_index], reader_args);

        KernelDescriptor::RTArgList writer_args;
        writer_args.push_back(output.buffer());
        writer_args.push_back(kCompletionSemaphoreId);
        writer_args.push_back(work_count);
        writer_args.append(work);
        writer.emplace_runtime_args(plan.cores[core_index], writer_args);

        compute.emplace_runtime_args(plan.cores[core_index], {work_count});
    }

    SemaphoreDescriptor semaphore{};
    semaphore.id = kCompletionSemaphoreId;
    semaphore.core_type = tt::CoreType::WORKER;
    semaphore.core_ranges = core_ranges;
    semaphore.initial_value = kInitialSemaphoreValue;

    ProgramDescriptor descriptor{};
    descriptor.kernels.push_back(std::move(reader));
    descriptor.kernels.push_back(std::move(writer));
    descriptor.kernels.push_back(std::move(compute));
    descriptor.semaphores.push_back(std::move(semaphore));
    descriptor.cbs = make_decay_cbs(core_ranges);
    return descriptor;
}

ProgramDescriptor make_rank_update_program_descriptor(
    const std::filesystem::path& kernel_root,
    const OwnershipPlan& plan,
    const Tensor& a,
    const Tensor& b,
    const Tensor& state,
    Tensor& output) {
    const CoreRangeSet core_ranges = make_core_ranges(plan.cores);

    KernelDescriptor reader{};
    reader.kernel_source = (kernel_root / kRankUpdateReaderKernelName).string();
    reader.core_ranges = core_ranges;
    reader.compile_time_args = reader_compile_args(a, state);
    reader.config = tt::tt_metal::ReaderConfigDescriptor{};

    KernelDescriptor writer{};
    writer.kernel_source = (kernel_root / kRankUpdateWriterKernelName).string();
    writer.core_ranges = core_ranges;
    tt::tt_metal::TensorAccessorArgs(*output.buffer()).append_to(writer.compile_time_args);
    writer.config = tt::tt_metal::WriterConfigDescriptor{};

    tt::tt_metal::ComputeConfigDescriptor compute_config{};
    compute_config.math_fidelity = tt::tt_metal::MathFidelity::HiFi4;
    compute_config.fp32_dest_acc_en = true;
    compute_config.dst_full_sync_en = false;
    compute_config.math_approx_mode = false;
    KernelDescriptor compute{};
    compute.kernel_source = (kernel_root / kRankUpdateComputeKernelName).string();
    compute.core_ranges = core_ranges;
    compute.config = std::move(compute_config);

    for (std::size_t core_index = 0U; core_index < plan.cores.size(); ++core_index) {
        const auto& work = plan.work_by_core[core_index];
        require(
            work.size() <= std::numeric_limits<std::uint32_t>::max(), "rank_update diagnostic work count overflows");
        const std::uint32_t work_count = static_cast<std::uint32_t>(work.size());

        KernelDescriptor::RTArgList reader_args;
        reader_args.push_back(a.buffer());
        reader_args.push_back(b.buffer());
        reader_args.push_back(state.buffer());
        reader_args.push_back(work_count);
        reader_args.append(work);
        reader.emplace_runtime_args(plan.cores[core_index], reader_args);

        KernelDescriptor::RTArgList writer_args;
        writer_args.push_back(output.buffer());
        writer_args.push_back(kCompletionSemaphoreId);
        writer_args.push_back(work_count);
        writer_args.append(work);
        writer.emplace_runtime_args(plan.cores[core_index], writer_args);

        compute.emplace_runtime_args(plan.cores[core_index], {work_count});
    }

    SemaphoreDescriptor semaphore{};
    semaphore.id = kCompletionSemaphoreId;
    semaphore.core_type = tt::CoreType::WORKER;
    semaphore.core_ranges = core_ranges;
    semaphore.initial_value = kInitialSemaphoreValue;

    ProgramDescriptor descriptor{};
    descriptor.kernels.push_back(std::move(reader));
    descriptor.kernels.push_back(std::move(writer));
    descriptor.kernels.push_back(std::move(compute));
    descriptor.semaphores.push_back(std::move(semaphore));
    descriptor.cbs = make_rank_update_cbs(core_ranges);
    return descriptor;
}

ProgramDescriptor make_outer_update_program_descriptor(
    const std::filesystem::path& kernel_root,
    const OwnershipPlan& plan,
    const Tensor& v,
    const Tensor& k,
    Tensor& output) {
    const CoreRangeSet core_ranges = make_core_ranges(plan.cores);

    KernelDescriptor reader{};
    reader.kernel_source = (kernel_root / kOuterUpdateReaderKernelName).string();
    reader.core_ranges = core_ranges;
    tt::tt_metal::TensorAccessorArgs(*v.buffer()).append_to(reader.compile_time_args);
    reader.config = tt::tt_metal::ReaderConfigDescriptor{};

    KernelDescriptor writer{};
    writer.kernel_source = (kernel_root / kOuterUpdateWriterKernelName).string();
    writer.core_ranges = core_ranges;
    tt::tt_metal::TensorAccessorArgs(*output.buffer()).append_to(writer.compile_time_args);
    writer.config = tt::tt_metal::WriterConfigDescriptor{};

    tt::tt_metal::ComputeConfigDescriptor compute_config{};
    compute_config.math_fidelity = tt::tt_metal::MathFidelity::HiFi4;
    compute_config.fp32_dest_acc_en = true;
    compute_config.dst_full_sync_en = false;
    compute_config.math_approx_mode = false;
    KernelDescriptor compute{};
    compute.kernel_source = (kernel_root / kOuterUpdateComputeKernelName).string();
    compute.core_ranges = core_ranges;
    compute.config = std::move(compute_config);

    for (std::size_t core_index = 0U; core_index < plan.cores.size(); ++core_index) {
        const auto& work = plan.work_by_core[core_index];
        require(
            work.size() <= std::numeric_limits<std::uint32_t>::max(), "outer_update diagnostic work count overflows");
        const std::uint32_t work_count = static_cast<std::uint32_t>(work.size());

        KernelDescriptor::RTArgList reader_args;
        reader_args.push_back(v.buffer());
        reader_args.push_back(k.buffer());
        reader_args.push_back(work_count);
        reader_args.append(work);
        reader.emplace_runtime_args(plan.cores[core_index], reader_args);

        KernelDescriptor::RTArgList writer_args;
        writer_args.push_back(output.buffer());
        writer_args.push_back(kCompletionSemaphoreId);
        writer_args.push_back(work_count);
        writer_args.append(work);
        writer.emplace_runtime_args(plan.cores[core_index], writer_args);

        compute.emplace_runtime_args(plan.cores[core_index], {work_count});
    }

    SemaphoreDescriptor semaphore{};
    semaphore.id = kCompletionSemaphoreId;
    semaphore.core_type = tt::CoreType::WORKER;
    semaphore.core_ranges = core_ranges;
    semaphore.initial_value = kInitialSemaphoreValue;

    ProgramDescriptor descriptor{};
    descriptor.kernels.push_back(std::move(reader));
    descriptor.kernels.push_back(std::move(writer));
    descriptor.kernels.push_back(std::move(compute));
    descriptor.semaphores.push_back(std::move(semaphore));
    descriptor.cbs = make_outer_update_cbs(core_ranges);
    return descriptor;
}

ProgramDescriptor make_first_add_program_descriptor(
    const std::filesystem::path& kernel_root,
    const OwnershipPlan& plan,
    const Tensor& lhs,
    const Tensor& rhs,
    Tensor& output) {
    const CoreRangeSet core_ranges = make_core_ranges(plan.cores);

    KernelDescriptor reader{};
    reader.kernel_source = (kernel_root / kFirstAddReaderKernelName).string();
    reader.core_ranges = core_ranges;
    tt::tt_metal::TensorAccessorArgs(*lhs.buffer()).append_to(reader.compile_time_args);
    reader.config = tt::tt_metal::ReaderConfigDescriptor{};

    KernelDescriptor writer{};
    writer.kernel_source = (kernel_root / kFirstAddWriterKernelName).string();
    writer.core_ranges = core_ranges;
    tt::tt_metal::TensorAccessorArgs(*output.buffer()).append_to(writer.compile_time_args);
    writer.config = tt::tt_metal::WriterConfigDescriptor{};

    tt::tt_metal::ComputeConfigDescriptor compute_config{};
    compute_config.math_fidelity = tt::tt_metal::MathFidelity::HiFi4;
    compute_config.fp32_dest_acc_en = true;
    compute_config.dst_full_sync_en = false;
    compute_config.math_approx_mode = false;
    compute_config.unpack_to_dest_mode.assign(kCircularBufferCount, tt::tt_metal::UnpackToDestMode::Default);
    compute_config.unpack_to_dest_mode[kFirstAddLhsCb] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    compute_config.unpack_to_dest_mode[kFirstAddRhsCb] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    KernelDescriptor compute{};
    compute.kernel_source = (kernel_root / kFirstAddComputeKernelName).string();
    compute.core_ranges = core_ranges;
    compute.config = std::move(compute_config);

    for (std::size_t core_index = 0U; core_index < plan.cores.size(); ++core_index) {
        const auto& work = plan.work_by_core[core_index];
        require(work.size() <= std::numeric_limits<std::uint32_t>::max(), "first_add diagnostic work count overflows");
        const std::uint32_t work_count = static_cast<std::uint32_t>(work.size());

        KernelDescriptor::RTArgList reader_args;
        reader_args.push_back(lhs.buffer());
        reader_args.push_back(rhs.buffer());
        reader_args.push_back(work_count);
        reader_args.append(work);
        reader.emplace_runtime_args(plan.cores[core_index], reader_args);

        KernelDescriptor::RTArgList writer_args;
        writer_args.push_back(output.buffer());
        writer_args.push_back(kCompletionSemaphoreId);
        writer_args.push_back(work_count);
        writer_args.append(work);
        writer.emplace_runtime_args(plan.cores[core_index], writer_args);

        compute.emplace_runtime_args(plan.cores[core_index], {work_count});
    }

    SemaphoreDescriptor semaphore{};
    semaphore.id = kCompletionSemaphoreId;
    semaphore.core_type = tt::CoreType::WORKER;
    semaphore.core_ranges = core_ranges;
    semaphore.initial_value = kInitialSemaphoreValue;

    ProgramDescriptor descriptor{};
    descriptor.kernels.push_back(std::move(reader));
    descriptor.kernels.push_back(std::move(writer));
    descriptor.kernels.push_back(std::move(compute));
    descriptor.semaphores.push_back(std::move(semaphore));
    descriptor.cbs = make_first_add_cbs(core_ranges);
    return descriptor;
}

ProgramDescriptor make_bf16_readout_program_descriptor(
    const std::filesystem::path& kernel_root,
    const OwnershipPlan& plan,
    const Tensor& r,
    const Tensor& state,
    Tensor& output) {
    const CoreRangeSet core_ranges = make_core_ranges(plan.cores);

    KernelDescriptor reader{};
    reader.kernel_source = (kernel_root / kBf16ReadoutReaderKernelName).string();
    reader.core_ranges = core_ranges;
    reader.compile_time_args = reader_compile_args(r, state);
    reader.config = tt::tt_metal::ReaderConfigDescriptor{};

    KernelDescriptor writer{};
    writer.kernel_source = (kernel_root / kBf16ReadoutWriterKernelName).string();
    writer.core_ranges = core_ranges;
    tt::tt_metal::TensorAccessorArgs(*output.buffer()).append_to(writer.compile_time_args);
    writer.config = tt::tt_metal::WriterConfigDescriptor{};

    tt::tt_metal::ComputeConfigDescriptor compute_config{};
    compute_config.math_fidelity = tt::tt_metal::MathFidelity::HiFi2;
    compute_config.fp32_dest_acc_en = false;
    compute_config.dst_full_sync_en = false;
    compute_config.math_approx_mode = false;
    KernelDescriptor compute{};
    compute.kernel_source = (kernel_root / kBf16ReadoutComputeKernelName).string();
    compute.core_ranges = core_ranges;
    compute.config = std::move(compute_config);

    for (std::size_t core_index = 0U; core_index < plan.cores.size(); ++core_index) {
        const auto& work = plan.work_by_core[core_index];
        require(
            work.size() <= std::numeric_limits<std::uint32_t>::max(), "BF16 readout diagnostic work count overflows");
        const std::uint32_t work_count = static_cast<std::uint32_t>(work.size());

        KernelDescriptor::RTArgList reader_args;
        reader_args.push_back(r.buffer());
        reader_args.push_back(state.buffer());
        reader_args.push_back(work_count);
        reader_args.append(work);
        reader.emplace_runtime_args(plan.cores[core_index], reader_args);

        KernelDescriptor::RTArgList writer_args;
        writer_args.push_back(output.buffer());
        writer_args.push_back(kCompletionSemaphoreId);
        writer_args.push_back(work_count);
        writer_args.append(work);
        writer.emplace_runtime_args(plan.cores[core_index], writer_args);

        compute.emplace_runtime_args(plan.cores[core_index], {work_count});
    }

    SemaphoreDescriptor semaphore{};
    semaphore.id = kCompletionSemaphoreId;
    semaphore.core_type = tt::CoreType::WORKER;
    semaphore.core_ranges = core_ranges;
    semaphore.initial_value = kInitialSemaphoreValue;

    ProgramDescriptor descriptor{};
    descriptor.kernels.push_back(std::move(reader));
    descriptor.kernels.push_back(std::move(writer));
    descriptor.kernels.push_back(std::move(compute));
    descriptor.semaphores.push_back(std::move(semaphore));
    descriptor.cbs = make_bf16_readout_cbs(core_ranges);
    return descriptor;
}

std::vector<Tensor> device_tensors(const Tensor &tensor) {
  return ttnn::distributed::get_device_tensors(tensor);
}

void validate_kernel_root(const std::filesystem::path &kernel_root) {
  require(kernel_root.is_absolute(),
          "fused WKV kernel root must be an absolute path");
  for (const std::string_view kernel :
       {kReaderKernelName,
        kWriterKernelName,
        kComputeKernelName,
        kStateDotAReaderKernelName,
        kStateDotAWriterKernelName,
        kStateDotAComputeKernelName,
        kDecayReaderKernelName,
        kDecayWriterKernelName,
        kDecayComputeKernelName,
        kRankUpdateReaderKernelName,
        kRankUpdateWriterKernelName,
        kRankUpdateComputeKernelName,
        kOuterUpdateReaderKernelName,
        kOuterUpdateWriterKernelName,
        kOuterUpdateComputeKernelName,
        kFirstAddReaderKernelName,
        kFirstAddWriterKernelName,
        kFirstAddComputeKernelName,
        kBf16ReadoutReaderKernelName,
        kBf16ReadoutWriterKernelName,
        kBf16ReadoutComputeKernelName}) {
      require(std::filesystem::is_regular_file(kernel_root / kernel), "fused WKV kernel source is missing");
  }
}

} // namespace

DispatchDecision select_dispatch(const Inputs &inputs,
                                 bool physically_admitted) {
  const auto &state_shape = inputs.state.logical_shape();
  if (state_shape.rank() != kTensorRank ||
      state_shape[kBatchDimension] == 0U ||
      state_shape[kBatchDimension] > kMaximumBatchSize) {
    return reject(physically_admitted, "unsupported_batch_size");
  }
  const std::uint32_t batch_size = state_shape[kBatchDimension];
  if (!has_state_shape(inputs.state, batch_size)) {
    return reject(physically_admitted, "unsupported_state_shape");
  }
  if (inputs.state.dtype() != tt::tt_metal::DataType::FLOAT32) {
    return reject(physically_admitted, "unsupported_state_dtype");
  }
  if (inputs.state.layout() != ttnn::Layout::TILE) {
    return reject(physically_admitted, "unsupported_state_layout");
  }
  if (inputs.state.device() == nullptr ||
      inputs.state.device()->arch() != tt::ARCH::BLACKHOLE) {
    return reject(physically_admitted, "unsupported_architecture");
  }
  if (inputs.state.is_sharded()) {
    return reject(physically_admitted, "unsupported_state_sharding");
  }

  const auto input_vectors = vectors(inputs);
  for (std::size_t vector_index = 0U;
       vector_index < input_vectors.size(); ++vector_index) {
    const Tensor *vector = input_vectors[vector_index];
    const bool shared_across_batch =
        vector_index >= kFirstSharedVectorIndex &&
        vector_index <= kLastSharedVectorIndex;
    const std::uint32_t expected_batch =
        shared_across_batch ? 1U : batch_size;
    if (!has_vector_shape(*vector, expected_batch)) {
      return reject(physically_admitted, "unsupported_vector_shape");
    }
    if (vector->dtype() != tt::tt_metal::DataType::BFLOAT16) {
      return reject(physically_admitted, "unsupported_vector_dtype");
    }
    if (vector->layout() != ttnn::Layout::TILE) {
      return reject(physically_admitted, "unsupported_vector_layout");
    }
    if (vector->device() != inputs.state.device()) {
      return reject(physically_admitted, "vector_device_mismatch");
    }
    if (vector->is_sharded()) {
      return reject(physically_admitted, "unsupported_vector_sharding");
    }
    if (vector->memory_config() != inputs.a.memory_config()) {
      return reject(physically_admitted, "vector_memory_config_mismatch");
    }
  }

  if (!physically_admitted) {
    return DispatchDecision{true, false, false, "physical_admission_missing"};
  }
  return DispatchDecision{true, true, true, "fused_wkv_admitted"};
}

Tensor run_state_dot_a_diagnostic(MeshDevice& mesh, const Inputs& inputs, const std::filesystem::path& kernel_root) {
    const DispatchDecision decision = select_dispatch(inputs, true);
    require(decision.use_fused_path, "state_dot_a diagnostic does not satisfy the fused capability contract");
    require(inputs.state.device() == &mesh, "state_dot_a diagnostic state is on a different mesh");
    validate_kernel_root(kernel_root);

    const std::uint32_t batch_size = inputs.state.logical_shape()[kBatchDimension];
    Tensor output = ttnn::empty(
        ttnn::Shape({batch_size, kHeadCount, kHeadSize, 1U}),
        tt::tt_metal::DataType::FLOAT32,
        ttnn::Layout::TILE,
        &mesh,
        inputs.state.memory_config());

    const std::vector<Tensor> a_shards = device_tensors(inputs.a);
    const std::vector<Tensor> state_shards = device_tensors(inputs.state);
    std::vector<Tensor> output_shards = device_tensors(output);
    const std::size_t device_count = state_shards.size();
    require(
        device_count > 0U && device_count == mesh.num_devices(),
        "state_dot_a diagnostic state shard count differs from the mesh");
    require(
        a_shards.size() == device_count && output_shards.size() == device_count,
        "state_dot_a diagnostic shard count differs from the state");

    tt::tt_metal::experimental::MeshProgramDescriptor mesh_descriptor;
    for (std::size_t device_index = 0U; device_index < device_count; ++device_index) {
        const CoreCoord grid = state_shards[device_index].device()->compute_with_storage_grid_size();
        const OwnershipPlan plan = make_ownership_plan(grid, batch_size);
        ProgramDescriptor descriptor = make_state_dot_a_program_descriptor(
            kernel_root, plan, a_shards[device_index], state_shards[device_index], output_shards[device_index]);
        const ttnn::MeshCoordinate coordinate(kMeshRow, static_cast<std::uint32_t>(device_index));
        mesh_descriptor.mesh_programs.emplace_back(ttnn::MeshCoordinateRange(coordinate), std::move(descriptor));
    }

    std::vector<Tensor> io_tensors{inputs.a, inputs.state, output};
    return ttnn::generic_op(io_tensors, mesh_descriptor);
}

Tensor run_decay_diagnostic(MeshDevice& mesh, const Inputs& inputs, const std::filesystem::path& kernel_root) {
    const DispatchDecision decision = select_dispatch(inputs, true);
    require(decision.use_fused_path, "decay diagnostic does not satisfy the fused capability contract");
    require(inputs.state.device() == &mesh, "decay diagnostic state is on a different mesh");
    validate_kernel_root(kernel_root);

    const std::uint32_t batch_size = inputs.state.logical_shape()[kBatchDimension];
    Tensor output = ttnn::empty(
        inputs.state.logical_shape(),
        tt::tt_metal::DataType::FLOAT32,
        ttnn::Layout::TILE,
        &mesh,
        inputs.state.memory_config());

    const std::vector<Tensor> w_shards = device_tensors(inputs.w);
    const std::vector<Tensor> state_shards = device_tensors(inputs.state);
    std::vector<Tensor> output_shards = device_tensors(output);
    const std::size_t device_count = state_shards.size();
    require(
        device_count > 0U && device_count == mesh.num_devices(),
        "decay diagnostic state shard count differs from the mesh");
    require(
        w_shards.size() == device_count && output_shards.size() == device_count,
        "decay diagnostic shard count differs from the state");

    tt::tt_metal::experimental::MeshProgramDescriptor mesh_descriptor;
    for (std::size_t device_index = 0U; device_index < device_count; ++device_index) {
        const CoreCoord grid = state_shards[device_index].device()->compute_with_storage_grid_size();
        const OwnershipPlan plan = make_ownership_plan(grid, batch_size);
        ProgramDescriptor descriptor = make_decay_program_descriptor(
            kernel_root, plan, w_shards[device_index], state_shards[device_index], output_shards[device_index]);
        const ttnn::MeshCoordinate coordinate(kMeshRow, static_cast<std::uint32_t>(device_index));
        mesh_descriptor.mesh_programs.emplace_back(ttnn::MeshCoordinateRange(coordinate), std::move(descriptor));
    }

    std::vector<Tensor> io_tensors{inputs.w, inputs.state, output};
    return ttnn::generic_op(io_tensors, mesh_descriptor);
}

Tensor run_rank_update_diagnostic(MeshDevice& mesh, const Inputs& inputs, const std::filesystem::path& kernel_root) {
    const DispatchDecision decision = select_dispatch(inputs, true);
    require(decision.use_fused_path, "rank_update diagnostic does not satisfy the fused capability contract");
    require(inputs.state.device() == &mesh, "rank_update diagnostic state is on a different mesh");
    validate_kernel_root(kernel_root);

    const std::uint32_t batch_size = inputs.state.logical_shape()[kBatchDimension];
    Tensor output = ttnn::empty(
        inputs.state.logical_shape(),
        tt::tt_metal::DataType::FLOAT32,
        ttnn::Layout::TILE,
        &mesh,
        inputs.state.memory_config());

    const std::vector<Tensor> a_shards = device_tensors(inputs.a);
    const std::vector<Tensor> b_shards = device_tensors(inputs.b);
    const std::vector<Tensor> state_shards = device_tensors(inputs.state);
    std::vector<Tensor> output_shards = device_tensors(output);
    const std::size_t device_count = state_shards.size();
    require(
        device_count > 0U && device_count == mesh.num_devices(),
        "rank_update diagnostic state shard count differs from the mesh");
    require(
        a_shards.size() == device_count && b_shards.size() == device_count && output_shards.size() == device_count,
        "rank_update diagnostic shard count differs from the state");

    tt::tt_metal::experimental::MeshProgramDescriptor mesh_descriptor;
    for (std::size_t device_index = 0U; device_index < device_count; ++device_index) {
        const CoreCoord grid = state_shards[device_index].device()->compute_with_storage_grid_size();
        const OwnershipPlan plan = make_ownership_plan(grid, batch_size);
        ProgramDescriptor descriptor = make_rank_update_program_descriptor(
            kernel_root,
            plan,
            a_shards[device_index],
            b_shards[device_index],
            state_shards[device_index],
            output_shards[device_index]);
        const ttnn::MeshCoordinate coordinate(kMeshRow, static_cast<std::uint32_t>(device_index));
        mesh_descriptor.mesh_programs.emplace_back(ttnn::MeshCoordinateRange(coordinate), std::move(descriptor));
    }

    std::vector<Tensor> io_tensors{inputs.a, inputs.b, inputs.state, output};
    return ttnn::generic_op(io_tensors, mesh_descriptor);
}

Tensor run_outer_update_diagnostic(MeshDevice& mesh, const Inputs& inputs, const std::filesystem::path& kernel_root) {
    const DispatchDecision decision = select_dispatch(inputs, true);
    require(decision.use_fused_path, "outer_update diagnostic does not satisfy the fused capability contract");
    require(inputs.state.device() == &mesh, "outer_update diagnostic state is on a different mesh");
    validate_kernel_root(kernel_root);

    const std::uint32_t batch_size = inputs.state.logical_shape()[kBatchDimension];
    Tensor output = ttnn::empty(
        inputs.state.logical_shape(),
        tt::tt_metal::DataType::FLOAT32,
        ttnn::Layout::TILE,
        &mesh,
        inputs.state.memory_config());

    const std::vector<Tensor> v_shards = device_tensors(inputs.v);
    const std::vector<Tensor> k_shards = device_tensors(inputs.k);
    std::vector<Tensor> output_shards = device_tensors(output);
    const std::size_t device_count = output_shards.size();
    require(
        device_count > 0U && device_count == mesh.num_devices(),
        "outer_update diagnostic output shard count differs from the mesh");
    require(
        v_shards.size() == device_count && k_shards.size() == device_count,
        "outer_update diagnostic shard count differs from the output");

    tt::tt_metal::experimental::MeshProgramDescriptor mesh_descriptor;
    for (std::size_t device_index = 0U; device_index < device_count; ++device_index) {
        const CoreCoord grid = output_shards[device_index].device()->compute_with_storage_grid_size();
        const OwnershipPlan plan = make_ownership_plan(grid, batch_size);
        ProgramDescriptor descriptor = make_outer_update_program_descriptor(
            kernel_root, plan, v_shards[device_index], k_shards[device_index], output_shards[device_index]);
        const ttnn::MeshCoordinate coordinate(kMeshRow, static_cast<std::uint32_t>(device_index));
        mesh_descriptor.mesh_programs.emplace_back(ttnn::MeshCoordinateRange(coordinate), std::move(descriptor));
    }

    std::vector<Tensor> io_tensors{inputs.v, inputs.k, output};
    return ttnn::generic_op(io_tensors, mesh_descriptor);
}

Tensor run_first_add_diagnostic(
    MeshDevice& mesh,
    const Inputs& inputs,
    const Tensor& lhs,
    const Tensor& rhs,
    const std::filesystem::path& kernel_root) {
    const DispatchDecision decision = select_dispatch(inputs, true);
    require(decision.use_fused_path, "first_add diagnostic does not satisfy the fused capability contract");
    require(lhs.device() == &mesh && rhs.device() == &mesh, "first_add diagnostic input is on a different mesh");
    require(
        lhs.logical_shape() == inputs.state.logical_shape() && rhs.logical_shape() == inputs.state.logical_shape(),
        "first_add diagnostic input has the wrong shape");
    require(
        lhs.dtype() == tt::tt_metal::DataType::FLOAT32 && rhs.dtype() == tt::tt_metal::DataType::FLOAT32,
        "first_add diagnostic input has the wrong dtype");
    validate_kernel_root(kernel_root);

    const std::uint32_t batch_size = inputs.state.logical_shape()[kBatchDimension];
    Tensor output = ttnn::empty(
        inputs.state.logical_shape(),
        tt::tt_metal::DataType::FLOAT32,
        ttnn::Layout::TILE,
        &mesh,
        inputs.state.memory_config());

    const std::vector<Tensor> lhs_shards = device_tensors(lhs);
    const std::vector<Tensor> rhs_shards = device_tensors(rhs);
    std::vector<Tensor> output_shards = device_tensors(output);
    const std::size_t device_count = output_shards.size();
    require(
        device_count > 0U && device_count == mesh.num_devices(),
        "first_add diagnostic output shard count differs from the mesh");
    require(
        lhs_shards.size() == device_count && rhs_shards.size() == device_count,
        "first_add diagnostic shard count differs from the output");

    tt::tt_metal::experimental::MeshProgramDescriptor mesh_descriptor;
    for (std::size_t device_index = 0U; device_index < device_count; ++device_index) {
        const CoreCoord grid = output_shards[device_index].device()->compute_with_storage_grid_size();
        const OwnershipPlan plan = make_ownership_plan(grid, batch_size);
        ProgramDescriptor descriptor = make_first_add_program_descriptor(
            kernel_root, plan, lhs_shards[device_index], rhs_shards[device_index], output_shards[device_index]);
        const ttnn::MeshCoordinate coordinate(kMeshRow, static_cast<std::uint32_t>(device_index));
        mesh_descriptor.mesh_programs.emplace_back(ttnn::MeshCoordinateRange(coordinate), std::move(descriptor));
    }

    std::vector<Tensor> io_tensors{lhs, rhs, output};
    return ttnn::generic_op(io_tensors, mesh_descriptor);
}

Tensor run_bf16_readout_diagnostic(
    MeshDevice& mesh, const Inputs& inputs, const Tensor& state, const std::filesystem::path& kernel_root) {
    const DispatchDecision decision = select_dispatch(inputs, true);
    require(decision.use_fused_path, "BF16 readout diagnostic does not satisfy the fused capability contract");
    require(
        state.device() == &mesh && state.logical_shape() == inputs.state.logical_shape(),
        "BF16 readout diagnostic state is incompatible");
    require(state.dtype() == tt::tt_metal::DataType::FLOAT32, "BF16 readout diagnostic state has the wrong dtype");
    validate_kernel_root(kernel_root);

    const std::uint32_t batch_size = state.logical_shape()[kBatchDimension];
    Tensor output = ttnn::empty(
        ttnn::Shape({batch_size, kHeadCount, kHeadSize, 1U}),
        tt::tt_metal::DataType::BFLOAT16,
        ttnn::Layout::TILE,
        &mesh,
        inputs.r.memory_config());

    const std::vector<Tensor> r_shards = device_tensors(inputs.r);
    const std::vector<Tensor> state_shards = device_tensors(state);
    std::vector<Tensor> output_shards = device_tensors(output);
    const std::size_t device_count = state_shards.size();
    require(
        device_count > 0U && device_count == mesh.num_devices(),
        "BF16 readout diagnostic state shard count differs from the mesh");
    require(
        r_shards.size() == device_count && output_shards.size() == device_count,
        "BF16 readout diagnostic shard count differs from the state");

    tt::tt_metal::experimental::MeshProgramDescriptor mesh_descriptor;
    for (std::size_t device_index = 0U; device_index < device_count; ++device_index) {
        const CoreCoord grid = state_shards[device_index].device()->compute_with_storage_grid_size();
        const OwnershipPlan plan = make_ownership_plan(grid, batch_size);
        ProgramDescriptor descriptor = make_bf16_readout_program_descriptor(
            kernel_root, plan, r_shards[device_index], state_shards[device_index], output_shards[device_index]);
        const ttnn::MeshCoordinate coordinate(kMeshRow, static_cast<std::uint32_t>(device_index));
        mesh_descriptor.mesh_programs.emplace_back(ttnn::MeshCoordinateRange(coordinate), std::move(descriptor));
    }

    std::vector<Tensor> io_tensors{inputs.r, state, output};
    return ttnn::generic_op(io_tensors, mesh_descriptor);
}

Outputs run(MeshDevice &mesh, const Inputs &inputs,
            const std::filesystem::path &kernel_root) {
  const DispatchDecision decision = select_dispatch(inputs, true);
  require(decision.use_fused_path,
          "fused WKV invocation does not satisfy its capability contract");
  require(inputs.state.device() == &mesh,
          "fused WKV state is on a different mesh");
  validate_kernel_root(kernel_root);

  const std::uint32_t batch_size =
      inputs.state.logical_shape()[kBatchDimension];
  Tensor output = ttnn::empty(
      ttnn::Shape({batch_size, kHeadCount, 1U, kHeadSize}),
      tt::tt_metal::DataType::BFLOAT16, ttnn::Layout::TILE, &mesh,
      inputs.a.memory_config());
  Tensor next_state = ttnn::empty(
      inputs.state.logical_shape(), tt::tt_metal::DataType::FLOAT32,
      ttnn::Layout::TILE, &mesh, inputs.state.memory_config());

  const auto input_vectors = vectors(inputs);
  std::array<std::vector<Tensor>, kVectorCount> vector_shards;
  for (std::size_t vector = 0U; vector < input_vectors.size(); ++vector) {
    vector_shards[vector] = device_tensors(*input_vectors[vector]);
  }
  const std::vector<Tensor> state_shards = device_tensors(inputs.state);
  std::vector<Tensor> output_shards = device_tensors(output);
  std::vector<Tensor> next_state_shards = device_tensors(next_state);
  const std::size_t device_count = state_shards.size();
  require(device_count > 0U && device_count == mesh.num_devices(),
          "fused WKV state shard count differs from the mesh");
  require(output_shards.size() == device_count &&
              next_state_shards.size() == device_count,
          "fused WKV output shard count differs from the state");
  for (const auto &shards : vector_shards) {
    require(shards.size() == device_count,
            "fused WKV vector shard count differs from the state");
  }

  tt::tt_metal::experimental::MeshProgramDescriptor mesh_descriptor;
  for (std::size_t device_index = 0U; device_index < device_count;
       ++device_index) {
    std::array<const Tensor *, kVectorCount> local_vectors{};
    for (std::size_t vector = 0U; vector < local_vectors.size(); ++vector) {
      local_vectors[vector] = &vector_shards[vector][device_index];
    }
    const CoreCoord grid =
        state_shards[device_index].device()->compute_with_storage_grid_size();
    const OwnershipPlan plan = make_ownership_plan(grid, batch_size);
    ProgramDescriptor descriptor = make_program_descriptor(
        kernel_root, plan, local_vectors, state_shards[device_index],
        output_shards[device_index], next_state_shards[device_index]);
    const ttnn::MeshCoordinate coordinate(
        kMeshRow, static_cast<std::uint32_t>(device_index));
    mesh_descriptor.mesh_programs.emplace_back(
        ttnn::MeshCoordinateRange(coordinate), std::move(descriptor));
  }

  std::vector<Tensor> io_tensors;
  for (const Tensor *vector : input_vectors) {
    io_tensors.push_back(*vector);
  }
  io_tensors.push_back(inputs.state);
  io_tensors.push_back(output);
  io_tensors.push_back(next_state);
  Tensor returned_state = ttnn::generic_op(io_tensors, mesh_descriptor);
  if (kSplitReadout) {
      Tensor raw_readout = run_bf16_readout_diagnostic(mesh, inputs, returned_state, kernel_root);
      return Outputs{std::move(returned_state), std::move(raw_readout), false};
  }
  return Outputs{std::move(returned_state), std::move(output), true};
}

} // namespace rwkv7::fused_wkv
