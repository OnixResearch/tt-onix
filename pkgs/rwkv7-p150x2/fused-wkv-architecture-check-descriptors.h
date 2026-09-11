#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "llk_defs.h"

namespace rwkv7_fused_wkv_architecture_check {

constexpr std::size_t kCircularBufferCount = 64;
constexpr std::uint8_t kFloat32DataFormat = 0;
constexpr std::uint8_t kBfloat16DataFormat = 5;
constexpr std::uint8_t kInvalidDataFormat = 255;
constexpr std::uint8_t kTileFaceCount = 4;
constexpr std::uint8_t kTileFaceDimension = 16;
constexpr std::uint8_t kTileDimension = 32;
constexpr std::uint16_t kFloat32TileSizeBytes = 4096;
constexpr std::uint16_t kBfloat16TileSizeBytes = 2048;
constexpr std::uint16_t kInvalidTileSizeBytes = 1088;
constexpr std::uint8_t kTileFacesPerDimension = 2;

constexpr std::size_t kFirstRecurrenceVectorCb = 0;
constexpr std::size_t kLastRecurrenceVectorCb = 5;
constexpr std::size_t kStateCb = 6;
constexpr std::size_t kRkCb = 7;
constexpr std::size_t kStateDotACb = 8;
constexpr std::size_t kNextStateCb = 9;
constexpr std::size_t kReadoutCb = 10;
constexpr std::size_t kGroupNormWeightCb = 11;
constexpr std::size_t kFirstOrientedVectorCb = 12;
constexpr std::size_t kLastOrientedVectorCb = 15;
constexpr std::size_t kOutputCb = 16;
constexpr std::size_t kNextStateOutputCb = 17;
constexpr std::size_t kFirstTrailingVectorCb = 18;
constexpr std::size_t kLastCorrectionInputCb = 21;
constexpr std::size_t kFirstCorrectionScratchCb = 22;
constexpr std::size_t kLastCorrectionScratchCb = 31;

template <typename Value, std::size_t Count>
constexpr std::array<Value, Count> repeat_value(Value value) {
  std::array<Value, Count> result{};
  for (Value &item : result) {
    item = value;
  }
  return result;
}

template <typename Value, std::size_t Count, typename Generator>
constexpr std::array<Value, Count> generate_values(Generator generator) {
  std::array<Value, Count> result{};
  for (std::size_t index = 0; index < Count; ++index) {
    result[index] = generator(index);
  }
  return result;
}

constexpr bool is_bfloat16_cb(std::size_t index) {
  return (index >= kFirstRecurrenceVectorCb &&
          index <= kLastRecurrenceVectorCb) ||
         index == kRkCb || index == kGroupNormWeightCb ||
         (index >= kFirstOrientedVectorCb &&
          index <= kLastOrientedVectorCb) ||
         index == kOutputCb ||
         (index >= kFirstTrailingVectorCb &&
          index <= kLastCorrectionInputCb);
}

constexpr bool is_float32_cb(std::size_t index) {
  return index == kStateCb || index == kStateDotACb ||
         index == kNextStateCb || index == kReadoutCb ||
         index == kNextStateOutputCb ||
         (index >= kFirstCorrectionScratchCb &&
          index <= kLastCorrectionScratchCb);
}

constexpr std::uint8_t data_format_for_index(std::size_t index) {
  if (is_bfloat16_cb(index)) {
    return kBfloat16DataFormat;
  }
  if (is_float32_cb(index)) {
    return kFloat32DataFormat;
  }
  return kInvalidDataFormat;
}

constexpr std::uint16_t tile_size_for_index(std::size_t index) {
  if (is_bfloat16_cb(index)) {
    return kBfloat16TileSizeBytes;
  }
  if (is_float32_cb(index)) {
    return kFloat32TileSizeBytes;
  }
  return kInvalidTileSizeBytes;
}

} // namespace rwkv7_fused_wkv_architecture_check

#if defined(UCK_CHLKC_MATH)
constexpr ckernel::MathFidelity MATH_FIDELITY = ckernel::MathFidelity::HiFi4;
constexpr bool APPROX = false;
#endif

#if !defined(UCK_CHLKC_PACK)
constexpr auto unpack_src_format =
    rwkv7_fused_wkv_architecture_check::generate_values<
        std::uint8_t,
        rwkv7_fused_wkv_architecture_check::kCircularBufferCount>(
        rwkv7_fused_wkv_architecture_check::data_format_for_index);
constexpr auto unpack_dst_format = unpack_src_format;
constexpr auto unpack_tile_num_faces =
    rwkv7_fused_wkv_architecture_check::repeat_value<
        std::uint8_t,
        rwkv7_fused_wkv_architecture_check::kCircularBufferCount>(
        rwkv7_fused_wkv_architecture_check::kTileFaceCount);
constexpr auto unpack_partial_face =
    rwkv7_fused_wkv_architecture_check::repeat_value<
        std::uint8_t,
        rwkv7_fused_wkv_architecture_check::kCircularBufferCount>(0);
constexpr auto unpack_tile_face_r_dim =
    rwkv7_fused_wkv_architecture_check::repeat_value<
        std::uint8_t,
        rwkv7_fused_wkv_architecture_check::kCircularBufferCount>(
        rwkv7_fused_wkv_architecture_check::kTileFaceDimension);
constexpr auto unpack_narrow_tile = unpack_partial_face;
constexpr auto unpack_tile_r_dim =
    rwkv7_fused_wkv_architecture_check::repeat_value<
        std::uint8_t,
        rwkv7_fused_wkv_architecture_check::kCircularBufferCount>(
        rwkv7_fused_wkv_architecture_check::kTileDimension);
constexpr auto unpack_tile_c_dim = unpack_tile_r_dim;
constexpr auto unpack_tile_size =
    rwkv7_fused_wkv_architecture_check::generate_values<
        std::uint16_t,
        rwkv7_fused_wkv_architecture_check::kCircularBufferCount>(
        rwkv7_fused_wkv_architecture_check::tile_size_for_index);
constexpr auto unpack_num_faces_r_dim =
    rwkv7_fused_wkv_architecture_check::repeat_value<
        std::uint8_t,
        rwkv7_fused_wkv_architecture_check::kCircularBufferCount>(
        rwkv7_fused_wkv_architecture_check::kTileFacesPerDimension);
constexpr auto unpack_num_faces_c_dim = unpack_num_faces_r_dim;
#endif

#if defined(UCK_CHLKC_MATH) || defined(UCK_CHLKC_PACK) ||                  \
    defined(UCK_CHLKC_UNPACK) || defined(UCK_CHLKC_ISOLATE_SFPU)
constexpr bool DST_ACCUM_MODE = true;
#define DST_SYNC_MODE DstSync::SyncHalf
#endif
