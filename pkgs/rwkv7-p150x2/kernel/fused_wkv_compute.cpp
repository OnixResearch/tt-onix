#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reduce.h"
#include "api/compute/reg_api.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_wh.h"

#if !defined(ARCH_BLACKHOLE)
#error "fused RWKV-7 WKV compute supports Blackhole only"
#endif

namespace {

constexpr std::uint32_t kTileWidth = 32;
constexpr std::uint32_t kHeadTileCount = 2;
constexpr std::uint32_t kStateTileCount =
    kHeadTileCount * kHeadTileCount;
constexpr std::uint32_t kHeadElementCount = kTileWidth * kHeadTileCount;
constexpr std::uint32_t kRecurrenceVectorCount = 6;
constexpr std::uint32_t kSharedVectorCount = 3;
constexpr std::uint32_t kPerWorkCorrectionVectorCount = 1;
constexpr std::uint32_t kRawVectorCount =
    kRecurrenceVectorCount + kSharedVectorCount +
    kPerWorkCorrectionVectorCount;
constexpr std::uint32_t kWorkCountArgIndex = 0;
constexpr std::uint32_t kScalarTileCount = 1;
constexpr std::uint32_t kFirstTile = 0;
constexpr std::uint32_t kSecondTile = 1;
constexpr float kLayerNormEpsilon = 1.0e-5F;
constexpr float kGroupNormEpsilon =
    static_cast<float>(kHeadElementCount) * kLayerNormEpsilon;
constexpr float kMeanReductionScale =
    1.0F / static_cast<float>(kHeadElementCount);
constexpr float kSumReductionScale = 1.0F;

// The reader supplies ten sanitized row vectors in this order: a, w, k, v,
// r, b, r_k, group-norm weight, group-norm bias, and gate. Compute orients
// only the six recurrence vectors. The correction vectors remain rows.
constexpr std::uint32_t kRawACb =
    static_cast<std::uint32_t>(tt::CBIndex::c_0);
constexpr std::uint32_t kRawWCb =
    static_cast<std::uint32_t>(tt::CBIndex::c_1);
constexpr std::uint32_t kRawKCb =
    static_cast<std::uint32_t>(tt::CBIndex::c_2);
constexpr std::uint32_t kRawVCb =
    static_cast<std::uint32_t>(tt::CBIndex::c_3);
constexpr std::uint32_t kRawRCb =
    static_cast<std::uint32_t>(tt::CBIndex::c_4);
constexpr std::uint32_t kRawBCb =
    static_cast<std::uint32_t>(tt::CBIndex::c_5);
constexpr std::uint32_t kStateCb =
    static_cast<std::uint32_t>(tt::CBIndex::c_6);
constexpr std::uint32_t kRawRkCb =
    static_cast<std::uint32_t>(tt::CBIndex::c_7);
constexpr std::uint32_t kStateDotACb =
    static_cast<std::uint32_t>(tt::CBIndex::c_8);
constexpr std::uint32_t kNextStateCb =
    static_cast<std::uint32_t>(tt::CBIndex::c_9);
constexpr std::uint32_t kReadoutCb =
    static_cast<std::uint32_t>(tt::CBIndex::c_10);
constexpr std::uint32_t kRawGroupNormWeightCb =
    static_cast<std::uint32_t>(tt::CBIndex::c_11);
constexpr std::uint32_t kACb =
    static_cast<std::uint32_t>(tt::CBIndex::c_12);
constexpr std::uint32_t kWCb =
    static_cast<std::uint32_t>(tt::CBIndex::c_13);
constexpr std::uint32_t kKCb =
    static_cast<std::uint32_t>(tt::CBIndex::c_14);
constexpr std::uint32_t kVCb =
    static_cast<std::uint32_t>(tt::CBIndex::c_15);
constexpr std::uint32_t kOutputCb =
    static_cast<std::uint32_t>(tt::CBIndex::c_16);
constexpr std::uint32_t kNextStateOutputCb =
    static_cast<std::uint32_t>(tt::CBIndex::c_17);
constexpr std::uint32_t kRCb =
    static_cast<std::uint32_t>(tt::CBIndex::c_18);
constexpr std::uint32_t kBCb =
    static_cast<std::uint32_t>(tt::CBIndex::c_19);
constexpr std::uint32_t kRawGroupNormBiasCb =
    static_cast<std::uint32_t>(tt::CBIndex::c_20);
constexpr std::uint32_t kRawGateCb =
    static_cast<std::uint32_t>(tt::CBIndex::c_21);
constexpr std::uint32_t kReadoutRowCb =
    static_cast<std::uint32_t>(tt::CBIndex::c_22);
constexpr std::uint32_t kMeanCb =
    static_cast<std::uint32_t>(tt::CBIndex::c_23);
constexpr std::uint32_t kCenteredCb =
    static_cast<std::uint32_t>(tt::CBIndex::c_24);
constexpr std::uint32_t kSquaredCb =
    static_cast<std::uint32_t>(tt::CBIndex::c_25);
constexpr std::uint32_t kVarianceCb =
    static_cast<std::uint32_t>(tt::CBIndex::c_26);
constexpr std::uint32_t kInverseStdCb =
    static_cast<std::uint32_t>(tt::CBIndex::c_27);
constexpr std::uint32_t kAffineCb =
    static_cast<std::uint32_t>(tt::CBIndex::c_28);
constexpr std::uint32_t kCorrectionScalarCb =
    static_cast<std::uint32_t>(tt::CBIndex::c_29);
constexpr std::uint32_t kCorrectionVectorCb =
    static_cast<std::uint32_t>(tt::CBIndex::c_30);
constexpr std::uint32_t kReductionScalerCb =
    static_cast<std::uint32_t>(tt::CBIndex::c_31);

constexpr std::uint32_t kRawVectorCbs[kRawVectorCount] = {
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
constexpr std::uint32_t
    kOrientedVectorCbs[kRecurrenceVectorCount] = {
        kACb,
        kWCb,
        kKCb,
        kVCb,
        kRCb,
        kBCb,
};
constexpr bool kTransposeVector[kRecurrenceVectorCount] = {
    true,
    false,
    false,
    true,
    true,
    false,
};

void wait_for_work_item() {
  for (std::uint32_t vector = 0; vector < kRawVectorCount; ++vector) {
    cb_wait_front(kRawVectorCbs[vector], kHeadTileCount);
  }
  cb_wait_front(kStateCb, kStateTileCount);
}

void release_work_item() {
  for (std::uint32_t vector = 0; vector < kRawVectorCount; ++vector) {
    cb_pop_front(kRawVectorCbs[vector], kHeadTileCount);
  }
  cb_pop_front(kStateCb, kStateTileCount);
}

void copy_vector(std::uint32_t input_cb, std::uint32_t output_cb) {
  cb_reserve_back(output_cb, kHeadTileCount);
  reconfig_data_format(input_cb, input_cb);
  copy_tile_init(input_cb);
  tile_regs_acquire();
  for (std::uint32_t tile = 0; tile < kHeadTileCount; ++tile) {
    copy_tile(input_cb, tile, tile);
  }
  tile_regs_commit();
  tile_regs_wait();
  pack_reconfig_data_format(output_cb);
  for (std::uint32_t tile = 0; tile < kHeadTileCount; ++tile) {
    pack_tile(tile, output_cb, tile);
  }
  tile_regs_release();
  cb_push_back(output_cb, kHeadTileCount);
}

void transpose_vector(std::uint32_t input_cb, std::uint32_t output_cb) {
  cb_reserve_back(output_cb, kHeadTileCount);
  transpose_wh_init(input_cb, output_cb);
  tile_regs_acquire();
  for (std::uint32_t tile = 0; tile < kHeadTileCount; ++tile) {
    transpose_wh_tile(input_cb, tile, tile);
  }
  tile_regs_commit();
  tile_regs_wait();
  pack_reconfig_data_format(output_cb);
  for (std::uint32_t tile = 0; tile < kHeadTileCount; ++tile) {
    pack_tile(tile, output_cb, tile);
  }
  tile_regs_release();
  cb_push_back(output_cb, kHeadTileCount);
}

void orient_vectors() {
  for (std::uint32_t vector = 0; vector < kRecurrenceVectorCount;
       ++vector) {
    const std::uint32_t raw_cb = kRawVectorCbs[vector];
    const std::uint32_t oriented_cb = kOrientedVectorCbs[vector];
    if (kTransposeVector[vector]) {
      transpose_vector(raw_cb, oriented_cb);
    } else {
      copy_vector(raw_cb, oriented_cb);
    }
  }
  for (std::uint32_t vector = 0; vector < kRecurrenceVectorCount;
       ++vector) {
    cb_wait_front(kOrientedVectorCbs[vector], kHeadTileCount);
  }
}

void release_oriented_vectors() {
  for (std::uint32_t vector = 0; vector < kRecurrenceVectorCount;
       ++vector) {
    cb_pop_front(kOrientedVectorCbs[vector], kHeadTileCount);
  }
}

void compute_state_dot_a() {
  cb_reserve_back(kStateDotACb, kHeadTileCount);
  mm_init(kStateCb, kACb, kStateDotACb);
  tile_regs_acquire();
  for (std::uint32_t row_tile = 0; row_tile < kHeadTileCount;
       ++row_tile) {
    for (std::uint32_t column_tile = 0;
         column_tile < kHeadTileCount; ++column_tile) {
      const std::uint32_t state_tile =
          row_tile * kHeadTileCount + column_tile;
      matmul_tiles(kStateCb, kACb, state_tile, column_tile, row_tile);
    }
  }
  tile_regs_commit();
  tile_regs_wait();
  pack_reconfig_data_format(kStateDotACb);
  for (std::uint32_t row_tile = 0; row_tile < kHeadTileCount;
       ++row_tile) {
    pack_tile(row_tile, kStateDotACb, row_tile);
  }
  tile_regs_release();
  cb_push_back(kStateDotACb, kHeadTileCount);
  cb_wait_front(kStateDotACb, kHeadTileCount);
}

void compute_next_state() {
  cb_reserve_back(kNextStateCb, kStateTileCount);
  tile_regs_acquire();

  // S' = S * diag(w) + (S @ a) outer b + v outer k.
  reconfig_data_format(kStateCb, kWCb);
  mul_bcast_rows_init_short(kStateCb, kWCb);
  for (std::uint32_t output_tile = 0; output_tile < kStateTileCount;
       ++output_tile) {
    const std::uint32_t column_tile = output_tile % kHeadTileCount;
    mul_tiles_bcast_rows(
        kStateCb, kWCb, output_tile, column_tile, output_tile);
  }

  reconfig_data_format(kBCb, kStateDotACb);
  mm_init_short(kStateDotACb, kBCb, 0);
  for (std::uint32_t output_tile = 0; output_tile < kStateTileCount;
       ++output_tile) {
    const std::uint32_t row_tile = output_tile / kHeadTileCount;
    const std::uint32_t column_tile = output_tile % kHeadTileCount;
    matmul_tiles(
        kStateDotACb, kBCb, row_tile, column_tile, output_tile);
  }

  reconfig_data_format(kKCb, kVCb);
  mm_init_short(kVCb, kKCb, 0);
  for (std::uint32_t output_tile = 0; output_tile < kStateTileCount;
       ++output_tile) {
    const std::uint32_t row_tile = output_tile / kHeadTileCount;
    const std::uint32_t column_tile = output_tile % kHeadTileCount;
    matmul_tiles(kVCb, kKCb, row_tile, column_tile, output_tile);
  }

  tile_regs_commit();
  tile_regs_wait();
  pack_reconfig_data_format(kNextStateCb);
  for (std::uint32_t output_tile = 0; output_tile < kStateTileCount;
       ++output_tile) {
    pack_tile(output_tile, kNextStateCb, output_tile);
  }
  tile_regs_release();
  cb_push_back(kNextStateCb, kStateTileCount);
  cb_wait_front(kNextStateCb, kStateTileCount);
}

void compute_readout() {
  cb_reserve_back(kReadoutCb, kHeadTileCount);
  mm_init(kNextStateCb, kRCb, kReadoutCb);
  tile_regs_acquire();
  for (std::uint32_t row_tile = 0; row_tile < kHeadTileCount;
       ++row_tile) {
    for (std::uint32_t column_tile = 0;
         column_tile < kHeadTileCount; ++column_tile) {
      const std::uint32_t state_tile =
          row_tile * kHeadTileCount + column_tile;
      matmul_tiles(
          kNextStateCb, kRCb, state_tile, column_tile, row_tile);
    }
  }
  tile_regs_commit();
  tile_regs_wait();
  pack_reconfig_data_format(kReadoutCb);
  for (std::uint32_t row_tile = 0; row_tile < kHeadTileCount;
       ++row_tile) {
    pack_tile(row_tile, kReadoutCb, row_tile);
  }
  tile_regs_release();
  cb_push_back(kReadoutCb, kHeadTileCount);
  cb_wait_front(kReadoutCb, kHeadTileCount);
}

void transpose_readout_to_row() {
  cb_reserve_back(kReadoutRowCb, kHeadTileCount);
  transpose_wh_init(kReadoutCb, kReadoutRowCb);
  tile_regs_acquire();
  for (std::uint32_t tile = 0; tile < kHeadTileCount; ++tile) {
    transpose_wh_tile(kReadoutCb, tile, tile);
  }
  tile_regs_commit();
  tile_regs_wait();
  pack_reconfig_data_format(kReadoutRowCb);
  for (std::uint32_t tile = 0; tile < kHeadTileCount; ++tile) {
    pack_tile(tile, kReadoutRowCb, tile);
  }
  tile_regs_release();
  cb_push_back(kReadoutRowCb, kHeadTileCount);
  cb_wait_front(kReadoutRowCb, kHeadTileCount);
}

void make_reduction_scaler(float value) {
  cb_reserve_back(kReductionScalerCb, kScalarTileCount);
  tile_regs_acquire();
  fill_tile_init();
  fill_tile(kFirstTile, value);
  tile_regs_commit();
  tile_regs_wait();
  pack_reconfig_data_format(kReductionScalerCb);
  pack_tile(kFirstTile, kReductionScalerCb, kFirstTile);
  tile_regs_release();
  cb_push_back(kReductionScalerCb, kScalarTileCount);
  cb_wait_front(kReductionScalerCb, kScalarTileCount);
}

void reduce_row_vector(std::uint32_t input_cb, std::uint32_t output_cb) {
  cb_reserve_back(output_cb, kScalarTileCount);
  reconfig_data_format(kReductionScalerCb, input_cb);
  reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(
      input_cb, kReductionScalerCb, output_cb);
  tile_regs_acquire();
  reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(
      input_cb, kReductionScalerCb, kFirstTile, kFirstTile, kFirstTile);
  reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(
      input_cb, kReductionScalerCb, kSecondTile, kFirstTile,
      kSecondTile);
  reduce_uninit(input_cb);
  add_binary_tile_init();
  add_binary_tile(kFirstTile, kSecondTile, kFirstTile);
  tile_regs_commit();
  tile_regs_wait();
  pack_tile(kFirstTile, output_cb, kFirstTile);
  tile_regs_release();
  cb_push_back(output_cb, kScalarTileCount);
  cb_wait_front(output_cb, kScalarTileCount);
}

void center_readout() {
  cb_reserve_back(kCenteredCb, kHeadTileCount);
  reconfig_data_format(kReadoutRowCb, kMeanCb);
  sub_tiles_bcast_scalar_init_short(kReadoutRowCb, kMeanCb);
  tile_regs_acquire();
  for (std::uint32_t tile = 0; tile < kHeadTileCount; ++tile) {
    sub_tiles_bcast_scalar(
        kReadoutRowCb, kMeanCb, tile, kFirstTile, tile);
  }
  tile_regs_commit();
  tile_regs_wait();
  pack_reconfig_data_format(kCenteredCb);
  for (std::uint32_t tile = 0; tile < kHeadTileCount; ++tile) {
    pack_tile(tile, kCenteredCb, tile);
  }
  tile_regs_release();
  cb_push_back(kCenteredCb, kHeadTileCount);
  cb_wait_front(kCenteredCb, kHeadTileCount);
  cb_pop_front(kReadoutRowCb, kHeadTileCount);
  cb_pop_front(kMeanCb, kScalarTileCount);
}

void square_centered_readout() {
  cb_reserve_back(kSquaredCb, kHeadTileCount);
  reconfig_data_format(kCenteredCb, kCenteredCb);
  mul_tiles_init(kCenteredCb, kCenteredCb);
  tile_regs_acquire();
  for (std::uint32_t tile = 0; tile < kHeadTileCount; ++tile) {
    mul_tiles(kCenteredCb, kCenteredCb, tile, tile, tile);
  }
  tile_regs_commit();
  tile_regs_wait();
  pack_reconfig_data_format(kSquaredCb);
  for (std::uint32_t tile = 0; tile < kHeadTileCount; ++tile) {
    pack_tile(tile, kSquaredCb, tile);
  }
  tile_regs_release();
  cb_push_back(kSquaredCb, kHeadTileCount);
  cb_wait_front(kSquaredCb, kHeadTileCount);
}

void compute_inverse_standard_deviation() {
  cb_reserve_back(kInverseStdCb, kScalarTileCount);
  reconfig_data_format(kVarianceCb, kVarianceCb);
  copy_tile_init(kVarianceCb);
  tile_regs_acquire();
  copy_tile(kVarianceCb, kFirstTile, kFirstTile);
  fill_tile_init();
  fill_tile(kSecondTile, kGroupNormEpsilon);
  add_binary_tile_init();
  add_binary_tile(kFirstTile, kSecondTile, kFirstTile);
  rsqrt_tile_init<true>();
  rsqrt_tile<true>(kFirstTile);
  tile_regs_commit();
  tile_regs_wait();
  pack_reconfig_data_format(kInverseStdCb);
  pack_tile(kFirstTile, kInverseStdCb, kFirstTile);
  tile_regs_release();
  cb_push_back(kInverseStdCb, kScalarTileCount);
  cb_wait_front(kInverseStdCb, kScalarTileCount);
  cb_pop_front(kVarianceCb, kScalarTileCount);
}

void normalize_readout() {
  cb_reserve_back(kAffineCb, kHeadTileCount);
  reconfig_data_format(kCenteredCb, kInverseStdCb);
  mul_tiles_bcast_scalar_init_short(kCenteredCb, kInverseStdCb);
  tile_regs_acquire();
  for (std::uint32_t tile = 0; tile < kHeadTileCount; ++tile) {
    mul_tiles_bcast_scalar(
        kCenteredCb, kInverseStdCb, tile, kFirstTile, tile);
  }
  tile_regs_commit();
  tile_regs_wait();
  pack_reconfig_data_format(kAffineCb);
  for (std::uint32_t tile = 0; tile < kHeadTileCount; ++tile) {
    pack_tile(tile, kAffineCb, tile);
  }
  tile_regs_release();
  cb_push_back(kAffineCb, kHeadTileCount);
  cb_wait_front(kAffineCb, kHeadTileCount);
  cb_pop_front(kCenteredCb, kHeadTileCount);
  cb_pop_front(kInverseStdCb, kScalarTileCount);
}

void apply_group_norm_affine() {
  cb_reserve_back(kCorrectionVectorCb, kHeadTileCount);
  reconfig_data_format(kAffineCb, kRawGroupNormWeightCb);
  mul_tiles_init(kAffineCb, kRawGroupNormWeightCb);
  tile_regs_acquire();
  for (std::uint32_t tile = 0; tile < kHeadTileCount; ++tile) {
    mul_tiles(
        kAffineCb, kRawGroupNormWeightCb, tile, tile, tile);
  }
  tile_regs_commit();
  tile_regs_wait();
  pack_reconfig_data_format(kCorrectionVectorCb);
  for (std::uint32_t tile = 0; tile < kHeadTileCount; ++tile) {
    pack_tile(tile, kCorrectionVectorCb, tile);
  }
  tile_regs_release();
  cb_push_back(kCorrectionVectorCb, kHeadTileCount);
  cb_wait_front(kCorrectionVectorCb, kHeadTileCount);
  cb_pop_front(kAffineCb, kHeadTileCount);

  cb_reserve_back(kAffineCb, kHeadTileCount);
  reconfig_data_format(kCorrectionVectorCb, kRawGroupNormBiasCb);
  add_tiles_init(kCorrectionVectorCb, kRawGroupNormBiasCb);
  tile_regs_acquire();
  for (std::uint32_t tile = 0; tile < kHeadTileCount; ++tile) {
    add_tiles(
        kCorrectionVectorCb, kRawGroupNormBiasCb, tile, tile, tile);
  }
  tile_regs_commit();
  tile_regs_wait();
  pack_reconfig_data_format(kAffineCb);
  for (std::uint32_t tile = 0; tile < kHeadTileCount; ++tile) {
    pack_tile(tile, kAffineCb, tile);
  }
  tile_regs_release();
  cb_push_back(kAffineCb, kHeadTileCount);
  cb_wait_front(kAffineCb, kHeadTileCount);
  cb_pop_front(kCorrectionVectorCb, kHeadTileCount);
}

void multiply_rows(
    std::uint32_t left_cb,
    std::uint32_t right_cb,
    std::uint32_t output_cb) {
  cb_reserve_back(output_cb, kHeadTileCount);
  reconfig_data_format(left_cb, right_cb);
  mul_tiles_init(left_cb, right_cb);
  tile_regs_acquire();
  for (std::uint32_t tile = 0; tile < kHeadTileCount; ++tile) {
    mul_tiles(left_cb, right_cb, tile, tile, tile);
  }
  tile_regs_commit();
  tile_regs_wait();
  pack_reconfig_data_format(output_cb);
  for (std::uint32_t tile = 0; tile < kHeadTileCount; ++tile) {
    pack_tile(tile, output_cb, tile);
  }
  tile_regs_release();
  cb_push_back(output_cb, kHeadTileCount);
  cb_wait_front(output_cb, kHeadTileCount);
}

void compute_attention_correction() {
  // correction_scalar = sum(r * k * r_k).
  multiply_rows(kRawRCb, kRawKCb, kCorrectionScalarCb);
  multiply_rows(kCorrectionScalarCb, kRawRkCb, kCorrectionVectorCb);
  cb_pop_front(kCorrectionScalarCb, kHeadTileCount);

  reduce_row_vector(kCorrectionVectorCb, kCorrectionScalarCb);
  cb_pop_front(kCorrectionVectorCb, kHeadTileCount);
  cb_pop_front(kReductionScalerCb, kScalarTileCount);

  cb_reserve_back(kCorrectionVectorCb, kHeadTileCount);
  reconfig_data_format(kRawVCb, kCorrectionScalarCb);
  mul_tiles_bcast_scalar_init_short(kRawVCb, kCorrectionScalarCb);
  tile_regs_acquire();
  for (std::uint32_t tile = 0; tile < kHeadTileCount; ++tile) {
    mul_tiles_bcast_scalar(
        kRawVCb, kCorrectionScalarCb, tile, kFirstTile, tile);
  }
  tile_regs_commit();
  tile_regs_wait();
  pack_reconfig_data_format(kCorrectionVectorCb);
  for (std::uint32_t tile = 0; tile < kHeadTileCount; ++tile) {
    pack_tile(tile, kCorrectionVectorCb, tile);
  }
  tile_regs_release();
  cb_push_back(kCorrectionVectorCb, kHeadTileCount);
  cb_wait_front(kCorrectionVectorCb, kHeadTileCount);
  cb_pop_front(kCorrectionScalarCb, kScalarTileCount);

  cb_reserve_back(kCenteredCb, kHeadTileCount);
  reconfig_data_format(kAffineCb, kCorrectionVectorCb);
  add_tiles_init(kAffineCb, kCorrectionVectorCb);
  tile_regs_acquire();
  for (std::uint32_t tile = 0; tile < kHeadTileCount; ++tile) {
    add_tiles(kAffineCb, kCorrectionVectorCb, tile, tile, tile);
  }
  tile_regs_commit();
  tile_regs_wait();
  pack_reconfig_data_format(kCenteredCb);
  for (std::uint32_t tile = 0; tile < kHeadTileCount; ++tile) {
    pack_tile(tile, kCenteredCb, tile);
  }
  tile_regs_release();
  cb_push_back(kCenteredCb, kHeadTileCount);
  cb_wait_front(kCenteredCb, kHeadTileCount);
  cb_pop_front(kAffineCb, kHeadTileCount);
  cb_pop_front(kCorrectionVectorCb, kHeadTileCount);
}

void apply_gate_and_emit_output() {
  cb_reserve_back(kOutputCb, kHeadTileCount);
  reconfig_data_format(kCenteredCb, kRawGateCb);
  mul_tiles_init(kCenteredCb, kRawGateCb);
  tile_regs_acquire();
  for (std::uint32_t tile = 0; tile < kHeadTileCount; ++tile) {
    mul_tiles(kCenteredCb, kRawGateCb, tile, tile, tile);
  }
  tile_regs_commit();
  tile_regs_wait();

  // This is the only BF16 boundary in the fused attention result.
  pack_reconfig_data_format(kOutputCb);
  for (std::uint32_t tile = 0; tile < kHeadTileCount; ++tile) {
    pack_tile(tile, kOutputCb, tile);
  }
  tile_regs_release();
  cb_push_back(kOutputCb, kHeadTileCount);
  cb_pop_front(kCenteredCb, kHeadTileCount);
}

void emit_next_state() {
  cb_reserve_back(kNextStateOutputCb, kStateTileCount);
  reconfig_data_format(kNextStateCb, kNextStateCb);
  copy_tile_init(kNextStateCb);
  tile_regs_acquire();
  for (std::uint32_t state_tile = 0; state_tile < kStateTileCount;
       ++state_tile) {
    copy_tile(kNextStateCb, state_tile, state_tile);
  }
  tile_regs_commit();
  tile_regs_wait();
  pack_reconfig_data_format(kOutputCb, kNextStateOutputCb);
  for (std::uint32_t state_tile = 0; state_tile < kStateTileCount;
       ++state_tile) {
    pack_tile(state_tile, kNextStateOutputCb, state_tile);
  }
  tile_regs_release();
  cb_push_back(kNextStateOutputCb, kStateTileCount);
}

void release_recurrence_intermediates() {
  cb_pop_front(kStateDotACb, kHeadTileCount);
  cb_pop_front(kNextStateCb, kStateTileCount);
  cb_pop_front(kReadoutCb, kHeadTileCount);
}

void compute_corrected_output() {
  transpose_readout_to_row();
  make_reduction_scaler(kMeanReductionScale);
  reduce_row_vector(kReadoutRowCb, kMeanCb);
  center_readout();
  square_centered_readout();
  reduce_row_vector(kSquaredCb, kVarianceCb);
  cb_pop_front(kSquaredCb, kHeadTileCount);
  cb_pop_front(kReductionScalerCb, kScalarTileCount);
  compute_inverse_standard_deviation();
  normalize_readout();
  apply_group_norm_affine();

  make_reduction_scaler(kSumReductionScale);
  compute_attention_correction();
  apply_gate_and_emit_output();
}

} // namespace

void kernel_main() {
  const std::uint32_t work_count =
      get_arg_val<std::uint32_t>(kWorkCountArgIndex);

  compute_kernel_hw_startup(kStateCb, kACb, kOutputCb);

  for (std::uint32_t work_index = 0; work_index < work_count;
       ++work_index) {
    wait_for_work_item();
    orient_vectors();
    compute_state_dot_a();
    compute_next_state();
    compute_readout();
    compute_corrected_output();
    emit_next_state();
    release_recurrence_intermediates();
    release_oriented_vectors();
    release_work_item();
  }
}
