#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_wh.h"

#if !defined(ARCH_BLACKHOLE)
#error "fused RWKV-7 WKV compute supports Blackhole only"
#endif

namespace {

constexpr std::uint32_t kHeadTileCount = 2;
constexpr std::uint32_t kStateTileCount =
    kHeadTileCount * kHeadTileCount;
constexpr std::uint32_t kWorkCountArgIndex = 0;

// The reader supplies column vectors a, v, and r. It supplies row vectors w,
// k, and b. Each vector contains two 32-element tiles for one 64-wide head.
constexpr std::uint32_t kACb = static_cast<std::uint32_t>(tt::CBIndex::c_0);
constexpr std::uint32_t kWCb = static_cast<std::uint32_t>(tt::CBIndex::c_1);
constexpr std::uint32_t kKCb = static_cast<std::uint32_t>(tt::CBIndex::c_2);
constexpr std::uint32_t kVCb = static_cast<std::uint32_t>(tt::CBIndex::c_3);
constexpr std::uint32_t kRCb = static_cast<std::uint32_t>(tt::CBIndex::c_4);
constexpr std::uint32_t kBCb = static_cast<std::uint32_t>(tt::CBIndex::c_5);

// State and internal circular buffers use FP32 tiles. kOutputCb uses BF16.
constexpr std::uint32_t kStateCb =
    static_cast<std::uint32_t>(tt::CBIndex::c_6);
constexpr std::uint32_t kStateDotACb =
    static_cast<std::uint32_t>(tt::CBIndex::c_8);
constexpr std::uint32_t kNextStateCb =
    static_cast<std::uint32_t>(tt::CBIndex::c_9);
constexpr std::uint32_t kReadoutCb =
    static_cast<std::uint32_t>(tt::CBIndex::c_10);
constexpr std::uint32_t kOutputCb =
    static_cast<std::uint32_t>(tt::CBIndex::c_16);
constexpr std::uint32_t kNextStateOutputCb =
    static_cast<std::uint32_t>(tt::CBIndex::c_17);

void wait_for_work_item() {
  cb_wait_front(kACb, kHeadTileCount);
  cb_wait_front(kWCb, kHeadTileCount);
  cb_wait_front(kKCb, kHeadTileCount);
  cb_wait_front(kVCb, kHeadTileCount);
  cb_wait_front(kRCb, kHeadTileCount);
  cb_wait_front(kBCb, kHeadTileCount);
  cb_wait_front(kStateCb, kStateTileCount);
}

void release_work_item() {
  cb_pop_front(kACb, kHeadTileCount);
  cb_pop_front(kWCb, kHeadTileCount);
  cb_pop_front(kKCb, kHeadTileCount);
  cb_pop_front(kVCb, kHeadTileCount);
  cb_pop_front(kRCb, kHeadTileCount);
  cb_pop_front(kBCb, kHeadTileCount);
  cb_pop_front(kStateCb, kStateTileCount);
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
  mul_bcast_rows_init_short(kStateCb, kWCb);
  for (std::uint32_t output_tile = 0; output_tile < kStateTileCount;
       ++output_tile) {
    const std::uint32_t column_tile = output_tile % kHeadTileCount;
    mul_tiles_bcast_rows(
        kStateCb, kWCb, output_tile, column_tile, output_tile);
  }

  mm_init_short(kStateDotACb, kBCb, 0);
  for (std::uint32_t output_tile = 0; output_tile < kStateTileCount;
       ++output_tile) {
    const std::uint32_t row_tile = output_tile / kHeadTileCount;
    const std::uint32_t column_tile = output_tile % kHeadTileCount;
    matmul_tiles(
        kStateDotACb, kBCb, row_tile, column_tile, output_tile);
  }

  mm_init_short(kVCb, kKCb, 0);
  for (std::uint32_t output_tile = 0; output_tile < kStateTileCount;
       ++output_tile) {
    const std::uint32_t row_tile = output_tile / kHeadTileCount;
    const std::uint32_t column_tile = output_tile % kHeadTileCount;
    matmul_tiles(kVCb, kKCb, row_tile, column_tile, output_tile);
  }

  tile_regs_commit();
  tile_regs_wait();
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
  for (std::uint32_t row_tile = 0; row_tile < kHeadTileCount;
       ++row_tile) {
    pack_tile(row_tile, kReadoutCb, row_tile);
  }
  tile_regs_release();
  cb_push_back(kReadoutCb, kHeadTileCount);
  cb_wait_front(kReadoutCb, kHeadTileCount);
}

void emit_result() {
  cb_reserve_back(kOutputCb, kHeadTileCount);
  transpose_wh_init(kReadoutCb, kOutputCb);
  tile_regs_acquire();
  for (std::uint32_t row_tile = 0; row_tile < kHeadTileCount;
       ++row_tile) {
    transpose_wh_tile(kReadoutCb, row_tile, row_tile);
  }
  tile_regs_commit();
  tile_regs_wait();
  pack_reconfig_data_format(kOutputCb);
  for (std::uint32_t row_tile = 0; row_tile < kHeadTileCount;
       ++row_tile) {
    pack_tile(row_tile, kOutputCb, row_tile);
  }
  tile_regs_release();
  cb_push_back(kOutputCb, kHeadTileCount);

  cb_reserve_back(kNextStateOutputCb, kStateTileCount);
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

void release_intermediates() {
  cb_pop_front(kStateDotACb, kHeadTileCount);
  cb_pop_front(kNextStateCb, kStateTileCount);
  cb_pop_front(kReadoutCb, kHeadTileCount);
}

} // namespace

void kernel_main() {
  const std::uint32_t work_count =
      get_arg_val<std::uint32_t>(kWorkCountArgIndex);

  compute_kernel_hw_startup(kStateCb, kACb, kOutputCb);

  for (std::uint32_t work_index = 0; work_index < work_count;
       ++work_index) {
    wait_for_work_item();
    compute_state_dot_a();
    compute_next_state();
    compute_readout();
    emit_result();
    release_intermediates();
    release_work_item();
  }
}
