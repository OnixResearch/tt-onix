#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_wh.h"

#if !defined(ARCH_BLACKHOLE)
#error "fused RWKV-7 outer_update diagnostic compute supports Blackhole only"
#endif

namespace {

constexpr std::uint32_t kHeadTileCount = 2U;
constexpr std::uint32_t kStateTileCount = kHeadTileCount * kHeadTileCount;
constexpr std::uint32_t kWorkCountArgIndex = 0U;

constexpr std::uint32_t kRawVCb = static_cast<std::uint32_t>(tt::CBIndex::c_0);
constexpr std::uint32_t kRawKCb = static_cast<std::uint32_t>(tt::CBIndex::c_1);
constexpr std::uint32_t kOrientedVCb = static_cast<std::uint32_t>(tt::CBIndex::c_2);
constexpr std::uint32_t kOrientedKCb = static_cast<std::uint32_t>(tt::CBIndex::c_3);
constexpr std::uint32_t kOuterUpdateOutputCb = static_cast<std::uint32_t>(tt::CBIndex::c_16);

void orient_v() {
    cb_reserve_back(kOrientedVCb, kHeadTileCount);
    transpose_wh_init(kRawVCb, kOrientedVCb);
    tile_regs_acquire();
    for (std::uint32_t tile = 0U; tile < kHeadTileCount; ++tile) {
        transpose_wh_tile(kRawVCb, tile, tile);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(kOrientedVCb);
    for (std::uint32_t tile = 0U; tile < kHeadTileCount; ++tile) {
        pack_tile(tile, kOrientedVCb, tile);
    }
    tile_regs_release();
    cb_push_back(kOrientedVCb, kHeadTileCount);
    cb_wait_front(kOrientedVCb, kHeadTileCount);
}

void orient_k() {
    cb_reserve_back(kOrientedKCb, kHeadTileCount);
    reconfig_data_format(kRawKCb, kRawKCb);
    copy_tile_init(kRawKCb);
    tile_regs_acquire();
    for (std::uint32_t tile = 0U; tile < kHeadTileCount; ++tile) {
        copy_tile(kRawKCb, tile, tile);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(kOrientedKCb);
    for (std::uint32_t tile = 0U; tile < kHeadTileCount; ++tile) {
        pack_tile(tile, kOrientedKCb, tile);
    }
    tile_regs_release();
    cb_push_back(kOrientedKCb, kHeadTileCount);
    cb_wait_front(kOrientedKCb, kHeadTileCount);
}

void compute_outer_update() {
    cb_reserve_back(kOuterUpdateOutputCb, kStateTileCount);
    mm_init(kOrientedVCb, kOrientedKCb, kOuterUpdateOutputCb);
    tile_regs_acquire();
    for (std::uint32_t output_tile = 0U; output_tile < kStateTileCount; ++output_tile) {
        const std::uint32_t row_tile = output_tile / kHeadTileCount;
        const std::uint32_t column_tile = output_tile % kHeadTileCount;
        matmul_tiles(kOrientedVCb, kOrientedKCb, row_tile, column_tile, output_tile);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(kOuterUpdateOutputCb);
    for (std::uint32_t tile = 0U; tile < kStateTileCount; ++tile) {
        pack_tile(tile, kOuterUpdateOutputCb, tile);
    }
    tile_regs_release();
    cb_push_back(kOuterUpdateOutputCb, kStateTileCount);
}

}  // namespace

void kernel_main() {
    const std::uint32_t work_count = get_arg_val<std::uint32_t>(kWorkCountArgIndex);
    compute_kernel_hw_startup(kOrientedVCb, kOrientedKCb, kOuterUpdateOutputCb);

    for (std::uint32_t work_index = 0U; work_index < work_count; ++work_index) {
        cb_wait_front(kRawVCb, kHeadTileCount);
        cb_wait_front(kRawKCb, kHeadTileCount);
        orient_v();
        orient_k();
        compute_outer_update();
        cb_pop_front(kOrientedVCb, kHeadTileCount);
        cb_pop_front(kOrientedKCb, kHeadTileCount);
        cb_pop_front(kRawVCb, kHeadTileCount);
        cb_pop_front(kRawKCb, kHeadTileCount);
    }
}
