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
#error "fused RWKV-7 rank_update diagnostic compute supports Blackhole only"
#endif

namespace {

constexpr std::uint32_t kHeadTileCount = 2U;
constexpr std::uint32_t kStateTileCount = kHeadTileCount * kHeadTileCount;
constexpr std::uint32_t kWorkCountArgIndex = 0U;

constexpr std::uint32_t kRawACb = static_cast<std::uint32_t>(tt::CBIndex::c_0);
constexpr std::uint32_t kRawBCb = static_cast<std::uint32_t>(tt::CBIndex::c_1);
constexpr std::uint32_t kStateCb = static_cast<std::uint32_t>(tt::CBIndex::c_2);
constexpr std::uint32_t kOrientedACb = static_cast<std::uint32_t>(tt::CBIndex::c_3);
constexpr std::uint32_t kOrientedBCb = static_cast<std::uint32_t>(tt::CBIndex::c_4);
constexpr std::uint32_t kStateDotACb = static_cast<std::uint32_t>(tt::CBIndex::c_5);
constexpr std::uint32_t kRankUpdateOutputCb = static_cast<std::uint32_t>(tt::CBIndex::c_16);

void orient_a() {
    cb_reserve_back(kOrientedACb, kHeadTileCount);
    transpose_wh_init(kRawACb, kOrientedACb);
    tile_regs_acquire();
    for (std::uint32_t tile = 0U; tile < kHeadTileCount; ++tile) {
        transpose_wh_tile(kRawACb, tile, tile);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(kOrientedACb);
    for (std::uint32_t tile = 0U; tile < kHeadTileCount; ++tile) {
        pack_tile(tile, kOrientedACb, tile);
    }
    tile_regs_release();
    cb_push_back(kOrientedACb, kHeadTileCount);
    cb_wait_front(kOrientedACb, kHeadTileCount);
}

void orient_b() {
    cb_reserve_back(kOrientedBCb, kHeadTileCount);
    reconfig_data_format(kRawBCb, kRawBCb);
    copy_tile_init(kRawBCb);
    tile_regs_acquire();
    for (std::uint32_t tile = 0U; tile < kHeadTileCount; ++tile) {
        copy_tile(kRawBCb, tile, tile);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(kOrientedBCb);
    for (std::uint32_t tile = 0U; tile < kHeadTileCount; ++tile) {
        pack_tile(tile, kOrientedBCb, tile);
    }
    tile_regs_release();
    cb_push_back(kOrientedBCb, kHeadTileCount);
    cb_wait_front(kOrientedBCb, kHeadTileCount);
}

void compute_state_dot_a() {
    cb_reserve_back(kStateDotACb, kHeadTileCount);
    mm_init(kStateCb, kOrientedACb, kStateDotACb);
    tile_regs_acquire();
    for (std::uint32_t row_tile = 0U; row_tile < kHeadTileCount; ++row_tile) {
        for (std::uint32_t column_tile = 0U; column_tile < kHeadTileCount; ++column_tile) {
            const std::uint32_t state_tile = row_tile * kHeadTileCount + column_tile;
            matmul_tiles(kStateCb, kOrientedACb, state_tile, column_tile, row_tile);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(kStateDotACb);
    for (std::uint32_t tile = 0U; tile < kHeadTileCount; ++tile) {
        pack_tile(tile, kStateDotACb, tile);
    }
    tile_regs_release();
    cb_push_back(kStateDotACb, kHeadTileCount);
    cb_wait_front(kStateDotACb, kHeadTileCount);
}

void compute_rank_update() {
    cb_reserve_back(kRankUpdateOutputCb, kStateTileCount);
    mm_init(kStateDotACb, kOrientedBCb, kRankUpdateOutputCb);
    tile_regs_acquire();
    for (std::uint32_t output_tile = 0U; output_tile < kStateTileCount; ++output_tile) {
        const std::uint32_t row_tile = output_tile / kHeadTileCount;
        const std::uint32_t column_tile = output_tile % kHeadTileCount;
        matmul_tiles(kStateDotACb, kOrientedBCb, row_tile, column_tile, output_tile);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(kRankUpdateOutputCb);
    for (std::uint32_t tile = 0U; tile < kStateTileCount; ++tile) {
        pack_tile(tile, kRankUpdateOutputCb, tile);
    }
    tile_regs_release();
    cb_push_back(kRankUpdateOutputCb, kStateTileCount);
}

}  // namespace

void kernel_main() {
    const std::uint32_t work_count = get_arg_val<std::uint32_t>(kWorkCountArgIndex);
    compute_kernel_hw_startup(kStateCb, kOrientedACb, kRankUpdateOutputCb);

    for (std::uint32_t work_index = 0U; work_index < work_count; ++work_index) {
        cb_wait_front(kRawACb, kHeadTileCount);
        cb_wait_front(kRawBCb, kHeadTileCount);
        cb_wait_front(kStateCb, kStateTileCount);
        orient_a();
        orient_b();
        compute_state_dot_a();
        compute_rank_update();
        cb_pop_front(kStateDotACb, kHeadTileCount);
        cb_pop_front(kOrientedACb, kHeadTileCount);
        cb_pop_front(kOrientedBCb, kHeadTileCount);
        cb_pop_front(kRawACb, kHeadTileCount);
        cb_pop_front(kRawBCb, kHeadTileCount);
        cb_pop_front(kStateCb, kStateTileCount);
    }
}
