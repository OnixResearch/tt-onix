#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/transpose_wh.h"

#if !defined(ARCH_BLACKHOLE)
#error "fused RWKV-7 state_dot_a diagnostic compute supports Blackhole only"
#endif

namespace {

constexpr std::uint32_t kHeadTileCount = 2;
constexpr std::uint32_t kStateTileCount = kHeadTileCount * kHeadTileCount;
constexpr std::uint32_t kWorkCountArgIndex = 0;

constexpr std::uint32_t kRawACb = static_cast<std::uint32_t>(tt::CBIndex::c_0);
constexpr std::uint32_t kStateCb = static_cast<std::uint32_t>(tt::CBIndex::c_1);
constexpr std::uint32_t kOrientedACb = static_cast<std::uint32_t>(tt::CBIndex::c_2);
constexpr std::uint32_t kStateDotAOutputCb = static_cast<std::uint32_t>(tt::CBIndex::c_16);

void transpose_a() {
    cb_reserve_back(kOrientedACb, kHeadTileCount);
    transpose_wh_init(kRawACb, kOrientedACb);
    tile_regs_acquire();
    for (std::uint32_t tile = 0; tile < kHeadTileCount; ++tile) {
        transpose_wh_tile(kRawACb, tile, tile);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(kOrientedACb);
    for (std::uint32_t tile = 0; tile < kHeadTileCount; ++tile) {
        pack_tile(tile, kOrientedACb, tile);
    }
    tile_regs_release();
    cb_push_back(kOrientedACb, kHeadTileCount);
    cb_wait_front(kOrientedACb, kHeadTileCount);
}

void compute_state_dot_a() {
    cb_reserve_back(kStateDotAOutputCb, kHeadTileCount);
    mm_init(kStateCb, kOrientedACb, kStateDotAOutputCb);
    tile_regs_acquire();
    for (std::uint32_t row_tile = 0; row_tile < kHeadTileCount; ++row_tile) {
        for (std::uint32_t column_tile = 0; column_tile < kHeadTileCount; ++column_tile) {
            const std::uint32_t state_tile = row_tile * kHeadTileCount + column_tile;
            matmul_tiles(kStateCb, kOrientedACb, state_tile, column_tile, row_tile);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(kStateDotAOutputCb);
    for (std::uint32_t row_tile = 0; row_tile < kHeadTileCount; ++row_tile) {
        pack_tile(row_tile, kStateDotAOutputCb, row_tile);
    }
    tile_regs_release();
    cb_push_back(kStateDotAOutputCb, kHeadTileCount);
}

}  // namespace

void kernel_main() {
    const std::uint32_t work_count = get_arg_val<std::uint32_t>(kWorkCountArgIndex);
    compute_kernel_hw_startup(kStateCb, kOrientedACb, kStateDotAOutputCb);

    for (std::uint32_t work_index = 0; work_index < work_count; ++work_index) {
        cb_wait_front(kRawACb, kHeadTileCount);
        cb_wait_front(kStateCb, kStateTileCount);
        transpose_a();
        compute_state_dot_a();
        cb_pop_front(kOrientedACb, kHeadTileCount);
        cb_pop_front(kRawACb, kHeadTileCount);
        cb_pop_front(kStateCb, kStateTileCount);
    }
}
