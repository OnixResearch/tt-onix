#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"

#if !defined(ARCH_BLACKHOLE)
#error "fused RWKV-7 decay diagnostic compute supports Blackhole only"
#endif

namespace {

constexpr std::uint32_t kHeadTileCount = 2U;
constexpr std::uint32_t kStateTileCount = kHeadTileCount * kHeadTileCount;
constexpr std::uint32_t kWorkCountArgIndex = 0U;

constexpr std::uint32_t kRawWCb = static_cast<std::uint32_t>(tt::CBIndex::c_0);
constexpr std::uint32_t kStateCb = static_cast<std::uint32_t>(tt::CBIndex::c_1);
constexpr std::uint32_t kWCb = static_cast<std::uint32_t>(tt::CBIndex::c_2);
constexpr std::uint32_t kDecayOutputCb = static_cast<std::uint32_t>(tt::CBIndex::c_16);

void copy_w_to_fp32() {
    cb_reserve_back(kWCb, kHeadTileCount);
    reconfig_data_format(kRawWCb, kRawWCb);
    copy_tile_init(kRawWCb);
    tile_regs_acquire();
    for (std::uint32_t tile = 0U; tile < kHeadTileCount; ++tile) {
        copy_tile(kRawWCb, tile, tile);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(kWCb);
    for (std::uint32_t tile = 0U; tile < kHeadTileCount; ++tile) {
        pack_tile(tile, kWCb, tile);
    }
    tile_regs_release();
    cb_push_back(kWCb, kHeadTileCount);
    cb_wait_front(kWCb, kHeadTileCount);
}

void compute_decay() {
    cb_reserve_back(kDecayOutputCb, kStateTileCount);
    unary_op_init_common(kStateCb, kDecayOutputCb);
    mul_binary_tile_init();
    tile_regs_acquire();

    copy_tile_to_dst_init_short_with_dt(kWCb, kStateCb);
    for (std::uint32_t output_tile = 0U; output_tile < kStateTileCount; ++output_tile) {
        copy_tile(kStateCb, output_tile, output_tile * 2U);
    }

    copy_tile_to_dst_init_short_with_dt(kStateCb, kWCb);
    for (std::uint32_t output_tile = 0U; output_tile < kStateTileCount; ++output_tile) {
        const std::uint32_t column_tile = output_tile % kHeadTileCount;
        const std::uint32_t state_dst = output_tile * 2U;
        const std::uint32_t decay_dst = state_dst + 1U;
        copy_tile(kWCb, column_tile, decay_dst);
        mul_binary_tile(state_dst, decay_dst, state_dst);
    }

    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(kDecayOutputCb);
    for (std::uint32_t output_tile = 0U; output_tile < kStateTileCount; ++output_tile) {
        pack_tile(output_tile * 2U, kDecayOutputCb, output_tile);
    }
    tile_regs_release();
    cb_push_back(kDecayOutputCb, kStateTileCount);
}

}  // namespace

void kernel_main() {
    const std::uint32_t work_count = get_arg_val<std::uint32_t>(kWorkCountArgIndex);
    compute_kernel_hw_startup(kStateCb, kWCb, kDecayOutputCb);

    for (std::uint32_t work_index = 0U; work_index < work_count; ++work_index) {
        cb_wait_front(kRawWCb, kHeadTileCount);
        cb_wait_front(kStateCb, kStateTileCount);
        copy_w_to_fp32();
        compute_decay();
        cb_pop_front(kWCb, kHeadTileCount);
        cb_pop_front(kRawWCb, kHeadTileCount);
        cb_pop_front(kStateCb, kStateTileCount);
    }
}
