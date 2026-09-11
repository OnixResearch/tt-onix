#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"

#if !defined(ARCH_BLACKHOLE)
#error "fused RWKV-7 first_add diagnostic compute supports Blackhole only"
#endif

namespace {

constexpr std::uint32_t kHeadTileCount = 2U;
constexpr std::uint32_t kStateTileCount = kHeadTileCount * kHeadTileCount;
constexpr std::uint32_t kOperandCount = 2U;
constexpr std::uint32_t kWorkCountArgIndex = 0U;
constexpr std::uint32_t kLhsCb = static_cast<std::uint32_t>(tt::CBIndex::c_0);
constexpr std::uint32_t kRhsCb = static_cast<std::uint32_t>(tt::CBIndex::c_1);
constexpr std::uint32_t kFirstAddOutputCb = static_cast<std::uint32_t>(tt::CBIndex::c_16);

void compute_first_add() {
    cb_reserve_back(kFirstAddOutputCb, kStateTileCount);
    unary_op_init_common(kLhsCb, kFirstAddOutputCb);
    add_binary_tile_init();
    tile_regs_acquire();

    copy_tile_to_dst_init_short_with_dt(kLhsCb, kLhsCb);
    for (std::uint32_t tile = 0U; tile < kStateTileCount; ++tile) {
        copy_tile(kLhsCb, tile, tile * kOperandCount);
    }
    copy_tile_to_dst_init_short_with_dt(kLhsCb, kRhsCb);
    for (std::uint32_t tile = 0U; tile < kStateTileCount; ++tile) {
        const std::uint32_t lhs_dst = tile * kOperandCount;
        const std::uint32_t rhs_dst = lhs_dst + 1U;
        copy_tile(kRhsCb, tile, rhs_dst);
        add_binary_tile(lhs_dst, rhs_dst, lhs_dst);
    }

    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(kFirstAddOutputCb);
    for (std::uint32_t tile = 0U; tile < kStateTileCount; ++tile) {
        pack_tile(tile * kOperandCount, kFirstAddOutputCb, tile);
    }
    tile_regs_release();
    cb_push_back(kFirstAddOutputCb, kStateTileCount);
}

}  // namespace

void kernel_main() {
    const std::uint32_t work_count = get_arg_val<std::uint32_t>(kWorkCountArgIndex);
    compute_kernel_hw_startup(kLhsCb, kRhsCb, kFirstAddOutputCb);

    for (std::uint32_t work_index = 0U; work_index < work_count; ++work_index) {
        cb_wait_front(kLhsCb, kStateTileCount);
        cb_wait_front(kRhsCb, kStateTileCount);
        compute_first_add();
        cb_pop_front(kLhsCb, kStateTileCount);
        cb_pop_front(kRhsCb, kStateTileCount);
    }
}
