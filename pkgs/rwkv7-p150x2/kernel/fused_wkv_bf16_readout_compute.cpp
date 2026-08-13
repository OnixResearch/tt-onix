#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"

#if !defined(ARCH_BLACKHOLE)
#error "fused RWKV-7 BF16 readout diagnostic compute supports Blackhole only"
#endif

namespace {

constexpr std::uint32_t kHeadTileCount = 2U;
constexpr std::uint32_t kStateTileCount = kHeadTileCount * kHeadTileCount;
constexpr std::uint32_t kWorkCountArgIndex = 0U;
constexpr std::uint32_t kStateCb = static_cast<std::uint32_t>(tt::CBIndex::c_0);
constexpr std::uint32_t kRawRCb = static_cast<std::uint32_t>(tt::CBIndex::c_1);
constexpr std::uint32_t kBf16ReadoutOutputCb = static_cast<std::uint32_t>(tt::CBIndex::c_16);
constexpr std::uint32_t kOutputBlockWidth = 1U;
constexpr std::uint32_t kOutputBlockHeight = kHeadTileCount;
constexpr std::uint32_t kInnerBlockWidth = kHeadTileCount;
constexpr bool kTransposeR = true;

void compute_bf16_readout() {
    tile_regs_acquire();
    std::uint32_t state_tile = 0U;
    std::uint32_t r_tile = 0U;
    constexpr std::uint32_t kOutputTile = 0U;
    for (std::uint32_t inner_tile = 0U; inner_tile < kInnerBlockWidth; ++inner_tile) {
        matmul_block(
            kStateCb,
            kRawRCb,
            state_tile,
            r_tile,
            kOutputTile,
            kTransposeR,
            kOutputBlockWidth,
            kOutputBlockHeight,
            kInnerBlockWidth);
        ++state_tile;
        r_tile += kOutputBlockWidth;
    }
    tile_regs_commit();
    cb_reserve_back(kBf16ReadoutOutputCb, kHeadTileCount);
    tile_regs_wait();
    for (std::uint32_t tile = 0U; tile < kHeadTileCount; ++tile) {
        pack_tile(tile, kBf16ReadoutOutputCb, tile);
    }
    tile_regs_release();
    cb_push_back(kBf16ReadoutOutputCb, kHeadTileCount);
}

}  // namespace

void kernel_main() {
    const std::uint32_t work_count = get_arg_val<std::uint32_t>(kWorkCountArgIndex);
    compute_kernel_hw_startup<SrcOrder::Reverse>(kStateCb, kRawRCb, kBf16ReadoutOutputCb);
    matmul_block_init(kStateCb, kRawRCb, kTransposeR, kOutputBlockWidth, kOutputBlockHeight, kInnerBlockWidth);

    for (std::uint32_t work_index = 0U; work_index < work_count; ++work_index) {
        cb_wait_front(kStateCb, kStateTileCount);
        cb_wait_front(kRawRCb, kHeadTileCount);
        compute_bf16_readout();
        cb_pop_front(kStateCb, kStateTileCount);
        cb_pop_front(kRawRCb, kHeadTileCount);
    }
}
