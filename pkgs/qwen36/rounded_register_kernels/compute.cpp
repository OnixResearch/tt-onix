// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
// SPDX-License-Identifier: LGPL-2.1-only AND Apache-2.0
// BF16 SFPU product call derived from TT-Metal eltwise_binary_sfpu.h
// (Apache-2.0); Dst sequence inspired by Thatch-cloud's simulator-only
// mlp_register_epilogue.py at 95589fb2ebdd7aa400e1daa89dcb3a8d457756bc.
// Scheduling, reader, and strict Qwen TP2 B1/B8 integration are local.
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    const uint32_t pairs = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(0, 1, 16);
    for (uint32_t pair = 0; pair < pairs; ++pair) {
        matmul_init(0, 1);
        tile_regs_acquire();
        for (uint32_t k = 0; k < 160; ++k) {
            cb_wait_front(0, 1);
            cb_wait_front(1, 2);
            matmul_tiles(0, 1, 0, 0, 0);
            matmul_tiles(0, 1, 0, 1, 1);
            cb_pop_front(0, 1);
            cb_pop_front(1, 2);
        }
        // Preserve the composed path's two BF16 rounding boundaries without
        // packing/reloading the activated gate and up projection intermediates.
        silu_tile_init();
        silu_tile(0);
        constexpr uint32_t fp32 = static_cast<uint32_t>(DataFormat::Float32);
        constexpr uint32_t bf16 = static_cast<uint32_t>(DataFormat::Float16_b);
        typecast_tile_init<fp32, bf16>();
        // The original matmul's native BF16 pack rounds ties away, whereas
        // the public SFPU cast defaults to ties-even. Keep the exact pinned
        // typecast instruction chain but select its native-pack bias:
        // ((word + 0x8000) & 0xffff0000), without the retained-LSB term.
        MATH((sfpi::vConstIntPrgm0 = 0));
        MATH((sfpi::vConstIntPrgm1 = 0x8000));
        typecast_tile<fp32, bf16>(0);
        typecast_tile<fp32, bf16>(1);
        mul_binary_tile_init();
        // Deliberately BF16 product semantics in FP32 Dst: public
        // mul_binary_tile would select DST_ACCUM_MODE=true and skip narrowing.
        MATH((SFPU_BINARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sfpu_binary_mul,
            (APPROX, ckernel::BinaryOp::MUL, 8, false), 0, 1, 0, VectorMode::RC)));
        tile_regs_commit();
        cb_reserve_back(16, 1);
        tile_regs_wait();
        pack_tile(0, 16);
        tile_regs_release();
        cb_push_back(16, 1);
    }
}
