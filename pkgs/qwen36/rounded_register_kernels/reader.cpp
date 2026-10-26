// SPDX-License-Identifier: LGPL-2.1-only
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t gate_addr = get_arg_val<uint32_t>(1);
    const uint32_t up_addr = get_arg_val<uint32_t>(2);
    uint32_t tile = get_arg_val<uint32_t>(3);
    const uint32_t count = get_arg_val<uint32_t>(4);
    const uint32_t stride = get_arg_val<uint32_t>(5);
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto gate_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto up_args = TensorAccessorArgs<gate_args.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(x_args, x_addr, get_tile_size(0));
    const auto gate = TensorAccessor(gate_args, gate_addr, get_tile_size(1));
    const auto up = TensorAccessor(up_args, up_addr, get_tile_size(1));
    for (uint32_t pair = 0; pair < count; ++pair, tile += stride) {
        for (uint32_t k = 0; k < 160; ++k) {
            cb_reserve_back(0, 1);
            cb_reserve_back(1, 2);
            noc_async_read_tile(k, x, get_write_ptr(0));
            noc_async_read_tile(k * 272 + tile, gate, get_write_ptr(1));
            noc_async_read_tile(k * 272 + tile, up, get_write_ptr(1) + get_tile_size(1));
            noc_async_read_barrier();
            cb_push_back(0, 1);
            cb_push_back(1, 2);
        }
    }
}
