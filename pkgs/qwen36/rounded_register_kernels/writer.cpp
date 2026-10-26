// SPDX-License-Identifier: LGPL-2.1-only
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t address = get_arg_val<uint32_t>(0);
    uint32_t tile = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    const uint32_t stride = get_arg_val<uint32_t>(3);
    constexpr auto args = TensorAccessorArgs<0>();
    const auto output = TensorAccessor(args, address, get_tile_size(16));
    for (uint32_t i = 0; i < count; ++i, tile += stride) {
        cb_wait_front(16, 1);
        noc_async_write_tile(tile, output, get_read_ptr(16));
        noc_async_write_barrier();
        cb_pop_front(16, 1);
    }
}
