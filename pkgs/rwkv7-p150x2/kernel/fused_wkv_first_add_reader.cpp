#include <cstdint>

#if !defined(ARCH_BLACKHOLE)
#error "fused RWKV-7 first_add diagnostic reader supports Blackhole only"
#endif

namespace {

constexpr std::uint32_t kHeadTileCount = 2U;
constexpr std::uint32_t kStateTileCount = kHeadTileCount * kHeadTileCount;
constexpr std::uint32_t kLhsCb = static_cast<std::uint32_t>(tt::CBIndex::c_0);
constexpr std::uint32_t kRhsCb = static_cast<std::uint32_t>(tt::CBIndex::c_1);
constexpr std::uint32_t kLhsAddressArg = 0U;
constexpr std::uint32_t kRhsAddressArg = 1U;
constexpr std::uint32_t kWorkCountArg = 2U;
constexpr std::uint32_t kWorkListStartArg = 3U;
constexpr std::uint32_t kAccessorArgOffset = 0U;

template <typename AccessorArgs>
void read_matrix(std::uint32_t cb, std::uint32_t address, std::uint32_t work_item, const AccessorArgs& accessor_args) {
    const auto accessor = TensorAccessor(accessor_args, address);
    const std::uint32_t tile_bytes = get_tile_size(cb);
    const std::uint32_t first_page = work_item * kStateTileCount;

    cb_reserve_back(cb, kStateTileCount);
    std::uint32_t write_address = get_write_ptr(cb);
    for (std::uint32_t tile = 0U; tile < kStateTileCount; ++tile) {
        noc_async_read_page(first_page + tile, accessor, write_address);
        write_address += tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb, kStateTileCount);
}

}  // namespace

void kernel_main() {
    const std::uint32_t lhs_address = get_arg_val<std::uint32_t>(kLhsAddressArg);
    const std::uint32_t rhs_address = get_arg_val<std::uint32_t>(kRhsAddressArg);
    const std::uint32_t work_count = get_arg_val<std::uint32_t>(kWorkCountArg);
    constexpr auto accessor_args = TensorAccessorArgs<kAccessorArgOffset>();

    for (std::uint32_t work_index = 0U; work_index < work_count; ++work_index) {
        const std::uint32_t work_item = get_arg_val<std::uint32_t>(kWorkListStartArg + work_index);
        read_matrix(kLhsCb, lhs_address, work_item, accessor_args);
        read_matrix(kRhsCb, rhs_address, work_item, accessor_args);
    }
}
