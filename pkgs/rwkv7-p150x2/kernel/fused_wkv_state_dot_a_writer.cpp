#include <cstdint>

#if !defined(ARCH_BLACKHOLE)
#error "fused RWKV-7 state_dot_a diagnostic writer supports Blackhole only"
#endif

namespace {

constexpr std::uint32_t kHeadTileCount = 2;
constexpr std::uint32_t kStateDotAOutputCb = static_cast<std::uint32_t>(tt::CBIndex::c_16);

constexpr std::uint32_t kOutputAddressArg = 0;
constexpr std::uint32_t kCompletionSemaphoreIdArg = 1;
constexpr std::uint32_t kWorkCountArg = 2;
constexpr std::uint32_t kWorkListStartArg = 3;
constexpr std::uint32_t kOutputAccessorArgOffset = 0;

}  // namespace

void kernel_main() {
    const std::uint32_t output_address = get_arg_val<std::uint32_t>(kOutputAddressArg);
    const std::uint32_t completion_semaphore_id = get_arg_val<std::uint32_t>(kCompletionSemaphoreIdArg);
    const std::uint32_t work_count = get_arg_val<std::uint32_t>(kWorkCountArg);
    constexpr auto output_accessor_args = TensorAccessorArgs<kOutputAccessorArgOffset>();
    const auto output_accessor = TensorAccessor(output_accessor_args, output_address);
    const std::uint32_t tile_bytes = get_tile_size(kStateDotAOutputCb);

    std::uint32_t completed_count = 0;
    for (std::uint32_t work_index = 0; work_index < work_count; ++work_index) {
        const std::uint32_t work_item = get_arg_val<std::uint32_t>(kWorkListStartArg + work_index);
        cb_wait_front(kStateDotAOutputCb, kHeadTileCount);
        std::uint32_t read_address = get_read_ptr(kStateDotAOutputCb);
        const std::uint32_t first_page = work_item * kHeadTileCount;
        for (std::uint32_t tile = 0; tile < kHeadTileCount; ++tile) {
            noc_async_write_page(first_page + tile, output_accessor, read_address);
            read_address += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(kStateDotAOutputCb, kHeadTileCount);
        ++completed_count;
    }

    auto* completion_semaphore =
        reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(get_semaphore(completion_semaphore_id));
    noc_semaphore_set(completion_semaphore, completed_count);
}
