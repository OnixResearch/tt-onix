#include <cstdint>

#if !defined(ARCH_BLACKHOLE)
#error "fused RWKV-7 decay diagnostic writer supports Blackhole only"
#endif

namespace {

constexpr std::uint32_t kHeadTileCount = 2U;
constexpr std::uint32_t kStateTileCount = kHeadTileCount * kHeadTileCount;
constexpr std::uint32_t kDecayOutputCb = static_cast<std::uint32_t>(tt::CBIndex::c_16);

constexpr std::uint32_t kOutputAddressArg = 0U;
constexpr std::uint32_t kCompletionSemaphoreIdArg = 1U;
constexpr std::uint32_t kWorkCountArg = 2U;
constexpr std::uint32_t kWorkListStartArg = 3U;
constexpr std::uint32_t kOutputAccessorArgOffset = 0U;

}  // namespace

void kernel_main() {
    const std::uint32_t output_address = get_arg_val<std::uint32_t>(kOutputAddressArg);
    const std::uint32_t completion_semaphore_id = get_arg_val<std::uint32_t>(kCompletionSemaphoreIdArg);
    const std::uint32_t work_count = get_arg_val<std::uint32_t>(kWorkCountArg);
    constexpr auto output_accessor_args = TensorAccessorArgs<kOutputAccessorArgOffset>();
    const auto output_accessor = TensorAccessor(output_accessor_args, output_address);
    const std::uint32_t tile_bytes = get_tile_size(kDecayOutputCb);

    std::uint32_t completed_count = 0U;
    for (std::uint32_t work_index = 0U; work_index < work_count; ++work_index) {
        const std::uint32_t work_item = get_arg_val<std::uint32_t>(kWorkListStartArg + work_index);
        cb_wait_front(kDecayOutputCb, kStateTileCount);
        std::uint32_t read_address = get_read_ptr(kDecayOutputCb);
        const std::uint32_t first_page = work_item * kStateTileCount;
        for (std::uint32_t tile = 0U; tile < kStateTileCount; ++tile) {
            noc_async_write_page(first_page + tile, output_accessor, read_address);
            read_address += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(kDecayOutputCb, kStateTileCount);
        ++completed_count;
    }

    auto* completion_semaphore =
        reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(get_semaphore(completion_semaphore_id));
    noc_semaphore_set(completion_semaphore, completed_count);
}
