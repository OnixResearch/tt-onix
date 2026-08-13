#include <cstdint>

#if !defined(ARCH_BLACKHOLE)
#error "fused RWKV-7 WKV writer supports Blackhole only"
#endif

namespace {

constexpr std::uint32_t kHeadTileCount = 2;
constexpr std::uint32_t kStateTileCount =
    kHeadTileCount * kHeadTileCount;
constexpr std::uint32_t kOutputCb =
    static_cast<std::uint32_t>(tt::CBIndex::c_16);
constexpr std::uint32_t kNextStateOutputCb =
    static_cast<std::uint32_t>(tt::CBIndex::c_17);
constexpr std::uint32_t kSplitNextStateCb = static_cast<std::uint32_t>(tt::CBIndex::c_9);

constexpr std::uint32_t kOutputAddressArg = 0;
constexpr std::uint32_t kStateAddressArg = 1;
constexpr std::uint32_t kCompletionSemaphoreIdArg = 2;
constexpr std::uint32_t kSplitReadoutArg = 3;
constexpr std::uint32_t kWorkCountArg = 4;
constexpr std::uint32_t kWorkListStartArg = 5;

constexpr std::uint32_t kOutputAccessorArgOffset = 0;
constexpr std::uint32_t kOutputAccessorArgCount =
    TensorAccessorArgs<kOutputAccessorArgOffset>::num_compile_time_args();
constexpr std::uint32_t kStateAccessorArgOffset =
    kOutputAccessorArgOffset + kOutputAccessorArgCount;

template <typename AccessorArgs>
void write_tiles(
    std::uint32_t cb,
    std::uint32_t tile_count,
    std::uint32_t first_page,
    std::uint32_t address,
    const AccessorArgs &accessor_args) {
  const auto accessor = TensorAccessor(accessor_args, address);
  const std::uint32_t tile_bytes = get_tile_size(cb);

  cb_wait_front(cb, tile_count);
  std::uint32_t read_address = get_read_ptr(cb);
  for (std::uint32_t tile = 0; tile < tile_count; ++tile) {
    noc_async_write_page(first_page + tile, accessor, read_address);
    read_address += tile_bytes;
  }
  noc_async_write_barrier();
  cb_pop_front(cb, tile_count);
}

} // namespace

void kernel_main() {
  const std::uint32_t output_address =
      get_arg_val<std::uint32_t>(kOutputAddressArg);
  const std::uint32_t next_state_address =
      get_arg_val<std::uint32_t>(kStateAddressArg);
  const std::uint32_t completion_semaphore_id =
      get_arg_val<std::uint32_t>(kCompletionSemaphoreIdArg);
  const bool split_readout = get_arg_val<std::uint32_t>(kSplitReadoutArg) == 1U;
  const std::uint32_t work_count =
      get_arg_val<std::uint32_t>(kWorkCountArg);

  constexpr auto output_accessor_args =
      TensorAccessorArgs<kOutputAccessorArgOffset>();
  constexpr auto state_accessor_args =
      TensorAccessorArgs<kStateAccessorArgOffset>();

  std::uint32_t completed_count = 0;
  for (std::uint32_t work_index = 0; work_index < work_count;
       ++work_index) {
    const std::uint32_t work_item = get_arg_val<std::uint32_t>(
        kWorkListStartArg + work_index);
    if (!split_readout) {
        write_tiles(kOutputCb, kHeadTileCount, work_item * kHeadTileCount, output_address, output_accessor_args);
    }
    write_tiles(
        split_readout ? kSplitNextStateCb : kNextStateOutputCb,
        kStateTileCount,
        work_item * kStateTileCount,
        next_state_address,
        state_accessor_args);
    ++completed_count;
  }

  // Each write_tiles call completes its NoC writes before this signal.
  auto *completion_semaphore =
      reinterpret_cast<volatile tt_l1_ptr std::uint32_t *>(
          get_semaphore(completion_semaphore_id));
  noc_semaphore_set(completion_semaphore, completed_count);
}
