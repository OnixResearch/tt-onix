#include <cstdint>

#if !defined(ARCH_BLACKHOLE)
#error "fused RWKV-7 BF16 readout diagnostic reader supports Blackhole only"
#endif

namespace {

constexpr std::uint32_t kHeadTileCount = 2U;
constexpr std::uint32_t kStateTileCount = kHeadTileCount * kHeadTileCount;
constexpr std::uint32_t kTileFaceCount = 4U;
constexpr std::uint32_t kTopFaceCount = 2U;
constexpr std::uint32_t kFaceHeight = 16U;
constexpr std::uint32_t kFaceWidth = 16U;
constexpr std::uint32_t kBf16Bytes = 2U;
constexpr std::uint32_t kL1WordBytes = sizeof(std::uint32_t);
constexpr std::uint32_t kWordsPerFace = kFaceHeight * kFaceWidth * kBf16Bytes / kL1WordBytes;
constexpr std::uint32_t kWordsPerFaceRow = kFaceWidth * kBf16Bytes / kL1WordBytes;

constexpr std::uint32_t kStateCb = static_cast<std::uint32_t>(tt::CBIndex::c_0);
constexpr std::uint32_t kRawRCb = static_cast<std::uint32_t>(tt::CBIndex::c_1);
constexpr std::uint32_t kRAddressArg = 0U;
constexpr std::uint32_t kStateAddressArg = 1U;
constexpr std::uint32_t kWorkCountArg = 2U;
constexpr std::uint32_t kWorkListStartArg = 3U;
constexpr std::uint32_t kRAccessorArgOffset = 0U;
constexpr std::uint32_t kRAccessorArgCount = TensorAccessorArgs<kRAccessorArgOffset>::num_compile_time_args();
constexpr std::uint32_t kStateAccessorArgOffset = kRAccessorArgOffset + kRAccessorArgCount;

void clear_vector_padding(std::uint32_t tile_address) {
    auto* tile_words = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(tile_address);
    for (std::uint32_t face = 0U; face < kTileFaceCount; ++face) {
        const std::uint32_t face_start = face * kWordsPerFace;
        const std::uint32_t clear_start = face < kTopFaceCount ? face_start + kWordsPerFaceRow : face_start;
        const std::uint32_t clear_end = face_start + kWordsPerFace;
        for (std::uint32_t word = clear_start; word < clear_end; ++word) {
            tile_words[word] = 0U;
        }
    }
}

template <typename AccessorArgs>
void read_r(std::uint32_t address, std::uint32_t work_item, const AccessorArgs& accessor_args) {
    const auto accessor = TensorAccessor(accessor_args, address);
    const std::uint32_t tile_bytes = get_tile_size(kRawRCb);
    const std::uint32_t first_page = work_item * kHeadTileCount;

    cb_reserve_back(kRawRCb, kHeadTileCount);
    const std::uint32_t first_write_address = get_write_ptr(kRawRCb);
    std::uint32_t write_address = first_write_address;
    for (std::uint32_t tile = 0U; tile < kHeadTileCount; ++tile) {
        noc_async_read_page(first_page + tile, accessor, write_address);
        write_address += tile_bytes;
    }
    noc_async_read_barrier();
    for (std::uint32_t tile = 0U; tile < kHeadTileCount; ++tile) {
        clear_vector_padding(first_write_address + tile * tile_bytes);
    }
    cb_push_back(kRawRCb, kHeadTileCount);
}

template <typename AccessorArgs>
void read_state(std::uint32_t address, std::uint32_t work_item, const AccessorArgs& accessor_args) {
    const auto accessor = TensorAccessor(accessor_args, address);
    const std::uint32_t tile_bytes = get_tile_size(kStateCb);
    const std::uint32_t first_page = work_item * kStateTileCount;

    cb_reserve_back(kStateCb, kStateTileCount);
    std::uint32_t write_address = get_write_ptr(kStateCb);
    for (std::uint32_t tile = 0U; tile < kStateTileCount; ++tile) {
        noc_async_read_page(first_page + tile, accessor, write_address);
        write_address += tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(kStateCb, kStateTileCount);
}

}  // namespace

void kernel_main() {
    const std::uint32_t r_address = get_arg_val<std::uint32_t>(kRAddressArg);
    const std::uint32_t state_address = get_arg_val<std::uint32_t>(kStateAddressArg);
    const std::uint32_t work_count = get_arg_val<std::uint32_t>(kWorkCountArg);
    constexpr auto r_accessor_args = TensorAccessorArgs<kRAccessorArgOffset>();
    constexpr auto state_accessor_args = TensorAccessorArgs<kStateAccessorArgOffset>();

    for (std::uint32_t work_index = 0U; work_index < work_count; ++work_index) {
        const std::uint32_t work_item = get_arg_val<std::uint32_t>(kWorkListStartArg + work_index);
        read_state(state_address, work_item, state_accessor_args);
        read_r(r_address, work_item, r_accessor_args);
    }
}
