#include <cstdint>

#if !defined(ARCH_BLACKHOLE)
#error "fused RWKV-7 state_dot_a diagnostic reader supports Blackhole only"
#endif

namespace {

constexpr std::uint32_t kHeadTileCount = 2;
constexpr std::uint32_t kStateTileCount = kHeadTileCount * kHeadTileCount;
constexpr std::uint32_t kTileFaceCount = 4;
constexpr std::uint32_t kTopFaceCount = 2;
constexpr std::uint32_t kFaceHeight = 16;
constexpr std::uint32_t kFaceWidth = 16;
constexpr std::uint32_t kBf16Bytes = 2;
constexpr std::uint32_t kL1WordBytes = sizeof(std::uint32_t);
constexpr std::uint32_t kWordsPerFace = kFaceHeight * kFaceWidth * kBf16Bytes / kL1WordBytes;
constexpr std::uint32_t kWordsPerFaceRow = kFaceWidth * kBf16Bytes / kL1WordBytes;

constexpr std::uint32_t kRawACb = static_cast<std::uint32_t>(tt::CBIndex::c_0);
constexpr std::uint32_t kStateCb = static_cast<std::uint32_t>(tt::CBIndex::c_1);

constexpr std::uint32_t kAAddressArg = 0;
constexpr std::uint32_t kStateAddressArg = 1;
constexpr std::uint32_t kWorkCountArg = 2;
constexpr std::uint32_t kWorkListStartArg = 3;

constexpr std::uint32_t kAAccessorArgOffset = 0;
constexpr std::uint32_t kAAccessorArgCount = TensorAccessorArgs<kAAccessorArgOffset>::num_compile_time_args();
constexpr std::uint32_t kStateAccessorArgOffset = kAAccessorArgOffset + kAAccessorArgCount;

void clear_vector_padding(std::uint32_t tile_address) {
    auto* tile_words = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(tile_address);
    for (std::uint32_t face = 0; face < kTileFaceCount; ++face) {
        const std::uint32_t face_start = face * kWordsPerFace;
        const std::uint32_t clear_start = face < kTopFaceCount ? face_start + kWordsPerFaceRow : face_start;
        const std::uint32_t clear_end = face_start + kWordsPerFace;
        for (std::uint32_t word = clear_start; word < clear_end; ++word) {
            tile_words[word] = 0U;
        }
    }
}

template <typename AccessorArgs>
void read_a(std::uint32_t address, std::uint32_t work_item, const AccessorArgs& accessor_args) {
    const auto accessor = TensorAccessor(accessor_args, address);
    const std::uint32_t tile_bytes = get_tile_size(kRawACb);
    const std::uint32_t first_page = work_item * kHeadTileCount;

    cb_reserve_back(kRawACb, kHeadTileCount);
    const std::uint32_t first_write_address = get_write_ptr(kRawACb);
    std::uint32_t write_address = first_write_address;
    for (std::uint32_t tile = 0; tile < kHeadTileCount; ++tile) {
        noc_async_read_page(first_page + tile, accessor, write_address);
        write_address += tile_bytes;
    }
    noc_async_read_barrier();
    for (std::uint32_t tile = 0; tile < kHeadTileCount; ++tile) {
        clear_vector_padding(first_write_address + tile * tile_bytes);
    }
    cb_push_back(kRawACb, kHeadTileCount);
}

template <typename AccessorArgs>
void read_state(std::uint32_t address, std::uint32_t work_item, const AccessorArgs& accessor_args) {
    const auto accessor = TensorAccessor(accessor_args, address);
    const std::uint32_t tile_bytes = get_tile_size(kStateCb);
    const std::uint32_t first_page = work_item * kStateTileCount;

    cb_reserve_back(kStateCb, kStateTileCount);
    std::uint32_t write_address = get_write_ptr(kStateCb);
    for (std::uint32_t tile = 0; tile < kStateTileCount; ++tile) {
        noc_async_read_page(first_page + tile, accessor, write_address);
        write_address += tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(kStateCb, kStateTileCount);
}

}  // namespace

void kernel_main() {
    const std::uint32_t a_address = get_arg_val<std::uint32_t>(kAAddressArg);
    const std::uint32_t state_address = get_arg_val<std::uint32_t>(kStateAddressArg);
    const std::uint32_t work_count = get_arg_val<std::uint32_t>(kWorkCountArg);
    constexpr auto a_accessor_args = TensorAccessorArgs<kAAccessorArgOffset>();
    constexpr auto state_accessor_args = TensorAccessorArgs<kStateAccessorArgOffset>();

    for (std::uint32_t work_index = 0; work_index < work_count; ++work_index) {
        const std::uint32_t work_item = get_arg_val<std::uint32_t>(kWorkListStartArg + work_index);
        read_state(state_address, work_item, state_accessor_args);
        read_a(a_address, work_item, a_accessor_args);
    }
}
