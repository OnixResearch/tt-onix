#include <cstdint>

#if !defined(ARCH_BLACKHOLE)
#error "fused RWKV-7 WKV reader supports Blackhole only"
#endif

namespace {

constexpr std::uint32_t kHeadCount = 12;
constexpr std::uint32_t kHeadTileCount = 2;
constexpr std::uint32_t kStateTileCount =
    kHeadTileCount * kHeadTileCount;
constexpr std::uint32_t kRecurrenceVectorCount = 6;
constexpr std::uint32_t kSharedVectorCount = 3;
constexpr std::uint32_t kPerWorkCorrectionVectorCount = 1;
constexpr std::uint32_t kVectorCount =
    kRecurrenceVectorCount + kSharedVectorCount +
    kPerWorkCorrectionVectorCount;
constexpr std::uint32_t kTileFaceCount = 4;
constexpr std::uint32_t kTopFaceCount = 2;
constexpr std::uint32_t kFaceHeight = 16;
constexpr std::uint32_t kFaceWidth = 16;
constexpr std::uint32_t kBf16Bytes = 2;
constexpr std::uint32_t kL1WordBytes = sizeof(std::uint32_t);
constexpr std::uint32_t kWordsPerFace =
    kFaceHeight * kFaceWidth * kBf16Bytes / kL1WordBytes;
constexpr std::uint32_t kWordsPerFaceRow =
    kFaceWidth * kBf16Bytes / kL1WordBytes;

constexpr std::uint32_t kACb = static_cast<std::uint32_t>(tt::CBIndex::c_0);
constexpr std::uint32_t kWCb = static_cast<std::uint32_t>(tt::CBIndex::c_1);
constexpr std::uint32_t kKCb = static_cast<std::uint32_t>(tt::CBIndex::c_2);
constexpr std::uint32_t kVCb = static_cast<std::uint32_t>(tt::CBIndex::c_3);
constexpr std::uint32_t kRCb = static_cast<std::uint32_t>(tt::CBIndex::c_4);
constexpr std::uint32_t kBCb = static_cast<std::uint32_t>(tt::CBIndex::c_5);
constexpr std::uint32_t kStateCb =
    static_cast<std::uint32_t>(tt::CBIndex::c_6);
constexpr std::uint32_t kRkCb =
    static_cast<std::uint32_t>(tt::CBIndex::c_7);
constexpr std::uint32_t kGroupNormWeightCb =
    static_cast<std::uint32_t>(tt::CBIndex::c_11);
constexpr std::uint32_t kGroupNormBiasCb =
    static_cast<std::uint32_t>(tt::CBIndex::c_20);
constexpr std::uint32_t kGateCb =
    static_cast<std::uint32_t>(tt::CBIndex::c_21);

constexpr std::uint32_t kAAddressArg = 0;
constexpr std::uint32_t kWAddressArg = 1;
constexpr std::uint32_t kKAddressArg = 2;
constexpr std::uint32_t kVAddressArg = 3;
constexpr std::uint32_t kRAddressArg = 4;
constexpr std::uint32_t kBAddressArg = 5;
constexpr std::uint32_t kRkAddressArg = 6;
constexpr std::uint32_t kGroupNormWeightAddressArg = 7;
constexpr std::uint32_t kGroupNormBiasAddressArg = 8;
constexpr std::uint32_t kGateAddressArg = 9;
constexpr std::uint32_t kStateAddressArg = 10;
constexpr std::uint32_t kWorkCountArg = 11;
constexpr std::uint32_t kWorkListStartArg = 12;

constexpr std::uint32_t kVectorAccessorArgOffset = 0;
constexpr std::uint32_t kVectorAccessorArgCount =
    TensorAccessorArgs<kVectorAccessorArgOffset>::num_compile_time_args();
constexpr std::uint32_t kStateAccessorArgOffset =
    kVectorAccessorArgOffset + kVectorAccessorArgCount;

void clear_vector_padding(std::uint32_t tile_address) {
  auto *tile_words =
      reinterpret_cast<volatile tt_l1_ptr std::uint32_t *>(tile_address);
  for (std::uint32_t face = 0; face < kTileFaceCount; ++face) {
    const std::uint32_t face_start = face * kWordsPerFace;
    const std::uint32_t clear_start =
        face < kTopFaceCount ? face_start + kWordsPerFaceRow : face_start;
    const std::uint32_t clear_end = face_start + kWordsPerFace;
    for (std::uint32_t word = clear_start; word < clear_end; ++word) {
      tile_words[word] = 0U;
    }
  }
}

template <typename AccessorArgs>
void read_vector(
    std::uint32_t cb,
    std::uint32_t address,
    std::uint32_t work_item,
    const AccessorArgs &accessor_args) {
  const auto accessor = TensorAccessor(accessor_args, address);
  const std::uint32_t tile_bytes = get_tile_size(cb);
  const std::uint32_t first_page = work_item * kHeadTileCount;

  cb_reserve_back(cb, kHeadTileCount);
  const std::uint32_t first_write_address = get_write_ptr(cb);
  std::uint32_t write_address = first_write_address;
  for (std::uint32_t tile = 0; tile < kHeadTileCount; ++tile) {
    noc_async_read_page(first_page + tile, accessor, write_address);
    write_address += tile_bytes;
  }
  noc_async_read_barrier();
  for (std::uint32_t tile = 0; tile < kHeadTileCount; ++tile) {
    clear_vector_padding(first_write_address + tile * tile_bytes);
  }
  cb_push_back(cb, kHeadTileCount);
}

template <typename AccessorArgs>
void read_state(
    std::uint32_t state_address,
    std::uint32_t work_item,
    const AccessorArgs &accessor_args) {
  const auto accessor = TensorAccessor(accessor_args, state_address);
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

} // namespace

void kernel_main() {
  const std::uint32_t vector_addresses[kVectorCount] = {
      get_arg_val<std::uint32_t>(kAAddressArg),
      get_arg_val<std::uint32_t>(kWAddressArg),
      get_arg_val<std::uint32_t>(kKAddressArg),
      get_arg_val<std::uint32_t>(kVAddressArg),
      get_arg_val<std::uint32_t>(kRAddressArg),
      get_arg_val<std::uint32_t>(kBAddressArg),
      get_arg_val<std::uint32_t>(kRkAddressArg),
      get_arg_val<std::uint32_t>(kGroupNormWeightAddressArg),
      get_arg_val<std::uint32_t>(kGroupNormBiasAddressArg),
      get_arg_val<std::uint32_t>(kGateAddressArg),
  };
  constexpr std::uint32_t vector_cbs[kVectorCount] = {
      kACb,
      kWCb,
      kKCb,
      kVCb,
      kRCb,
      kBCb,
      kRkCb,
      kGroupNormWeightCb,
      kGroupNormBiasCb,
      kGateCb,
  };
  constexpr bool shared_across_batch[kVectorCount] = {
      false,
      false,
      false,
      false,
      false,
      false,
      true,
      true,
      true,
      false,
  };
  const std::uint32_t state_address =
      get_arg_val<std::uint32_t>(kStateAddressArg);
  const std::uint32_t work_count =
      get_arg_val<std::uint32_t>(kWorkCountArg);

  constexpr auto vector_accessor_args =
      TensorAccessorArgs<kVectorAccessorArgOffset>();
  constexpr auto state_accessor_args =
      TensorAccessorArgs<kStateAccessorArgOffset>();

  for (std::uint32_t work_index = 0; work_index < work_count;
       ++work_index) {
    const std::uint32_t work_item = get_arg_val<std::uint32_t>(
        kWorkListStartArg + work_index);
    read_state(state_address, work_item, state_accessor_args);
    for (std::uint32_t vector = 0; vector < kVectorCount; ++vector) {
      const std::uint32_t vector_work_item =
          shared_across_batch[vector] ? work_item % kHeadCount : work_item;
      read_vector(
          vector_cbs[vector],
          vector_addresses[vector],
          vector_work_item,
          vector_accessor_args);
    }
  }
}
