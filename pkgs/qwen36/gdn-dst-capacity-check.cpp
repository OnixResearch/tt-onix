// Host-only execution of the real kernel body against a bounded symbolic API.
// This checks Dst indices and per-tile expression order, not hardware numerics,
// CB/NoC ordering, or concurrent TRISC behavior.
#include <array>
#include <cstdint>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

#define TENSIX_FIRMWARE
#include "tensix_types.h"
#include "tt-metalium/constants.hpp"

constexpr uint32_t kHeadTiles = 4;
constexpr uint32_t kStateTiles = kHeadTiles * kHeadTiles;
constexpr uint32_t kTileRows =
    tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH / DEST_FACE_WIDTH;
constexpr uint32_t kDstCapacity = BIT32_DEST_REGISTER_HALF_SIZE / kTileRows;
constexpr uint32_t kB8InstancesPerCore = 2;
constexpr uint32_t kB16InstancesPerCore = 4;
constexpr auto kCapacityError = "Dst index exceeds configured capacity";
static_assert(kDstCapacity == kHeadTiles);

namespace tt {
struct CBIndex {
  enum : uint32_t {
    c_0,
    c_1,
    c_2,
    c_3,
    c_4,
    c_5,
    c_6,
    c_7,
    c_8,
    c_9,
    c_10,
    c_11,
    c_12,
    c_13,
    c_16 = 16
  };
};
} // namespace tt
namespace ckernel {
enum class DstTileShape { Tile32x32 };
enum class DstSync { SyncHalf };
template <DstSync, bool, DstTileShape> constexpr uint32_t get_dest_max_tiles() {
  return kDstCapacity;
}
} // namespace ckernel
#define DST_SYNC_MODE ckernel::DstSync::SyncHalf
#define DST_ACCUM_MODE true

enum class Stage { Released, Math, Committed, Pack };
using Tiles = std::map<std::pair<uint32_t, uint32_t>, std::string>;
struct State {
  uint32_t capacity = kDstCapacity;
  uint32_t instances = 1;
  Stage stage = Stage::Released;
  std::array<std::string, kDstCapacity> dst{};
  Tiles tiles;
};
State state;
void require(bool condition, const char *diagnostic) {
  if (!condition) {
    throw std::runtime_error(diagnostic);
  }
}
constexpr uint32_t get_compile_time_arg_val(uint32_t index) {
  if (index > 1) {
    throw std::runtime_error("unknown compile argument");
  }
  return kHeadTiles;
}
template <typename T> T get_arg_val(uint32_t index) {
  require(index == 0, "unknown runtime argument");
  return static_cast<T>(state.instances);
}
struct CircularBuffer {
  explicit CircularBuffer(uint32_t) {}
  void reserve_back(uint32_t) {}
  void wait_front(uint32_t) {}
  void push_back(uint32_t) {}
  void pop_front(uint32_t) {}
};
enum class SrcOrder { Reverse };
template <SrcOrder>
void compute_kernel_hw_startup(uint32_t, uint32_t, uint32_t) {}
void init_sfpu(uint32_t, uint32_t) {}
void copy_tile_to_dst_init_short(uint32_t) {}
template <bool> void exp_tile_init() {}
void mul_tiles_bcast_scalar_init_short(uint32_t, uint32_t) {}
void matmul_init(uint32_t, uint32_t) {}
void sub_tiles_init(uint32_t, uint32_t) {}
void add_tiles_init(uint32_t, uint32_t) {}
void transpose_init(uint32_t) {}

std::string &destination(uint32_t tile) {
  require(tile < state.capacity && tile < state.dst.size(), kCapacityError);
  return state.dst.at(tile);
}
const std::string &input(uint32_t cb, uint32_t tile) {
  return state.tiles.at({cb, tile});
}
void store(uint32_t dst, std::string expression) {
  require(state.stage == Stage::Math, "math outside acquired Dst");
  destination(dst) = std::move(expression);
}
void tile_regs_acquire() {
  require(state.stage == Stage::Released, "nested Dst acquire");
  state.dst.fill("");
  state.stage = Stage::Math;
}
void tile_regs_commit() {
  require(state.stage == Stage::Math, "commit outside math stage");
  state.stage = Stage::Committed;
}
void tile_regs_wait() {
  require(state.stage == Stage::Committed, "wait before commit");
  state.stage = Stage::Pack;
}
void pack_tile(uint32_t dst, uint32_t cb, uint32_t tile) {
  require(state.stage == Stage::Pack, "pack before wait");
  const auto &value = destination(dst);
  require(!value.empty(), "pack unwritten Dst");
  state.tiles[{cb, tile}] = value;
}
void tile_regs_release() {
  require(state.stage == Stage::Pack, "release before pack stage");
  state.stage = Stage::Released;
}
void copy_tile(uint32_t cb, uint32_t tile, uint32_t dst) {
  store(dst, input(cb, tile));
}
template <bool> void exp_tile(uint32_t dst) {
  store(dst, "exp(" + destination(dst) + ")");
}
void transpose_tile(uint32_t cb, uint32_t tile, uint32_t dst) {
  store(dst, "transpose(" + input(cb, tile) + ")");
}
void binary(const char *operation, uint32_t a, uint32_t b, uint32_t ai,
            uint32_t bi, uint32_t dst) {
  store(dst,
        std::string(operation) + "(" + input(a, ai) + "," + input(b, bi) + ")");
}
void mul_tiles_bcast_scalar(uint32_t a, uint32_t b, uint32_t ai, uint32_t bi,
                            uint32_t dst) {
  binary("multiply", a, b, ai, bi, dst);
}
void sub_tiles(uint32_t a, uint32_t b, uint32_t ai, uint32_t bi, uint32_t dst) {
  binary("subtract", a, b, ai, bi, dst);
}
void add_tiles(uint32_t a, uint32_t b, uint32_t ai, uint32_t bi, uint32_t dst) {
  binary("add", a, b, ai, bi, dst);
}
void matmul_tiles(uint32_t a, uint32_t b, uint32_t ai, uint32_t bi,
                  uint32_t dst) {
  const auto prior = destination(dst);
  binary("matmul", a, b, ai, bi, dst);
  if (!prior.empty()) {
    store(dst, "accumulate(" + prior + "," + destination(dst) + ")");
  }
}

namespace control {
#include GDN_CONTROL_KERNEL
}
namespace candidate {
#include GDN_CANDIDATE_KERNEL
}

Tiles run(void (*kernel)(), uint32_t instances, uint32_t capacity) {
  state = State{};
  state.capacity = capacity;
  state.instances = instances;
  for (const auto cb : {tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_2,
                        tt::CBIndex::c_3, tt::CBIndex::c_4, tt::CBIndex::c_5}) {
    const uint32_t tiles = cb == tt::CBIndex::c_5 ? kStateTiles : kHeadTiles;
    for (uint32_t tile = 0; tile < tiles; ++tile) {
      state.tiles[{cb, tile}] =
          "input" + std::to_string(cb) + ":" + std::to_string(tile);
    }
  }
  kernel();
  require(state.stage == Stage::Released, "kernel retained Dst");
  return state.tiles;
}
int main() {
  try {
    for (const uint32_t instances :
         {1U, kB8InstancesPerCore, kB16InstancesPerCore}) {
      const auto reference = run(control::kernel_main, instances, kDstCapacity);
      const auto observed =
          run(candidate::kernel_main, instances, kDstCapacity);
      require(observed == reference,
              "per-tile expression or accumulation order changed");
    }
    bool rejected = false;
    try {
      (void)run(candidate::kernel_main, 1, kDstCapacity - 1);
    } catch (const std::runtime_error &error) {
      require(error.what() == std::string(kCapacityError),
              "wrong negative diagnostic");
      rejected = true;
    }
    require(rejected, "undersized Dst fixture was accepted");
    std::cout << "PASS GDN Dst bounds and symbolic tile order; capacity="
              << kDstCapacity << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
