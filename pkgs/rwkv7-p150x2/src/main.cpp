#include "config.hpp"

#include <blake3.h>
#include <nlohmann/json.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include <ttnn/distributed/api.hpp>
#include <ttnn/distributed/distributed_tensor.hpp>
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/copy/typecast/typecast.hpp>
#include <ttnn/operations/data_movement/transpose/transpose.hpp>
#include <ttnn/operations/data_movement/untilize/untilize.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/eltwise/binary/binary_composite.hpp>
#include <ttnn/operations/eltwise/ternary/ternary.hpp>
#include <ttnn/operations/eltwise/unary/unary.hpp>
#include <ttnn/operations/embedding/embedding.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/normalization/layernorm/layernorm.hpp>
#include <ttnn/operations/reduction/argmax/argmax.hpp>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/types.hpp>

#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

using Json = nlohmann::json;
using Tensor = ttnn::Tensor;
using MeshDevice = ttnn::distributed::MeshDevice;
using TensorToMesh = ttnn::distributed::TensorToMesh;

constexpr std::uint32_t kSchemaVersion = 1U;
constexpr std::uint32_t kHiddenSize = RWKV7_PINNED_HIDDEN_SIZE;
constexpr std::uint32_t kHeadSize = 64U;
constexpr std::uint32_t kHeadCount = 12U;
constexpr std::uint32_t kIntermediateSize = 3072U;
constexpr std::uint32_t kDecayRank = 64U;
constexpr std::uint32_t kAttentionRank = 64U;
constexpr std::uint32_t kGateRank = 128U;
constexpr std::uint32_t kValueRank = 32U;
constexpr std::uint32_t kLayerCount = RWKV7_PINNED_LAYER_COUNT;
constexpr std::uint32_t kVocabularySize = RWKV7_PINNED_VOCABULARY_SIZE;
constexpr std::uint32_t kLayerZero = 0U;
constexpr std::uint32_t kMinimumGeneratedTokens = 2U;
constexpr std::uint32_t kMaximumGeneratedTokens = 4U;
constexpr std::uint32_t kMaximumPromptTokens = RWKV7_PINNED_MAXIMUM_PROMPT_TOKENS;
constexpr std::uint32_t kGenerationEosToken = 0U;
constexpr std::uint32_t kMeshRows = 1U;
constexpr std::uint32_t kMeshColumns = 2U;
constexpr std::size_t kExpectedDeviceCount = 2U;
constexpr int kPhysicalDeviceZero = 0;
constexpr int kPhysicalDeviceOne = 1;
constexpr int kFilteredLogicalDevice = 0;
constexpr int kLastDimension = 3;
constexpr int kMatrixRowDimension = 2;
constexpr int kMatrixColumnDimension = 3;
constexpr float kLayerNormEpsilon = 1.0e-5F;
constexpr float kGroupNormEpsilon =
    static_cast<float>(kHeadSize) * kLayerNormEpsilon;
constexpr float kNormalizationFloor = 1.0e-12F;
constexpr float kDecayScale = -0.60653067F;
constexpr float kOne = 1.0F;
constexpr std::size_t kBf16Bytes = 2U;
constexpr std::size_t kSafetensorsHeaderLengthBytes = 8U;
constexpr std::size_t kBlake3DigestBytes = 32U;
constexpr std::size_t kHexCharactersPerByte = 2U;
constexpr std::uint32_t kHexRadix = 16U;
constexpr std::uint32_t kBitsPerByte = 8U;
constexpr std::uint64_t kExpectedCheckpointBytes = RWKV7_PINNED_CHECKPOINT_BYTES;
constexpr std::string_view kExpectedCheckpointBlake3 = RWKV7_PINNED_CHECKPOINT_BLAKE3;
constexpr std::string_view kTargetName = RWKV7_PINNED_TARGET_NAME;
constexpr std::string_view kModelRevision = RWKV7_PINNED_MODEL_REVISION;
constexpr std::string_view kModelId = RWKV7_PINNED_MODEL_ID;
constexpr std::string_view kArchitectureFamily = RWKV7_PINNED_ARCHITECTURE_FAMILY;
constexpr std::string_view kStateKind = RWKV7_PINNED_STATE_KIND;
constexpr std::string_view kModelPrecision = RWKV7_PINNED_MODEL_PRECISION;
constexpr std::string_view kArchitectureAdapter = RWKV7_PINNED_ARCHITECTURE_ADAPTER;
constexpr std::string_view kEvidenceAdapter = RWKV7_PINNED_EVIDENCE_ADAPTER;
constexpr std::string_view kTopologyId = RWKV7_PINNED_TOPOLOGY_ID;
constexpr std::string_view kReplication = RWKV7_PINNED_REPLICATION;
constexpr std::string_view kSelfTestArgument = "--self-test";
constexpr std::string_view kFastWorkerArgument = "--fast-worker";
constexpr std::string_view kSimulatorSmokeArgument = "--simulator-smoke";
constexpr std::string_view kFastDeviceArgument = "--device";
constexpr std::string_view kFastRequestsArgument = "--requests";
constexpr std::string_view kFastOutputArgument = "--output";
constexpr std::string_view kTokensArgument = "--tokens";
constexpr std::string_view kMaxNewArgument = "--max-new";
constexpr std::string_view kEvidenceRootArgument = "--evidence-root";
constexpr std::string_view kEnabledEnvironmentValue = "1";
constexpr std::string_view kDisabledEnvironmentValue = "0";
constexpr std::string_view kLogsDirectoryName = "logs";
constexpr std::string_view kProfilerDirectoryName = "profiler";
constexpr std::string_view kInspectorRelativeDirectory =
    "logs/generated/inspector";
constexpr std::string_view kProfilerCsvRelativePath =
    "profiler/.logs/profile_log_device.csv";
constexpr std::size_t kSelfTestArgumentCount = 2U;
constexpr std::size_t kRuntimeArgumentCount = 7U;
constexpr std::size_t kFastWorkerArgumentCount = 8U;
constexpr std::size_t kSimulatorSmokeArgumentCount = 4U;
constexpr std::size_t kSimulatorOutputNameIndex = 2U;
constexpr std::size_t kSimulatorOutputValueIndex = 3U;
constexpr std::size_t kFastDeviceNameIndex = 2U;
constexpr std::size_t kFastDeviceValueIndex = 3U;
constexpr std::size_t kFastRequestsNameIndex = 4U;
constexpr std::size_t kFastRequestsValueIndex = 5U;
constexpr std::size_t kFastOutputNameIndex = 6U;
constexpr std::size_t kFastOutputValueIndex = 7U;
constexpr std::size_t kFastPerCardSlotCount = 4U;
constexpr std::uint32_t kFastWarmupTokenCount = 4U;
constexpr std::uint32_t kFastMaximumGeneratedTokens = 64U;
constexpr std::size_t kProgramArgumentIndex = 0U;
constexpr std::size_t kModeArgumentIndex = 1U;
constexpr std::size_t kTokensValueIndex = 2U;
constexpr std::size_t kMaxNewNameIndex = 3U;
constexpr std::size_t kMaxNewValueIndex = 4U;
constexpr std::size_t kEvidenceNameIndex = 5U;
constexpr std::size_t kEvidenceValueIndex = 6U;
constexpr std::size_t kFirstIndex = 0U;
constexpr std::size_t kSecondIndex = 1U;
constexpr std::uint32_t kSelfTestTokenTwo = 2U;
constexpr std::uint32_t kSelfTestRows = 2U;
constexpr std::uint32_t kSelfTestColumns = 3U;
constexpr std::uint16_t kSelfTestBitOne = 1U;
constexpr std::uint16_t kSelfTestBitTwo = 2U;
constexpr std::uint16_t kSelfTestBitThree = 3U;
constexpr std::uint16_t kSelfTestBitFour = 4U;
constexpr std::uint16_t kSelfTestBitFive = 5U;
constexpr std::uint16_t kSelfTestBitSix = 6U;

static_assert(sizeof(bfloat16) == kBf16Bytes);
static_assert(std::is_trivially_copyable_v<bfloat16>);
static_assert(kHiddenSize == kHeadSize * kHeadCount);

[[noreturn]] void fail(std::string_view message) {
  throw std::runtime_error(std::string(message));
}

void require(bool condition, std::string_view message) {
  if (!condition) {
    fail(message);
  }
}

std::uint32_t parse_u32(std::string_view text, std::string_view name) {
  require(!text.empty(), std::string(name) + " must not be empty");
  std::uint32_t value = 0U;
  const char *const first = text.data();
  const char *const last = first + text.size();
  const auto [end, error] = std::from_chars(first, last, value);
  require(error == std::errc{} && end == last,
          std::string(name) + " must be an exact unsigned decimal");
  return value;
}

std::uint32_t parse_positive_u32(std::string_view text, std::string_view name) {
  const std::uint32_t value = parse_u32(text, name);
  require(value > 0U, std::string(name) + " must be positive");
  return value;
}

std::vector<std::uint32_t> parse_tokens(std::string_view text) {
  require(!text.empty(), "token list must not be empty");
  std::vector<std::uint32_t> tokens;
  std::size_t begin = 0U;
  while (begin < text.size()) {
    const std::size_t separator = text.find(',', begin);
    const std::size_t end =
        separator == std::string_view::npos ? text.size() : separator;
    const std::string_view field = text.substr(begin, end - begin);
    const std::uint32_t token = parse_u32(field, "token ID");
    require(token < kVocabularySize, "token ID exceeds the pinned vocabulary");
    tokens.push_back(token);
    require(tokens.size() <= kMaximumPromptTokens,
            "prompt token count exceeds the package cap");
    if (separator == std::string_view::npos) {
      break;
    }
    begin = separator + 1U;
    require(begin < text.size(), "token list must not end with a separator");
  }
  require(!tokens.empty(), "token list must contain at least one token");
  return tokens;
}

struct RuntimeRequest {
  std::vector<std::uint32_t> prompt_tokens;
  std::uint32_t max_new_tokens;
  std::filesystem::path evidence_root;
};

RuntimeRequest
parse_runtime_request(const std::vector<std::string> &arguments) {
  require(arguments.size() == kRuntimeArgumentCount,
          "runtime invocation has the wrong argument count");
  require(arguments[kModeArgumentIndex] == kTokensArgument,
          "runtime invocation must start with --tokens");
  require(arguments[kMaxNewNameIndex] == kMaxNewArgument,
          "runtime invocation must place --max-new second");
  require(arguments[kEvidenceNameIndex] == kEvidenceRootArgument,
          "runtime invocation must place --evidence-root last");
  const std::vector<std::uint32_t> prompt_tokens =
      parse_tokens(arguments[kTokensValueIndex]);
  const std::uint32_t max_new_tokens =
      parse_positive_u32(arguments[kMaxNewValueIndex], "max-new");
  require(max_new_tokens >= kMinimumGeneratedTokens,
          "max-new is below the retained-state proof minimum");
  require(max_new_tokens <= kMaximumGeneratedTokens,
          "max-new exceeds the package cap");
  const std::filesystem::path evidence_root(arguments[kEvidenceValueIndex]);
  require(evidence_root.is_absolute(), "evidence root must be absolute");
  require(!evidence_root.string().starts_with("/nix/store/"),
          "evidence root must be outside the Nix store");
  return RuntimeRequest{prompt_tokens, max_new_tokens, evidence_root};
}

std::string
digest_hex(const std::array<std::uint8_t, kBlake3DigestBytes> &digest) {
  constexpr std::array<char, kHexRadix> digits = {'0', '1', '2', '3', '4', '5',
                                                  '6', '7', '8', '9', 'a', 'b',
                                                  'c', 'd', 'e', 'f'};
  std::string output;
  output.reserve(kBlake3DigestBytes * kHexCharactersPerByte);
  for (const std::uint8_t byte : digest) {
    output.push_back(digits[byte >> (kBitsPerByte / kHexCharactersPerByte)]);
    output.push_back(digits[byte & static_cast<std::uint8_t>(kHexRadix - 1U)]);
  }
  return output;
}

std::string blake3_hex(const void *data, std::size_t size) {
  blake3_hasher hasher;
  blake3_hasher_init(&hasher);
  blake3_hasher_update(&hasher, data, size);
  std::array<std::uint8_t, kBlake3DigestBytes> digest{};
  blake3_hasher_finalize(&hasher, digest.data(), digest.size());
  return digest_hex(digest);
}

class MappedFile {
public:
  explicit MappedFile(const std::filesystem::path &path) {
    descriptor_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    require(descriptor_ >= 0, "failed to open pinned checkpoint");
    struct stat metadata{};
    if (::fstat(descriptor_, &metadata) != 0) {
      close_descriptor();
      fail("failed to stat pinned checkpoint");
    }
    require(metadata.st_size > 0, "pinned checkpoint is empty");
    size_ = static_cast<std::size_t>(metadata.st_size);
    mapping_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, descriptor_, 0);
    if (mapping_ == MAP_FAILED) {
      mapping_ = nullptr;
      close_descriptor();
      fail("failed to map pinned checkpoint");
    }
  }

  MappedFile(const MappedFile &) = delete;
  MappedFile &operator=(const MappedFile &) = delete;

  ~MappedFile() {
    if (mapping_ != nullptr) {
      static_cast<void>(::munmap(mapping_, size_));
    }
    close_descriptor();
  }

  [[nodiscard]] const std::uint8_t *data() const {
    return static_cast<const std::uint8_t *>(mapping_);
  }
  [[nodiscard]] std::size_t size() const { return size_; }

private:
  void close_descriptor() {
    if (descriptor_ >= 0) {
      static_cast<void>(::close(descriptor_));
      descriptor_ = -1;
    }
  }

  int descriptor_ = -1;
  void *mapping_ = nullptr;
  std::size_t size_ = 0U;
};

std::uint64_t decode_u64_little_endian(const std::uint8_t *bytes) {
  std::uint64_t value = 0U;
  for (std::size_t index = 0U; index < kSafetensorsHeaderLengthBytes; ++index) {
    value |= static_cast<std::uint64_t>(bytes[index]) << (index * kBitsPerByte);
  }
  return value;
}

std::size_t checked_element_count(const std::vector<std::uint32_t> &shape,
                                  std::string_view name) {
  require(!shape.empty(), std::string(name) + " shape must not be empty");
  std::size_t count = 1U;
  for (const std::uint32_t dimension : shape) {
    require(dimension > 0U,
            std::string(name) + " shape dimensions must be positive");
    require(count <= std::numeric_limits<std::size_t>::max() / dimension,
            std::string(name) + " element count overflows");
    count *= dimension;
  }
  return count;
}

struct TensorDescriptor {
  std::vector<std::uint32_t> shape;
  std::size_t begin;
  std::size_t end;
};

class CheckpointIndex {
public:
  explicit CheckpointIndex(const MappedFile &file) : file_(file) {
    require(file_.size() == kExpectedCheckpointBytes,
            "checkpoint byte count does not match the pinned authority");
    require(blake3_hex(file_.data(), file_.size()) == kExpectedCheckpointBlake3,
            "checkpoint BLAKE3 does not match the pinned authority");
    require(file_.size() > kSafetensorsHeaderLengthBytes,
            "checkpoint is shorter than the safetensors prefix");
    const std::uint64_t header_length_u64 =
        decode_u64_little_endian(file_.data());
    require(header_length_u64 <= std::numeric_limits<std::size_t>::max(),
            "safetensors header length overflows");
    const std::size_t header_length =
        static_cast<std::size_t>(header_length_u64);
    require(header_length <= file_.size() - kSafetensorsHeaderLengthBytes,
            "safetensors header exceeds the file");
    data_begin_ = kSafetensorsHeaderLengthBytes + header_length;
    const char *const header_begin = reinterpret_cast<const char *>(
        file_.data() + kSafetensorsHeaderLengthBytes);
    const char *const header_end = header_begin + header_length;
    const Json header = Json::parse(header_begin, header_end);
    require(header.is_object(), "safetensors header must be an object");
    for (const auto &[name, value] : header.items()) {
      if (name == "__metadata__") {
        continue;
      }
      require(value.is_object(), "safetensors tensor entry must be an object");
      require(value.at("dtype").get<std::string>() == "BF16",
              "checkpoint tensor dtype must be BF16");
      const std::vector<std::uint64_t> raw_shape =
          value.at("shape").get<std::vector<std::uint64_t>>();
      std::vector<std::uint32_t> shape;
      shape.reserve(raw_shape.size());
      for (const std::uint64_t dimension : raw_shape) {
        require(dimension <= std::numeric_limits<std::uint32_t>::max(),
                "checkpoint shape dimension overflows");
        shape.push_back(static_cast<std::uint32_t>(dimension));
      }
      const std::vector<std::uint64_t> offsets =
          value.at("data_offsets").get<std::vector<std::uint64_t>>();
      require(offsets.size() == kExpectedDeviceCount,
              "checkpoint tensor offsets must contain exactly two values");
      require(offsets[kFirstIndex] <= offsets[kSecondIndex],
              "checkpoint tensor offsets are reversed");
      require(offsets[kSecondIndex] <= file_.size() - data_begin_,
              "checkpoint tensor exceeds the data region");
      const std::size_t begin = static_cast<std::size_t>(offsets[kFirstIndex]);
      const std::size_t end = static_cast<std::size_t>(offsets[kSecondIndex]);
      const std::size_t element_count = checked_element_count(shape, name);
      require(element_count <=
                  std::numeric_limits<std::size_t>::max() / kBf16Bytes,
              "checkpoint tensor byte count overflows");
      require(end - begin == element_count * kBf16Bytes,
              "checkpoint tensor byte count does not match its shape");
      const auto [_, inserted] =
          tensors_.emplace(name, TensorDescriptor{shape, begin, end});
      require(inserted, "checkpoint contains a duplicate tensor name");
    }
  }

  void validate(const std::string &name,
                const std::vector<std::uint32_t> &expected_shape) const {
    const auto found = tensors_.find(name);
    require(found != tensors_.end(), "checkpoint is missing tensor " + name);
    require(found->second.shape == expected_shape,
            "checkpoint tensor shape mismatch for " + name);
  }

  [[nodiscard]] std::vector<bfloat16>
  read(const std::string &name,
       const std::vector<std::uint32_t> &expected_shape) const {
    validate(name, expected_shape);
    const auto found = tensors_.find(name);
    const TensorDescriptor &descriptor = found->second;
    const std::size_t element_count =
        checked_element_count(descriptor.shape, name);
    std::vector<bfloat16> values;
    values.reserve(element_count);
    const std::uint8_t *const bytes =
        file_.data() + data_begin_ + descriptor.begin;
    for (std::size_t index = 0U; index < element_count; ++index) {
      const std::size_t byte_index = index * kBf16Bytes;
      const std::uint16_t bits =
          static_cast<std::uint16_t>(bytes[byte_index]) |
          (static_cast<std::uint16_t>(bytes[byte_index + 1U]) << kBitsPerByte);
      const bfloat16 value = std::bit_cast<bfloat16>(bits);
      require(std::isfinite(static_cast<float>(value)),
              "checkpoint tensor contains a non-finite BF16 value");
      values.push_back(value);
    }
    return values;
  }

private:
  const MappedFile &file_;
  std::size_t data_begin_ = 0U;
  std::map<std::string, TensorDescriptor> tensors_;
};

std::vector<bfloat16> transpose_matrix_bits(const std::vector<bfloat16> &input,
                                            std::uint32_t rows,
                                            std::uint32_t columns) {
  require(input.size() == static_cast<std::size_t>(rows) * columns,
          "matrix transpose input size mismatch");
  std::vector<bfloat16> output(input.size());
  for (std::uint32_t row = 0U; row < rows; ++row) {
    for (std::uint32_t column = 0U; column < columns; ++column) {
      output[static_cast<std::size_t>(column) * rows + row] =
          input[static_cast<std::size_t>(row) * columns + column];
    }
  }
  return output;
}

std::string layer_tensor_name(std::uint32_t layer, std::string_view suffix) {
  return "model.layers." + std::to_string(layer) + "." + std::string(suffix);
}

void validate_checkpoint_schema(const CheckpointIndex &checkpoint) {
  checkpoint.validate("model.embeddings.weight",
                      {kVocabularySize, kHiddenSize});
  checkpoint.validate("model.norm.weight", {kHiddenSize});
  checkpoint.validate("model.norm.bias", {kHiddenSize});
  checkpoint.validate("lm_head.weight", {kVocabularySize, kHiddenSize});
  for (std::uint32_t layer = 0U; layer < kLayerCount; ++layer) {
    const auto vector = [&](std::string_view suffix,
                            std::vector<std::uint32_t> shape = {kHiddenSize}) {
      checkpoint.validate(layer_tensor_name(layer, suffix), shape);
    };
    const auto matrix = [&](std::string_view suffix, std::uint32_t rows,
                            std::uint32_t columns) {
      checkpoint.validate(layer_tensor_name(layer, suffix), {rows, columns});
    };
    if (layer == kLayerZero) {
      vector("pre_norm.weight");
      vector("pre_norm.bias");
    }
    vector("attn_norm.weight");
    vector("attn_norm.bias");
    for (std::string_view suffix : {"attn.x_r", "attn.x_w", "attn.x_k",
                                    "attn.x_v", "attn.x_a", "attn.x_g"}) {
      vector(suffix, {1U, 1U, kHiddenSize});
    }
    matrix("attn.r_proj.weight", kHiddenSize, kHiddenSize);
    matrix("attn.k_proj.weight", kHiddenSize, kHiddenSize);
    matrix("attn.v_proj.weight", kHiddenSize, kHiddenSize);
    matrix("attn.o_proj.weight", kHiddenSize, kHiddenSize);
    matrix("attn.w_lora.lora.0.weight", kDecayRank, kHiddenSize);
    matrix("attn.w_lora.lora.2.weight", kHiddenSize, kDecayRank);
    vector("attn.w_lora.lora.2.bias");
    matrix("attn.a_lora.lora.0.weight", kAttentionRank, kHiddenSize);
    matrix("attn.a_lora.lora.2.weight", kHiddenSize, kAttentionRank);
    vector("attn.a_lora.lora.2.bias");
    matrix("attn.g_lora.lora.0.weight", kGateRank, kHiddenSize);
    matrix("attn.g_lora.lora.2.weight", kHiddenSize, kGateRank);
    if (layer != kLayerZero) {
      matrix("attn.v_lora.lora.0.weight", kValueRank, kHiddenSize);
      matrix("attn.v_lora.lora.2.weight", kHiddenSize, kValueRank);
      vector("attn.v_lora.lora.2.bias");
    }
    vector("attn.k_k");
    vector("attn.k_a");
    vector("attn.r_k", {kHeadCount, kHeadSize});
    vector("attn.g_norm.weight");
    vector("attn.g_norm.bias");
    vector("ffn_norm.weight");
    vector("ffn_norm.bias");
    vector("ffn.x_k");
    matrix("ffn.key.weight", kIntermediateSize, kHiddenSize);
    matrix("ffn.value.weight", kHiddenSize, kIntermediateSize);
  }
}

ttnn::Shape make_shape(const std::vector<std::uint32_t> &dimensions) {
  return ttnn::Shape(
      ttnn::Shape::Container(dimensions.begin(), dimensions.end()));
}

struct OperationEvent {
  std::uint64_t ordinal;
  std::string stage;
  std::string operation;
};

struct ProfilerRange {
  std::string stage;
  std::uint64_t begin_byte;
  std::uint64_t end_byte;
};

Json map_to_json(const std::map<std::string, std::uint64_t> &values);

std::string sanitize_stage(std::string stage) {
  for (char &character : stage) {
    if (character == '/' || character == ' ' || character == ':') {
      character = '-';
    }
  }
  return stage;
}

std::string_view mesh_descriptor_for_device_count(std::size_t device_count) {
  require(device_count > 0U && device_count <= kExpectedDeviceCount,
          "mesh descriptor device count is outside the supported range");
  if (device_count == 1U) {
    return RWKV7_PINNED_SINGLE_CARD_MESH_DESCRIPTOR_PATH;
  }
  return RWKV7_PINNED_MESH_DESCRIPTOR_PATH;
}

// r[impl tenstorrent.rwkv7_p150x2.full_device_runtime]
class DeviceContext {
public:
  DeviceContext()
      : DeviceContext({kPhysicalDeviceZero, kPhysicalDeviceOne}) {
    require(expected_device_count_ == kExpectedDeviceCount,
            "evidence context must open both physical cards");
  }

  explicit DeviceContext(int physical_device_id)
      : DeviceContext(std::vector<int>{physical_device_id}) {
    require(physical_device_id == kPhysicalDeviceZero ||
                physical_device_id == kPhysicalDeviceOne,
            "fast worker device must be physical device 0 or 1");
  }

  explicit DeviceContext(std::vector<int> physical_device_ids)
      : physical_device_ids_(std::move(physical_device_ids)),
        expected_device_count_(physical_device_ids_.size()) {
    require(!physical_device_ids_.empty(),
            "device context must select at least one physical card");
    require(expected_device_count_ <= kExpectedDeviceCount,
            "device context selected too many physical cards");
    const char *const descriptor = std::getenv("TT_MESH_GRAPH_DESC_PATH");
    require(descriptor != nullptr, "TT_MESH_GRAPH_DESC_PATH is required");
    const std::string_view expected_descriptor =
        mesh_descriptor_for_device_count(expected_device_count_);
    require(std::string_view(descriptor) == expected_descriptor,
            "TT_MESH_GRAPH_DESC_PATH does not match the selected pinned descriptor");
    const char *const metalium_home = std::getenv("TT_METAL_HOME");
    require(metalium_home != nullptr, "TT_METAL_HOME is required");
    require(std::string_view(metalium_home) == RWKV7_PINNED_METALIUM_ROOT,
            "TT_METAL_HOME does not match the pinned runtime");
    const std::uint32_t mesh_columns =
        static_cast<std::uint32_t>(expected_device_count_);
    mesh_ = ttnn::distributed::open_mesh_device(
        ttnn::distributed::MeshShape{kMeshRows, mesh_columns},
        DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1U,
        tt::tt_metal::DispatchCoreConfig{}, std::nullopt,
        physical_device_ids_);
    require(mesh_ != nullptr, "failed to open the selected P150 mesh");
    require(mesh_->num_devices() == expected_device_count_,
            "opened mesh does not contain the selected device count");
    require(mesh_->shape() ==
                ttnn::distributed::MeshShape{kMeshRows, mesh_columns},
            "opened mesh shape does not match the selected cards");
    const auto device_ids = mesh_->get_device_ids();
    require(device_ids.size() == expected_device_count_,
            "opened mesh reported the wrong number of physical IDs");
    for (std::size_t index = 0U; index < device_ids.size(); ++index) {
      require(static_cast<int>(device_ids[index]) == physical_device_ids_[index],
              "opened mesh physical ID order differs from the selection");
    }
    mesh_->enable_program_cache();
    const char *const profiler_root = std::getenv("TT_METAL_PROFILER_DIR");
    require(profiler_root != nullptr, "TT_METAL_PROFILER_DIR is required");
    profiler_csv_path_ = std::filesystem::path(profiler_root) / ".logs" /
                         "profile_log_device.csv";
    mapper_ = ttnn::distributed::replicate_tensor_to_mesh_mapper(*mesh_);
    require(mapper_ != nullptr, "failed to create replicated mesh mapper");
  }

  DeviceContext(const DeviceContext &) = delete;
  DeviceContext &operator=(const DeviceContext &) = delete;

  ~DeviceContext() {
    if (mesh_ != nullptr) {
      ttnn::distributed::close_mesh_device(mesh_);
    }
  }

  [[nodiscard]] MeshDevice &mesh() const { return *mesh_; }

  [[nodiscard]] std::size_t program_cache_entries() const {
    return mesh_->num_program_cache_entries();
  }

  void set_stage(std::string stage) {
    if (!operation_recording_enabled_) {
      return;
    }
    stage_ = sanitize_stage(std::move(stage));
  }

  void set_profiler_flush_enabled(bool enabled) {
    profiler_flush_enabled_ = enabled;
  }

  void set_operation_recording_enabled(bool enabled) {
    operation_recording_enabled_ = enabled;
  }

  void flush_profiler(std::string stage) {
    if (!profiler_flush_enabled_) {
      return;
    }
    require(mesh_ != nullptr, "cannot flush profiler without an open mesh");
    // The mesh-level read drains the profiler thread pool (wait_for_thread_pool)
    // before returning, so the processed device CSV exists synchronously. A
    // per-device detail::ReadDeviceProfilerResults only enqueues the
    // processor to that pool and returns before the file is written.
    tt::tt_metal::ReadMeshDeviceProfilerResults(*mesh_);
    require(std::filesystem::exists(profiler_csv_path_),
            "device profiler did not create its raw CSV");
    const std::uint64_t end_byte =
        std::filesystem::file_size(profiler_csv_path_);
    require(end_byte > profiler_cursor_,
            "device profiler flush appended no raw CSV evidence");
    profiler_ranges_.push_back(ProfilerRange{sanitize_stage(std::move(stage)),
                                             profiler_cursor_, end_byte});
    profiler_cursor_ = end_byte;
  }

  Tensor checked(Tensor tensor, std::string_view operation) {
    require(tensor.device() == mesh_.get(),
            "TT-NN operation returned a tensor on the wrong mesh");
    require(ttnn::distributed::get_device_tensors(tensor).size() ==
                expected_device_count_,
            "TT-NN operation did not preserve the selected device shards");
    if (operation_recording_enabled_) {
      ++operation_counts_[std::string(operation)];
      ++stage_operation_counts_[stage_][std::string(operation)];
      operation_events_.push_back(
          OperationEvent{static_cast<std::uint64_t>(operation_events_.size()),
                         stage_, std::string(operation)});
    }
    return tensor;
  }

  Tensor bf16_tensor(std::vector<bfloat16> values,
                     const std::vector<std::uint32_t> &shape,
                     ttnn::Layout target_layout) {
    const std::size_t expected =
        checked_element_count(shape, "host BF16 tensor");
    require(values.size() == expected,
            "host BF16 tensor size does not match its shape");
    // Create the host tensor directly in its target layout. Tensor::from_vector
    // pads a TILE-layout tensor to its tile-aligned padded shape at creation, so
    // a (1, hidden) norm/bias vector whose height is below the tile height still
    // converts. The previous ROW_MAJOR -> to_layout(TILE) path demanded an
    // explicit 32-multiple height and aborted on every such vector.
    const ttnn::TensorSpec spec(
        make_shape(shape),
        tt::tt_metal::TensorLayout(
            ttnn::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(target_layout),
            tt::tt_metal::MemoryConfig{}));
    Tensor host = Tensor::from_vector(std::move(values), spec);
    return checked(
        ttnn::distributed::distribute_tensor(host, *mapper_, std::ref(*mesh_)),
        "host-to-device-bf16");
  }

  Tensor float32_tensor(std::vector<float> values,
                        const std::vector<std::uint32_t> &shape,
                        ttnn::Layout target_layout) {
    const std::size_t expected =
        checked_element_count(shape, "host FLOAT32 tensor");
    require(values.size() == expected,
            "host FLOAT32 tensor size does not match its shape");
    const ttnn::TensorSpec spec(
        make_shape(shape),
        tt::tt_metal::TensorLayout(
            ttnn::DataType::FLOAT32,
            tt::tt_metal::PageConfig(target_layout),
            tt::tt_metal::MemoryConfig{}));
    Tensor host = Tensor::from_vector(std::move(values), spec);
    return checked(
        ttnn::distributed::distribute_tensor(host, *mapper_, std::ref(*mesh_)),
        "host-to-device-fp32");
  }

  Tensor uint32_tensor(std::vector<std::uint32_t> values,
                       const std::vector<std::uint32_t> &shape) {
    const std::size_t expected =
        checked_element_count(shape, "host UINT32 tensor");
    require(values.size() == expected,
            "host UINT32 tensor size does not match its shape");
    const ttnn::TensorSpec spec(
        make_shape(shape),
        tt::tt_metal::TensorLayout(
            ttnn::DataType::UINT32,
            tt::tt_metal::PageConfig(ttnn::Layout::ROW_MAJOR),
            tt::tt_metal::MemoryConfig{}));
    const Tensor host = Tensor::from_vector(std::move(values), spec);
    return checked(
        ttnn::distributed::distribute_tensor(host, *mapper_, std::ref(*mesh_)),
        "host-to-device-u32");
  }

  Tensor zeros(const std::vector<std::uint32_t> &shape) {
    return bf16_tensor(
        std::vector<bfloat16>(checked_element_count(shape, "zero tensor"),
                              bfloat16(0.0F)),
        shape, ttnn::Layout::TILE);
  }

  Tensor zeros_fp32(const std::vector<std::uint32_t> &shape) {
    return float32_tensor(
        std::vector<float>(checked_element_count(shape, "FP32 zero tensor"),
                           0.0F),
        shape, ttnn::Layout::TILE);
  }

  [[nodiscard]] const std::map<std::string, std::uint64_t> &
  operation_counts() const {
    return operation_counts_;
  }

  [[nodiscard]] const std::map<std::string,
                               std::map<std::string, std::uint64_t>> &
  stage_operation_counts() const {
    return stage_operation_counts_;
  }

  [[nodiscard]] const std::vector<OperationEvent> &operation_events() const {
    return operation_events_;
  }

  [[nodiscard]] const std::vector<ProfilerRange> &profiler_ranges() const {
    return profiler_ranges_;
  }

private:
  std::vector<int> physical_device_ids_;
  std::size_t expected_device_count_;
  std::shared_ptr<MeshDevice> mesh_;
  std::unique_ptr<TensorToMesh> mapper_;
  bool profiler_flush_enabled_ = true;
  bool operation_recording_enabled_ = true;
  std::string stage_ = "initialization";
  std::map<std::string, std::uint64_t> operation_counts_;
  std::map<std::string, std::map<std::string, std::uint64_t>>
      stage_operation_counts_;
  std::vector<OperationEvent> operation_events_;
  std::filesystem::path profiler_csv_path_;
  std::uint64_t profiler_cursor_ = 0U;
  std::vector<ProfilerRange> profiler_ranges_;
};

struct LayerWeights {
  std::uint32_t layer_index;
  std::map<std::string, Tensor> tensors;

  [[nodiscard]] const Tensor &at(std::string_view key) const {
    const auto found = tensors.find(std::string(key));
    require(found != tensors.end(),
            "loaded layer is missing tensor key " + std::string(key));
    return found->second;
  }
};

struct ModelWeights {
  Tensor embeddings;
  Tensor final_norm_weight;
  Tensor final_norm_bias;
  Tensor head;
  std::vector<LayerWeights> layers;
};

Tensor load_vector(
    DeviceContext &context, const CheckpointIndex &checkpoint,
    const std::string &name, std::uint32_t length,
    const std::vector<std::uint32_t> &checkpoint_shape = {},
    const std::vector<std::uint32_t> &device_shape = {}) {
  const std::vector<std::uint32_t> source_shape =
      checkpoint_shape.empty() ? std::vector<std::uint32_t>{length}
                               : checkpoint_shape;
  const std::vector<std::uint32_t> logical_shape =
      device_shape.empty() ? std::vector<std::uint32_t>{1U, 1U, 1U, length}
                           : device_shape;
  return context.bf16_tensor(checkpoint.read(name, source_shape), logical_shape,
                             ttnn::Layout::TILE);
}

Tensor load_matrix(DeviceContext &context, const CheckpointIndex &checkpoint,
                   const std::string &name, std::uint32_t rows,
                   std::uint32_t columns) {
  std::vector<bfloat16> values = checkpoint.read(name, {rows, columns});
  values = transpose_matrix_bits(values, rows, columns);
  return context.bf16_tensor(std::move(values), {1U, 1U, columns, rows},
                             ttnn::Layout::TILE);
}

LayerWeights load_layer(DeviceContext &context,
                        const CheckpointIndex &checkpoint,
                        std::uint32_t layer_index) {
  LayerWeights layer{layer_index, {}};
  const auto vector =
      [&](std::string_view key, std::string_view suffix,
          std::vector<std::uint32_t> checkpoint_shape = {},
          std::vector<std::uint32_t> device_shape = {}) {
        layer.tensors.emplace(
            std::string(key),
            load_vector(context, checkpoint,
                        layer_tensor_name(layer_index, suffix), kHiddenSize,
                        checkpoint_shape, device_shape));
      };
  const auto matrix = [&](std::string_view key, std::string_view suffix,
                          std::uint32_t rows, std::uint32_t columns) {
    layer.tensors.emplace(std::string(key),
                          load_matrix(context, checkpoint,
                                      layer_tensor_name(layer_index, suffix),
                                      rows, columns));
  };

  if (layer_index == kLayerZero) {
    vector("pre_norm_weight", "pre_norm.weight");
    vector("pre_norm_bias", "pre_norm.bias");
  }
  vector("attn_norm_weight", "attn_norm.weight");
  vector("attn_norm_bias", "attn_norm.bias");
  vector("x_r", "attn.x_r", {1U, 1U, kHiddenSize});
  vector("x_w", "attn.x_w", {1U, 1U, kHiddenSize});
  vector("x_k", "attn.x_k", {1U, 1U, kHiddenSize});
  vector("x_v", "attn.x_v", {1U, 1U, kHiddenSize});
  vector("x_a", "attn.x_a", {1U, 1U, kHiddenSize});
  vector("x_g", "attn.x_g", {1U, 1U, kHiddenSize});
  matrix("r_projection", "attn.r_proj.weight", kHiddenSize, kHiddenSize);
  matrix("k_projection", "attn.k_proj.weight", kHiddenSize, kHiddenSize);
  matrix("v_projection", "attn.v_proj.weight", kHiddenSize, kHiddenSize);
  matrix("output_projection", "attn.o_proj.weight", kHiddenSize, kHiddenSize);
  matrix("w_down", "attn.w_lora.lora.0.weight", kDecayRank, kHiddenSize);
  matrix("w_up", "attn.w_lora.lora.2.weight", kHiddenSize, kDecayRank);
  vector("w_bias", "attn.w_lora.lora.2.bias");
  matrix("a_down", "attn.a_lora.lora.0.weight", kAttentionRank, kHiddenSize);
  matrix("a_up", "attn.a_lora.lora.2.weight", kHiddenSize, kAttentionRank);
  vector("a_bias", "attn.a_lora.lora.2.bias");
  matrix("gate_down", "attn.g_lora.lora.0.weight", kGateRank, kHiddenSize);
  matrix("gate_up", "attn.g_lora.lora.2.weight", kHiddenSize, kGateRank);
  if (layer_index != kLayerZero) {
    matrix("value_down", "attn.v_lora.lora.0.weight", kValueRank, kHiddenSize);
    matrix("value_up", "attn.v_lora.lora.2.weight", kHiddenSize, kValueRank);
    vector("value_bias", "attn.v_lora.lora.2.bias");
  }
  vector("k_k", "attn.k_k");
  vector("k_a", "attn.k_a");
  const std::vector<std::uint32_t> head_row_shape{1U, kHeadCount, 1U,
                                                   kHeadSize};
  vector("r_k", "attn.r_k", {kHeadCount, kHeadSize}, head_row_shape);
  vector("group_norm_weight", "attn.g_norm.weight", {}, head_row_shape);
  vector("group_norm_bias", "attn.g_norm.bias", {}, head_row_shape);
  vector("ffn_norm_weight", "ffn_norm.weight");
  vector("ffn_norm_bias", "ffn_norm.bias");
  vector("ffn_x_k", "ffn.x_k");
  matrix("ffn_key", "ffn.key.weight", kIntermediateSize, kHiddenSize);
  matrix("ffn_value", "ffn.value.weight", kHiddenSize, kIntermediateSize);
  return layer;
}

ModelWeights load_model(DeviceContext &context,
                        const CheckpointIndex &checkpoint) {
  context.set_stage("model-load");
  const Tensor embeddings = context.bf16_tensor(
      checkpoint.read("model.embeddings.weight",
                      {kVocabularySize, kHiddenSize}),
      {kVocabularySize, kHiddenSize}, ttnn::Layout::ROW_MAJOR);
  std::vector<LayerWeights> layers;
  layers.reserve(kLayerCount);
  for (std::uint32_t layer_index = 0U; layer_index < kLayerCount;
       ++layer_index) {
    layers.push_back(load_layer(context, checkpoint, layer_index));
  }
  const Tensor final_norm_weight =
      load_vector(context, checkpoint, "model.norm.weight", kHiddenSize);
  const Tensor final_norm_bias =
      load_vector(context, checkpoint, "model.norm.bias", kHiddenSize);
  const Tensor head = load_matrix(context, checkpoint, "lm_head.weight",
                                  kVocabularySize, kHiddenSize);
  return ModelWeights{embeddings, final_norm_weight, final_norm_bias, head,
                      std::move(layers)};
}

struct LayerState {
  Tensor attention_previous;
  Tensor ffn_previous;
  Tensor matrix;
};

struct ModelState {
  std::vector<LayerState> layers;
};

ModelState zero_model_state(DeviceContext &context) {
  context.set_stage("state-zero-initialization");
  std::vector<LayerState> layers;
  layers.reserve(kLayerCount);
  for (std::uint32_t layer_index = 0U; layer_index < kLayerCount;
       ++layer_index) {
    layers.push_back(
        LayerState{context.zeros({1U, 1U, 1U, kHiddenSize}),
                   context.zeros({1U, 1U, 1U, kHiddenSize}),
                   context.zeros({1U, kHeadCount, kHeadSize, kHeadSize})});
  }
  return ModelState{std::move(layers)};
}

ModelState zero_fast_model_state(DeviceContext &context,
                                 std::uint32_t batch_size) {
  require(batch_size > 0U && batch_size <= kFastPerCardSlotCount,
          "fast state batch size is outside the card-local range");
  context.set_stage("fast-state-zero-initialization");
  std::vector<LayerState> layers;
  layers.reserve(kLayerCount);
  for (std::uint32_t layer_index = 0U; layer_index < kLayerCount;
       ++layer_index) {
    layers.push_back(LayerState{
        context.zeros({1U, 1U, batch_size, kHiddenSize}),
        context.zeros({1U, 1U, batch_size, kHiddenSize}),
        context.zeros_fp32(
            {batch_size, kHeadCount, kHeadSize, kHeadSize})});
  }
  return ModelState{std::move(layers)};
}

Tensor op_reshape(DeviceContext &context, const Tensor &input,
                  const std::vector<std::uint32_t> &shape) {
  return context.checked(ttnn::reshape(input, make_shape(shape)), "reshape");
}

Tensor op_typecast(DeviceContext &context, const Tensor &input,
                   ttnn::DataType dtype, std::string_view operation) {
  return context.checked(ttnn::typecast(input, dtype), operation);
}

Tensor op_transpose(DeviceContext &context, const Tensor &input, int first,
                    int second) {
  return context.checked(ttnn::transpose(input, first, second), "transpose");
}

Tensor op_add(DeviceContext &context, const Tensor &left, const Tensor &right) {
  return context.checked(ttnn::add(left, right), "add");
}

Tensor op_add_scalar(DeviceContext &context, const Tensor &input,
                     float scalar) {
  return context.checked(ttnn::add(input, scalar), "add-scalar");
}

Tensor op_subtract(DeviceContext &context, const Tensor &left,
                   const Tensor &right) {
  return context.checked(ttnn::subtract(left, right), "subtract");
}

Tensor op_subtract_scalar(DeviceContext &context, const Tensor &input,
                          float scalar) {
  return context.checked(ttnn::subtract(input, scalar), "subtract-scalar");
}

Tensor op_multiply(DeviceContext &context, const Tensor &left,
                   const Tensor &right) {
  return context.checked(ttnn::multiply(left, right), "multiply");
}

Tensor op_multiply_scalar(DeviceContext &context, const Tensor &input,
                          float scalar) {
  return context.checked(ttnn::multiply(input, scalar), "multiply-scalar");
}

Tensor op_divide(DeviceContext &context, const Tensor &left,
                 const Tensor &right) {
  return context.checked(ttnn::divide(left, right), "divide");
}

Tensor op_lerp(DeviceContext &context, const Tensor &input, const Tensor &end,
               const Tensor &weight) {
  return context.checked(ttnn::lerp(input, end, weight), "lerp");
}

Tensor op_matmul(DeviceContext &context, const Tensor &left,
                 const Tensor &right) {
  return context.checked(ttnn::matmul(left, right), "matmul");
}

Tensor op_linear_bias(DeviceContext &context, const Tensor &input,
                      const Tensor &weight, const Tensor &bias) {
  return context.checked(ttnn::linear(input, weight, bias),
                         "linear-with-bias");
}

Tensor op_matmul_transpose_left(DeviceContext &context, const Tensor &left,
                                const Tensor &right) {
  return context.checked(ttnn::matmul(left, right, true, false),
                         "matmul-transpose-left");
}

Tensor op_matmul_transpose_right(DeviceContext &context, const Tensor &left,
                                 const Tensor &right) {
  return context.checked(ttnn::matmul(left, right, false, true),
                         "matmul-transpose-right");
}

Tensor op_matmul_transpose_right_bf16(DeviceContext &context,
                                      const Tensor &left,
                                      const Tensor &right) {
  return context.checked(
      ttnn::matmul(left, right, false, true, std::nullopt,
                   ttnn::DataType::BFLOAT16),
      "matmul-transpose-right-bf16");
}

Tensor op_layer_norm(DeviceContext &context, const Tensor &input,
                     const Tensor &weight, const Tensor &bias, float epsilon) {
  return context.checked(ttnn::layer_norm(input, epsilon, weight, bias),
                         "layer-norm");
}

Tensor op_layer_norm(DeviceContext &context, const Tensor &input,
                     float epsilon) {
  return context.checked(ttnn::layer_norm(input, epsilon),
                         "layer-norm-unaffined");
}

Tensor op_sigmoid(DeviceContext &context, const Tensor &input) {
  return context.checked(ttnn::sigmoid(input), "sigmoid");
}

Tensor op_tanh(DeviceContext &context, const Tensor &input) {
  return context.checked(ttnn::tanh(input), "tanh");
}

Tensor op_exp(DeviceContext &context, const Tensor &input) {
  return context.checked(ttnn::exp(input, false), "exp");
}

Tensor op_relu(DeviceContext &context, const Tensor &input) {
  return context.checked(ttnn::relu(input), "relu");
}

Tensor op_square(DeviceContext &context, const Tensor &input) {
  return context.checked(ttnn::square(input), "square");
}

Tensor op_sqrt(DeviceContext &context, const Tensor &input) {
  return context.checked(ttnn::sqrt(input, false), "sqrt");
}

Tensor op_maximum_scalar(DeviceContext &context, const Tensor &input,
                         float scalar) {
  return context.checked(ttnn::maximum(input, scalar), "maximum-scalar");
}

Tensor op_neg(DeviceContext &context, const Tensor &input) {
  return context.checked(ttnn::neg(input), "neg");
}

Tensor op_sum_last(DeviceContext &context, const Tensor &input) {
  return context.checked(ttnn::sum(input, kLastDimension, true), "sum-last");
}

Tensor mixed(DeviceContext &context, const Tensor &input,
             const Tensor &previous, const Tensor &factor) {
  return op_lerp(context, input, previous, factor);
}

Tensor normalize_heads(DeviceContext &context, const Tensor &input) {
  const std::uint32_t batch_size = input.logical_shape()[-2];
  const Tensor rows = op_reshape(
      context, input, {batch_size, kHeadCount, 1U, kHeadSize});
  const Tensor norm = op_maximum_scalar(
      context, op_sqrt(context, op_sum_last(context, op_square(context, rows))),
      kNormalizationFloor);
  return op_reshape(context, op_divide(context, rows, norm),
                    {1U, 1U, batch_size, kHiddenSize});
}

struct TimePreparation {
  Tensor projected_value;
  Tensor r;
  Tensor w;
  Tensor k;
  Tensor v;
  Tensor a;
  Tensor b;
  Tensor gate;
};

TimePreparation prepare_time_mix(DeviceContext &context,
                                 const LayerWeights &weights,
                                 const Tensor &input, const Tensor &previous,
                                 const std::optional<Tensor> &value_anchor) {
  const Tensor x_r = mixed(context, input, previous, weights.at("x_r"));
  const Tensor x_w = mixed(context, input, previous, weights.at("x_w"));
  const Tensor x_k = mixed(context, input, previous, weights.at("x_k"));
  const Tensor x_v = mixed(context, input, previous, weights.at("x_v"));
  const Tensor x_a = mixed(context, input, previous, weights.at("x_a"));
  const Tensor x_g = mixed(context, input, previous, weights.at("x_g"));

  const Tensor r = op_matmul(context, x_r, weights.at("r_projection"));
  const Tensor raw_k = op_matmul(context, x_k, weights.at("k_projection"));
  const Tensor projected_value =
      op_matmul(context, x_v, weights.at("v_projection"));
  Tensor v = projected_value;
  if (weights.layer_index == kLayerZero) {
    require(!value_anchor.has_value(),
            "layer zero received a stale value anchor");
  } else {
    require(value_anchor.has_value(),
            "later layer is missing the token-local value anchor");
    const Tensor value_hidden =
        op_matmul(context, x_v, weights.at("value_down"));
    const Tensor value_logits =
        op_linear_bias(context, value_hidden, weights.at("value_up"),
                       weights.at("value_bias"));
    const Tensor value_gate = op_sigmoid(context, value_logits);
    v = op_add(
        context, projected_value,
        op_multiply(context, value_gate,
                    op_subtract(context, *value_anchor, projected_value)));
  }

  const Tensor w_hidden =
      op_tanh(context, op_matmul(context, x_w, weights.at("w_down")));
  const Tensor w_logits =
      op_linear_bias(context, w_hidden, weights.at("w_up"),
                     weights.at("w_bias"));
  const Tensor w =
      op_exp(context, op_multiply_scalar(context, op_sigmoid(context, w_logits),
                                         kDecayScale));

  const Tensor a_hidden = op_matmul(context, x_a, weights.at("a_down"));
  const Tensor a_logits =
      op_linear_bias(context, a_hidden, weights.at("a_up"),
                     weights.at("a_bias"));
  const Tensor a_gate = op_sigmoid(context, a_logits);

  const Tensor gate_hidden =
      op_sigmoid(context, op_matmul(context, x_g, weights.at("gate_down")));
  const Tensor gate = op_matmul(context, gate_hidden, weights.at("gate_up"));

  const Tensor kk =
      normalize_heads(context, op_multiply(context, raw_k, weights.at("k_k")));
  const Tensor k = op_multiply(
      context, raw_k,
      op_add_scalar(context,
                    op_multiply(context,
                                op_subtract_scalar(context, a_gate, kOne),
                                weights.at("k_a")),
                    kOne));
  const Tensor a = op_neg(context, kk);
  const Tensor b = op_multiply(context, kk, a_gate);
  return TimePreparation{projected_value, r, w, k, v, a, b, gate};
}

struct WkvOutput {
  Tensor state;
  Tensor output_for_hidden;
  Tensor r_rows;
  Tensor k_rows;
  Tensor v_rows;
};

WkvOutput wkv_step(DeviceContext &context, const Tensor &state,
                   const TimePreparation &preparation) {
  const bool fp32_state = state.dtype() == ttnn::DataType::FLOAT32;
  const std::uint32_t batch_size = state.logical_shape()[0];
  const auto state_dtype = [&](const Tensor &tensor) {
    return fp32_state
               ? op_typecast(context, tensor, ttnn::DataType::FLOAT32,
                             "typecast-recurrence-fp32")
               : tensor;
  };
  const auto state_row = [&](const Tensor &tensor) {
    return op_reshape(context, state_dtype(tensor),
                      {batch_size, kHeadCount, 1U, kHeadSize});
  };
  const auto source_row = [&](const Tensor &tensor) {
    return op_reshape(context, tensor,
                      {batch_size, kHeadCount, 1U, kHeadSize});
  };
  const auto source_to_state = [&](const Tensor &rows) {
    return fp32_state
               ? op_typecast(context, rows, ttnn::DataType::FLOAT32,
                             "typecast-recurrence-fp32")
               : rows;
  };
  const Tensor a_rows = state_row(preparation.a);
  const Tensor b_rows = state_row(preparation.b);
  const Tensor w_rows = state_row(preparation.w);
  const Tensor v_rows = source_row(preparation.v);
  const Tensor k_rows = source_row(preparation.k);
  const Tensor r_rows = source_row(preparation.r);
  const Tensor v_state_rows = source_to_state(v_rows);
  const Tensor k_state_rows = source_to_state(k_rows);
  const Tensor r_state_rows = source_to_state(r_rows);
  const Tensor state_dot_a =
      op_matmul_transpose_right(context, state, a_rows);
  const Tensor rank_update = op_matmul(context, state_dot_a, b_rows);
  const Tensor decay = op_multiply(context, state, w_rows);
  const Tensor outer =
      op_matmul_transpose_left(context, v_state_rows, k_state_rows);
  const Tensor next_state =
      op_add(context, op_add(context, decay, rank_update), outer);
  const Tensor output_for_hidden =
      fp32_state
          ? op_matmul_transpose_right_bf16(context, next_state, r_state_rows)
          : op_matmul_transpose_right(context, next_state, r_state_rows);
  return WkvOutput{next_state, output_for_hidden, r_rows, k_rows, v_rows};
}

Tensor finish_attention(DeviceContext &context, const LayerWeights &weights,
                        const TimePreparation &preparation,
                        const WkvOutput &wkv) {
  const std::uint32_t batch_size = wkv.output_for_hidden.logical_shape()[0];
  const Tensor rows = op_reshape(
      context, wkv.output_for_hidden,
      {batch_size, kHeadCount, 1U, kHeadSize});
  const Tensor normalized_rows =
      op_layer_norm(context, rows, kGroupNormEpsilon);
  const Tensor scaled_rows =
      op_add(context,
             op_multiply(context, normalized_rows,
                         weights.at("group_norm_weight")),
             weights.at("group_norm_bias"));
  const Tensor scalar = op_sum_last(
      context,
      op_multiply(context,
                  op_multiply(context, wkv.r_rows, wkv.k_rows),
                  weights.at("r_k")));
  const Tensor corrected_rows = op_multiply(
      context, op_add(context, scaled_rows,
                      op_multiply(context, scalar, wkv.v_rows)),
      op_reshape(context, preparation.gate,
                 {batch_size, kHeadCount, 1U, kHeadSize}));
  return op_matmul(
      context,
      op_reshape(context, corrected_rows,
                 {1U, 1U, batch_size, kHiddenSize}),
      weights.at("output_projection"));
}

Tensor channel_mix(DeviceContext &context, const LayerWeights &weights,
                   const Tensor &input, const Tensor &previous) {
  const Tensor mixed_input =
      mixed(context, input, previous, weights.at("ffn_x_k"));
  const Tensor activated = op_square(
      context,
      op_relu(context, op_matmul(context, mixed_input, weights.at("ffn_key"))));
  return op_matmul(context, activated, weights.at("ffn_value"));
}

Tensor execute_layer(DeviceContext &context, const LayerWeights &weights,
                     const Tensor &hidden, LayerState &state,
                     std::optional<Tensor> &value_anchor,
                     std::string_view stage_prefix) {
  context.set_stage(std::string(stage_prefix) + "/pre-norm");
  Tensor residual = hidden;
  if (weights.layer_index == kLayerZero) {
    residual = op_layer_norm(context, hidden, weights.at("pre_norm_weight"),
                             weights.at("pre_norm_bias"), kLayerNormEpsilon);
  }
  const Tensor attention_input =
      op_layer_norm(context, residual, weights.at("attn_norm_weight"),
                    weights.at("attn_norm_bias"), kLayerNormEpsilon);

  context.set_stage(std::string(stage_prefix) + "/time-mix-preparation");
  const TimePreparation preparation =
      prepare_time_mix(context, weights, attention_input,
                       state.attention_previous, value_anchor);
  if (weights.layer_index == kLayerZero) {
    value_anchor = preparation.projected_value;
  }
  state.attention_previous = attention_input;

  context.set_stage(std::string(stage_prefix) + "/wkv");
  const WkvOutput wkv = wkv_step(context, state.matrix, preparation);
  state.matrix = wkv.state;

  context.set_stage(std::string(stage_prefix) + "/attention-finish");
  const Tensor attention_output =
      finish_attention(context, weights, preparation, wkv);
  const Tensor after_attention = op_add(context, residual, attention_output);

  context.set_stage(std::string(stage_prefix) + "/channel-mix");
  const Tensor ffn_input =
      op_layer_norm(context, after_attention, weights.at("ffn_norm_weight"),
                    weights.at("ffn_norm_bias"), kLayerNormEpsilon);
  const Tensor ffn_output =
      channel_mix(context, weights, ffn_input, state.ffn_previous);
  state.ffn_previous = ffn_input;
  const Tensor output = op_add(context, after_attention, ffn_output);
  context.flush_profiler(std::string(stage_prefix) + "/complete");
  return output;
}

struct HostBf16Shards {
  std::array<std::vector<bfloat16>, kExpectedDeviceCount> shards;
};

HostBf16Shards read_bf16_shards(DeviceContext &context, const Tensor &input) {
  context.set_stage("evidence/readback-bf16");
  Tensor row_major = input;
  if (input.layout() == ttnn::Layout::TILE) {
    row_major = context.checked(ttnn::untilize(input), "untilize-evidence");
  }
  const Tensor host = ttnn::from_device(row_major, true);
  const std::vector<Tensor> shards =
      ttnn::distributed::get_device_tensors(host);
  require(shards.size() == kExpectedDeviceCount,
          "BF16 readback did not contain two card shards");
  return HostBf16Shards{{shards[kFirstIndex].to_vector<bfloat16>(),
                         shards[kSecondIndex].to_vector<bfloat16>()}};
}

std::array<std::uint32_t, kExpectedDeviceCount>
read_u32_shards(DeviceContext &context, const Tensor &input) {
  context.set_stage("selection/readback-u32");
  const Tensor host = ttnn::from_device(input, true);
  const std::vector<Tensor> shards =
      ttnn::distributed::get_device_tensors(host);
  require(shards.size() == kExpectedDeviceCount,
          "UINT32 readback did not contain two card shards");
  const std::vector<std::uint32_t> first =
      shards[kFirstIndex].to_vector<std::uint32_t>();
  const std::vector<std::uint32_t> second =
      shards[kSecondIndex].to_vector<std::uint32_t>();
  require(first.size() == 1U && second.size() == 1U,
          "device argmax did not return one value per card");
  return {first[kFirstIndex], second[kFirstIndex]};
}

std::vector<std::uint32_t>
read_fast_u32(DeviceContext &context, const Tensor &input,
              std::size_t batch_size) {
  require(batch_size > 0U && batch_size <= kFastPerCardSlotCount,
          "fast UINT32 readback batch size is outside the card-local range");
  context.set_stage("fast-selection/readback-u32");
  const Tensor host = ttnn::from_device(input, true);
  const std::vector<Tensor> shards =
      ttnn::distributed::get_device_tensors(host);
  require(shards.size() == 1U,
          "fast UINT32 readback did not contain one card shard");
  std::vector<std::uint32_t> values =
      shards[kFirstIndex].to_vector<std::uint32_t>();
  require(values.size() == batch_size,
          "fast device argmax did not return one value per batch row");
  for (const std::uint32_t value : values) {
    require(value < kVocabularySize,
            "fast device argmax selected an invalid token ID");
  }
  return values;
}

std::vector<std::uint8_t> bf16_bytes(const std::vector<bfloat16> &values) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(values.size() * kBf16Bytes);
  for (const bfloat16 value : values) {
    const std::uint16_t bits = std::bit_cast<std::uint16_t>(value);
    bytes.push_back(static_cast<std::uint8_t>(bits));
    bytes.push_back(static_cast<std::uint8_t>(bits >> kBitsPerByte));
  }
  return bytes;
}

Json write_bf16_artifact(const std::filesystem::path &root,
                         std::string_view name, std::uint32_t card,
                         const std::vector<bfloat16> &values) {
  const std::string filename =
      std::string(name) + "-card-" + std::to_string(card) + ".bf16";
  const std::filesystem::path path = root / filename;
  const std::vector<std::uint8_t> bytes = bf16_bytes(values);
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  require(output.good(), "failed to create BF16 evidence artifact");
  output.write(reinterpret_cast<const char *>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  output.flush();
  require(output.good(), "failed to write BF16 evidence artifact");
  return Json{{"path", filename},
              {"card", card},
              {"element_count", values.size()},
              {"byte_count", bytes.size()},
              {"blake3", blake3_hex(bytes.data(), bytes.size())}};
}

Json persist_sharded_artifact(const std::filesystem::path &root,
                              std::string_view name,
                              const HostBf16Shards &shards) {
  Json artifacts = Json::array();
  for (std::size_t card = 0U; card < kExpectedDeviceCount; ++card) {
    artifacts.push_back(write_bf16_artifact(
        root, name, static_cast<std::uint32_t>(card), shards.shards[card]));
  }
  return artifacts;
}

HostBf16Shards read_model_state(DeviceContext &context,
                                const ModelState &state) {
  HostBf16Shards combined;
  for (const LayerState &layer : state.layers) {
    for (const Tensor *tensor :
         {&layer.attention_previous, &layer.ffn_previous, &layer.matrix}) {
      const HostBf16Shards current = read_bf16_shards(context, *tensor);
      for (std::size_t card = 0U; card < kExpectedDeviceCount; ++card) {
        combined.shards[card].insert(combined.shards[card].end(),
                                     current.shards[card].begin(),
                                     current.shards[card].end());
      }
    }
  }
  return combined;
}

void require_zero_state(const HostBf16Shards &state) {
  for (const auto &shard : state.shards) {
    require(!shard.empty(), "initial recurrent state shard must not be empty");
    for (const bfloat16 value : shard) {
      require(std::bit_cast<std::uint16_t>(value) == 0U,
              "initial recurrent state contains a nonzero BF16 bit pattern");
    }
  }
}

struct StepResult {
  std::uint32_t next_token;
  Json receipt;
};

StepResult advance_token(DeviceContext &context, const ModelWeights &weights,
                         ModelState &state, std::uint32_t input_token,
                         const std::filesystem::path &evidence_root,
                         std::string_view session, std::uint32_t step_index) {
  const std::string step_name =
      std::string(session) + "-step-" + std::to_string(step_index);
  context.set_stage(step_name + "/embedding");
  const Tensor token = context.uint32_tensor({input_token}, {1U, 1U});
  Tensor hidden =
      context.checked(ttnn::embedding(token, weights.embeddings, std::nullopt,
                                      ttnn::Layout::TILE),
                      "embedding");
  hidden = op_reshape(context, hidden, {1U, 1U, 1U, kHiddenSize});
  context.flush_profiler(step_name + "/embedding");

  std::optional<Tensor> value_anchor;
  for (std::uint32_t layer_index = 0U; layer_index < kLayerCount;
       ++layer_index) {
    const std::string stage_prefix =
        step_name + "/layer-" + std::to_string(layer_index);
    hidden =
        execute_layer(context, weights.layers[layer_index], hidden,
                      state.layers[layer_index], value_anchor, stage_prefix);
  }
  require(value_anchor.has_value(),
          "full model did not establish the token-local value anchor");

  context.set_stage(step_name + "/final-normalization");
  const Tensor final_hidden =
      op_layer_norm(context, hidden, weights.final_norm_weight,
                    weights.final_norm_bias, kLayerNormEpsilon);
  context.set_stage(step_name + "/language-model-head");
  const Tensor logits = op_matmul(context, final_hidden, weights.head);
  const Tensor logits_row_major =
      context.checked(ttnn::untilize(logits), "untilize-logits");
  context.set_stage(step_name + "/greedy-selection");
  const Tensor selected = context.checked(
      ttnn::argmax(logits_row_major, kLastDimension, true), "argmax");
  const auto selected_by_card = read_u32_shards(context, selected);
  require(selected_by_card[kFirstIndex] == selected_by_card[kSecondIndex],
          "the two P150 cards selected different greedy token IDs");
  require(selected_by_card[kFirstIndex] < kVocabularySize,
          "device argmax selected an invalid token ID");
  context.flush_profiler(step_name + "/selection");

  const HostBf16Shards final_hidden_shards =
      read_bf16_shards(context, final_hidden);
  const HostBf16Shards logits_shards =
      read_bf16_shards(context, logits_row_major);
  const HostBf16Shards state_shards = read_model_state(context, state);
  Json receipt{
      {"step_index", step_index},
      {"input_token_id", input_token},
      {"selected_token_ids_by_card", selected_by_card},
      {"selected_token_id", selected_by_card[kFirstIndex]},
      {"final_hidden",
       persist_sharded_artifact(evidence_root, step_name + "-final-hidden",
                                final_hidden_shards)},
      {"logits", persist_sharded_artifact(evidence_root, step_name + "-logits",
                                          logits_shards)},
      {"recurrent_state",
       persist_sharded_artifact(evidence_root, step_name + "-state",
                                state_shards)}};
  context.flush_profiler(step_name + "/evidence-readback");
  return StepResult{selected_by_card[kFirstIndex], std::move(receipt)};
}

struct SessionResult {
  std::vector<std::uint32_t> generated_tokens;
  Json receipt;
  ModelState control_state;
  std::uint32_t control_input_token;
  bool has_control_state;
};

SessionResult run_session(DeviceContext &context, const ModelWeights &weights,
                          const RuntimeRequest &request,
                          const std::filesystem::path &evidence_root,
                          std::string_view session_name) {
  ModelState state = zero_model_state(context);
  const HostBf16Shards initial_state = read_model_state(context, state);
  require_zero_state(initial_state);
  const Json initial_state_artifacts = persist_sharded_artifact(
      evidence_root, std::string(session_name) + "-initial-state",
      initial_state);
  Json steps = Json::array();
  std::uint32_t step_index = 0U;
  StepResult current{0U, Json{}};
  for (const std::uint32_t prompt_token : request.prompt_tokens) {
    current = advance_token(context, weights, state, prompt_token,
                            evidence_root, session_name, step_index);
    steps.push_back(current.receipt);
    ++step_index;
  }

  std::vector<std::uint32_t> generated;
  generated.reserve(request.max_new_tokens);
  generated.push_back(current.next_token);
  ModelState control_state;
  std::uint32_t control_input_token = 0U;
  bool has_control_state = false;
  while (generated.size() < request.max_new_tokens) {
    require(generated.back() != kGenerationEosToken,
            "generation-config EOS was selected before the retained-state "
            "proof minimum");
    if (!has_control_state) {
      control_state = state;
      control_input_token = generated.back();
      has_control_state = true;
    }
    current = advance_token(context, weights, state, generated.back(),
                            evidence_root, session_name, step_index);
    steps.push_back(current.receipt);
    generated.push_back(current.next_token);
    ++step_index;
    if (current.next_token == kGenerationEosToken &&
        generated.size() < kMinimumGeneratedTokens) {
      fail("generation-config EOS prevented the retained-state proof minimum");
    }
    if (current.next_token == kGenerationEosToken) {
      break;
    }
  }
  require(generated.size() >= kMinimumGeneratedTokens,
          "session did not generate the retained-state proof minimum");
  require(has_control_state, "session did not retain a pre-control state");
  return SessionResult{
      generated,
      Json{{"name", session_name},
           {"initial_recurrent_state", initial_state_artifacts},
           {"generated_token_ids", generated},
           {"steps", std::move(steps)}},
      std::move(control_state), control_input_token, has_control_state};
}

Json run_controls(DeviceContext &context, const ModelWeights &weights,
                  const SessionResult &cold,
                  const std::filesystem::path &evidence_root) {
  require(cold.has_control_state,
          "cold session lacks a retained control state");
  ModelState reset = zero_model_state(context);
  const StepResult reset_result =
      advance_token(context, weights, reset, cold.control_input_token,
                    evidence_root, "control-reset", 0U);

  ModelState transposed = cold.control_state;
  for (std::uint32_t layer_index = 0U; layer_index < kLayerCount;
       ++layer_index) {
    context.set_stage("control-transposed/layer-" +
                      std::to_string(layer_index));
    transposed.layers[layer_index].matrix =
        op_transpose(context, transposed.layers[layer_index].matrix,
                     kMatrixRowDimension, kMatrixColumnDimension);
  }
  const StepResult transposed_result =
      advance_token(context, weights, transposed, cold.control_input_token,
                    evidence_root, "control-transposed", 0U);
  return Json{{"input_token_id", cold.control_input_token},
              {"reset", reset_result.receipt},
              {"transposed", transposed_result.receipt}};
}

struct FastRequest {
  std::string request_id;
  std::vector<std::uint32_t> prompt_tokens;
  std::uint32_t max_new_tokens;
  std::size_t state_slot;
};

struct FastWorkerRequest {
  std::string run_id;
  std::vector<FastRequest> requests;
};

struct FastWorkerArguments {
  int physical_device_id;
  std::filesystem::path requests_path;
  std::filesystem::path output_path;
};

struct FastSlot {
  const FastRequest *request;
  std::size_t prompt_index;
  std::vector<std::uint32_t> generated_tokens;
  std::uint64_t processed_token_count;
  std::chrono::steady_clock::time_point started_at;
};

struct FastBatchStepPlan {
  std::vector<std::uint32_t> input_tokens;
  std::vector<bool> consumes_prompt;
  std::vector<bool> selection_required;
  bool any_selection_required;
};

FastWorkerArguments
parse_fast_worker_arguments(const std::vector<std::string> &arguments) {
  require(arguments.size() == kFastWorkerArgumentCount,
          "fast worker invocation has the wrong argument count");
  require(arguments[kModeArgumentIndex] == kFastWorkerArgument,
          "fast worker invocation must start with --fast-worker");
  require(arguments[kFastDeviceNameIndex] == kFastDeviceArgument,
          "fast worker invocation must place --device first");
  require(arguments[kFastRequestsNameIndex] == kFastRequestsArgument,
          "fast worker invocation must place --requests second");
  require(arguments[kFastOutputNameIndex] == kFastOutputArgument,
          "fast worker invocation must place --output last");
  const std::uint32_t device_id =
      parse_u32(arguments[kFastDeviceValueIndex], "fast worker device");
  require(device_id <= static_cast<std::uint32_t>(kPhysicalDeviceOne),
          "fast worker device must be physical device 0 or 1");
  const std::filesystem::path requests_path(
      arguments[kFastRequestsValueIndex]);
  const std::filesystem::path output_path(arguments[kFastOutputValueIndex]);
  require(requests_path.is_absolute() && output_path.is_absolute(),
          "fast worker request and output paths must be absolute");
  require(std::filesystem::is_regular_file(requests_path),
          "fast worker request path is not a regular file");
  require(!std::filesystem::exists(output_path),
          "fast worker output path already exists");
  require(!output_path.string().starts_with("/nix/store/"),
          "fast worker output must be outside the Nix store");
  return FastWorkerArguments{static_cast<int>(device_id), requests_path,
                             output_path};
}

FastWorkerRequest read_fast_worker_request(
    const std::filesystem::path &path) {
  std::ifstream input(path);
  require(input.good(), "failed to open fast worker request");
  const Json document = Json::parse(input);
  require(document.is_object() && document.size() == 3U,
          "fast worker request must contain exactly three fields");
  require(document.at("schema_version").get<std::uint32_t>() ==
              kSchemaVersion,
          "fast worker request schema mismatch");
  FastWorkerRequest request{document.at("run_id").get<std::string>(), {}};
  require(!request.run_id.empty(), "fast worker run ID must not be empty");
  const Json &raw_requests = document.at("requests");
  require(raw_requests.is_array() && !raw_requests.empty(),
          "fast worker request list must not be empty");
  require(raw_requests.size() <= kFastPerCardSlotCount,
          "fast worker request list exceeds card-local state slots");
  std::map<std::string, bool> request_ids;
  std::array<bool, kFastPerCardSlotCount> used_slots{};
  for (const Json &raw : raw_requests) {
    require(raw.is_object() && raw.size() == 4U,
            "fast request must contain exactly four fields");
    FastRequest current{
        raw.at("request_id").get<std::string>(),
        raw.at("prompt_token_ids").get<std::vector<std::uint32_t>>(),
        raw.at("max_new_tokens").get<std::uint32_t>(),
        raw.at("state_slot").get<std::size_t>()};
    require(!current.request_id.empty(),
            "fast request ID must not be empty");
    require(request_ids.emplace(current.request_id, true).second,
            "fast request IDs must be unique");
    require(!current.prompt_tokens.empty() &&
                current.prompt_tokens.size() <= kMaximumPromptTokens,
            "fast request prompt length is outside the package bounds");
    for (const std::uint32_t token : current.prompt_tokens) {
      require(token < kVocabularySize,
              "fast request prompt token exceeds the vocabulary");
    }
    require(current.max_new_tokens > 0U &&
                current.max_new_tokens <= kFastMaximumGeneratedTokens,
            "fast request generation count is outside the package bounds");
    require(current.state_slot < kFastPerCardSlotCount,
            "fast request state slot is outside the card-local range");
    require(!used_slots[current.state_slot],
            "fast worker received duplicate active state slots");
    used_slots[current.state_slot] = true;
    request.requests.push_back(std::move(current));
  }
  return request;
}

FastBatchStepPlan
plan_fast_batch_step(const std::vector<FastSlot> &slots,
                     const std::vector<bool> &completed) {
  require(!slots.empty() && slots.size() <= kFastPerCardSlotCount,
          "fast batch planner slot count is outside the card-local range");
  require(completed.size() == slots.size(),
          "fast batch planner completion count differs from slot count");
  FastBatchStepPlan plan{{}, {}, {}, false};
  plan.input_tokens.reserve(slots.size());
  plan.consumes_prompt.reserve(slots.size());
  plan.selection_required.reserve(slots.size());
  for (std::size_t index = 0U; index < slots.size(); ++index) {
    const FastSlot &slot = slots[index];
    require(slot.request != nullptr, "fast batch planner found an empty slot");
    if (completed[index]) {
      plan.input_tokens.push_back(kGenerationEosToken);
      plan.consumes_prompt.push_back(false);
      plan.selection_required.push_back(false);
      continue;
    }
    if (slot.prompt_index < slot.request->prompt_tokens.size()) {
      plan.input_tokens.push_back(
          slot.request->prompt_tokens[slot.prompt_index]);
      plan.consumes_prompt.push_back(true);
      const bool final_prompt_token =
          slot.prompt_index + 1U == slot.request->prompt_tokens.size();
      plan.selection_required.push_back(final_prompt_token);
      plan.any_selection_required |= final_prompt_token;
      continue;
    }
    require(!slot.generated_tokens.empty(),
            "fast batch planner has no generated token to continue");
    plan.input_tokens.push_back(slot.generated_tokens.back());
    plan.consumes_prompt.push_back(false);
    plan.selection_required.push_back(true);
    plan.any_selection_required = true;
  }
  return plan;
}

void apply_fast_batch_step(
    std::vector<FastSlot> &slots, std::vector<bool> &completed,
    const FastBatchStepPlan &plan,
    const std::optional<std::vector<std::uint32_t>> &selected_tokens) {
  require(completed.size() == slots.size() &&
              plan.input_tokens.size() == slots.size() &&
              plan.consumes_prompt.size() == slots.size() &&
              plan.selection_required.size() == slots.size(),
          "fast batch step shape differs from slot count");
  require(selected_tokens.has_value() == plan.any_selection_required,
          "fast batch selection result does not match the plan");
  if (selected_tokens.has_value()) {
    require(selected_tokens->size() == slots.size(),
            "fast batch selection count differs from slot count");
  }
  for (std::size_t index = 0U; index < slots.size(); ++index) {
    if (completed[index]) {
      continue;
    }
    FastSlot &slot = slots[index];
    require(slot.request != nullptr, "fast batch step found an empty slot");
    if (plan.consumes_prompt[index]) {
      ++slot.prompt_index;
    }
    ++slot.processed_token_count;
    if (plan.selection_required[index]) {
      slot.generated_tokens.push_back((*selected_tokens)[index]);
    }
    completed[index] =
        slot.generated_tokens.size() >= slot.request->max_new_tokens;
  }
}

std::optional<std::vector<std::uint32_t>>
advance_tokens_fast(DeviceContext &context, const ModelWeights &weights,
                    ModelState &state,
                    const std::vector<std::uint32_t> &input_tokens,
                    bool select_tokens, std::uint64_t step_index) {
  require(!input_tokens.empty() &&
              input_tokens.size() <= kFastPerCardSlotCount,
          "fast model batch size is outside the card-local range");
  require(state.layers.size() == kLayerCount,
          "fast model state has the wrong layer count");
  const auto batch_size = static_cast<std::uint32_t>(input_tokens.size());
  require(state.layers[kFirstIndex].matrix.logical_shape()[0] == batch_size,
          "fast model state batch size differs from token batch size");
  const std::string step_name =
      "fast-batch-step-" + std::to_string(step_index);
  context.set_stage(step_name + "/embedding");
  const Tensor token =
      context.uint32_tensor(input_tokens, {batch_size, 1U});
  Tensor hidden =
      context.checked(ttnn::embedding(token, weights.embeddings, std::nullopt,
                                      ttnn::Layout::TILE),
                      "embedding");
  hidden =
      op_reshape(context, hidden, {1U, 1U, batch_size, kHiddenSize});
  std::optional<Tensor> value_anchor;
  for (std::uint32_t layer_index = 0U; layer_index < kLayerCount;
       ++layer_index) {
    const std::string stage_prefix =
        step_name + "/layer-" + std::to_string(layer_index);
    hidden = execute_layer(context, weights.layers[layer_index], hidden,
                           state.layers[layer_index], value_anchor,
                           stage_prefix);
  }
  require(value_anchor.has_value(),
          "fast full model did not establish the token-local value anchor");
  if (!select_tokens) {
    return std::nullopt;
  }
  context.set_stage(step_name + "/final-normalization");
  const Tensor final_hidden =
      op_layer_norm(context, hidden, weights.final_norm_weight,
                    weights.final_norm_bias, kLayerNormEpsilon);
  context.set_stage(step_name + "/language-model-head");
  const Tensor logits = op_matmul(context, final_hidden, weights.head);
  const Tensor logits_row_major =
      context.checked(ttnn::untilize(logits), "untilize-logits");
  context.set_stage(step_name + "/greedy-selection");
  const Tensor selected = context.checked(
      ttnn::argmax(logits_row_major, kLastDimension, true), "argmax");
  return read_fast_u32(context, selected, input_tokens.size());
}

void write_json(const std::filesystem::path &path, const Json &value) {
  std::ofstream output(path, std::ios::trunc);
  require(output.good(), "failed to create runtime receipt");
  output << value.dump(2) << '\n';
  output.flush();
  require(output.good(), "failed to write runtime receipt");
}

std::string_view visible_device_value(int physical_device_id) {
  if (physical_device_id == kPhysicalDeviceZero) {
    return "0";
  }
  if (physical_device_id == kPhysicalDeviceOne) {
    return "1";
  }
  fail("visible device must be physical device 0 or 1");
}

void select_single_card_device(int physical_device_id) {
  const int descriptor_status =
      setenv("TT_MESH_GRAPH_DESC_PATH",
             RWKV7_PINNED_SINGLE_CARD_MESH_DESCRIPTOR_PATH, 1);
  require(descriptor_status == 0,
          "failed to select the pinned single-card mesh descriptor");
  const std::string_view visible_device =
      visible_device_value(physical_device_id);
  const int visibility_status =
      setenv("TT_VISIBLE_DEVICES", visible_device.data(), 1);
  require(visibility_status == 0, "failed to select the physical card");
}

std::filesystem::path parse_simulator_smoke_output(
    const std::vector<std::string> &arguments) {
  require(arguments.size() == kSimulatorSmokeArgumentCount,
          "simulator smoke invocation has the wrong argument count");
  require(arguments[kModeArgumentIndex] == kSimulatorSmokeArgument,
          "simulator smoke invocation must start with --simulator-smoke");
  require(arguments[kSimulatorOutputNameIndex] == kFastOutputArgument,
          "simulator smoke invocation must place --output last");
  const std::filesystem::path output(arguments[kSimulatorOutputValueIndex]);
  require(output.is_absolute() &&
              !output.string().starts_with("/nix/store/") &&
              !std::filesystem::exists(output),
          "simulator smoke output must be a new absolute non-store path");
  return output;
}

void run_simulator_smoke(const std::filesystem::path &output_path) {
  select_single_card_device(kPhysicalDeviceZero);
  const MappedFile checkpoint_file(RWKV7_PINNED_CHECKPOINT_PATH);
  const CheckpointIndex checkpoint(checkpoint_file);
  DeviceContext context(kPhysicalDeviceZero);
  context.set_profiler_flush_enabled(false);
  const ModelWeights weights = load_model(context, checkpoint);
  ModelState state = zero_fast_model_state(context, 1U);
  require(state.layers[kFirstIndex].matrix.dtype() == ttnn::DataType::FLOAT32,
          "simulator smoke recurrent matrix is not FP32");
  for (std::uint32_t step = 0U; step < kMinimumGeneratedTokens; ++step) {
    context.set_stage("simulator-smoke/embedding");
    const Tensor token = context.uint32_tensor({kGenerationEosToken}, {1U, 1U});
    Tensor hidden = context.checked(
        ttnn::embedding(token, weights.embeddings, std::nullopt,
                        ttnn::Layout::TILE),
        "embedding");
    hidden = op_reshape(context, hidden, {1U, 1U, 1U, kHiddenSize});
    std::optional<Tensor> value_anchor;
    static_cast<void>(execute_layer(
        context, weights.layers[kFirstIndex], hidden,
        state.layers[kFirstIndex], value_anchor,
        "simulator-smoke/layer-0/step-" + std::to_string(step)));
    require(value_anchor.has_value(),
            "simulator smoke layer did not establish its value anchor");
  }
  require(!context.operation_counts().empty(),
          "simulator smoke executed no TT-NN operations");
  const Json receipt{
      {"schema_version", kSchemaVersion},
      {"outcome", "simulated_single_card_layer_state_carry"},
      {"simulated", true},
      {"physical_hardware", false},
      {"architecture", "blackhole"},
      {"mesh_shape", {kMeshRows, kMeshRows}},
      {"selection_ordinal", kPhysicalDeviceZero},
      {"layer_count_executed", 1U},
      {"recurrent_steps", kMinimumGeneratedTokens},
      {"recurrent_matrix_residency", "device_fp32"},
      {"performance_claim", false},
      {"operation_counts", map_to_json(context.operation_counts())}};
  write_json(output_path, receipt);
  std::cout << output_path << '\n';
}

void run_fast_worker(const FastWorkerArguments &arguments) {
  select_single_card_device(arguments.physical_device_id);
  const char *const profiler_value = std::getenv("TT_METAL_DEVICE_PROFILER");
  const bool profiler_attributed =
      profiler_value != nullptr &&
      std::string_view(profiler_value) == kEnabledEnvironmentValue;
  const FastWorkerRequest request =
      read_fast_worker_request(arguments.requests_path);
  const MappedFile checkpoint_file(RWKV7_PINNED_CHECKPOINT_PATH);
  const CheckpointIndex checkpoint(checkpoint_file);
  DeviceContext context(kFilteredLogicalDevice);
  context.set_profiler_flush_enabled(false);
  context.set_operation_recording_enabled(false);
  const ModelWeights weights = load_model(context, checkpoint);
  const auto batch_size =
      static_cast<std::uint32_t>(request.requests.size());

  ModelState warmup_state = zero_fast_model_state(context, batch_size);
  std::vector<std::uint32_t> warmup_tokens(
      request.requests.size(), kGenerationEosToken);
  for (std::uint32_t index = 0U; index < kFastWarmupTokenCount; ++index) {
    const auto selected = advance_tokens_fast(
        context, weights, warmup_state, warmup_tokens, true, index);
    require(selected.has_value(),
            "fast worker warmup did not select tokens");
    warmup_tokens = *selected;
  }
  ModelState state = zero_fast_model_state(context, batch_size);
  const std::size_t cache_after_warmup = context.program_cache_entries();
  require(cache_after_warmup > 0U,
          "fast worker warmup did not populate the program cache");

  std::vector<FastSlot> slots;
  slots.reserve(request.requests.size());
  for (const FastRequest &current : request.requests) {
    slots.push_back(FastSlot{&current, 0U, {}, 0U, {}});
  }
  std::vector<bool> completed(slots.size(), false);
  Json request_metrics = Json::array();
  const auto timed_start = std::chrono::steady_clock::now();
  for (FastSlot &slot : slots) {
    slot.started_at = timed_start;
  }
  std::size_t completed_count = 0U;
  std::uint64_t batch_step_count = 0U;
  while (completed_count < slots.size()) {
    const std::vector<bool> completed_before = completed;
    const FastBatchStepPlan plan = plan_fast_batch_step(slots, completed);
    const auto selected = advance_tokens_fast(
        context, weights, state, plan.input_tokens,
        plan.any_selection_required, batch_step_count);
    apply_fast_batch_step(slots, completed, plan, selected);
    ++batch_step_count;
    const auto completed_at = std::chrono::steady_clock::now();
    for (std::size_t slot_index = 0U; slot_index < slots.size();
         ++slot_index) {
      if (completed_before[slot_index] || !completed[slot_index]) {
        continue;
      }
      FastSlot &slot = slots[slot_index];
      const double latency_seconds =
          std::chrono::duration<double>(completed_at - slot.started_at).count();
      require(std::isfinite(latency_seconds) && latency_seconds > 0.0,
              "fast request latency is not finite and positive");
      request_metrics.push_back(
          Json{{"request_id", slot.request->request_id},
               {"card_id", arguments.physical_device_id},
               {"state_slot", slot.request->state_slot},
               {"token_count", slot.processed_token_count},
               {"latency_seconds", latency_seconds},
               {"generated_token_ids", slot.generated_tokens}});
      ++completed_count;
    }
  }
  const auto timed_end = std::chrono::steady_clock::now();
  const double elapsed_seconds =
      std::chrono::duration<double>(timed_end - timed_start).count();
  require(std::isfinite(elapsed_seconds) && elapsed_seconds > 0.0,
          "fast worker elapsed time is not finite and positive");
  require(context.program_cache_entries() == cache_after_warmup,
          "fast worker compiled a new program during timed decode");

  context.set_profiler_flush_enabled(profiler_attributed);
  context.flush_profiler("fast-worker-post-timing-attribution");
  const Json receipt{
      {"schema_version", kSchemaVersion},
      {"run_id", request.run_id},
      {"physical_device_ids", {arguments.physical_device_id}},
      {"visible_physical_device_ids", {arguments.physical_device_id}},
      {"logical_device_ids", {kFilteredLogicalDevice}},
      {"physical_hardware", true},
      {"profiler_attributed", profiler_attributed},
      {"elapsed_seconds", elapsed_seconds},
      {"requests", std::move(request_metrics)},
      {"execution",
       {{"scheduler_mode", "deterministic_continuous_batching"},
        {"scheduler_tie_break", "least_loaded_card_then_lowest_card_then_lowest_slot"},
        {"replica_mode", "independent_card_replica"},
        {"recurrent_state_residency", "device_fp32"},
        {"timed_region", "decode_only"},
        {"warmup_token_count", kFastWarmupTokenCount},
        {"program_cache_stable", true},
        {"dynamic_token_and_state_bindings_validated", true},
        {"state_readback_during_timing", false},
        {"profiler_flush_during_timing", false},
        {"control_replay_during_timing", false},
        {"artifact_writes_during_timing", false}}}};
  write_json(arguments.output_path, receipt);
  std::cout << arguments.output_path << '\n';
}

struct EvidenceEnvironment {
  std::optional<std::string> logs_path;
  std::optional<std::string> profiler_directory;
  std::optional<std::string> inspector;
  std::optional<std::string> inspector_initialization;
  std::optional<std::string> inspector_runtime_entries;
  std::optional<std::string> inspector_rpc;
  std::optional<std::string> device_profiler;
  std::optional<std::string> mid_run_dump;
  std::optional<std::string> watcher;
  std::optional<std::string> dprint_cores;
};

void require_environment_value(const std::optional<std::string> &actual,
                               std::string_view name,
                               std::string_view expected) {
  require(actual.has_value(), std::string(name) + " is required");
  require(*actual == expected,
          std::string(name) + " does not match the required value");
}

// r[impl tenstorrent.rwkv7_p150x2.profiler_isolation]
void validate_evidence_environment(const EvidenceEnvironment &environment,
                                   const std::filesystem::path &root) {
  require_environment_value(environment.logs_path, "TT_METAL_LOGS_PATH",
                            (root / kLogsDirectoryName).string());
  require_environment_value(environment.profiler_directory,
                            "TT_METAL_PROFILER_DIR",
                            (root / kProfilerDirectoryName).string());
  require_environment_value(environment.inspector, "TT_METAL_INSPECTOR",
                            kEnabledEnvironmentValue);
  require_environment_value(environment.inspector_initialization,
                            "TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT",
                            kEnabledEnvironmentValue);
  require_environment_value(environment.inspector_runtime_entries,
                            "TT_METAL_INSPECTOR_LOG_RUNTIME_ENTRIES",
                            kEnabledEnvironmentValue);
  require_environment_value(environment.inspector_rpc, "TT_METAL_INSPECTOR_RPC",
                            kDisabledEnvironmentValue);
  require_environment_value(environment.device_profiler,
                            "TT_METAL_DEVICE_PROFILER",
                            kEnabledEnvironmentValue);
  // The runtime tracks per-stage profiler byte ranges into the device CSV.
  // That file is only appended mid-run when TT_METAL_PROFILER_MID_RUN_DUMP is
  // enabled; without it ProcessDeviceProfilerResults never calls
  // dumpDeviceResults and the CSV appears only at device close, which makes
  // every flush_profiler existence check fail.
  require_environment_value(environment.mid_run_dump,
                            "TT_METAL_PROFILER_MID_RUN_DUMP",
                            kEnabledEnvironmentValue);
  require(!environment.watcher.has_value(),
          "TT_METAL_WATCHER must be unset while profiling");
  require(!environment.dprint_cores.has_value(),
          "TT_METAL_DPRINT_CORES must be unset while profiling");
}

std::optional<std::string> read_environment_value(std::string_view name) {
  const std::string name_string(name);
  const char *const value = std::getenv(name_string.c_str());
  if (value == nullptr) {
    return std::nullopt;
  }
  return std::string(value);
}

EvidenceEnvironment capture_evidence_environment() {
  return EvidenceEnvironment{
      read_environment_value("TT_METAL_LOGS_PATH"),
      read_environment_value("TT_METAL_PROFILER_DIR"),
      read_environment_value("TT_METAL_INSPECTOR"),
      read_environment_value("TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT"),
      read_environment_value("TT_METAL_INSPECTOR_LOG_RUNTIME_ENTRIES"),
      read_environment_value("TT_METAL_INSPECTOR_RPC"),
      read_environment_value("TT_METAL_DEVICE_PROFILER"),
      read_environment_value("TT_METAL_PROFILER_MID_RUN_DUMP"),
      read_environment_value("TT_METAL_WATCHER"),
      read_environment_value("TT_METAL_DPRINT_CORES")};
}

void require_evidence_environment(const std::filesystem::path &root) {
  validate_evidence_environment(capture_evidence_environment(), root);
}

void prepare_evidence_root(const std::filesystem::path &root) {
  require(std::filesystem::exists(root),
          "evidence root must exist before runtime launch");
  require(std::filesystem::is_directory(root),
          "evidence root must be a directory");
  const std::filesystem::path canonical = std::filesystem::canonical(root);
  require(canonical == root, "evidence root must already be canonical");
  require(std::filesystem::is_empty(root),
          "evidence root must be empty before runtime launch");
  const std::filesystem::path probe = root / ".write-probe";
  {
    std::ofstream output(probe, std::ios::trunc);
    require(output.good(), "evidence root is not writable");
    output << "probe\n";
  }
  require(std::filesystem::remove(probe),
          "failed to remove evidence-root write probe");
  require(std::filesystem::create_directory(root / kLogsDirectoryName),
          "failed to create the Inspector log root");
  require(std::filesystem::create_directory(root / kProfilerDirectoryName),
          "failed to create the device-profiler root");
}

Json map_to_json(const std::map<std::string, std::uint64_t> &values) {
  Json output = Json::object();
  for (const auto &[name, value] : values) {
    output[name] = value;
  }
  return output;
}

Json stages_to_json(
    const std::map<std::string, std::map<std::string, std::uint64_t>> &stages) {
  Json output = Json::object();
  for (const auto &[stage, operations] : stages) {
    output[stage] = map_to_json(operations);
  }
  return output;
}

Json profiler_ranges_to_json(const std::vector<ProfilerRange> &ranges) {
  Json output = Json::array();
  for (const ProfilerRange &range : ranges) {
    output.push_back(Json{{"stage", range.stage},
                          {"begin_byte", range.begin_byte},
                          {"end_byte", range.end_byte}});
  }
  return output;
}

Json events_to_json(const std::vector<OperationEvent> &events) {
  Json output = Json::array();
  for (const OperationEvent &event : events) {
    output.push_back(Json{{"ordinal", event.ordinal},
                          {"stage", event.stage},
                          {"operation", event.operation}});
  }
  return output;
}

// r[impl tenstorrent.rwkv7_p150x2.full_device_runtime.complete_path]
void run_runtime(const RuntimeRequest &request) {
  prepare_evidence_root(request.evidence_root);
  require_evidence_environment(request.evidence_root);
  const MappedFile checkpoint_file(RWKV7_PINNED_CHECKPOINT_PATH);
  const CheckpointIndex checkpoint(checkpoint_file);
  DeviceContext context;
  const ModelWeights weights = load_model(context, checkpoint);

  const std::size_t cache_before_cold = context.program_cache_entries();
  SessionResult cold =
      run_session(context, weights, request, request.evidence_root, "cold");
  const std::size_t cache_after_cold = context.program_cache_entries();
  SessionResult warm =
      run_session(context, weights, request, request.evidence_root, "warm");
  const std::size_t cache_after_warm = context.program_cache_entries();
  require(cache_after_cold > cache_before_cold,
          "cold session did not populate the TT-NN program cache");
  require(cache_after_warm == cache_after_cold,
          "warm session compiled an unexpected new program");
  require(cold.generated_tokens == warm.generated_tokens,
          "cold and warm sessions selected different token IDs");
  const Json controls =
      run_controls(context, weights, cold, request.evidence_root);

  const auto device_ids = context.mesh().get_device_ids();
  const Json artifact_blake3 = Json::parse(RWKV7_PINNED_ARTIFACT_BLAKE3_JSON);
  require(artifact_blake3.is_object(),
          "pinned model artifact identities must be a JSON object");
  require(artifact_blake3.value("checkpoint", "") == kExpectedCheckpointBlake3,
          "pinned checkpoint identity differs from the artifact identity map");
  Json receipt{
      {"schema_version", kSchemaVersion},
      {"model",
       {{"target_name", kTargetName},
        {"id", kModelId},
        {"revision", kModelRevision},
        {"architecture_family", kArchitectureFamily},
        {"state_kind", kStateKind},
        {"precision", kModelPrecision},
        {"architecture_adapter", kArchitectureAdapter},
        {"evidence_adapter", kEvidenceAdapter},
        {"artifact_blake3", artifact_blake3},
        {"checkpoint_path", RWKV7_PINNED_CHECKPOINT_PATH},
        {"checkpoint_byte_count", kExpectedCheckpointBytes},
        {"checkpoint_blake3", kExpectedCheckpointBlake3}}},
      {"mesh",
       {{"id", kTopologyId},
        {"descriptor", RWKV7_PINNED_MESH_DESCRIPTOR_PATH},
        {"shape", {kMeshRows, kMeshColumns}},
        {"physical_device_ids", {kPhysicalDeviceZero, kPhysicalDeviceOne}},
        {"opened_device_ids",
         {static_cast<int>(device_ids[kFirstIndex]),
          static_cast<int>(device_ids[kSecondIndex])}},
        {"replication", kReplication}}},
      {"request",
       {{"prompt_token_ids", request.prompt_tokens},
        {"max_new_tokens", request.max_new_tokens}}},
      {"program_cache",
       {{"before_cold", cache_before_cold},
        {"after_cold", cache_after_cold},
        {"after_warm", cache_after_warm}}},
      {"cold", cold.receipt},
      {"warm", warm.receipt},
      {"controls", controls},
      {"raw_evidence_roots",
       {{"inspector",
         (request.evidence_root / kInspectorRelativeDirectory).string()},
        {"device_profiler_csv",
         (request.evidence_root / kProfilerCsvRelativePath).string()}}},
      {"operation_counts", map_to_json(context.operation_counts())},
      {"stage_operation_counts",
       stages_to_json(context.stage_operation_counts())},
      {"operation_events", events_to_json(context.operation_events())},
      {"profiler_ranges", profiler_ranges_to_json(context.profiler_ranges())},
      {"host_model_arithmetic", false},
      {"cpu_model_fallback", false},
      {"generated_text_published", false}};
  const std::filesystem::path receipt_path =
      request.evidence_root / "runtime-receipt.json";
  write_json(receipt_path, receipt);
  std::cout << receipt_path << '\n';
}

template <typename Operation>
void expect_operation_failure(Operation operation,
                              std::string_view expected_message) {
  try {
    operation();
  } catch (const std::exception &error) {
    require(std::string_view(error.what()).find(expected_message) !=
                std::string_view::npos,
            "negative self-test returned the wrong diagnostic");
    return;
  }
  fail("negative self-test unexpectedly accepted invalid input");
}

void expect_failure(const std::vector<std::string> &arguments,
                    std::string_view expected_message) {
  expect_operation_failure(
      [&arguments]() { static_cast<void>(parse_runtime_request(arguments)); },
      expected_message);
}

void run_self_test() {
  const std::vector<std::string> valid = {"rwkv7-p150x2-runtime",
                                          "--tokens",
                                          "0,2",
                                          "--max-new",
                                          "2",
                                          "--evidence-root",
                                          "/tmp/rwkv7-self-test"};
  const RuntimeRequest request = parse_runtime_request(valid);
  require(request.prompt_tokens ==
              std::vector<std::uint32_t>{0U, kSelfTestTokenTwo},
          "positive request parser self-test changed token IDs");
  require(request.max_new_tokens == kMinimumGeneratedTokens,
          "positive request parser self-test changed generation bound");
  require(mesh_descriptor_for_device_count(1U) ==
              RWKV7_PINNED_SINGLE_CARD_MESH_DESCRIPTOR_PATH,
          "single-card descriptor selection changed");
  require(mesh_descriptor_for_device_count(kExpectedDeviceCount) ==
              RWKV7_PINNED_MESH_DESCRIPTOR_PATH,
          "dual-card descriptor selection changed");
  require(visible_device_value(kPhysicalDeviceZero) == "0",
          "card 0 visibility changed");
  require(visible_device_value(kPhysicalDeviceOne) == "1",
          "card 1 visibility changed");
  expect_operation_failure(
      []() { static_cast<void>(mesh_descriptor_for_device_count(0U)); },
      "outside the supported range");
  expect_operation_failure(
      []() {
        static_cast<void>(
            mesh_descriptor_for_device_count(kExpectedDeviceCount + 1U));
      },
      "outside the supported range");
  expect_operation_failure(
      []() { static_cast<void>(visible_device_value(-1)); },
      "physical device 0 or 1");
  expect_operation_failure(
      []() {
        static_cast<void>(visible_device_value(kPhysicalDeviceOne + 1));
      },
      "physical device 0 or 1");

  std::vector<std::string> malformed = valid;
  malformed[kTokensValueIndex] = "1,";
  expect_failure(malformed, "end with a separator");
  malformed = valid;
  malformed[kMaxNewValueIndex] = "1";
  expect_failure(malformed, "proof minimum");
  malformed = valid;
  malformed[kEvidenceValueIndex] = "relative";
  expect_failure(malformed, "must be absolute");
  malformed = valid;
  malformed.push_back("unexpected");
  expect_failure(malformed, "wrong argument count");

  // r[verify tenstorrent.rwkv7_p150x2.profiler_isolation.clean]
  const std::filesystem::path environment_root("/tmp/rwkv7-environment-self-test");
  const EvidenceEnvironment valid_environment{
      (environment_root / kLogsDirectoryName).string(),
      (environment_root / kProfilerDirectoryName).string(),
      std::string(kEnabledEnvironmentValue),
      std::string(kEnabledEnvironmentValue),
      std::string(kEnabledEnvironmentValue),
      std::string(kDisabledEnvironmentValue),
      std::string(kEnabledEnvironmentValue),
      std::string(kEnabledEnvironmentValue),
      std::nullopt,
      std::nullopt};
  validate_evidence_environment(valid_environment, environment_root);

  EvidenceEnvironment missing_mid_run_dump = valid_environment;
  missing_mid_run_dump.mid_run_dump = std::nullopt;
  expect_operation_failure(
      [&missing_mid_run_dump, &environment_root]() {
        validate_evidence_environment(missing_mid_run_dump, environment_root);
      },
      "TT_METAL_PROFILER_MID_RUN_DUMP is required");

  // r[verify tenstorrent.rwkv7_p150x2.profiler_isolation.watcher_zero]
  for (const std::string_view watcher_value :
       {kDisabledEnvironmentValue, kEnabledEnvironmentValue}) {
    EvidenceEnvironment invalid_environment = valid_environment;
    invalid_environment.watcher = watcher_value;
    expect_operation_failure(
        [&invalid_environment, &environment_root]() {
          validate_evidence_environment(invalid_environment, environment_root);
        },
        "TT_METAL_WATCHER must be unset");
  }

  // r[verify tenstorrent.rwkv7_p150x2.profiler_isolation.dprint_present]
  for (const std::string_view dprint_value : {std::string_view{},
                                               std::string_view("all")}) {
    EvidenceEnvironment invalid_environment = valid_environment;
    invalid_environment.dprint_cores = dprint_value;
    expect_operation_failure(
        [&invalid_environment, &environment_root]() {
          validate_evidence_environment(invalid_environment, environment_root);
        },
        "TT_METAL_DPRINT_CORES must be unset");
  }

  constexpr std::uint32_t kFirstPromptToken = 10U;
  constexpr std::uint32_t kSecondPromptToken = 11U;
  constexpr std::uint32_t kThirdPromptToken = 12U;
  constexpr std::uint32_t kFirstSelectedToken = 20U;
  constexpr std::uint32_t kSecondSelectedToken = 21U;
  constexpr std::uint32_t kFirstFinalToken = 30U;
  constexpr std::uint32_t kSecondFinalToken = 31U;
  const FastRequest first_fast_request{
      "first", {kFirstPromptToken, kSecondPromptToken},
      kMinimumGeneratedTokens, kFirstIndex};
  const FastRequest second_fast_request{
      "second",
      {kFirstPromptToken, kSecondPromptToken, kThirdPromptToken},
      kMinimumGeneratedTokens, kSecondIndex};
  std::vector<FastSlot> fast_slots{
      FastSlot{&first_fast_request, 0U, {}, 0U, {}},
      FastSlot{&second_fast_request, 0U, {}, 0U, {}}};
  std::vector<bool> fast_completed(fast_slots.size(), false);
  const FastBatchStepPlan initial_plan =
      plan_fast_batch_step(fast_slots, fast_completed);
  require(initial_plan.input_tokens ==
              std::vector<std::uint32_t>{kFirstPromptToken,
                                         kFirstPromptToken} &&
              !initial_plan.any_selection_required,
          "positive fast batch planner self-test changed initial prompt plan");

  fast_slots[kFirstIndex].prompt_index = 1U;
  fast_slots[kSecondIndex].prompt_index = 2U;
  const FastBatchStepPlan final_prompt_plan =
      plan_fast_batch_step(fast_slots, fast_completed);
  require(final_prompt_plan.any_selection_required &&
              final_prompt_plan.selection_required ==
                  std::vector<bool>{true, true},
          "positive fast batch planner self-test omitted final prompt heads");
  apply_fast_batch_step(
      fast_slots, fast_completed, final_prompt_plan,
      std::vector<std::uint32_t>{kFirstSelectedToken,
                                 kSecondSelectedToken});
  require(fast_slots[kFirstIndex].generated_tokens ==
              std::vector<std::uint32_t>{kFirstSelectedToken} &&
              fast_slots[kSecondIndex].generated_tokens ==
                  std::vector<std::uint32_t>{kSecondSelectedToken} &&
              !fast_completed[kFirstIndex] && !fast_completed[kSecondIndex],
          "positive fast batch step self-test changed first selections");
  const FastBatchStepPlan generation_plan =
      plan_fast_batch_step(fast_slots, fast_completed);
  require(generation_plan.input_tokens ==
              std::vector<std::uint32_t>{kFirstSelectedToken,
                                         kSecondSelectedToken} &&
              generation_plan.any_selection_required,
          "positive fast batch planner self-test changed generation plan");
  apply_fast_batch_step(
      fast_slots, fast_completed, generation_plan,
      std::vector<std::uint32_t>{kFirstFinalToken, kSecondFinalToken});
  require(fast_completed[kFirstIndex] && fast_completed[kSecondIndex],
          "positive fast batch step self-test did not complete requests");

  expect_operation_failure(
      [&fast_slots]() {
        static_cast<void>(
            plan_fast_batch_step(fast_slots, std::vector<bool>{false}));
      },
      "completion count differs from slot count");
  expect_operation_failure(
      [&fast_slots]() {
        std::vector<bool> incomplete(fast_slots.size(), false);
        const FastBatchStepPlan plan =
            plan_fast_batch_step(fast_slots, incomplete);
        apply_fast_batch_step(fast_slots, incomplete, plan, std::nullopt);
      },
      "selection result does not match the plan");

  const std::vector<bfloat16> input = {
      std::bit_cast<bfloat16>(kSelfTestBitOne),
      std::bit_cast<bfloat16>(kSelfTestBitTwo),
      std::bit_cast<bfloat16>(kSelfTestBitThree),
      std::bit_cast<bfloat16>(kSelfTestBitFour),
      std::bit_cast<bfloat16>(kSelfTestBitFive),
      std::bit_cast<bfloat16>(kSelfTestBitSix)};
  const std::vector<bfloat16> transposed =
      transpose_matrix_bits(input, kSelfTestRows, kSelfTestColumns);
  const std::vector<std::uint16_t> expected_bits = {
      kSelfTestBitOne,  kSelfTestBitFour,  kSelfTestBitTwo,
      kSelfTestBitFive, kSelfTestBitThree, kSelfTestBitSix};
  require(transposed.size() == expected_bits.size(),
          "matrix transpose self-test changed element count");
  for (std::size_t index = 0U; index < expected_bits.size(); ++index) {
    require(std::bit_cast<std::uint16_t>(transposed[index]) ==
                expected_bits[index],
            "matrix transpose self-test changed orientation");
  }
  const MappedFile checkpoint_file(RWKV7_PINNED_CHECKPOINT_PATH);
  const CheckpointIndex checkpoint(checkpoint_file);
  validate_checkpoint_schema(checkpoint);
  std::cout << "rwkv7 P150x2 runtime self-test: PASS\n";
}

std::vector<std::string> collect_arguments(int argc, char **argv) {
  require(argc > 0, "process argument count is invalid");
  std::vector<std::string> arguments;
  arguments.reserve(static_cast<std::size_t>(argc));
  for (int index = 0; index < argc; ++index) {
    require(argv[index] != nullptr, "process argument is null");
    arguments.emplace_back(argv[index]);
  }
  return arguments;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const std::vector<std::string> arguments = collect_arguments(argc, argv);
    if (arguments.size() == kSelfTestArgumentCount &&
        arguments[kModeArgumentIndex] == kSelfTestArgument) {
      run_self_test();
      return EXIT_SUCCESS;
    }
    if (arguments.size() == kFastWorkerArgumentCount &&
        arguments[kModeArgumentIndex] == kFastWorkerArgument) {
      run_fast_worker(parse_fast_worker_arguments(arguments));
      return EXIT_SUCCESS;
    }
    if (arguments.size() == kSimulatorSmokeArgumentCount &&
        arguments[kModeArgumentIndex] == kSimulatorSmokeArgument) {
      run_simulator_smoke(parse_simulator_smoke_output(arguments));
      return EXIT_SUCCESS;
    }
    const RuntimeRequest request = parse_runtime_request(arguments);
    run_runtime(request);
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "rwkv7-p150x2-runtime: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
