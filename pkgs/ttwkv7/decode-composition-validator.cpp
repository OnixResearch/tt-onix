// Device-free composition contract validator for the TTWKV7 decode
// reader -> compute -> writer path.
//
// The validator never opens a device. It reads the reviewed contract fixture,
// the host construction source, the runtime-argument ABI header, and the three
// decode kernel sources, and checks that the ordered runtime argument schema,
// tile/layout expectations, circular-buffer indices/capacities/aliases, the
// bounded producer/consumer tile flow, and the host SetRuntimeArgs construction
// all still agree with the contract.
//
// Exit status: 0 contract satisfied, 1 contract violation, 2 usage/input error.

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <nlohmann/json.hpp>

#include "ttwkv7-decode-abi.h"

namespace {

using Json = nlohmann::ordered_json;

constexpr std::size_t kMaximumSourceBytes = 1U << 20U;

// Canonical ABI anchors. These are intentionally duplicated outside the
// header so a header edit that is rebuilt into this validator still fails.
constexpr std::size_t kContractReaderArgumentCount = 18;
constexpr std::size_t kContractComputeArgumentCount = 8;
constexpr std::size_t kContractWriterArgumentCount = 12;
constexpr std::size_t kContractInputCount = 6;
constexpr std::uint32_t kContractTileWidth = 32;
constexpr std::uint32_t kContractMaximumTokenCount = 32;
constexpr std::uint32_t kContractInstancePack = 4;
constexpr std::uint32_t kContractReaderChunkCount = 1;
constexpr std::uint32_t kContractTokensPerChunk = 1;
constexpr std::uint32_t kContractInstancesPerGroup = 1;

static_assert(ttwkv7::decode_abi::kReaderArgumentCount ==
              kContractReaderArgumentCount);
static_assert(ttwkv7::decode_abi::kComputeArgumentCount ==
              kContractComputeArgumentCount);
static_assert(ttwkv7::decode_abi::kWriterArgumentCount ==
              kContractWriterArgumentCount);
static_assert(ttwkv7::decode_abi::kInputCount == kContractInputCount);
static_assert(ttwkv7::decode_abi::kTileWidth == kContractTileWidth);
static_assert(ttwkv7::decode_abi::kMaximumTokenCount ==
              kContractMaximumTokenCount);
static_assert(ttwkv7::decode_abi::kDecodeInstancePack ==
              kContractInstancePack);
static_assert(ttwkv7::decode_abi::kDecodeReaderChunkCount ==
              kContractReaderChunkCount);
static_assert(ttwkv7::decode_abi::kDecodeTokensPerChunk ==
              kContractTokensPerChunk);
static_assert(ttwkv7::decode_abi::kDecodeInstancesPerGroup ==
              kContractInstancesPerGroup);

constexpr std::array<const char *,
                     ttwkv7::decode_abi::kReaderArgumentCount>
    kCanonicalReaderArgumentNames = {
        "HeadCount",      "TokenCount",        "TilesPerHeadDimension",
        "HeadTileRows",   "SequenceCount",     "Input0",
        "Input1",         "Input2",            "Input3",
        "Input4",         "Input5",            "State",
        "TokenCountCompatibility", "SelectorUnused", "ConstantsUnused",
        "ChunkCount",     "InstanceStart",     "InstanceEnd"};

constexpr std::array<const char *,
                     ttwkv7::decode_abi::kComputeArgumentCount>
    kCanonicalComputeArgumentNames = {
        "TilesPerHeadDimension", "InstanceCount", "TokenCount",
        "HeadTileRows",          "ChunkCount",    "InstanceStart",
        "InstanceEnd",           "InstancePack"};

constexpr std::array<const char *,
                     ttwkv7::decode_abi::kWriterArgumentCount>
    kCanonicalWriterArgumentNames = {
        "Output",         "InstanceStart",       "InstanceEnd",
        "InstanceCount",  "ChunkCount",          "HeadCount",
        "TilesPerHeadDimension", "ChannelTileCount", "TokenRows",
        "RealTokenCount", "TokensPerChunk",      "InstancesPerGroup"};

struct ContractViolation : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct UsageError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct SourceFile {
  std::string role;
  std::string path;
  std::string text;
};

[[nodiscard]] std::string trim(std::string_view value) {
  std::size_t begin = 0;
  std::size_t end = value.size();
  while (begin < end &&
         std::isspace(static_cast<unsigned char>(value[begin])) != 0) {
    ++begin;
  }
  while (end > begin &&
         std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
    --end;
  }
  return std::string(value.substr(begin, end - begin));
}

[[nodiscard]] std::string collapse_whitespace(std::string_view value) {
  std::string collapsed;
  collapsed.reserve(value.size());
  bool pending_space = false;
  for (const char character : value) {
    if (std::isspace(static_cast<unsigned char>(character)) != 0) {
      pending_space = !collapsed.empty();
      continue;
    }
    if (pending_space) {
      collapsed.push_back(' ');
      pending_space = false;
    }
    collapsed.push_back(character);
  }
  return collapsed;
}

[[nodiscard]] std::string read_bounded_file(const std::string &path,
                                            std::string_view description) {
  std::error_code error;
  if (!std::filesystem::is_regular_file(path, error) || error) {
    throw UsageError(std::string(description) +
                     " must be a readable regular file: " + path);
  }
  const std::uintmax_t size = std::filesystem::file_size(path, error);
  if (error || size == 0 || size > kMaximumSourceBytes) {
    throw UsageError(std::string(description) +
                     " size is outside the reviewed bound: " + path);
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw UsageError(std::string(description) + " could not be opened: " +
                     path);
  }
  std::ostringstream bytes;
  bytes << input.rdbuf();
  if (!input.good() && !input.eof()) {
    throw UsageError(std::string(description) +
                     " could not be read completely: " + path);
  }
  return bytes.str();
}

void require_contains(const SourceFile &source, std::string_view needle,
                      std::string_view message) {
  if (source.text.find(needle) == std::string::npos) {
    throw ContractViolation(source.role + ": " + std::string(message) +
                            " (missing '" + std::string(needle) + "')");
  }
}

void require_absent(const SourceFile &source, std::string_view needle,
                    std::string_view message) {
  if (source.text.find(needle) != std::string::npos) {
    throw ContractViolation(source.role + ": " + std::string(message) +
                            " (found forbidden '" + std::string(needle) +
                            "')");
  }
}

[[nodiscard]] std::size_t count_occurrences(std::string_view text,
                                            std::string_view needle) {
  if (needle.empty()) {
    return 0;
  }
  std::size_t count = 0;
  std::size_t position = 0;
  while ((position = text.find(needle, position)) != std::string_view::npos) {
    ++count;
    position += needle.size();
  }
  return count;
}

void require_occurrence_count(const SourceFile &source,
                              std::string_view needle, std::size_t expected,
                              std::string_view message) {
  const std::size_t actual = count_occurrences(source.text, needle);
  if (actual != expected) {
    throw ContractViolation(source.role + ": " + std::string(message) +
                            " (expected " + std::to_string(expected) +
                            " occurrence(s), found " + std::to_string(actual) +
                            ")");
  }
}

[[nodiscard]] std::string fingerprint(std::string_view bytes) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (const char character : bytes) {
    hash ^= static_cast<std::uint8_t>(character);
    hash *= 1099511628211ULL;
  }
  std::ostringstream hex;
  hex << std::hex << std::setw(16) << std::setfill('0') << hash;
  return hex.str();
}

[[nodiscard]] const Json &require_field(const Json &object,
                                        std::string_view key,
                                        std::string_view context) {
  const auto iterator = object.find(std::string(key));
  if (iterator == object.end()) {
    throw ContractViolation(std::string(context) + " lacks field '" +
                            std::string(key) + "'");
  }
  return *iterator;
}

[[nodiscard]] std::string require_string_field(const Json &object,
                                               std::string_view key,
                                               std::string_view context) {
  const Json &value = require_field(object, key, context);
  if (!value.is_string()) {
    throw ContractViolation(std::string(context) + "." + std::string(key) +
                            " must be a string");
  }
  return value.get<std::string>();
}

[[nodiscard]] std::uint32_t require_u32_field(const Json &object,
                                              std::string_view key,
                                              std::string_view context) {
  const Json &value = require_field(object, key, context);
  if (!value.is_number_unsigned()) {
    throw ContractViolation(std::string(context) + "." + std::string(key) +
                            " must be an unsigned integer");
  }
  return value.get<std::uint32_t>();
}

[[nodiscard]] std::vector<std::string> require_string_array(
    const Json &object, std::string_view key, std::string_view context) {
  const Json &value = require_field(object, key, context);
  if (!value.is_array()) {
    throw ContractViolation(std::string(context) + "." + std::string(key) +
                            " must be an array");
  }
  std::vector<std::string> items;
  items.reserve(value.size());
  for (const Json &item : value) {
    if (!item.is_string()) {
      throw ContractViolation(std::string(context) + "." + std::string(key) +
                              " must contain only strings");
    }
    items.push_back(item.get<std::string>());
  }
  return items;
}

[[nodiscard]] std::map<std::string, std::uint64_t>
parse_abi_constants(const std::string &text) {
  const std::regex pattern(
      R"(constexpr\s+std::(?:size_t|u?int(?:8|16|32|64)_t)\s+(k[A-Za-z0-9_]+)\s*=\s*([0-9]+)\s*;)");
  std::map<std::string, std::uint64_t> constants;
  for (std::sregex_iterator iterator(text.begin(), text.end(), pattern),
       end;
       iterator != end; ++iterator) {
    constants[(*iterator)[1].str()] =
        std::stoull((*iterator)[2].str());
  }
  return constants;
}

[[nodiscard]] std::vector<std::string>
parse_enum_members(const std::string &text, std::string_view enum_name) {
  const std::string marker =
      "enum class " + std::string(enum_name) + " : std::size_t {";
  const std::size_t begin = text.find(marker);
  if (begin == std::string::npos) {
    throw ContractViolation("ABI header lacks declaration for " +
                            std::string(enum_name));
  }
  const std::size_t body_begin = begin + marker.size();
  const std::size_t end = text.find("};", body_begin);
  if (end == std::string::npos) {
    throw ContractViolation("ABI header has an unterminated " +
                            std::string(enum_name) + " declaration");
  }
  const std::string body = text.substr(body_begin, end - body_begin);
  std::vector<std::string> members;
  std::size_t start = 0;
  while (start < body.size()) {
    const std::size_t comma = body.find(',', start);
    const std::string part =
        trim(body.substr(start, comma == std::string::npos
                                    ? std::string::npos
                                    : comma - start));
    if (!part.empty()) {
      std::size_t name_end = 0;
      while (name_end < part.size() &&
             (std::isalnum(static_cast<unsigned char>(part[name_end])) != 0 ||
              part[name_end] == '_')) {
        ++name_end;
      }
      if (name_end == 0) {
        throw ContractViolation("ABI header has a malformed " +
                                std::string(enum_name) + " member");
      }
      members.push_back(part.substr(0, name_end));
    }
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }
  return members;
}

struct RuntimeAssignment {
  std::string stage;
  std::string argument;
  std::string expression;
  bool input_loop = false;
};

[[nodiscard]] std::map<std::pair<std::string, std::string>, RuntimeAssignment>
parse_runtime_assignments(const std::string &text) {
  const std::regex pattern(
      R"(arguments\.(reader|compute|writer)\[index\(([A-Za-z_][A-Za-z0-9_]*)::([A-Za-z_][A-Za-z0-9_]*)\)\s*(\+\s*input)?\]\s*=\s*([^;]+);)");
  std::map<std::pair<std::string, std::string>, RuntimeAssignment> assignments;
  for (std::sregex_iterator iterator(text.begin(), text.end(), pattern), end;
       iterator != end; ++iterator) {
    RuntimeAssignment assignment;
    assignment.stage = (*iterator)[1].str();
    assignment.argument = (*iterator)[3].str();
    assignment.expression = collapse_whitespace((*iterator)[5].str());
    assignment.input_loop = (*iterator)[4].matched;
    const auto key = std::make_pair(assignment.stage, assignment.argument);
    if (assignments.find(key) != assignments.end()) {
      throw ContractViolation("ABI header assigns " + assignment.stage + "." +
                              assignment.argument + " more than once");
    }
    assignments.emplace(key, assignment);
  }
  return assignments;
}

[[nodiscard]] std::map<std::string, std::uint32_t>
parse_cb_constants(const std::string &text) {
  std::map<std::string, std::uint32_t> declarations;
  const std::string marker = "constexpr uint32_t ";
  const std::regex declarator(
      R"(^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([0-9]+)\s*$)");
  std::size_t position = 0;
  while ((position = text.find(marker, position)) != std::string::npos) {
    const std::size_t begin = position + marker.size();
    const std::size_t end = text.find(';', begin);
    if (end == std::string::npos) {
      break;
    }
    const std::string declaration = text.substr(begin, end - begin);
    std::size_t start = 0;
    while (start <= declaration.size()) {
      const std::size_t comma = declaration.find(',', start);
      const std::string part =
          declaration.substr(start, comma == std::string::npos
                                        ? std::string::npos
                                        : comma - start);
      std::smatch match;
      if (std::regex_match(part, match, declarator)) {
        declarations[match[1].str()] =
            static_cast<std::uint32_t>(std::stoul(match[2].str()));
      }
      if (comma == std::string::npos) {
        break;
      }
      start = comma + 1;
    }
    position = end + 1;
  }
  return declarations;
}

[[nodiscard]] std::set<std::uint32_t>
parse_numeric_argument_indices(const std::string &text) {
  const std::regex pattern(R"(get_arg_val<uint32_t>\(\s*([0-9]+)\s*\))");
  std::set<std::uint32_t> indices;
  for (std::sregex_iterator iterator(text.begin(), text.end(), pattern), end;
       iterator != end; ++iterator) {
    indices.insert(static_cast<std::uint32_t>(
        std::stoul((*iterator)[1].str())));
  }
  return indices;
}

void require_collapsed_contains(const SourceFile &source,
                                std::string_view needle,
                                std::string_view message) {
  const std::string collapsed = collapse_whitespace(source.text);
  if (collapsed.find(collapse_whitespace(needle)) == std::string::npos) {
    throw ContractViolation(source.role + ": " + std::string(message) +
                            " (missing '" +
                            collapse_whitespace(needle) + "')");
  }
}

void require_collapsed_occurrence_count(const SourceFile &source,
                                        std::string_view needle,
                                        std::size_t expected,
                                        std::string_view message) {
  const std::string collapsed = collapse_whitespace(source.text);
  const std::size_t actual =
      count_occurrences(collapsed, collapse_whitespace(needle));
  if (actual != expected) {
    throw ContractViolation(source.role + ": " + std::string(message) +
                            " (expected " + std::to_string(expected) +
                            " occurrence(s), found " + std::to_string(actual) +
                            ")");
  }
}

[[nodiscard]] std::string decode_cb_block(const SourceFile &host) {
  const std::string begin_marker =
      "for (int i = 0; i < 6; i++) MakeCB(prog, core, (CB)(CB::c_0 + i), P * "
      "St);";
  const std::string end_marker = "std::vector<uint32_t> rct, wct;";
  const std::size_t begin = host.text.find(begin_marker);
  if (begin == std::string::npos) {
    throw ContractViolation(
        "host construction lacks the decode circular-buffer block");
  }
  const std::size_t end = host.text.find(end_marker, begin);
  if (end == std::string::npos) {
    throw ContractViolation(
        "host construction has an unterminated decode circular-buffer block");
  }
  return host.text.substr(begin, end - begin);
}

void compare_argument_names(const std::vector<std::string> &actual,
                            const std::vector<std::string> &expected,
                            std::string_view stage) {
  if (actual.size() != expected.size()) {
    throw ContractViolation(std::string(stage) +
                            " argument-count drift (header has " +
                            std::to_string(actual.size()) + ", contract has " +
                            std::to_string(expected.size()) + ")");
  }
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (actual[index] != expected[index]) {
      throw ContractViolation(
          std::string(stage) + " argument order drift at index " +
          std::to_string(index) + " (header has '" + actual[index] +
          "', contract requires '" + expected[index] + "')");
    }
  }
}

void compare_canonical_names(
    const std::vector<std::string> &actual,
    const std::vector<std::string> &canonical,
    std::string_view stage) {
  if (actual.size() != canonical.size()) {
    throw ContractViolation(std::string(stage) +
                            " argument count does not match the compiled ABI "
                            "contract");
  }
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (actual[index] != canonical[index]) {
      throw ContractViolation(
          std::string(stage) + " argument meaning drift at index " +
          std::to_string(index) + " (header has '" + actual[index] +
          "', compiled contract requires '" + canonical[index] + "')");
    }
  }
}

[[nodiscard]] std::vector<std::uint32_t>
parse_u32_array(const Json &value, std::string_view context) {
  if (!value.is_array()) {
    throw ContractViolation(std::string(context) + " must be an array");
  }
  std::vector<std::uint32_t> items;
  items.reserve(value.size());
  for (const Json &item : value) {
    if (!item.is_number_unsigned()) {
      throw ContractViolation(std::string(context) +
                              " must contain only unsigned integers");
    }
    items.push_back(item.get<std::uint32_t>());
  }
  return items;
}

void require_argument_vector(const std::vector<std::uint32_t> &actual,
                             const Json &expected, std::string_view stage) {
  const std::vector<std::uint32_t> expected_values =
      parse_u32_array(expected, std::string("contract.") + std::string(stage));
  if (actual.size() != expected_values.size()) {
    throw ContractViolation(std::string(stage) +
                            " runtime argument count drift (ABI produced " +
                            std::to_string(actual.size()) +
                            ", contract requires " +
                            std::to_string(expected_values.size()) + ")");
  }
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (actual[index] != expected_values[index]) {
      throw ContractViolation(
          std::string(stage) + " runtime argument value drift at index " +
          std::to_string(index) + " (ABI produced " +
          std::to_string(actual[index]) + ", contract requires " +
          std::to_string(expected_values[index]) + ")");
    }
  }
}

struct CompositionConfig {
  std::uint32_t head_count;
  std::uint32_t head_size;
  std::uint32_t sequence_count;
  std::uint32_t token_count;
  std::uint32_t instance_start;
  std::uint32_t instance_end;
  ttwkv7::decode_abi::DecodeAddresses addresses;
};

[[nodiscard]] CompositionConfig parse_probe(
    const Json &runtime_arguments, std::string_view name) {
  const Json &probe =
      require_field(runtime_arguments, name, "contract.runtime_arguments");
  CompositionConfig config{};
  config.head_count = require_u32_field(probe, "head_count", "probe");
  config.head_size = require_u32_field(probe, "head_size", "probe");
  config.sequence_count =
      require_u32_field(probe, "sequence_count", "probe");
  config.token_count = require_u32_field(probe, "token_count", "probe");
  config.instance_start =
      require_u32_field(probe, "instance_start", "probe");
  config.instance_end = require_u32_field(probe, "instance_end", "probe");
  const Json &addresses = require_field(probe, "addresses", "probe");
  const std::vector<std::uint32_t> inputs =
      parse_u32_array(require_field(addresses, "inputs", "probe.addresses"),
                      "probe.addresses.inputs");
  if (inputs.size() != ttwkv7::decode_abi::kInputCount) {
    throw ContractViolation("probe.addresses.inputs must contain " +
                            std::to_string(ttwkv7::decode_abi::kInputCount) +
                            " addresses");
  }
  std::copy(inputs.begin(), inputs.end(), config.addresses.inputs.begin());
  config.addresses.state =
      require_u32_field(addresses, "state", "probe.addresses");
  config.addresses.output =
      require_u32_field(addresses, "output", "probe.addresses");
  return config;
}

[[nodiscard]] ttwkv7::decode_abi::DecodeRuntimeArguments
make_arguments_or_fail(const CompositionConfig &config,
                       std::string_view context) {
  const std::uint32_t tiles_per_head_dimension =
      config.head_size / ttwkv7::decode_abi::kTileWidth;
  const std::uint32_t head_tile_rows =
      (config.head_count + ttwkv7::decode_abi::kMaximumTokenCount - 1) /
      ttwkv7::decode_abi::kMaximumTokenCount;
  const std::uint32_t channel_tile_count =
      config.head_count * tiles_per_head_dimension;
  const ttwkv7::decode_abi::DecodeConfig abi_config{
      .head_count = config.head_count,
      .tiles_per_head_dimension = tiles_per_head_dimension,
      .head_tile_rows = head_tile_rows,
      .sequence_count = config.sequence_count,
      .token_count = config.token_count,
      .channel_tile_count = channel_tile_count,
  };
  const auto arguments = ttwkv7::decode_abi::make_decode_runtime_arguments(
      abi_config, config.addresses, config.instance_start,
      config.instance_end);
  if (!arguments) {
    throw ContractViolation(std::string(context) +
                            " is rejected by make_decode_runtime_arguments");
  }
  return *arguments;
}

void check_rejected_probe(const CompositionConfig &config,
                          std::string_view context) {
  const std::uint32_t tiles_per_head_dimension =
      config.head_size / ttwkv7::decode_abi::kTileWidth;
  const std::uint32_t head_tile_rows =
      (config.head_count + ttwkv7::decode_abi::kTileWidth - 1) /
      ttwkv7::decode_abi::kTileWidth;
  const std::uint32_t channel_tile_count =
      config.head_count * tiles_per_head_dimension;
  const ttwkv7::decode_abi::DecodeConfig abi_config{
      .head_count = config.head_count,
      .tiles_per_head_dimension = tiles_per_head_dimension,
      .head_tile_rows = head_tile_rows,
      .sequence_count = config.sequence_count,
      .token_count = config.token_count,
      .channel_tile_count = channel_tile_count,
  };
  if (ttwkv7::decode_abi::make_decode_runtime_arguments(
          abi_config, config.addresses, config.instance_start,
          config.instance_end)) {
    throw ContractViolation(std::string(context) +
                            " was accepted by make_decode_runtime_arguments");
  }
}

struct StageBinding {
  std::uint32_t index = 0;
  std::string variable;
  std::string meaning;
};

void validate_stage_runtime_arguments(const Json &runtime_arguments,
                                      std::string_view stage,
                                      const SourceFile &source,
                                      const std::vector<std::string> &meanings,
                                      std::size_t argument_count) {
  const std::string context = "contract.runtime_arguments." + std::string(stage);
  const Json &stage_contract =
      require_field(runtime_arguments, stage, "contract.runtime_arguments");
  const std::uint32_t declared_count =
      require_u32_field(stage_contract, "count", context);
  if (declared_count != argument_count) {
    throw ContractViolation(context + ".count does not match the compiled ABI "
                                       "argument count");
  }

  std::vector<StageBinding> bindings;
  const Json &binding_array = require_field(stage_contract, "bindings", context);
  if (!binding_array.is_array()) {
    throw ContractViolation(context + ".bindings must be an array");
  }
  std::set<std::uint32_t> covered;
  for (const Json &entry : binding_array) {
    StageBinding binding;
    binding.index = require_u32_field(entry, "index", context + ".bindings");
    binding.variable =
        require_string_field(entry, "variable", context + ".bindings");
    binding.meaning =
        require_string_field(entry, "meaning", context + ".bindings");
    if (binding.index >= argument_count) {
      throw ContractViolation(context + " binding index " +
                              std::to_string(binding.index) +
                              " is outside the argument count");
    }
    if (!covered.insert(binding.index).second) {
      throw ContractViolation(context + " binds index " +
                              std::to_string(binding.index) + " twice");
    }
    if (binding.meaning != meanings[binding.index]) {
      throw ContractViolation(context + " meaning drift at index " +
                              std::to_string(binding.index) + " (contract has '" +
                              binding.meaning + "', ABI requires '" +
                              meanings[binding.index] + "')");
    }
    const std::regex binding_pattern(
        "uint32_t\\s+" + binding.variable +
        "\\s*=\\s*get_arg_val<uint32_t>\\(\\s*" +
        std::to_string(binding.index) + "\\s*\\);");
    if (!std::regex_search(source.text, binding_pattern)) {
      throw ContractViolation(source.role + " no longer binds runtime argument " +
                              std::to_string(binding.index) + " to '" +
                              binding.variable + "' (" +
                              binding.meaning + ")");
    }
    bindings.push_back(binding);
  }

  std::set<std::uint32_t> expected_numeric;
  for (const StageBinding &binding : bindings) {
    expected_numeric.insert(binding.index);
  }

  const auto input_loop_iterator = stage_contract.find("input_loop");
  if (input_loop_iterator != stage_contract.end()) {
    const Json &input_loop = *input_loop_iterator;
    const std::uint32_t first =
        require_u32_field(input_loop, "first_index", context + ".input_loop");
    const std::uint32_t count =
        require_u32_field(input_loop, "count", context + ".input_loop");
    const std::string variable =
        require_string_field(input_loop, "variable", context + ".input_loop");
    const std::string expression = collapse_whitespace(
        require_string_field(input_loop, "expression", context + ".input_loop"));
    const std::string pattern =
        "for\\s*\\(\\s*uint32_t\\s+i\\s*=\\s*0\\s*;\\s*i\\s*<\\s*" +
        std::to_string(count) + "\\s*;\\s*i\\+\\+\\s*\\)\\s*" + variable +
        "\\[i\\]\\s*=\\s*get_arg_val<uint32_t>\\(\\s*" +
        std::to_string(first) + "\\s*\\+\\s*i\\s*\\);";
    if (!std::regex_search(source.text, std::regex(pattern))) {
      throw ContractViolation(source.role +
                              " no longer binds the ordered input address "
                              "arguments starting at index " +
                              std::to_string(first));
    }
    if (first + count > argument_count) {
      throw ContractViolation(context +
                              ".input_loop runs past the argument count");
    }
    for (std::uint32_t offset = 0; offset < count; ++offset) {
      expected_numeric.erase(first + offset);
      if (!covered.insert(first + offset).second) {
        throw ContractViolation(context + " input_loop overlaps a binding");
      }
    }
    (void)expression;
  }

  const auto unbound_iterator = stage_contract.find("unbound");
  if (unbound_iterator != stage_contract.end()) {
    const std::vector<std::uint32_t> unbound =
        parse_u32_array(*unbound_iterator, context + ".unbound");
    for (const std::uint32_t index : unbound) {
      if (index >= argument_count) {
        throw ContractViolation(context + ".unbound index " +
                                std::to_string(index) +
                                " is outside the argument count");
      }
      if (covered.find(index) != covered.end()) {
        throw ContractViolation(context + " marks bound index " +
                                std::to_string(index) + " as unbound");
      }
      covered.insert(index);
      expected_numeric.erase(index);
    }
  }

  if (covered.size() != argument_count) {
    throw ContractViolation(context + " does not bind every runtime argument");
  }

  const std::set<std::uint32_t> actual_numeric =
      parse_numeric_argument_indices(source.text);
  if (actual_numeric != expected_numeric) {
    throw ContractViolation(source.role +
                            " numeric get_arg_val index set drifted from the "
                            "ordered argument contract");
  }
}

void validate_stage_assignments(const Json &runtime_arguments,
                                std::string_view stage,
                                const SourceFile &abi,
                                const std::vector<RuntimeAssignment> &actual) {
  const std::string context = "contract.runtime_arguments." + std::string(stage);
  const Json &stage_contract =
      require_field(runtime_arguments, stage, "contract.runtime_arguments");
  const Json &assignment_array =
      require_field(stage_contract, "assignments", context);
  if (!assignment_array.is_array()) {
    throw ContractViolation(context + ".assignments must be an array");
  }
  for (const Json &entry : assignment_array) {
    const std::uint32_t index =
        require_u32_field(entry, "index", context + ".assignments");
    const std::string argument =
        require_string_field(entry, "argument", context + ".assignments");
    const std::string expression = collapse_whitespace(
        require_string_field(entry, "expression", context + ".assignments"));
    bool found = false;
    for (const RuntimeAssignment &assignment : actual) {
      if (assignment.stage == std::string(stage) &&
          assignment.argument == argument) {
        if (assignment.expression != expression) {
          throw ContractViolation(
              "ABI header assignment meaning drift for " + std::string(stage) +
              "." + argument + " at index " + std::to_string(index) + " (has '" +
              assignment.expression + "', contract requires '" + expression +
              "')");
        }
        found = true;
        break;
      }
    }
    if (!found) {
      throw ContractViolation("ABI header no longer assigns " +
                              std::string(stage) + "." + argument +
                              " at index " + std::to_string(index));
    }
  }
  (void)abi;
}

// ---- Ordered per-CB phase-window extraction and validation ----

enum class TokenKind {
  Identifier,
  Number,
  StringLiteral,
  CharLiteral,
  LParen,
  RParen,
  LBrace,
  RBrace,
  LBracket,
  RBracket,
  Semicolon,
  Comma,
  Plus,
  ColonColon,
  Less,
  Greater,
  Equal,
  Other,
};

struct Token {
  TokenKind kind = TokenKind::Other;
  std::string text;
  std::size_t line = 1;
  std::size_t start_offset = 0;
  std::size_t end_offset = 0;
};

[[nodiscard]] std::vector<std::string> split_source_lines(std::string_view text) {
  std::vector<std::string> lines;
  std::size_t start = 0;
  while (start <= text.size()) {
    const std::size_t newline = text.find('\n', start);
    if (newline == std::string_view::npos) {
      std::string line(text.substr(start));
      if (!line.empty() && line.back() == '\r') {
        line.pop_back();
      }
      lines.push_back(std::move(line));
      break;
    }
    std::string line(text.substr(start, newline - start));
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    lines.push_back(std::move(line));
    start = newline + 1;
  }
  return lines;
}

[[nodiscard]] std::string strip_whitespace(std::string_view value) {
  std::string result;
  result.reserve(value.size());
  for (const char ch : value) {
    if (std::isspace(static_cast<unsigned char>(ch)) == 0) {
      result.push_back(ch);
    }
  }
  return result;
}

[[nodiscard]] bool is_ident_start(char ch) {
  const unsigned char uch = static_cast<unsigned char>(ch);
  return std::isalpha(uch) != 0 || ch == '_';
}

[[nodiscard]] bool is_ident_continue(char ch) {
  const unsigned char uch = static_cast<unsigned char>(ch);
  return std::isalnum(uch) != 0 || ch == '_';
}

[[nodiscard]] std::vector<Token> tokenize_kernel_source(
    const SourceFile &source) {
  const std::string &text = source.text;
  std::vector<Token> tokens;
  std::size_t pos = 0;
  std::size_t line = 1;
  bool at_line_start = true;

  while (pos < text.size()) {
    const char ch = text[pos];
    if (ch == '\n') {
      ++pos;
      ++line;
      at_line_start = true;
      continue;
    }
    if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\f' || ch == '\v') {
      ++pos;
      continue;
    }
    if (ch == '/' && pos + 1 < text.size() && text[pos + 1] == '/') {
      pos += 2;
      while (pos < text.size() && text[pos] != '\n') {
        ++pos;
      }
      continue;
    }
    if (ch == '/' && pos + 1 < text.size() && text[pos + 1] == '*') {
      pos += 2;
      bool closed = false;
      while (pos + 1 < text.size()) {
        if (text[pos] == '\n') {
          ++line;
        }
        if (text[pos] == '*' && text[pos + 1] == '/') {
          pos += 2;
          closed = true;
          break;
        }
        ++pos;
      }
      if (!closed) {
        throw ContractViolation(source.role +
                                ": unterminated block comment at line " +
                                std::to_string(line));
      }
      continue;
    }
    if (ch == '#' && at_line_start) {
      const std::size_t directive_line = line;
      ++pos;
      while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t')) {
        ++pos;
      }
      const std::size_t kw_start = pos;
      while (pos < text.size() && is_ident_continue(text[pos])) {
        ++pos;
      }
      const std::string directive = text.substr(kw_start, pos - kw_start);
      if (directive == "define") {
        while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t')) {
          ++pos;
        }
        const std::size_t name_start = pos;
        while (pos < text.size() && is_ident_continue(text[pos])) {
          ++pos;
        }
        const std::string macro_name =
            pos > name_start ? text.substr(name_start, pos - name_start)
                             : std::string("<unnamed>");
        throw ContractViolation(
            source.role + ": unrecognized preprocessor macro '" + macro_name +
            "' at line " + std::to_string(directive_line) +
            " (macros cannot wrap or alias circular-buffer operations)");
      }
      // Consume the rest of this preprocessor directive, honoring backslash
      // line continuations.
      while (pos < text.size()) {
        if (text[pos] == '\\' && pos + 1 < text.size() &&
            text[pos + 1] == '\n') {
          pos += 2;
          ++line;
          continue;
        }
        if (text[pos] == '\\' && pos + 2 < text.size() &&
            text[pos + 1] == '\r' && text[pos + 2] == '\n') {
          pos += 3;
          ++line;
          continue;
        }
        if (text[pos] == '\n') {
          break;
        }
        ++pos;
      }
      continue;
    }

    at_line_start = false;
    const std::size_t start = pos;
    const std::size_t tok_line = line;

    if (ch == '"' || ch == '\'') {
      const char quote = ch;
      ++pos;
      while (pos < text.size() && text[pos] != quote) {
        if (text[pos] == '\\' && pos + 1 < text.size()) {
          if (text[pos + 1] == '\n') {
            ++line;
          }
          pos += 2;
          continue;
        }
        if (text[pos] == '\n') {
          ++line;
        }
        ++pos;
      }
      if (pos < text.size()) {
        ++pos;
      }
      tokens.push_back(
          Token{quote == '"' ? TokenKind::StringLiteral : TokenKind::CharLiteral,
                text.substr(start, pos - start), tok_line, start, pos});
      continue;
    }

    if (is_ident_start(ch)) {
      ++pos;
      while (pos < text.size() && is_ident_continue(text[pos])) {
        ++pos;
      }
      tokens.push_back(Token{TokenKind::Identifier,
                             text.substr(start, pos - start), tok_line, start,
                             pos});
      continue;
    }

    if (std::isdigit(static_cast<unsigned char>(ch)) != 0) {
      ++pos;
      while (pos < text.size() &&
             (is_ident_continue(text[pos]) || text[pos] == '.')) {
        ++pos;
      }
      tokens.push_back(Token{TokenKind::Number,
                             text.substr(start, pos - start), tok_line, start,
                             pos});
      continue;
    }

    if (ch == ':' && pos + 1 < text.size() && text[pos + 1] == ':') {
      pos += 2;
      tokens.push_back(
          Token{TokenKind::ColonColon, "::", tok_line, start, pos});
      continue;
    }

    TokenKind kind = TokenKind::Other;
    switch (ch) {
    case '(':
      kind = TokenKind::LParen;
      break;
    case ')':
      kind = TokenKind::RParen;
      break;
    case '{':
      kind = TokenKind::LBrace;
      break;
    case '}':
      kind = TokenKind::RBrace;
      break;
    case '[':
      kind = TokenKind::LBracket;
      break;
    case ']':
      kind = TokenKind::RBracket;
      break;
    case ';':
      kind = TokenKind::Semicolon;
      break;
    case ',':
      kind = TokenKind::Comma;
      break;
    case '+':
      kind = TokenKind::Plus;
      break;
    case '<':
      kind = TokenKind::Less;
      break;
    case '>':
      kind = TokenKind::Greater;
      break;
    case '=':
      kind = TokenKind::Equal;
      break;
    default:
      kind = TokenKind::Other;
      break;
    }
    ++pos;
    tokens.push_back(Token{kind, std::string(1, ch), tok_line, start, pos});
  }
  return tokens;
}

[[nodiscard]] std::string slice_tokens_clean(const std::string &source_text,
                                             const std::vector<Token> &tokens,
                                             std::size_t begin,
                                             std::size_t end) {
  if (begin >= end || end > tokens.size()) {
    return "";
  }
  const std::size_t start_off = tokens[begin].start_offset;
  const std::size_t end_off = tokens[end - 1].end_offset;
  std::string raw = source_text.substr(start_off, end_off - start_off);
  // Strip any inline comments that appeared between tokens.
  std::string cleaned;
  cleaned.reserve(raw.size());
  for (std::size_t i = 0; i < raw.size();) {
    if (raw[i] == '/' && i + 1 < raw.size() && raw[i + 1] == '/') {
      i += 2;
      while (i < raw.size() && raw[i] != '\n') {
        ++i;
      }
      continue;
    }
    if (raw[i] == '/' && i + 1 < raw.size() && raw[i + 1] == '*') {
      i += 2;
      while (i + 1 < raw.size() && !(raw[i] == '*' && raw[i + 1] == '/')) {
        ++i;
      }
      if (i + 1 < raw.size()) {
        i += 2;
      }
      cleaned.push_back(' ');
      continue;
    }
    cleaned.push_back(raw[i]);
    ++i;
  }
  return collapse_whitespace(cleaned);
}

struct CbSymbolTable {
  std::map<std::string, std::uint32_t> name_to_index;
  std::map<std::uint32_t, std::string> index_to_name;
};

[[nodiscard]] bool is_cb_symbol_name(std::string_view name) {
  return name.rfind("cb_", 0) == 0 || name.rfind("c_", 0) == 0;
}

[[nodiscard]] CbSymbolTable parse_namespace_cb_constants(
    const std::vector<Token> &tokens, const SourceFile &source,
    std::size_t kernel_main_token) {
  CbSymbolTable table;
  for (std::size_t i = 0; i + 4 < kernel_main_token; ++i) {
    if (tokens[i].kind == TokenKind::Identifier &&
        tokens[i].text == "constexpr" &&
        tokens[i + 1].kind == TokenKind::Identifier &&
        tokens[i + 1].text == "uint32_t") {
      std::size_t cursor = i + 2;
      while (cursor < kernel_main_token &&
             tokens[cursor].kind != TokenKind::Semicolon) {
        if (cursor + 2 < kernel_main_token &&
            tokens[cursor].kind == TokenKind::Identifier &&
            tokens[cursor + 1].kind == TokenKind::Equal &&
            tokens[cursor + 2].kind == TokenKind::Number) {
          const std::string &name = tokens[cursor].text;
          const std::string &num_text = tokens[cursor + 2].text;
          if (is_cb_symbol_name(name)) {
            std::size_t parsed_len = 0;
            const unsigned long value = std::stoul(num_text, &parsed_len, 10);
            if (parsed_len == num_text.size() ||
                (parsed_len + 1 == num_text.size() &&
                 (num_text[parsed_len] == 'u' ||
                  num_text[parsed_len] == 'U'))) {
              const auto cb_idx = static_cast<std::uint32_t>(value);
              const auto existing_name = table.name_to_index.find(name);
              if (existing_name != table.name_to_index.end()) {
                throw ContractViolation(source.role + ": duplicate CB symbol '" +
                                        name + "' declaration");
              }
              const auto existing_idx = table.index_to_name.find(cb_idx);
              if (existing_idx != table.index_to_name.end()) {
                throw ContractViolation(
                    source.role + ": duplicate CB index " +
                    std::to_string(cb_idx) + " declared by '" +
                    existing_idx->second + "' and '" + name + "'");
              }
              table.name_to_index.emplace(name, cb_idx);
              table.index_to_name.emplace(cb_idx, name);
            }
          }
          cursor += 3;
          if (cursor < kernel_main_token &&
              tokens[cursor].kind == TokenKind::Comma) {
            ++cursor;
          }
          continue;
        }
        ++cursor;
      }
      i = cursor;
    }
  }
  return table;
}

[[nodiscard]] std::size_t find_matching_delim(const std::vector<Token> &tokens,
                                              std::size_t open_idx,
                                              TokenKind open_kind,
                                              TokenKind close_kind,
                                              const SourceFile &source) {
  if (open_idx >= tokens.size() || tokens[open_idx].kind != open_kind) {
    throw ContractViolation(source.role + ": expected opening delimiter");
  }
  std::size_t depth = 1;
  for (std::size_t i = open_idx + 1; i < tokens.size(); ++i) {
    if (tokens[i].kind == open_kind) {
      ++depth;
    } else if (tokens[i].kind == close_kind) {
      --depth;
      if (depth == 0) {
        return i;
      }
    }
  }
  throw ContractViolation(source.role + ": unmatched delimiter at line " +
                          std::to_string(tokens[open_idx].line));
}

[[nodiscard]] std::vector<std::pair<std::size_t, std::size_t>>
split_call_arguments(const std::vector<Token> &tokens, std::size_t lparen_idx,
                     std::size_t rparen_idx) {
  std::vector<std::pair<std::size_t, std::size_t>> args;
  if (lparen_idx + 1 >= rparen_idx) {
    return args;
  }
  std::size_t arg_start = lparen_idx + 1;
  std::size_t paren_depth = 0;
  std::size_t bracket_depth = 0;
  std::size_t brace_depth = 0;
  for (std::size_t i = lparen_idx + 1; i < rparen_idx; ++i) {
    const TokenKind kind = tokens[i].kind;
    if (kind == TokenKind::LParen) {
      ++paren_depth;
    } else if (kind == TokenKind::RParen && paren_depth > 0) {
      --paren_depth;
    } else if (kind == TokenKind::LBracket) {
      ++bracket_depth;
    } else if (kind == TokenKind::RBracket && bracket_depth > 0) {
      --bracket_depth;
    } else if (kind == TokenKind::LBrace) {
      ++brace_depth;
    } else if (kind == TokenKind::RBrace && brace_depth > 0) {
      --brace_depth;
    } else if (kind == TokenKind::Comma && paren_depth == 0 &&
               bracket_depth == 0 && brace_depth == 0) {
      args.emplace_back(arg_start, i);
      arg_start = i + 1;
    }
  }
  args.emplace_back(arg_start, rparen_idx);
  return args;
}

struct LoopFrame {
  std::string kind;
  std::string header;
  std::size_t line = 0;
  std::string index_var;
  std::optional<std::uint32_t> const_bound;
};

struct BranchFrame {
  std::string kind; // "if" or "else"
  std::string condition;
  std::size_t line = 0;

  [[nodiscard]] std::string label() const {
    if (kind == "if") {
      return "if (" + condition + ")";
    }
    return "else (" + condition + ")";
  }
};

enum class PhaseEventKind {
  Reserve,
  Push,
  Wait,
  Pop,
  ProducerAccess,
  ConsumerAccess,
  ProducerInit,
  ConsumerInit,
  Query,
  Config,
  LoopExit,
  BranchExit,
  ControlJump,
};

[[nodiscard]] std::string phase_event_kind_name(PhaseEventKind kind) {
  switch (kind) {
  case PhaseEventKind::Reserve:
    return "reserve";
  case PhaseEventKind::Push:
    return "push";
  case PhaseEventKind::Wait:
    return "wait";
  case PhaseEventKind::Pop:
    return "pop";
  case PhaseEventKind::ProducerAccess:
    return "producer_access";
  case PhaseEventKind::ConsumerAccess:
    return "consumer_access";
  case PhaseEventKind::ProducerInit:
    return "producer_init";
  case PhaseEventKind::ConsumerInit:
    return "consumer_init";
  case PhaseEventKind::Query:
    return "query";
  case PhaseEventKind::Config:
    return "config";
  case PhaseEventKind::LoopExit:
    return "loop_exit";
  case PhaseEventKind::BranchExit:
    return "branch_exit";
  case PhaseEventKind::ControlJump:
    return "control_jump";
  }
  return "unknown";
}

struct ResolvedCbTarget {
  std::string name;
  std::uint32_t index = 0;
  PhaseEventKind kind = PhaseEventKind::Query;
};

struct PhaseEvent {
  PhaseEventKind kind = PhaseEventKind::Query;
  std::string operation;
  std::string cb_expression;
  std::string cb_name;
  std::uint32_t cb_index = 0;
  std::string count_expr;
  std::string normalized_call;
  std::size_t line = 0;
  std::vector<std::string> loop_scope;
  std::vector<std::string> branch_scope;
  std::optional<std::string> inlined_from_lambda;
  std::optional<std::size_t> lambda_body_line;
};

struct ReadbackEntry {
  std::size_t line = 0;
  std::string source_line;
  std::string operation;
  std::string kind;
  std::string cb_expression;
  std::vector<ResolvedCbTarget> cb_targets;
  std::optional<std::string> tile_count;
  std::size_t loop_depth = 0;
  std::vector<std::string> loop_scope;
  std::vector<std::string> branch_scope;
  std::optional<std::string> inlined_from_lambda;
  std::optional<std::size_t> lambda_body_line;
  std::optional<std::string> lambda_body_source_line;
};

struct KernelPhaseExtraction {
  std::vector<PhaseEvent> events;
  std::vector<ReadbackEntry> readback;
};

struct CbApiOperand {
  std::size_t arg_index = 0;
  PhaseEventKind kind = PhaseEventKind::Query;
};

struct CbApiSpec {
  std::size_t min_args = 0;
  std::size_t max_args = 0;
  std::optional<std::size_t> count_arg_index;
  std::vector<CbApiOperand> operands;
};

[[nodiscard]] const std::map<std::string, CbApiSpec> &recognized_cb_apis() {
  static const std::map<std::string, CbApiSpec> kApis = {
      // Handshakes
      {"cb_reserve_back", {2, 2, 1, {{0, PhaseEventKind::Reserve}}}},
      {"cb_push_back", {2, 2, 1, {{0, PhaseEventKind::Push}}}},
      {"cb_wait_front", {2, 2, 1, {{0, PhaseEventKind::Wait}}}},
      {"cb_pop_front", {2, 2, 1, {{0, PhaseEventKind::Pop}}}},
      // Producer data/pointer accesses
      {"pack_tile", {2, 3, std::nullopt, {{1, PhaseEventKind::ProducerAccess}}}},
      {"get_write_ptr",
       {1, 1, std::nullopt, {{0, PhaseEventKind::ProducerAccess}}}},
      // Consumer data/pointer accesses
      {"get_read_ptr",
       {1, 1, std::nullopt, {{0, PhaseEventKind::ConsumerAccess}}}},
      {"copy_tile", {3, 3, std::nullopt, {{0, PhaseEventKind::ConsumerAccess}}}},
      {"transpose_wh_tile",
       {3, 3, std::nullopt, {{0, PhaseEventKind::ConsumerAccess}}}},
      {"mul_tiles",
       {5,
        5,
        std::nullopt,
        {{0, PhaseEventKind::ConsumerAccess},
         {1, PhaseEventKind::ConsumerAccess}}}},
      {"matmul_tiles",
       {5,
        5,
        std::nullopt,
        {{0, PhaseEventKind::ConsumerAccess},
         {1, PhaseEventKind::ConsumerAccess}}}},
      {"mul_tiles_bcast_rows",
       {5,
        5,
        std::nullopt,
        {{0, PhaseEventKind::ConsumerAccess},
         {1, PhaseEventKind::ConsumerAccess}}}},
      // Init / config / metadata queries
      {"get_tile_size", {1, 1, std::nullopt, {{0, PhaseEventKind::Query}}}},
      {"compute_kernel_hw_startup",
       {3,
        3,
        std::nullopt,
        {{0, PhaseEventKind::Config},
         {1, PhaseEventKind::Config},
         {2, PhaseEventKind::Config}}}},
      {"copy_tile_init",
       {1, 1, std::nullopt, {{0, PhaseEventKind::ConsumerInit}}}},
      {"mul_tiles_init",
       {2,
        2,
        std::nullopt,
        {{0, PhaseEventKind::ConsumerInit},
         {1, PhaseEventKind::ConsumerInit}}}},
      {"mul_bcast_rows_init_short",
       {2,
        2,
        std::nullopt,
        {{0, PhaseEventKind::ConsumerInit},
         {1, PhaseEventKind::ConsumerInit}}}},
      {"mm_init_short",
       {3,
        3,
        std::nullopt,
        {{0, PhaseEventKind::ConsumerInit},
         {1, PhaseEventKind::ConsumerInit}}}},
      {"transpose_wh_init",
       {2,
        2,
        std::nullopt,
        {{0, PhaseEventKind::ConsumerInit},
         {1, PhaseEventKind::ProducerInit}}}},
      {"mm_init",
       {3,
        3,
        std::nullopt,
        {{0, PhaseEventKind::ConsumerInit},
         {1, PhaseEventKind::ConsumerInit},
         {2, PhaseEventKind::ProducerInit}}}},
  };
  return kApis;
}

[[nodiscard]] std::string to_lower_ascii(std::string_view input) {
  std::string out;
  out.reserve(input.size());
  for (const char ch : input) {
    out.push_back(
        static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
  }
  return out;
}

[[nodiscard]] bool looks_like_macro_call(std::string_view name) {
  if (name == "MATH") {
    return false;
  }
  std::size_t upper_count = 0;
  for (const char ch : name) {
    const unsigned char uch = static_cast<unsigned char>(ch);
    if (std::islower(uch) != 0) {
      return false;
    }
    if (std::isupper(uch) != 0) {
      ++upper_count;
    }
  }
  return upper_count >= 2;
}

[[nodiscard]] bool looks_like_unrecognized_cb_call(std::string_view name) {
  if (looks_like_macro_call(name)) {
    return true;
  }
  const std::string lower = to_lower_ascii(name);
  if (lower.rfind("cb_", 0) == 0 || lower.find("_cb_") != std::string::npos ||
      (lower.size() >= 3 && lower.compare(lower.size() - 3, 3, "_cb") == 0)) {
    return true;
  }
  for (const char *pattern :
       {"pack_tile", "copy_tile", "write_ptr", "read_ptr", "tile_size",
        "cbwait", "cbpop", "cbpush", "cbreserve"}) {
    if (lower.find(pattern) != std::string::npos) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] std::optional<std::pair<std::string, std::uint32_t>>
parse_static_for_bound(const std::string &header_text) {
  static const std::regex pattern(
      R"(^\(\s*(?:const\s+)?(?:uint32_t|int|std::size_t|size_t)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*0\s*;\s*([A-Za-z_][A-Za-z0-9_]*)\s*<\s*([0-9]+)[uU]?\s*;\s*(?:\+\+([A-Za-z_][A-Za-z0-9_]*)|([A-Za-z_][A-Za-z0-9_]*)\+\+)\s*\)$)");
  std::smatch match;
  if (!std::regex_match(header_text, match, pattern)) {
    return std::nullopt;
  }
  const std::string var = match[1].str();
  const std::string cmp_var = match[2].str();
  const std::string inc_var =
      match[4].matched ? match[4].str() : match[5].str();
  if (var != cmp_var || var != inc_var) {
    return std::nullopt;
  }
  const auto bound = static_cast<std::uint32_t>(std::stoul(match[3].str()));
  if (bound == 0) {
    return std::nullopt;
  }
  return std::make_pair(var, bound);
}

struct ResolvedCbExpression {
  std::vector<std::pair<std::string, std::uint32_t>> targets;
  std::optional<std::size_t> collapsed_loop_index;
};

[[nodiscard]] ResolvedCbExpression resolve_cb_expression(
    const SourceFile &source, const std::vector<Token> &tokens,
    std::size_t begin, std::size_t end, const CbSymbolTable &cbs,
    const std::vector<LoopFrame> &loop_stack, std::string_view call_name,
    std::size_t call_line) {
  const std::string expr_text =
      slice_tokens_clean(source.text, tokens, begin, end);
  // Case 1: Single declared constexpr CB identifier.
  if (end - begin == 1 && tokens[begin].kind == TokenKind::Identifier) {
    const std::string &ident = tokens[begin].text;
    const auto it = cbs.name_to_index.find(ident);
    if (it != cbs.name_to_index.end()) {
      ResolvedCbExpression resolved;
      resolved.targets.emplace_back(it->first, it->second);
      return resolved;
    }
    throw ContractViolation(
        source.role + ": unrecognized CB expression '" + expr_text + "' in '" +
        std::string(call_name) + "' at line " + std::to_string(call_line) +
        " (expected a declared constexpr CB symbol, not a runtime variable or "
        "macro)");
  }
  // Case 2: Declared CB constant + enclosing static constant-bound for-loop
  // index (e.g., cb_a + inp inside for (uint32_t inp = 0; inp < 6; inp++)).
  if (end - begin == 3 && tokens[begin].kind == TokenKind::Identifier &&
      tokens[begin + 1].kind == TokenKind::Plus &&
      tokens[begin + 2].kind == TokenKind::Identifier) {
    const std::string &base_ident = tokens[begin].text;
    const std::string &var_ident = tokens[begin + 2].text;
    const auto base_it = cbs.name_to_index.find(base_ident);
    if (base_it == cbs.name_to_index.end()) {
      throw ContractViolation(
          source.role + ": unrecognized CB base symbol '" + base_ident +
          "' in expression '" + expr_text + "' in '" + std::string(call_name) +
          "' at line " + std::to_string(call_line));
    }
    for (std::size_t idx = loop_stack.size(); idx > 0; --idx) {
      const LoopFrame &frame = loop_stack[idx - 1];
      if (frame.index_var == var_ident && frame.const_bound.has_value()) {
        ResolvedCbExpression resolved;
        resolved.collapsed_loop_index = idx - 1;
        for (std::uint32_t offset = 0; offset < *frame.const_bound; ++offset) {
          const std::uint32_t target_idx = base_it->second + offset;
          const auto target_it = cbs.index_to_name.find(target_idx);
          if (target_it == cbs.index_to_name.end()) {
            throw ContractViolation(
                source.role + ": CB expression '" + expr_text + "' in '" +
                std::string(call_name) + "' at line " +
                std::to_string(call_line) + " resolves to undeclared CB index " +
                std::to_string(target_idx));
          }
          resolved.targets.emplace_back(target_it->second, target_idx);
        }
        return resolved;
      }
    }
    throw ContractViolation(
        source.role + ": unrecognized CB expression '" + expr_text + "' in '" +
        std::string(call_name) + "' at line " + std::to_string(call_line) +
        " (index '" + var_ident +
        "' is not a compile-time constant-bound loop variable)");
  }

  throw ContractViolation(
      source.role + ": unrecognized CB expression '" + expr_text + "' in '" +
      std::string(call_name) + "' at line " + std::to_string(call_line) +
      " (expected a declared constexpr CB symbol or static constant-bound "
      "offset)");
}

struct RecordedLambdaEvent {
  PhaseEventKind kind = PhaseEventKind::Query;
  std::string operation;
  std::string cb_expression;
  std::vector<ResolvedCbTarget> targets;
  std::optional<std::string> count_expr;
  std::string normalized_call;
  std::size_t body_line = 0;
  std::string body_source_line;
};

struct LocalLambdaRecord {
  std::size_t def_line = 0;
  std::vector<RecordedLambdaEvent> events;
  bool invoked = false;
};

class KernelPhaseParser {
public:
  KernelPhaseParser(const SourceFile &source,
                    const std::vector<std::string> &lines,
                    const std::vector<Token> &tokens,
                    const CbSymbolTable &cbs)
      : source_(source), lines_(lines), tokens_(tokens), cbs_(cbs) {}

  [[nodiscard]] KernelPhaseExtraction extract() {
    // Locate void kernel_main() { ... }
    std::optional<std::size_t> main_ident_idx;
    std::optional<std::size_t> main_lbrace;
    std::optional<std::size_t> main_rbrace;
    for (std::size_t i = 0; i + 4 < tokens_.size(); ++i) {
      if (tokens_[i].kind == TokenKind::Identifier &&
          tokens_[i].text == "void" &&
          tokens_[i + 1].kind == TokenKind::Identifier &&
          tokens_[i + 1].text == "kernel_main" &&
          tokens_[i + 2].kind == TokenKind::LParen &&
          tokens_[i + 3].kind == TokenKind::RParen &&
          tokens_[i + 4].kind == TokenKind::LBrace) {
        if (main_lbrace.has_value()) {
          throw ContractViolation(source_.role +
                                  ": multiple kernel_main() definitions");
        }
        main_ident_idx = i + 1;
        main_lbrace = i + 4;
        main_rbrace = find_matching_delim(tokens_, *main_lbrace,
                                          TokenKind::LBrace, TokenKind::RBrace,
                                          source_);
      }
    }
    if (!main_lbrace.has_value() || !main_rbrace.has_value()) {
      throw ContractViolation(source_.role +
                              ": lacks a void kernel_main() definition");
    }
    (void)main_ident_idx;

    // Ensure no CB calls appear outside kernel_main().
    check_no_cb_calls_outside_main(0, *main_lbrace);
    check_no_cb_calls_outside_main(*main_rbrace + 1, tokens_.size());

    // Parse kernel_main() body statements.
    std::size_t pos = *main_lbrace + 1;
    while (pos < *main_rbrace) {
      pos = parse_statement(pos, *main_rbrace);
    }

    // Refuse any local lambda that captured CB operations but was never called.
    for (const auto &[name, record] : lambdas_) {
      if (!record.events.empty() && !record.invoked) {
        throw ContractViolation(
            source_.role + ": local lambda '" + name + "' at line " +
            std::to_string(record.def_line) +
            " contains circular-buffer operations but is never invoked");
      }
    }

    return std::move(result_);
  }

private:
  const SourceFile &source_;
  const std::vector<std::string> &lines_;
  const std::vector<Token> &tokens_;
  const CbSymbolTable &cbs_;

  std::vector<LoopFrame> loop_stack_;
  std::vector<BranchFrame> branch_stack_;
  std::map<std::string, LocalLambdaRecord> lambdas_;
  std::vector<RecordedLambdaEvent> *recording_lambda_ = nullptr;
  KernelPhaseExtraction result_;

  [[nodiscard]] std::string line_text(std::size_t line_no) const {
    if (line_no >= 1 && line_no <= lines_.size()) {
      return lines_[line_no - 1];
    }
    return "";
  }

  [[nodiscard]] std::vector<std::string> current_loop_scope(
      std::optional<std::size_t> exclude_index = std::nullopt) const {
    std::vector<std::string> scope;
    scope.reserve(loop_stack_.size());
    for (std::size_t i = 0; i < loop_stack_.size(); ++i) {
      if (exclude_index.has_value() && i == *exclude_index) {
        continue;
      }
      scope.push_back(loop_stack_[i].header);
    }
    return scope;
  }

  [[nodiscard]] std::vector<std::string> current_branch_scope() const {
    std::vector<std::string> scope;
    scope.reserve(branch_stack_.size());
    for (const BranchFrame &frame : branch_stack_) {
      scope.push_back(frame.label());
    }
    return scope;
  }

  void check_no_cb_calls_outside_main(std::size_t begin, std::size_t end) const {
    const auto &apis = recognized_cb_apis();
    for (std::size_t i = begin; i < end; ++i) {
      if (tokens_[i].kind != TokenKind::Identifier) {
        continue;
      }
      std::size_t call_lparen = i + 1;
      if (call_lparen < end && tokens_[call_lparen].kind == TokenKind::Less) {
        // Skip optional template argument list.
        std::size_t angle_depth = 1;
        std::size_t j = call_lparen + 1;
        while (j < end && angle_depth > 0) {
          if (tokens_[j].kind == TokenKind::Less) {
            ++angle_depth;
          } else if (tokens_[j].kind == TokenKind::Greater) {
            --angle_depth;
          } else if (tokens_[j].kind == TokenKind::Semicolon ||
                     tokens_[j].kind == TokenKind::LBrace) {
            break;
          }
          ++j;
        }
        if (angle_depth == 0) {
          call_lparen = j;
        }
      }
      if (call_lparen >= end || tokens_[call_lparen].kind != TokenKind::LParen) {
        continue;
      }
      const std::string &callee = tokens_[i].text;
      if (callee == "if" || callee == "for" || callee == "while" ||
          callee == "switch" || callee == "sizeof") {
        continue;
      }
      if (apis.find(callee) != apis.end() ||
          looks_like_unrecognized_cb_call(callee)) {
        throw ContractViolation(
            source_.role + ": circular-buffer/macro call '" + callee +
            "' outside kernel_main() at line " +
            std::to_string(tokens_[i].line));
      }
    }
  }

  [[nodiscard]] std::size_t parse_statement(std::size_t pos,
                                            std::size_t limit) {
    if (pos >= limit) {
      return limit;
    }
    // Empty statement
    if (tokens_[pos].kind == TokenKind::Semicolon) {
      return pos + 1;
    }
    // Compound block { ... }
    if (tokens_[pos].kind == TokenKind::LBrace) {
      const std::size_t rbrace = find_matching_delim(
          tokens_, pos, TokenKind::LBrace, TokenKind::RBrace, source_);
      std::size_t cur = pos + 1;
      while (cur < rbrace) {
        cur = parse_statement(cur, rbrace);
      }
      return rbrace + 1;
    }
    // for (...) <stmt>
    if (tokens_[pos].kind == TokenKind::Identifier &&
        tokens_[pos].text == "for") {
      const std::size_t for_line = tokens_[pos].line;
      if (pos + 1 >= limit || tokens_[pos + 1].kind != TokenKind::LParen) {
        throw ContractViolation(source_.role + ": malformed for loop at line " +
                                std::to_string(for_line));
      }
      const std::size_t lparen = pos + 1;
      const std::size_t rparen = find_matching_delim(
          tokens_, lparen, TokenKind::LParen, TokenKind::RParen, source_);
      scan_expression_range(lparen + 1, rparen);
      const std::string parens_text =
          slice_tokens_clean(source_.text, tokens_, lparen, rparen + 1);
      LoopFrame frame;
      frame.kind = "for";
      frame.header = "for " + parens_text;
      frame.line = for_line;
      if (const auto bound = parse_static_for_bound(parens_text)) {
        frame.index_var = bound->first;
        frame.const_bound = bound->second;
      }
      loop_stack_.push_back(frame);
      const std::size_t next = parse_statement(rparen + 1, limit);
      if (recording_lambda_ == nullptr) {
        PhaseEvent exit_ev;
        exit_ev.kind = PhaseEventKind::LoopExit;
        exit_ev.line = for_line;
        exit_ev.loop_scope = current_loop_scope();
        exit_ev.branch_scope = current_branch_scope();
        result_.events.push_back(std::move(exit_ev));
      }
      loop_stack_.pop_back();
      return next;
    }
    // while (...) <stmt>
    if (tokens_[pos].kind == TokenKind::Identifier &&
        tokens_[pos].text == "while") {
      const std::size_t while_line = tokens_[pos].line;
      if (pos + 1 >= limit || tokens_[pos + 1].kind != TokenKind::LParen) {
        throw ContractViolation(source_.role +
                                ": malformed while loop at line " +
                                std::to_string(while_line));
      }
      const std::size_t lparen = pos + 1;
      const std::size_t rparen = find_matching_delim(
          tokens_, lparen, TokenKind::LParen, TokenKind::RParen, source_);
      scan_expression_range(lparen + 1, rparen);
      const std::string parens_text =
          slice_tokens_clean(source_.text, tokens_, lparen, rparen + 1);
      LoopFrame frame;
      frame.kind = "while";
      frame.header = "while " + parens_text;
      frame.line = while_line;
      loop_stack_.push_back(frame);
      const std::size_t next = parse_statement(rparen + 1, limit);
      if (recording_lambda_ == nullptr) {
        PhaseEvent exit_ev;
        exit_ev.kind = PhaseEventKind::LoopExit;
        exit_ev.line = while_line;
        exit_ev.loop_scope = current_loop_scope();
        exit_ev.branch_scope = current_branch_scope();
        result_.events.push_back(std::move(exit_ev));
      }
      loop_stack_.pop_back();
      return next;
    }
    // do <stmt> while (...);
    if (tokens_[pos].kind == TokenKind::Identifier &&
        tokens_[pos].text == "do") {
      const std::size_t do_line = tokens_[pos].line;
      LoopFrame frame;
      frame.kind = "do";
      frame.header = "do";
      frame.line = do_line;
      loop_stack_.push_back(frame);
      std::size_t next = parse_statement(pos + 1, limit);
      if (recording_lambda_ == nullptr) {
        PhaseEvent exit_ev;
        exit_ev.kind = PhaseEventKind::LoopExit;
        exit_ev.line = do_line;
        exit_ev.loop_scope = current_loop_scope();
        exit_ev.branch_scope = current_branch_scope();
        result_.events.push_back(std::move(exit_ev));
      }
      loop_stack_.pop_back();
      if (next < limit && tokens_[next].kind == TokenKind::Identifier &&
          tokens_[next].text == "while" && next + 1 < limit &&
          tokens_[next + 1].kind == TokenKind::LParen) {
        const std::size_t rparen = find_matching_delim(
            tokens_, next + 1, TokenKind::LParen, TokenKind::RParen, source_);
        scan_expression_range(next + 2, rparen);
        next = rparen + 1;
        if (next < limit && tokens_[next].kind == TokenKind::Semicolon) {
          ++next;
        }
      }
      return next;
    }
    // if (...) <stmt> [else <stmt>]
    if (tokens_[pos].kind == TokenKind::Identifier &&
        tokens_[pos].text == "if") {
      const std::size_t if_line = tokens_[pos].line;
      if (pos + 1 >= limit || tokens_[pos + 1].kind != TokenKind::LParen) {
        throw ContractViolation(source_.role +
                                ": malformed if statement at line " +
                                std::to_string(if_line));
      }
      const std::size_t lparen = pos + 1;
      const std::size_t rparen = find_matching_delim(
          tokens_, lparen, TokenKind::LParen, TokenKind::RParen, source_);
      scan_expression_range(lparen + 1, rparen);
      const std::string cond =
          slice_tokens_clean(source_.text, tokens_, lparen + 1, rparen);
      branch_stack_.push_back(BranchFrame{"if", cond, if_line});
      std::size_t next = parse_statement(rparen + 1, limit);
      if (recording_lambda_ == nullptr) {
        PhaseEvent exit_ev;
        exit_ev.kind = PhaseEventKind::BranchExit;
        exit_ev.line = if_line;
        exit_ev.loop_scope = current_loop_scope();
        exit_ev.branch_scope = current_branch_scope();
        result_.events.push_back(std::move(exit_ev));
      }
      branch_stack_.pop_back();
      if (next < limit && tokens_[next].kind == TokenKind::Identifier &&
          tokens_[next].text == "else") {
        const std::size_t else_line = tokens_[next].line;
        branch_stack_.push_back(BranchFrame{"else", cond, else_line});
        next = parse_statement(next + 1, limit);
        if (recording_lambda_ == nullptr) {
          PhaseEvent exit_ev;
          exit_ev.kind = PhaseEventKind::BranchExit;
          exit_ev.line = else_line;
          exit_ev.loop_scope = current_loop_scope();
          exit_ev.branch_scope = current_branch_scope();
          result_.events.push_back(std::move(exit_ev));
        }
        branch_stack_.pop_back();
      }
      return next;
    }
    // switch (...) <stmt>
    if (tokens_[pos].kind == TokenKind::Identifier &&
        tokens_[pos].text == "switch") {
      if (pos + 1 < limit && tokens_[pos + 1].kind == TokenKind::LParen) {
        const std::size_t rparen = find_matching_delim(
            tokens_, pos + 1, TokenKind::LParen, TokenKind::RParen, source_);
        scan_expression_range(pos + 2, rparen);
        return parse_statement(rparen + 1, limit);
      }
    }
    // Local lambda definition: [const] auto <name> = [...] (...) { ... };
    {
      std::size_t cur = pos;
      if (cur < limit && tokens_[cur].kind == TokenKind::Identifier &&
          tokens_[cur].text == "const") {
        ++cur;
      }
      if (cur + 3 < limit && tokens_[cur].kind == TokenKind::Identifier &&
          tokens_[cur].text == "auto" &&
          tokens_[cur + 1].kind == TokenKind::Identifier &&
          tokens_[cur + 2].kind == TokenKind::Equal &&
          tokens_[cur + 3].kind == TokenKind::LBracket) {
        const std::string lambda_name = tokens_[cur + 1].text;
        const std::size_t def_line = tokens_[cur + 1].line;
        const std::size_t rbracket = find_matching_delim(
            tokens_, cur + 3, TokenKind::LBracket, TokenKind::RBracket,
            source_);
        std::size_t body_pos = rbracket + 1;
        if (body_pos < limit && tokens_[body_pos].kind == TokenKind::LParen) {
          body_pos = find_matching_delim(tokens_, body_pos, TokenKind::LParen,
                                         TokenKind::RParen, source_) +
                     1;
        }
        while (body_pos < limit && tokens_[body_pos].kind != TokenKind::LBrace &&
               tokens_[body_pos].kind != TokenKind::Semicolon) {
          ++body_pos;
        }
        if (body_pos < limit && tokens_[body_pos].kind == TokenKind::LBrace) {
          const std::size_t rbrace = find_matching_delim(
              tokens_, body_pos, TokenKind::LBrace, TokenKind::RBrace, source_);
          if (recording_lambda_ != nullptr) {
            throw ContractViolation(
                source_.role + ": nested local lambda '" + lambda_name +
                "' at line " + std::to_string(def_line) + " is not permitted");
          }
          LocalLambdaRecord record;
          record.def_line = def_line;
          recording_lambda_ = &record.events;
          std::size_t body_cur = body_pos + 1;
          while (body_cur < rbrace) {
            body_cur = parse_statement(body_cur, rbrace);
          }
          recording_lambda_ = nullptr;
          lambdas_[lambda_name] = std::move(record);
          std::size_t next = rbrace + 1;
          if (next < limit && tokens_[next].kind == TokenKind::Semicolon) {
            ++next;
          }
          return next;
        }
      }
    }

    // Simple statement ending in ';' (tracking nested delimiters).
    std::size_t end = pos;
    std::size_t paren_depth = 0;
    std::size_t bracket_depth = 0;
    std::size_t brace_depth = 0;
    while (end < limit) {
      const TokenKind kind = tokens_[end].kind;
      if (kind == TokenKind::LParen) {
        ++paren_depth;
      } else if (kind == TokenKind::RParen && paren_depth > 0) {
        --paren_depth;
      } else if (kind == TokenKind::LBracket) {
        ++bracket_depth;
      } else if (kind == TokenKind::RBracket && bracket_depth > 0) {
        --bracket_depth;
      } else if (kind == TokenKind::LBrace) {
        ++brace_depth;
      } else if (kind == TokenKind::RBrace && brace_depth > 0) {
        --brace_depth;
      } else if (kind == TokenKind::Semicolon && paren_depth == 0 &&
                 bracket_depth == 0 && brace_depth == 0) {
        break;
      }
      ++end;
    }
    scan_expression_range(pos, end);
    // Refuse early control-flow jumps inside kernel_main() that could bypass
    // a closing cb_push_back / cb_pop_front handshake.
    if (pos < end && tokens_[pos].kind == TokenKind::Identifier &&
        (tokens_[pos].text == "return" || tokens_[pos].text == "break" ||
         tokens_[pos].text == "continue" || tokens_[pos].text == "goto")) {
      if (recording_lambda_ == nullptr) {
        PhaseEvent jump_ev;
        jump_ev.kind = PhaseEventKind::ControlJump;
        jump_ev.operation = tokens_[pos].text;
        jump_ev.line = tokens_[pos].line;
        jump_ev.loop_scope = current_loop_scope();
        jump_ev.branch_scope = current_branch_scope();
        result_.events.push_back(std::move(jump_ev));
      }
    }
    return end < limit ? end + 1 : limit;
  }

  // Returns true if the token range [begin, end) contains a top-level (outside
  // nested recognized calls) reference to any declared CB symbol.
  [[nodiscard]] std::optional<std::string>
  find_direct_cb_reference(std::size_t begin, std::size_t end) const {
    const auto &apis = recognized_cb_apis();
    std::size_t i = begin;
    while (i < end) {
      if (tokens_[i].kind == TokenKind::Identifier) {
        // Skip inside nested recognized CB API calls (e.g.
        // aligned_l1_scratch_address(get_write_ptr(c_dram_read_scratch))).
        if (i + 1 < end && tokens_[i + 1].kind == TokenKind::LParen &&
            apis.find(tokens_[i].text) != apis.end()) {
          const std::size_t rparen = find_matching_delim(
              tokens_, i + 1, TokenKind::LParen, TokenKind::RParen, source_);
          i = rparen + 1;
          continue;
        }
        if (cbs_.name_to_index.find(tokens_[i].text) !=
            cbs_.name_to_index.end()) {
          return tokens_[i].text;
        }
      }
      ++i;
    }
    return std::nullopt;
  }

  void scan_expression_range(std::size_t begin, std::size_t end) {
    // Refuse local shadowing or reassignment of any declared CB constant.
    for (std::size_t idx = begin; idx < end; ++idx) {
      if (tokens_[idx].kind == TokenKind::Identifier &&
          cbs_.name_to_index.find(tokens_[idx].text) !=
              cbs_.name_to_index.end()) {
        const bool followed_by_assign =
            idx + 1 < end && tokens_[idx + 1].kind == TokenKind::Equal &&
            (idx + 2 >= end || tokens_[idx + 2].kind != TokenKind::Equal);
        const bool preceded_by_type =
            idx > begin && tokens_[idx - 1].kind == TokenKind::Identifier &&
            (tokens_[idx - 1].text == "uint32_t" ||
             tokens_[idx - 1].text == "int" ||
             tokens_[idx - 1].text == "auto" ||
             tokens_[idx - 1].text == "size_t" ||
             tokens_[idx - 1].text == "const" ||
             tokens_[idx - 1].text == "constexpr");
        if (followed_by_assign || preceded_by_type) {
          throw ContractViolation(
              source_.role + ": local shadowing or reassignment of CB symbol '" +
              tokens_[idx].text + "' at line " +
              std::to_string(tokens_[idx].line));
        }
      }
    }

    const auto &apis = recognized_cb_apis();
    std::size_t i = begin;
    while (i < end) {
      if (tokens_[i].kind != TokenKind::Identifier) {
        ++i;
        continue;
      }
      // Handle qualified call name (e.g., ttwkv7::read_dram_face_row(...))
      std::size_t name_start = i;
      std::size_t name_end = i;
      while (name_end + 2 < end &&
             tokens_[name_end + 1].kind == TokenKind::ColonColon &&
             tokens_[name_end + 2].kind == TokenKind::Identifier) {
        name_end += 2;
      }
      const std::string unqualified = tokens_[name_end].text;
      const std::string qualified =
          slice_tokens_clean(source_.text, tokens_, name_start, name_end + 1);

      // Skip optional template arguments <...>
      std::size_t after_name = name_end + 1;
      if (after_name < end && tokens_[after_name].kind == TokenKind::Less) {
        std::size_t angle_depth = 1;
        std::size_t j = after_name + 1;
        while (j < end && angle_depth > 0) {
          if (tokens_[j].kind == TokenKind::Less) {
            ++angle_depth;
          } else if (tokens_[j].kind == TokenKind::Greater) {
            --angle_depth;
          } else if (tokens_[j].kind == TokenKind::Semicolon ||
                     tokens_[j].kind == TokenKind::LParen ||
                     tokens_[j].kind == TokenKind::RParen) {
            break;
          }
          ++j;
        }
        if (angle_depth == 0) {
          after_name = j;
        }
      }

      if (after_name >= end || tokens_[after_name].kind != TokenKind::LParen) {
        i = name_end + 1;
        continue;
      }

      const std::size_t lparen = after_name;
      const std::size_t rparen = find_matching_delim(
          tokens_, lparen, TokenKind::LParen, TokenKind::RParen, source_);
      const std::size_t call_line = tokens_[name_start].line;

      // Ignore C++ keywords / type casts / method calls on objects ('.' or
      // '->') unless it matches a CB name or references a CB.
      const bool is_member_call =
          name_start > begin &&
          (tokens_[name_start - 1].text == "." ||
           tokens_[name_start - 1].text == ">");

      const auto args = split_call_arguments(tokens_, lparen, rparen);

      // First scan nested expressions inside each argument (C++ evaluates
      // arguments before the outer call).
      for (const auto &[arg_begin, arg_end] : args) {
        scan_expression_range(arg_begin, arg_end);
      }

      if (unqualified == "sizeof" || unqualified == "static_cast" ||
          unqualified == "reinterpret_cast" || unqualified == "const_cast") {
        i = rparen + 1;
        continue;
      }

      // Check recognized local lambda invocation.
      const auto lambda_it =
          (name_start == name_end && !is_member_call)
              ? lambdas_.find(unqualified)
              : lambdas_.end();
      if (lambda_it != lambdas_.end()) {
        // Refuse passing a CB symbol dynamically into a lambda.
        for (const auto &[arg_begin, arg_end] : args) {
          if (const auto cb_ref = find_direct_cb_reference(arg_begin, arg_end)) {
            throw ContractViolation(
                source_.role + ": unrecognized dynamic CB argument '" + *cb_ref +
                "' passed to lambda '" + unqualified + "' at line " +
                std::to_string(call_line));
          }
        }
        lambda_it->second.invoked = true;
        if (recording_lambda_ != nullptr) {
          throw ContractViolation(
              source_.role + ": nested invocation of lambda '" + unqualified +
              "' at line " + std::to_string(call_line));
        }
        const std::vector<std::string> call_loops = current_loop_scope();
        const std::vector<std::string> call_branches = current_branch_scope();
        for (const RecordedLambdaEvent &lam_ev : lambda_it->second.events) {
          ReadbackEntry rb;
          rb.line = call_line;
          rb.source_line = line_text(call_line);
          rb.operation = lam_ev.operation;
          rb.kind = phase_event_kind_name(lam_ev.kind);
          rb.cb_expression = lam_ev.cb_expression;
          rb.cb_targets = lam_ev.targets;
          rb.tile_count = lam_ev.count_expr;
          rb.loop_depth = call_loops.size();
          rb.loop_scope = call_loops;
          rb.branch_scope = call_branches;
          rb.inlined_from_lambda = unqualified;
          rb.lambda_body_line = lam_ev.body_line;
          rb.lambda_body_source_line = lam_ev.body_source_line;
          result_.readback.push_back(std::move(rb));

          for (const ResolvedCbTarget &target : lam_ev.targets) {
            PhaseEvent ev;
            ev.kind = target.kind;
            ev.operation = lam_ev.operation;
            ev.cb_expression = lam_ev.cb_expression;
            ev.cb_name = target.name;
            ev.cb_index = target.index;
            ev.count_expr = lam_ev.count_expr.value_or("");
            ev.normalized_call = lam_ev.normalized_call;
            ev.line = call_line;
            ev.loop_scope = call_loops;
            ev.branch_scope = call_branches;
            ev.inlined_from_lambda = unqualified;
            ev.lambda_body_line = lam_ev.body_line;
            result_.events.push_back(std::move(ev));
          }
        }
        i = rparen + 1;
        continue;
      }

      // Check recognized CB API call.
      const auto api_it =
          (name_start == name_end && !is_member_call)
              ? apis.find(unqualified)
              : apis.end();
      if (api_it != apis.end()) {
        const CbApiSpec &spec = api_it->second;
        if (args.size() < spec.min_args || args.size() > spec.max_args) {
          throw ContractViolation(
              source_.role + ": unrecognized arity (" +
              std::to_string(args.size()) + ") for CB call '" + unqualified +
              "' at line " + std::to_string(call_line));
        }
        const std::string normalized_call =
            slice_tokens_clean(source_.text, tokens_, name_start, rparen + 1);
        std::optional<std::string> count_expr;
        if (spec.count_arg_index.has_value()) {
          const auto [c_begin, c_end] = args[*spec.count_arg_index];
          count_expr = slice_tokens_clean(source_.text, tokens_, c_begin, c_end);
          if (count_expr->empty()) {
            throw ContractViolation(
                source_.role + ": empty tile-count expression in '" +
                unqualified + "' at line " + std::to_string(call_line));
          }
        }

        std::vector<ResolvedCbTarget> all_targets;
        std::vector<std::string> cb_expr_parts;
        std::vector<std::pair<ResolvedCbTarget, std::vector<std::string>>>
            emitted_targets;

        for (const CbApiOperand &operand : spec.operands) {
          const auto [a_begin, a_end] = args[operand.arg_index];
          const std::string expr_text =
              slice_tokens_clean(source_.text, tokens_, a_begin, a_end);
          cb_expr_parts.push_back(expr_text);
          const ResolvedCbExpression resolved = resolve_cb_expression(
              source_, tokens_, a_begin, a_end, cbs_, loop_stack_, unqualified,
              call_line);
          const std::vector<std::string> eff_loops =
              current_loop_scope(resolved.collapsed_loop_index);
          for (const auto &[cb_name, cb_idx] : resolved.targets) {
            ResolvedCbTarget target{cb_name, cb_idx, operand.kind};
            all_targets.push_back(target);
            emitted_targets.emplace_back(target, eff_loops);
          }
        }

        std::string combined_cb_expr;
        for (std::size_t part_i = 0; part_i < cb_expr_parts.size(); ++part_i) {
          if (part_i > 0) {
            combined_cb_expr += ", ";
          }
          combined_cb_expr += cb_expr_parts[part_i];
        }

        const PhaseEventKind primary_kind = spec.operands.front().kind;
        if (recording_lambda_ != nullptr) {
          RecordedLambdaEvent lam_ev;
          lam_ev.kind = primary_kind;
          lam_ev.operation = unqualified;
          lam_ev.cb_expression = combined_cb_expr;
          lam_ev.targets = all_targets;
          lam_ev.count_expr = count_expr;
          lam_ev.normalized_call = normalized_call;
          lam_ev.body_line = call_line;
          lam_ev.body_source_line = line_text(call_line);
          recording_lambda_->push_back(std::move(lam_ev));
        } else {
          const std::vector<std::string> raw_loops = current_loop_scope();
          const std::vector<std::string> branches = current_branch_scope();
          ReadbackEntry rb;
          rb.line = call_line;
          rb.source_line = line_text(call_line);
          rb.operation = unqualified;
          rb.kind = phase_event_kind_name(primary_kind);
          rb.cb_expression = combined_cb_expr;
          rb.cb_targets = all_targets;
          rb.tile_count = count_expr;
          rb.loop_depth = raw_loops.size();
          rb.loop_scope = raw_loops;
          rb.branch_scope = branches;
          result_.readback.push_back(std::move(rb));

          for (const auto &[target, eff_loops] : emitted_targets) {
            PhaseEvent ev;
            ev.kind = target.kind;
            ev.operation = unqualified;
            ev.cb_expression = combined_cb_expr;
            ev.cb_name = target.name;
            ev.cb_index = target.index;
            ev.count_expr = count_expr.value_or("");
            ev.normalized_call = normalized_call;
            ev.line = call_line;
            ev.loop_scope = eff_loops;
            ev.branch_scope = branches;
            result_.events.push_back(std::move(ev));
          }
        }
        i = rparen + 1;
        continue;
      }

      // Refusal by name: any unrecognized call whose name looks like a CB/macro
      // wrapper OR whose arguments directly pass a declared CB symbol.
      if (looks_like_unrecognized_cb_call(unqualified)) {
        throw ContractViolation(
            source_.role + ": unrecognized CB/macro call '" + qualified +
            "' at line " + std::to_string(call_line));
      }
      for (const auto &[arg_begin, arg_end] : args) {
        if (const auto cb_ref = find_direct_cb_reference(arg_begin, arg_end)) {
          throw ContractViolation(
              source_.role + ": unrecognized CB call '" + qualified +
              "' referencing CB '" + *cb_ref + "' at line " +
              std::to_string(call_line));
        }
      }

      i = rparen + 1;
    }
  }
};

struct OpenPhaseWindow {
  std::size_t open_line = 0;
  std::string count_expr;
  std::string count_key;
  std::vector<std::string> loop_scope;
  std::vector<std::string> branch_scope;
  bool accessed = false;
  std::vector<std::string> access_ops;
  std::vector<std::string> access_calls;
};

struct ClosedPhaseWindow {
  std::size_t open_line = 0;
  std::size_t close_line = 0;
  std::string count_expr;
  std::string count_key;
  std::vector<std::string> loop_scope;
  std::vector<std::string> branch_scope;
  std::vector<std::string> access_ops;
  std::vector<std::string> access_calls;
};

struct CbWindowState {
  std::string cb_name;
  std::uint32_t cb_index = 0;
  std::optional<OpenPhaseWindow> open_producer;
  std::optional<OpenPhaseWindow> open_consumer;
  std::vector<ClosedPhaseWindow> producer_windows;
  std::vector<ClosedPhaseWindow> consumer_windows;
  std::size_t total_producer_events = 0;
  std::size_t total_consumer_events = 0;
  // Loop-carried consumer state tracking (for compute's on-chip cb_state carry)
  std::optional<std::vector<std::string>> pending_carry_loop;
  std::string pending_carry_count_key;
  std::size_t pending_carry_pop_line = 0;
};

struct PhaseValidationPolicy {
  std::optional<std::string> resident_scratch_cb; // reader c_dram_read_scratch
  std::optional<std::string> resident_const_cb;   // compute cb_ident
  std::optional<std::string> loop_carry_cb;       // compute cb_state
  std::optional<std::string> compute_output_cb;   // compute cb_out
};

[[nodiscard]] bool is_scope_prefix(const std::vector<std::string> &outer,
                                   const std::vector<std::string> &inner) {
  if (outer.size() > inner.size()) {
    return false;
  }
  for (std::size_t i = 0; i < outer.size(); ++i) {
    if (outer[i] != inner[i]) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] std::string format_scope(const std::vector<std::string> &scope) {
  if (scope.empty()) {
    return "<top-level>";
  }
  std::string out;
  for (std::size_t i = 0; i < scope.size(); ++i) {
    if (i > 0) {
      out += " > ";
    }
    out += scope[i];
  }
  return out;
}

[[nodiscard]] std::map<std::uint32_t, CbWindowState> validate_cb_phase_windows(
    const SourceFile &source, const CbSymbolTable &cbs,
    const KernelPhaseExtraction &extraction,
    const PhaseValidationPolicy &policy) {
  std::map<std::uint32_t, CbWindowState> states;
  for (const auto &[idx, name] : cbs.index_to_name) {
    CbWindowState st;
    st.cb_name = name;
    st.cb_index = idx;
    states.emplace(idx, std::move(st));
  }

  for (const PhaseEvent &ev : extraction.events) {
    if (ev.kind == PhaseEventKind::LoopExit) {
      for (auto &[idx, st] : states) {
        (void)idx;
        if (st.pending_carry_loop.has_value() &&
            *st.pending_carry_loop == ev.loop_scope) {
          if (!st.open_consumer.has_value() ||
              st.open_consumer->loop_scope != ev.loop_scope ||
              st.open_consumer->branch_scope.empty() ||
              st.open_consumer->branch_scope.back() != "if (!last)" ||
              st.open_consumer->count_key != st.pending_carry_count_key) {
            throw ContractViolation(
                source.role + ": loop-carried consumer window on CB '" +
                st.cb_name + "' popped at line " +
                std::to_string(st.pending_carry_pop_line) +
                " lacks a matching if (!last) cb_wait_front re-establishment "
                "before loop exit");
          }
          // On the final iteration (!last is false), the carried consumer
          // window closes cleanly at loop exit.
          st.open_consumer.reset();
          st.pending_carry_loop.reset();
          st.pending_carry_count_key.clear();
          st.pending_carry_pop_line = 0;
          continue;
        }
        if (st.open_producer.has_value() &&
            is_scope_prefix(ev.loop_scope, st.open_producer->loop_scope)) {
          throw ContractViolation(
              source.role + ": unclosed cb_reserve_back window on CB '" +
              st.cb_name + "' (index " + std::to_string(st.cb_index) +
              ") opened at line " +
              std::to_string(st.open_producer->open_line) +
              " crosses end of loop '" + format_scope(ev.loop_scope) + "'");
        }
        if (st.open_consumer.has_value() &&
            is_scope_prefix(ev.loop_scope, st.open_consumer->loop_scope)) {
          throw ContractViolation(
              source.role + ": unclosed cb_wait_front window on CB '" +
              st.cb_name + "' (index " + std::to_string(st.cb_index) +
              ") opened at line " +
              std::to_string(st.open_consumer->open_line) +
              " crosses end of loop '" + format_scope(ev.loop_scope) + "'");
        }
      }
      continue;
    }

    if (ev.kind == PhaseEventKind::BranchExit) {
      for (auto &[idx, st] : states) {
        (void)idx;
        if (st.open_producer.has_value() &&
            is_scope_prefix(ev.branch_scope, st.open_producer->branch_scope)) {
          throw ContractViolation(
              source.role + ": unclosed cb_reserve_back window on CB '" +
              st.cb_name + "' (index " + std::to_string(st.cb_index) +
              ") opened at line " +
              std::to_string(st.open_producer->open_line) +
              " crosses end of branch '" + format_scope(ev.branch_scope) + "'");
        }
        if (st.open_consumer.has_value() &&
            is_scope_prefix(ev.branch_scope, st.open_consumer->branch_scope)) {
          const bool is_valid_carry_rewait =
              policy.loop_carry_cb.has_value() &&
              st.cb_name == *policy.loop_carry_cb &&
              st.pending_carry_loop.has_value() &&
              st.open_consumer->loop_scope == *st.pending_carry_loop &&
              !ev.branch_scope.empty() && ev.branch_scope.back() == "if (!last)";
          if (!is_valid_carry_rewait) {
            throw ContractViolation(
                source.role + ": unclosed cb_wait_front window on CB '" +
                st.cb_name + "' (index " + std::to_string(st.cb_index) +
                ") opened at line " +
                std::to_string(st.open_consumer->open_line) +
                " crosses end of branch '" + format_scope(ev.branch_scope) +
                "'");
          }
        }
      }
      continue;
    }

    if (ev.kind == PhaseEventKind::ControlJump) {
      for (const auto &[idx, st] : states) {
        (void)idx;
        if (st.open_producer.has_value()) {
          const bool is_resident_scratch =
              policy.resident_scratch_cb.has_value() &&
              st.cb_name == *policy.resident_scratch_cb &&
              st.open_producer->loop_scope.empty();
          if (!is_resident_scratch) {
            throw ContractViolation(
                source.role + ": control-flow jump '" + ev.operation +
                "' at line " + std::to_string(ev.line) +
                " occurs while cb_reserve_back on CB '" + st.cb_name +
                "' (opened at line " +
                std::to_string(st.open_producer->open_line) +
                ") is still open");
          }
        }
        if (st.open_consumer.has_value()) {
          const bool is_resident_const =
              policy.resident_const_cb.has_value() &&
              st.cb_name == *policy.resident_const_cb &&
              st.open_consumer->loop_scope.empty();
          if (!is_resident_const) {
            throw ContractViolation(
                source.role + ": control-flow jump '" + ev.operation +
                "' at line " + std::to_string(ev.line) +
                " occurs while cb_wait_front on CB '" + st.cb_name +
                "' (opened at line " +
                std::to_string(st.open_consumer->open_line) +
                ") is still open");
          }
        }
      }
      continue;
    }

    CbWindowState &st = states.at(ev.cb_index);
    const std::string count_key = strip_whitespace(ev.count_expr);

    switch (ev.kind) {
    case PhaseEventKind::Query:
    case PhaseEventKind::Config:
      break;

    case PhaseEventKind::Reserve:
      if (st.open_producer.has_value()) {
        throw ContractViolation(
            source.role + ": nested/duplicate cb_reserve_back on CB '" +
            st.cb_name + "' (index " + std::to_string(st.cb_index) +
            ") at line " + std::to_string(ev.line) +
            " while window from line " +
            std::to_string(st.open_producer->open_line) + " is still open");
      }
      st.open_producer = OpenPhaseWindow{ev.line, ev.count_expr, count_key,
                                         ev.loop_scope, ev.branch_scope,
                                         false, {}, {}};
      ++st.total_producer_events;
      break;

    case PhaseEventKind::Push:
      if (!st.open_producer.has_value()) {
        throw ContractViolation(
            source.role + ": cb_push_back on CB '" + st.cb_name + "' (index " +
            std::to_string(st.cb_index) + ") at line " +
            std::to_string(ev.line) +
            " without an open cb_reserve_back window");
      }
      if (st.open_producer->count_key != count_key) {
        std::string prefix;
        if (policy.compute_output_cb.has_value() &&
            st.cb_name == *policy.compute_output_cb) {
          prefix = "compute output tile-count mismatch: ";
        }
        throw ContractViolation(
            source.role + ": " + prefix + "cb_push_back(" + st.cb_name + ", " +
            ev.count_expr + ") at line " + std::to_string(ev.line) +
            " does not match cb_reserve_back(" + st.cb_name + ", " +
            st.open_producer->count_expr + ") at line " +
            std::to_string(st.open_producer->open_line));
      }
      if (st.open_producer->loop_scope != ev.loop_scope ||
          st.open_producer->branch_scope != ev.branch_scope) {
        throw ContractViolation(
            source.role + ": cb_push_back on CB '" + st.cb_name + "' (index " +
            std::to_string(st.cb_index) + ") at line " +
            std::to_string(ev.line) + " scope [" + format_scope(ev.loop_scope) +
            "] does not match cb_reserve_back at line " +
            std::to_string(st.open_producer->open_line) + " scope [" +
            format_scope(st.open_producer->loop_scope) + "]");
      }
      st.producer_windows.push_back(ClosedPhaseWindow{
          st.open_producer->open_line, ev.line, st.open_producer->count_expr,
          st.open_producer->count_key, st.open_producer->loop_scope,
          st.open_producer->branch_scope,
          std::move(st.open_producer->access_ops),
          std::move(st.open_producer->access_calls)});
      st.open_producer.reset();
      ++st.total_producer_events;
      break;

    case PhaseEventKind::ProducerAccess:
    case PhaseEventKind::ProducerInit:
      if (!st.open_producer.has_value()) {
        throw ContractViolation(
            source.role + ": producer access '" + ev.operation + "' on CB '" +
            st.cb_name + "' (index " + std::to_string(st.cb_index) +
            ") at line " + std::to_string(ev.line) +
            " occurs outside an open cb_reserve_back..cb_push_back window");
      }
      if (!is_scope_prefix(st.open_producer->loop_scope, ev.loop_scope) ||
          !is_scope_prefix(st.open_producer->branch_scope, ev.branch_scope)) {
        throw ContractViolation(
            source.role + ": producer access '" + ev.operation + "' on CB '" +
            st.cb_name + "' (index " + std::to_string(st.cb_index) +
            ") at line " + std::to_string(ev.line) +
            " is outside the loop/branch scope of cb_reserve_back at line " +
            std::to_string(st.open_producer->open_line));
      }
      if (ev.kind == PhaseEventKind::ProducerAccess) {
        st.open_producer->accessed = true;
        st.open_producer->access_ops.push_back(ev.operation);
        st.open_producer->access_calls.push_back(ev.normalized_call);
      }
      ++st.total_producer_events;
      break;

    case PhaseEventKind::Wait:
      if (st.open_consumer.has_value()) {
        throw ContractViolation(
            source.role + ": nested/duplicate cb_wait_front on CB '" +
            st.cb_name + "' (index " + std::to_string(st.cb_index) +
            ") at line " + std::to_string(ev.line) +
            " while window from line " +
            std::to_string(st.open_consumer->open_line) + " is still open");
      }
      st.open_consumer = OpenPhaseWindow{ev.line, ev.count_expr, count_key,
                                         ev.loop_scope, ev.branch_scope,
                                         false, {}, {}};
      ++st.total_consumer_events;
      break;

    case PhaseEventKind::Pop: {
      if (!st.open_consumer.has_value()) {
        throw ContractViolation(
            source.role + ": cb_pop_front on CB '" + st.cb_name + "' (index " +
            std::to_string(st.cb_index) + ") at line " +
            std::to_string(ev.line) + " without an open cb_wait_front window");
      }
      if (st.open_consumer->count_key != count_key) {
        throw ContractViolation(
            source.role + ": cb_pop_front(" + st.cb_name + ", " +
            ev.count_expr + ") at line " + std::to_string(ev.line) +
            " does not match cb_wait_front(" + st.cb_name + ", " +
            st.open_consumer->count_expr + ") at line " +
            std::to_string(st.open_consumer->open_line));
      }
      const bool exact_scope =
          st.open_consumer->loop_scope == ev.loop_scope &&
          st.open_consumer->branch_scope == ev.branch_scope;
      const bool valid_loop_carry_pop =
          policy.loop_carry_cb.has_value() &&
          st.cb_name == *policy.loop_carry_cb &&
          !st.pending_carry_loop.has_value() &&
          st.open_consumer->branch_scope == ev.branch_scope &&
          is_scope_prefix(st.open_consumer->loop_scope, ev.loop_scope) &&
          ev.loop_scope.size() == st.open_consumer->loop_scope.size() + 1;
      if (!exact_scope && !valid_loop_carry_pop) {
        throw ContractViolation(
            source.role + ": cb_pop_front on CB '" + st.cb_name + "' (index " +
            std::to_string(st.cb_index) + ") at line " +
            std::to_string(ev.line) + " scope [" + format_scope(ev.loop_scope) +
            "] does not match cb_wait_front at line " +
            std::to_string(st.open_consumer->open_line) + " scope [" +
            format_scope(st.open_consumer->loop_scope) + "]");
      }
      if (valid_loop_carry_pop) {
        st.pending_carry_loop = ev.loop_scope;
        st.pending_carry_count_key = count_key;
        st.pending_carry_pop_line = ev.line;
      }
      st.consumer_windows.push_back(ClosedPhaseWindow{
          st.open_consumer->open_line, ev.line, st.open_consumer->count_expr,
          st.open_consumer->count_key, st.open_consumer->loop_scope,
          st.open_consumer->branch_scope,
          std::move(st.open_consumer->access_ops),
          std::move(st.open_consumer->access_calls)});
      st.open_consumer.reset();
      ++st.total_consumer_events;
      break;
    }

    case PhaseEventKind::ConsumerAccess:
    case PhaseEventKind::ConsumerInit:
      if (!st.open_consumer.has_value()) {
        throw ContractViolation(
            source.role + ": consumer access '" + ev.operation + "' on CB '" +
            st.cb_name + "' (index " + std::to_string(st.cb_index) +
            ") at line " + std::to_string(ev.line) +
            " occurs outside an open cb_wait_front..cb_pop_front window");
      }
      if (!is_scope_prefix(st.open_consumer->loop_scope, ev.loop_scope) ||
          !is_scope_prefix(st.open_consumer->branch_scope, ev.branch_scope)) {
        throw ContractViolation(
            source.role + ": consumer access '" + ev.operation + "' on CB '" +
            st.cb_name + "' (index " + std::to_string(st.cb_index) +
            ") at line " + std::to_string(ev.line) +
            " is outside the loop/branch scope of cb_wait_front at line " +
            std::to_string(st.open_consumer->open_line));
      }
      if (ev.kind == PhaseEventKind::ConsumerAccess) {
        st.open_consumer->accessed = true;
        st.open_consumer->access_ops.push_back(ev.operation);
        st.open_consumer->access_calls.push_back(ev.normalized_call);
      }
      ++st.total_consumer_events;
      break;

    case PhaseEventKind::LoopExit:
    case PhaseEventKind::BranchExit:
    case PhaseEventKind::ControlJump:
      break;
    }
  }

  // End-of-kernel closure validation for every declared CB.
  for (const auto &[idx, st] : states) {
    (void)idx;
    if (st.open_producer.has_value()) {
      const bool allowed_resident_scratch =
          policy.resident_scratch_cb.has_value() &&
          st.cb_name == *policy.resident_scratch_cb &&
          st.open_producer->loop_scope.empty() &&
          st.open_producer->branch_scope.empty() &&
          st.open_producer->count_key == "1" && st.open_producer->accessed &&
          st.producer_windows.empty() && st.total_consumer_events == 0;
      if (!allowed_resident_scratch) {
        throw ContractViolation(
            source.role + ": unclosed cb_reserve_back window on CB '" +
            st.cb_name + "' (index " + std::to_string(st.cb_index) +
            ") opened at line " + std::to_string(st.open_producer->open_line));
      }
    }
    if (st.open_consumer.has_value()) {
      const bool allowed_resident_const =
          policy.resident_const_cb.has_value() &&
          st.cb_name == *policy.resident_const_cb &&
          st.open_consumer->loop_scope.empty() &&
          st.open_consumer->branch_scope.empty() &&
          st.open_consumer->count_key == "4" && st.open_consumer->accessed &&
          st.consumer_windows.empty() && st.producer_windows.size() == 1;
      if (!allowed_resident_const) {
        throw ContractViolation(
            source.role + ": unclosed cb_wait_front window on CB '" +
            st.cb_name + "' (index " + std::to_string(st.cb_index) +
            ") opened at line " + std::to_string(st.open_consumer->open_line));
      }
    }
  }

  return states;
}

[[nodiscard]] bool contains_op(const std::vector<std::string> &ops,
                               std::string_view expected) {
  return std::find(ops.begin(), ops.end(), expected) != ops.end();
}

[[nodiscard]] Json build_stage_readback_json(
    const KernelPhaseExtraction &extraction) {
  Json arr = Json::array();
  for (const ReadbackEntry &entry : extraction.readback) {
    Json targets = Json::array();
    for (const ResolvedCbTarget &target : entry.cb_targets) {
      targets.push_back({{"name", target.name},
                         {"index", target.index},
                         {"role", phase_event_kind_name(target.kind)}});
    }
    Json obj = {
        {"line", entry.line},
        {"source_line", entry.source_line},
        {"operation", entry.operation},
        {"kind", entry.kind},
        {"cb_expression", entry.cb_expression},
        {"cb_targets", targets},
        {"loop_depth", entry.loop_depth},
        {"loop_scope", entry.loop_scope},
        {"branch_scope", entry.branch_scope},
    };
    if (entry.tile_count.has_value()) {
      obj["tile_count"] = *entry.tile_count;
    }
    if (entry.inlined_from_lambda.has_value()) {
      obj["inlined_from_lambda"] = *entry.inlined_from_lambda;
      obj["lambda_body_line"] = *entry.lambda_body_line;
      obj["lambda_body_source_line"] = *entry.lambda_body_source_line;
    }
    arr.push_back(std::move(obj));
  }
  return arr;
}

struct KernelPhaseAnalysis {
  CbSymbolTable cbs;
  KernelPhaseExtraction extraction;
  std::map<std::uint32_t, CbWindowState> windows;
};

[[nodiscard]] KernelPhaseAnalysis analyze_kernel_phases(
    const SourceFile &source, const PhaseValidationPolicy &policy) {
  const std::vector<std::string> lines = split_source_lines(source.text);
  const std::vector<Token> tokens = tokenize_kernel_source(source);
  std::optional<std::size_t> main_token;
  for (std::size_t i = 0; i + 4 < tokens.size(); ++i) {
    if (tokens[i].kind == TokenKind::Identifier && tokens[i].text == "void" &&
        tokens[i + 1].kind == TokenKind::Identifier &&
        tokens[i + 1].text == "kernel_main" &&
        tokens[i + 2].kind == TokenKind::LParen &&
        tokens[i + 3].kind == TokenKind::RParen &&
        tokens[i + 4].kind == TokenKind::LBrace) {
      main_token = i;
      break;
    }
  }
  if (!main_token.has_value()) {
    throw ContractViolation(source.role +
                            ": lacks a void kernel_main() definition");
  }
  KernelPhaseAnalysis analysis;
  analysis.cbs = parse_namespace_cb_constants(tokens, source, *main_token);
  KernelPhaseParser parser(source, lines, tokens, analysis.cbs);
  analysis.extraction = parser.extract();
  analysis.windows = validate_cb_phase_windows(source, analysis.cbs,
                                               analysis.extraction, policy);
  return analysis;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 8) {
      std::fprintf(stderr,
                   "usage: %s CONTRACT_JSON ABI_HEADER RUNNER_SOURCE "
                   "READER_SOURCE COMPUTE_SOURCE WRITER_SOURCE FIXTURE_JSON\n",
                   argc > 0 ? argv[0] : "wkv7-decode-composition-validator");
      return 2;
    }

    const SourceFile contract_file{
        "contract", argv[1],
        read_bounded_file(argv[1], "contract fixture")};
    const SourceFile abi{"abi_header", argv[2],
                         read_bounded_file(argv[2], "ABI header")};
    const SourceFile host{"host", argv[3],
                          read_bounded_file(argv[3], "host construction")};
    const SourceFile reader{"decode_reader", argv[4],
                            read_bounded_file(argv[4], "decode reader")};
    const SourceFile compute{"decode_compute", argv[5],
                             read_bounded_file(argv[5], "decode compute")};
    const SourceFile writer{"decode_writer", argv[6],
                            read_bounded_file(argv[6], "decode writer")};
    const SourceFile boundary{"boundary_fixture", argv[7],
                              read_bounded_file(argv[7], "boundary fixture")};

    Json contract;
    try {
      contract = Json::parse(contract_file.text);
    } catch (const Json::parse_error &error) {
      throw UsageError("contract fixture is not valid JSON: " +
                       std::string(error.what()));
    }
    Json boundary_json;
    try {
      boundary_json = Json::parse(boundary.text);
    } catch (const Json::parse_error &error) {
      throw UsageError("boundary fixture is not valid JSON: " +
                       std::string(error.what()));
    }

    if (require_u32_field(contract, "schema_version", "contract") != 1) {
      throw ContractViolation("contract.schema_version must be 1");
    }
    if (require_string_field(contract, "target", "contract") !=
        "ttwkv7-decode-composition") {
      throw ContractViolation("contract.target is not the decode composition");
    }

    // ---- boundary fixture binding ----
    const Json &fixture_contract =
        require_field(contract, "fixture", "contract");
    const Json &dimensions_contract =
        require_field(fixture_contract, "dimensions", "contract.fixture");
    const Json &dimensions =
        require_field(boundary_json, "dimensions", "boundary fixture");
    for (const char *dimension :
         {"head_count", "head_size", "hidden_size", "intermediate_size"}) {
      if (require_u32_field(dimensions, dimension, "boundary dimensions") !=
          require_u32_field(dimensions_contract, dimension,
                            "contract.fixture.dimensions")) {
        throw ContractViolation(std::string("boundary fixture dimension ") +
                                dimension +
                                " drifted from the composition contract");
      }
    }
    const std::vector<std::string> expected_input_order =
        require_string_array(fixture_contract, "input_order",
                             "contract.fixture");
    const Json &input_order =
        require_field(boundary_json, "input_order", "boundary fixture");
    if (!input_order.is_array() ||
        input_order.size() != expected_input_order.size()) {
      throw ContractViolation(
          "boundary fixture input order drifted from the composition "
          "contract");
    }
    for (std::size_t index = 0; index < expected_input_order.size(); ++index) {
      if (!input_order[index].is_string() ||
          input_order[index].get<std::string>() !=
              expected_input_order[index]) {
        throw ContractViolation(
            "boundary fixture input order drifted from the composition "
            "contract at index " +
            std::to_string(index));
      }
    }
    if (require_u32_field(boundary_json, "schema_version",
                          "boundary fixture") !=
            require_u32_field(fixture_contract, "schema_version",
                              "contract.fixture") ||
        require_u32_field(boundary_json, "layer_index", "boundary fixture") !=
            require_u32_field(fixture_contract, "layer_index",
                              "contract.fixture")) {
      throw ContractViolation(
          "boundary fixture schema/layer authority drifted from the "
          "composition contract");
    }
    for (const char *artifact :
         {"pre_state_artifact", "expected_output_artifact",
          "expected_post_state_artifact"}) {
      if (boundary_json.find(artifact) == boundary_json.end()) {
        throw ContractViolation(std::string("boundary fixture lacks ") +
                                artifact);
      }
    }
    const std::uint32_t fixture_sequence_count =
        require_u32_field(fixture_contract, "sequence_count",
                          "contract.fixture");
    const std::uint32_t fixture_token_count =
        require_u32_field(fixture_contract, "token_count", "contract.fixture");

    // ---- tile shape/pages ----
    const Json &tile = require_field(contract, "tile", "contract");
    const std::uint32_t tile_width = require_u32_field(tile, "width", "tile");
    const std::uint32_t tile_height =
        require_u32_field(tile, "height", "tile");
    const std::uint32_t tile_element_bytes =
        require_u32_field(tile, "element_bytes", "tile");
    const std::uint32_t tile_page_bytes =
        require_u32_field(tile, "page_bytes", "tile");
    const std::string tile_format =
        require_string_field(tile, "format", "tile");
    if (tile_width != ttwkv7::decode_abi::kTileWidth ||
        tile_height != ttwkv7::decode_abi::kTileWidth ||
        tile_element_bytes != 2 ||
        tile_page_bytes != tile_width * tile_height * tile_element_bytes ||
        tile_page_bytes != 2048 || tile_format != "Float16_b") {
      throw ContractViolation(
          "contract tile shape/format/page size is not the reviewed 32x32 "
          "bf16 tile");
    }
    require_contains(host, require_string_field(tile, "runner_tile_constant",
                                                "tile"),
                     "tile width/height constants drifted");
    require_contains(host, require_string_field(tile, "host_page_allocation",
                                                "tile"),
                     "tile page-size allocation drifted");
    require_contains(host, require_string_field(tile, "host_config", "tile"),
                     "tile page size/format drift");
    require_contains(host, require_string_field(tile, "host_create", "tile"),
                     "circular-buffer creation drifted");

    // ---- ABI header constants and ordered enum meanings ----
    const Json &abi_contract = require_field(contract, "abi", "contract");
    const std::map<std::string, std::uint64_t> parsed_constants =
        parse_abi_constants(abi.text);
    const Json &constant_contract =
        require_field(abi_contract, "constants", "contract.abi");
    for (const auto &entry : constant_contract.items()) {
      const auto parsed = parsed_constants.find(entry.key());
      if (parsed == parsed_constants.end()) {
        throw ContractViolation("ABI header lacks constant " + entry.key());
      }
      if (!entry.value().is_number_unsigned()) {
        throw ContractViolation("contract.abi.constants." + entry.key() +
                                " must be an unsigned integer");
      }
      if (parsed->second != entry.value().get<std::uint64_t>()) {
        throw ContractViolation("ABI constant drift for " + entry.key() +
                                " (header has " +
                                std::to_string(parsed->second) +
                                ", contract requires " +
                                std::to_string(
                                    entry.value().get<std::uint64_t>()) +
                                ")");
      }
    }
    // The compiled header is the authority for these anchors.
    if (parsed_constants.at("kReaderArgumentCount") !=
            ttwkv7::decode_abi::kReaderArgumentCount ||
        parsed_constants.at("kComputeArgumentCount") !=
            ttwkv7::decode_abi::kComputeArgumentCount ||
        parsed_constants.at("kWriterArgumentCount") !=
            ttwkv7::decode_abi::kWriterArgumentCount ||
        parsed_constants.at("kInputCount") !=
            ttwkv7::decode_abi::kInputCount ||
        parsed_constants.at("kTileWidth") != ttwkv7::decode_abi::kTileWidth ||
        parsed_constants.at("kMaximumTokenCount") !=
            ttwkv7::decode_abi::kMaximumTokenCount ||
        parsed_constants.at("kDecodeInstancePack") !=
            ttwkv7::decode_abi::kDecodeInstancePack ||
        parsed_constants.at("kDecodeReaderChunkCount") !=
            ttwkv7::decode_abi::kDecodeReaderChunkCount ||
        parsed_constants.at("kDecodeTokensPerChunk") !=
            ttwkv7::decode_abi::kDecodeTokensPerChunk ||
        parsed_constants.at("kDecodeInstancesPerGroup") !=
            ttwkv7::decode_abi::kDecodeInstancesPerGroup) {
      throw ContractViolation(
          "ABI header constants no longer match the compiled composition "
          "contract");
    }

    const std::vector<std::string> reader_meanings =
        require_string_array(abi_contract, "reader_arguments", "contract.abi");
    const std::vector<std::string> compute_meanings =
        require_string_array(abi_contract, "compute_arguments", "contract.abi");
    const std::vector<std::string> writer_meanings =
        require_string_array(abi_contract, "writer_arguments", "contract.abi");

    const std::vector<std::string> canonical_reader(
        kCanonicalReaderArgumentNames.begin(),
        kCanonicalReaderArgumentNames.end());
    const std::vector<std::string> canonical_compute(
        kCanonicalComputeArgumentNames.begin(),
        kCanonicalComputeArgumentNames.end());
    const std::vector<std::string> canonical_writer(
        kCanonicalWriterArgumentNames.begin(),
        kCanonicalWriterArgumentNames.end());

    const std::vector<std::string> parsed_reader =
        parse_enum_members(abi.text, "ReaderArgument");
    const std::vector<std::string> parsed_compute =
        parse_enum_members(abi.text, "ComputeArgument");
    const std::vector<std::string> parsed_writer =
        parse_enum_members(abi.text, "WriterArgument");

    compare_canonical_names(parsed_reader, canonical_reader, "reader");
    compare_canonical_names(parsed_compute, canonical_compute, "compute");
    compare_canonical_names(parsed_writer, canonical_writer, "writer");
    compare_argument_names(parsed_reader, reader_meanings, "reader");
    compare_argument_names(parsed_compute, compute_meanings, "compute");
    compare_argument_names(parsed_writer, writer_meanings, "writer");

    for (const std::string &assertion :
         require_string_array(abi_contract, "static_asserts", "contract.abi")) {
      require_collapsed_contains(abi, assertion,
                                 "ABI argument-count assertion drifted");
    }

    // ---- ABI assignment expressions (ordered meanings) ----
    const auto parsed_assignments = parse_runtime_assignments(abi.text);
    std::vector<RuntimeAssignment> assignments;
    assignments.reserve(parsed_assignments.size());
    for (const auto &entry : parsed_assignments) {
      assignments.push_back(entry.second);
    }

    const Json &runtime_arguments =
        require_field(contract, "runtime_arguments", "contract");
    validate_stage_runtime_arguments(runtime_arguments, "reader", reader,
                                     reader_meanings,
                                     ttwkv7::decode_abi::kReaderArgumentCount);
    validate_stage_runtime_arguments(runtime_arguments, "compute", compute,
                                     compute_meanings,
                                     ttwkv7::decode_abi::kComputeArgumentCount);
    validate_stage_runtime_arguments(runtime_arguments, "writer", writer,
                                     writer_meanings,
                                     ttwkv7::decode_abi::kWriterArgumentCount);
    validate_stage_assignments(runtime_arguments, "reader", abi, assignments);
    validate_stage_assignments(runtime_arguments, "compute", abi, assignments);
    validate_stage_assignments(runtime_arguments, "writer", abi, assignments);

    // ---- runtime argument evaluation against the real ABI function ----
    const CompositionConfig fixture_probe =
        parse_probe(runtime_arguments, "fixture_probe");
    if (fixture_probe.head_count !=
            require_u32_field(dimensions, "head_count", "boundary dimensions") ||
        fixture_probe.head_size !=
            require_u32_field(dimensions, "head_size", "boundary dimensions") ||
        fixture_probe.sequence_count != fixture_sequence_count ||
        fixture_probe.token_count != fixture_token_count) {
      throw ContractViolation(
          "runtime_arguments.fixture_probe does not describe the accepted "
          "boundary fixture");
    }
    const CompositionConfig distinct_probe =
        parse_probe(runtime_arguments, "distinct_probe");
    if (distinct_probe.token_count <= 1 ||
        distinct_probe.token_count > ttwkv7::decode_abi::kMaximumTokenCount ||
        distinct_probe.sequence_count <= 1 ||
        distinct_probe.head_count <= distinct_probe.sequence_count ||
        distinct_probe.instance_end >
            distinct_probe.head_count * distinct_probe.sequence_count) {
      throw ContractViolation(
          "runtime_arguments.distinct_probe is not a distinct bounded probe");
    }

    const Json &probe_expectations =
        require_field(runtime_arguments, "expected", "contract.runtime_arguments");
    const auto fixture_arguments = make_arguments_or_fail(fixture_probe,
                                                          "fixture probe");
    const auto distinct_arguments = make_arguments_or_fail(distinct_probe,
                                                           "distinct probe");
    require_argument_vector(
        std::vector<std::uint32_t>(fixture_arguments.reader.begin(),
                                   fixture_arguments.reader.end()),
        require_field(require_field(probe_expectations, "fixture",
                                    "contract.runtime_arguments.expected"),
                      "reader",
                      "contract.runtime_arguments.expected.fixture"),
        "reader");
    require_argument_vector(
        std::vector<std::uint32_t>(fixture_arguments.compute.begin(),
                                   fixture_arguments.compute.end()),
        require_field(require_field(probe_expectations, "fixture",
                                    "contract.runtime_arguments.expected"),
                      "compute",
                      "contract.runtime_arguments.expected.fixture"),
        "compute");
    require_argument_vector(
        std::vector<std::uint32_t>(fixture_arguments.writer.begin(),
                                   fixture_arguments.writer.end()),
        require_field(require_field(probe_expectations, "fixture",
                                    "contract.runtime_arguments.expected"),
                      "writer",
                      "contract.runtime_arguments.expected.fixture"),
        "writer");
    require_argument_vector(
        std::vector<std::uint32_t>(distinct_arguments.reader.begin(),
                                   distinct_arguments.reader.end()),
        require_field(require_field(probe_expectations, "distinct",
                                    "contract.runtime_arguments.expected"),
                      "reader",
                      "contract.runtime_arguments.expected.distinct"),
        "reader");
    require_argument_vector(
        std::vector<std::uint32_t>(distinct_arguments.compute.begin(),
                                   distinct_arguments.compute.end()),
        require_field(require_field(probe_expectations, "distinct",
                                    "contract.runtime_arguments.expected"),
                      "compute",
                      "contract.runtime_arguments.expected.distinct"),
        "compute");
    require_argument_vector(
        std::vector<std::uint32_t>(distinct_arguments.writer.begin(),
                                   distinct_arguments.writer.end()),
        require_field(require_field(probe_expectations, "distinct",
                                    "contract.runtime_arguments.expected"),
                      "writer",
                      "contract.runtime_arguments.expected.distinct"),
        "writer");

    // Semantic slot checks: the ordered ABI names must carry the values the
    // contract describes.
    const auto reader_index = [](ttwkv7::decode_abi::ReaderArgument argument) {
      return ttwkv7::decode_abi::index(argument);
    };
    const auto compute_index = [](ttwkv7::decode_abi::ComputeArgument argument) {
      return ttwkv7::decode_abi::index(argument);
    };
    const auto writer_index = [](ttwkv7::decode_abi::WriterArgument argument) {
      return ttwkv7::decode_abi::index(argument);
    };
    const auto expect_slot = [](const auto &arguments, std::size_t index,
                                std::uint32_t expected,
                                std::string_view stage) {
      if (arguments[index] != expected) {
        throw ContractViolation(std::string(stage) +
                                " ordered meaning drift at index " +
                                std::to_string(index) + " (has " +
                                std::to_string(arguments[index]) +
                                ", contract requires " +
                                std::to_string(expected) + ")");
      }
    };
    const std::uint32_t tiles_per_head =
        fixture_probe.head_size / ttwkv7::decode_abi::kTileWidth;
    const std::uint32_t head_tile_rows =
        (fixture_probe.head_count + ttwkv7::decode_abi::kTileWidth - 1) /
        ttwkv7::decode_abi::kTileWidth;
    const std::uint32_t instance_count =
        fixture_probe.sequence_count * fixture_probe.head_count;
    const std::uint32_t expected_channel_tiles =
        fixture_probe.head_count * tiles_per_head;
    const std::uint32_t token_rows =
        fixture_probe.sequence_count * fixture_probe.token_count;
    expect_slot(fixture_arguments.reader,
                reader_index(ttwkv7::decode_abi::ReaderArgument::HeadCount),
                fixture_probe.head_count, "reader");
    expect_slot(fixture_arguments.reader,
                reader_index(ttwkv7::decode_abi::ReaderArgument::TokenCount),
                fixture_probe.token_count, "reader");
    expect_slot(
        fixture_arguments.reader,
        reader_index(
            ttwkv7::decode_abi::ReaderArgument::TilesPerHeadDimension),
        tiles_per_head, "reader");
    expect_slot(fixture_arguments.reader,
                reader_index(ttwkv7::decode_abi::ReaderArgument::HeadTileRows),
                head_tile_rows, "reader");
    expect_slot(
        fixture_arguments.reader,
        reader_index(ttwkv7::decode_abi::ReaderArgument::SequenceCount),
        fixture_probe.sequence_count, "reader");
    expect_slot(fixture_arguments.reader,
                reader_index(ttwkv7::decode_abi::ReaderArgument::State),
                fixture_probe.addresses.state, "reader");
    expect_slot(
        fixture_arguments.reader,
        reader_index(
            ttwkv7::decode_abi::ReaderArgument::TokenCountCompatibility),
        fixture_probe.token_count, "reader");
    expect_slot(fixture_arguments.reader,
                reader_index(ttwkv7::decode_abi::ReaderArgument::SelectorUnused),
                0, "reader");
    expect_slot(
        fixture_arguments.reader,
        reader_index(ttwkv7::decode_abi::ReaderArgument::ConstantsUnused), 0,
        "reader");
    expect_slot(
        fixture_arguments.reader,
        reader_index(ttwkv7::decode_abi::ReaderArgument::ChunkCount),
        ttwkv7::decode_abi::kDecodeReaderChunkCount, "reader");
    expect_slot(
        fixture_arguments.reader,
        reader_index(ttwkv7::decode_abi::ReaderArgument::InstanceStart),
        fixture_probe.instance_start, "reader");
    expect_slot(fixture_arguments.reader,
                reader_index(ttwkv7::decode_abi::ReaderArgument::InstanceEnd),
                fixture_probe.instance_end, "reader");
    for (std::size_t input = 0; input < fixture_probe.addresses.inputs.size();
         ++input) {
      expect_slot(fixture_arguments.reader,
                  reader_index(ttwkv7::decode_abi::ReaderArgument::Input0) +
                      input,
                  fixture_probe.addresses.inputs[input], "reader");
    }
    expect_slot(
        fixture_arguments.compute,
        compute_index(
            ttwkv7::decode_abi::ComputeArgument::TilesPerHeadDimension),
        tiles_per_head, "compute");
    expect_slot(fixture_arguments.compute,
                compute_index(ttwkv7::decode_abi::ComputeArgument::InstanceCount),
                instance_count, "compute");
    expect_slot(fixture_arguments.compute,
                compute_index(ttwkv7::decode_abi::ComputeArgument::TokenCount),
                fixture_probe.token_count, "compute");
    expect_slot(fixture_arguments.compute,
                compute_index(ttwkv7::decode_abi::ComputeArgument::HeadTileRows),
                head_tile_rows, "compute");
    expect_slot(fixture_arguments.compute,
                compute_index(ttwkv7::decode_abi::ComputeArgument::ChunkCount),
                ttwkv7::decode_abi::kDecodeReaderChunkCount, "compute");
    expect_slot(
        fixture_arguments.compute,
        compute_index(ttwkv7::decode_abi::ComputeArgument::InstanceStart),
        fixture_probe.instance_start, "compute");
    expect_slot(fixture_arguments.compute,
                compute_index(ttwkv7::decode_abi::ComputeArgument::InstanceEnd),
                fixture_probe.instance_end, "compute");
    expect_slot(
        fixture_arguments.compute,
        compute_index(ttwkv7::decode_abi::ComputeArgument::InstancePack),
        ttwkv7::decode_abi::kDecodeInstancePack, "compute");
    expect_slot(fixture_arguments.writer,
                writer_index(ttwkv7::decode_abi::WriterArgument::Output),
                fixture_probe.addresses.output, "writer");
    expect_slot(fixture_arguments.writer,
                writer_index(ttwkv7::decode_abi::WriterArgument::InstanceStart),
                fixture_probe.instance_start, "writer");
    expect_slot(fixture_arguments.writer,
                writer_index(ttwkv7::decode_abi::WriterArgument::InstanceEnd),
                fixture_probe.instance_end, "writer");
    expect_slot(fixture_arguments.writer,
                writer_index(ttwkv7::decode_abi::WriterArgument::InstanceCount),
                instance_count, "writer");
    expect_slot(fixture_arguments.writer,
                writer_index(ttwkv7::decode_abi::WriterArgument::ChunkCount),
                fixture_probe.token_count, "writer");
    expect_slot(fixture_arguments.writer,
                writer_index(ttwkv7::decode_abi::WriterArgument::HeadCount),
                fixture_probe.head_count, "writer");
    expect_slot(
        fixture_arguments.writer,
        writer_index(
            ttwkv7::decode_abi::WriterArgument::TilesPerHeadDimension),
        tiles_per_head, "writer");
    expect_slot(
        fixture_arguments.writer,
        writer_index(ttwkv7::decode_abi::WriterArgument::ChannelTileCount),
        expected_channel_tiles, "writer");
    expect_slot(fixture_arguments.writer,
                writer_index(ttwkv7::decode_abi::WriterArgument::TokenRows),
                token_rows, "writer");
    expect_slot(fixture_arguments.writer,
                writer_index(ttwkv7::decode_abi::WriterArgument::RealTokenCount),
                fixture_probe.token_count, "writer");
    expect_slot(
        fixture_arguments.writer,
        writer_index(ttwkv7::decode_abi::WriterArgument::TokensPerChunk),
        ttwkv7::decode_abi::kDecodeTokensPerChunk, "writer");
    expect_slot(
        fixture_arguments.writer,
        writer_index(ttwkv7::decode_abi::WriterArgument::InstancesPerGroup),
        ttwkv7::decode_abi::kDecodeInstancesPerGroup, "writer");

    // Bounded rejection probes: the ABI must refuse out-of-contract configs.
    CompositionConfig rejected = fixture_probe;
    rejected.token_count = ttwkv7::decode_abi::kMaximumTokenCount + 1;
    check_rejected_probe(rejected, "token count above kMaximumTokenCount");
    rejected = fixture_probe;
    rejected.token_count = 0;
    check_rejected_probe(rejected, "zero token count");
    rejected = fixture_probe;
    rejected.addresses.output = 0;
    check_rejected_probe(rejected, "zero output address");
    rejected = fixture_probe;
    rejected.instance_end = instance_count + 1;
    check_rejected_probe(rejected, "instance range beyond the instance count");
    rejected = fixture_probe;
    rejected.head_count = 0;
    check_rejected_probe(rejected, "zero head count");

    // ---- circular-buffer indices, capacities, and aliases ----
    const Json &cb_contract = require_field(contract, "circular_buffers",
                                            "contract");
    const std::map<std::string, std::uint32_t> reader_cbs =
        parse_cb_constants(reader.text);
    const std::map<std::string, std::uint32_t> compute_cbs =
        parse_cb_constants(compute.text);
    const std::map<std::string, std::uint32_t> writer_cbs =
        parse_cb_constants(writer.text);
    const std::map<std::string,
                   const std::map<std::string, std::uint32_t> *>
        stage_cbs = {{"reader", &reader_cbs},
                     {"compute", &compute_cbs},
                     {"writer", &writer_cbs}};

    const Json &alias_array = require_field(cb_contract, "aliases",
                                            "contract.circular_buffers");
    if (!alias_array.is_array()) {
      throw ContractViolation(
          "contract.circular_buffers.aliases must be an array");
    }
    std::vector<std::uint32_t> alias_indices;
    for (const Json &alias : alias_array) {
      const std::uint32_t index =
          require_u32_field(alias, "index", "contract.circular_buffers.alias");
      alias_indices.push_back(index);
      const Json &participants =
          require_field(alias, "participants",
                        "contract.circular_buffers.alias");
      if (!participants.is_array() || participants.empty()) {
        throw ContractViolation(
            "contract.circular_buffers.alias.participants must be non-empty");
      }
      for (const Json &participant : participants) {
        const std::string stage =
            require_string_field(participant, "stage",
                                 "circular_buffers.alias.participant");
        const std::string name =
            require_string_field(participant, "name",
                                 "circular_buffers.alias.participant");
        const auto stage_iterator = stage_cbs.find(stage);
        if (stage_iterator == stage_cbs.end()) {
          throw ContractViolation("circular_buffers.alias names unknown stage " +
                                  stage);
        }
        const auto declaration = stage_iterator->second->find(name);
        if (declaration == stage_iterator->second->end()) {
          throw ContractViolation(stage + " kernel no longer declares " + name);
        }
        if (declaration->second != index) {
          throw ContractViolation(
              "staging CB index drift: " + stage + " " + name + " is " +
              std::to_string(declaration->second) +
              ", contract requires " + std::to_string(index));
        }
      }
    }

    // Only the declared alias participants may occupy an aliased index.
    for (const std::uint32_t index : alias_indices) {
      for (const auto &stage : stage_cbs) {
        for (const auto &declaration : *stage.second) {
          if (declaration.second != index) {
            continue;
          }
          bool allowed = false;
          for (const Json &alias : alias_array) {
            if (require_u32_field(alias, "index",
                                  "contract.circular_buffers.alias") != index) {
              continue;
            }
            for (const Json &participant :
                 require_field(alias, "participants",
                               "contract.circular_buffers.alias")) {
              if (require_string_field(participant, "stage",
                                       "circular_buffers.alias.participant") ==
                      stage.first &&
                  require_string_field(participant, "name",
                                       "circular_buffers.alias.participant") ==
                      declaration.first) {
                allowed = true;
              }
            }
          }
          if (!allowed) {
            throw ContractViolation("unexpected CB alias: " + stage.first +
                                    " declares " + declaration.first + " at " +
                                    std::to_string(index));
          }
        }
      }
    }

    const Json &exclusive_array =
        require_field(cb_contract, "exclusive", "contract.circular_buffers");
    if (!exclusive_array.is_array()) {
      throw ContractViolation(
          "contract.circular_buffers.exclusive must be an array");
    }
    for (const Json &entry : exclusive_array) {
      const std::uint32_t index =
          require_u32_field(entry, "index", "contract.circular_buffers");
      const std::string stage =
          require_string_field(entry, "stage", "contract.circular_buffers");
      const std::string name =
          require_string_field(entry, "name", "contract.circular_buffers");
      const auto stage_iterator = stage_cbs.find(stage);
      if (stage_iterator == stage_cbs.end()) {
        throw ContractViolation("exclusive CB names unknown stage " + stage);
      }
      const auto declaration = stage_iterator->second->find(name);
      if (declaration == stage_iterator->second->end() ||
          declaration->second != index) {
        throw ContractViolation("exclusive CB index drift for " + stage + " " +
                                name);
      }
      for (const auto &other_stage : stage_cbs) {
        if (other_stage.first == stage) {
          continue;
        }
        for (const auto &candidate : *other_stage.second) {
          if (candidate.second == index) {
            throw ContractViolation(
                "exclusive CB " + std::to_string(index) + " is also declared "
                "by " + other_stage.first + " " + candidate.first);
          }
        }
      }
    }

    // Decode host allocation statements and capacities.
    const std::string decode_cbs = decode_cb_block(host);
    const std::string collapsed_cbs = collapse_whitespace(decode_cbs);
    const Json &capacity_array =
        require_field(cb_contract, "capacity", "contract.circular_buffers");
    if (!capacity_array.is_array()) {
      throw ContractViolation(
          "contract.circular_buffers.capacity must be an array");
    }
    Json capacity_receipt = Json::array();
    for (const Json &capacity : capacity_array) {
      const std::uint32_t index =
          require_u32_field(capacity, "index",
                            "contract.circular_buffers.capacity");
      const std::string role = require_string_field(
          capacity, "role", "contract.circular_buffers.capacity");
      const std::uint32_t expected_tiles =
          require_u32_field(capacity, "expected_tiles",
                            "contract.circular_buffers.capacity");
      const std::uint32_t minimum_tiles =
          require_u32_field(capacity, "minimum_tiles",
                            "contract.circular_buffers.capacity");
      const std::regex numeric_pattern(
          "CB::c_" + std::to_string(index) + ",\\s*([0-9]+)u\\s*\\)");
      std::smatch match;
      if (!std::regex_search(decode_cbs, match, numeric_pattern)) {
        throw ContractViolation(
            "decode circular buffer c_" + std::to_string(index) +
            " capacity contract is missing or malformed");
      }
      const std::uint32_t actual =
          static_cast<std::uint32_t>(std::stoul(match[1].str()));
      if (actual < minimum_tiles) {
        throw ContractViolation(
            "staging buffer capacity below the contract minimum for c_" +
            std::to_string(index) + " (" + role + "): " +
            std::to_string(actual) + " < " + std::to_string(minimum_tiles) +
            " tiles");
      }
      if (actual != expected_tiles) {
        throw ContractViolation(
            "decode circular buffer capacity drift for c_" +
            std::to_string(index) + " (" + role + "): " +
            std::to_string(actual) + " != " + std::to_string(expected_tiles) +
            " tiles");
      }
      capacity_receipt.push_back(
          {{"index", index},
           {"role", role},
           {"capacity_tiles", actual},
           {"minimum_tiles", minimum_tiles}});
    }

    for (const std::string &statement : require_string_array(
             cb_contract, "decode_statements",
             "contract.circular_buffers")) {
      const std::string collapsed_statement = collapse_whitespace(statement);
      if (collapsed_cbs.find(collapsed_statement) == std::string::npos) {
        throw ContractViolation(
            "decode circular-buffer allocation drifted from the contract "
            "(missing '" +
            collapsed_statement + "')");
      }
    }

    // ---- bounded producer/consumer tile flow ----
    const Json &tile_flow = require_field(contract, "tile_flow", "contract");
    const Json &flow_reader =
        require_field(tile_flow, "reader", "contract.tile_flow");
    const Json &flow_compute =
        require_field(tile_flow, "compute", "contract.tile_flow");
    const Json &flow_writer =
        require_field(tile_flow, "writer", "contract.tile_flow");
    (void)flow_writer;

    const std::string staging_name =
        require_string_field(flow_reader, "staging", "tile_flow.reader");
    const std::string scratch_name =
        require_string_field(flow_reader, "scratch", "tile_flow.reader");
    const std::string output_name =
        require_string_field(flow_compute, "output", "tile_flow.compute");
    if (reader_cbs.at(staging_name) != compute_cbs.at(staging_name) ||
        compute_cbs.at(output_name) != writer_cbs.at(output_name)) {
      throw ContractViolation(
          "producer/consumer tile-flow endpoints do not share the contracted "
          "CB indices");
    }

    // Extract and validate ordered per-CB phase windows across all three
    // decodeL kernels (replacing substring-count checks).
    PhaseValidationPolicy reader_policy;
    reader_policy.resident_scratch_cb = scratch_name;
    const KernelPhaseAnalysis reader_phases =
        analyze_kernel_phases(reader, reader_policy);

    PhaseValidationPolicy compute_policy;
    compute_policy.resident_const_cb = "cb_ident";
    compute_policy.loop_carry_cb = "cb_state";
    compute_policy.compute_output_cb = output_name;
    const KernelPhaseAnalysis compute_phases =
        analyze_kernel_phases(compute, compute_policy);

    PhaseValidationPolicy writer_policy;
    const KernelPhaseAnalysis writer_phases =
        analyze_kernel_phases(writer, writer_policy);

    const std::uint32_t staging_idx = reader_cbs.at(staging_name);
    const std::uint32_t scratch_idx = reader_cbs.at(scratch_name);
    const std::uint32_t output_idx = compute_cbs.at(output_name);

    // ---- Reader phase-order protocol on staging & scratch CBs ----
    const CbWindowState &reader_staging = reader_phases.windows.at(staging_idx);
    if (reader_staging.total_consumer_events != 0) {
      throw ContractViolation(
          "decode_reader: unexpected consumer operations on staging CB '" +
          staging_name + "'");
    }
    if (reader_staging.producer_windows.size() != 2) {
      throw ContractViolation(
          "decode_reader: staging CB '" + staging_name +
          "' must have exactly 2 ordered producer windows (state seed NS, "
          "then token inputs St), found " +
          std::to_string(reader_staging.producer_windows.size()));
    }
    {
      const ClosedPhaseWindow &seed_win = reader_staging.producer_windows[0];
      const ClosedPhaseWindow &tok_win = reader_staging.producer_windows[1];
      if (seed_win.count_key != "NS") {
        throw ContractViolation(
            "decode_reader: first staging producer window must reserve/push "
            "'NS' state-seed tiles (found '" +
            seed_win.count_expr + "')");
      }
      if (seed_win.loop_scope.size() != 1 ||
          seed_win.loop_scope[0] !=
              "for (uint32_t inst = inst_start; inst < inst_end; inst++)") {
        throw ContractViolation(
            "decode_reader: state-seed staging window loop scope drifted "
            "(got [" +
            format_scope(seed_win.loop_scope) + "])");
      }
      if (!contains_op(seed_win.access_ops, "get_write_ptr")) {
        throw ContractViolation(
            "decode_reader: state-seed staging window lacks get_write_ptr "
            "inside reserve..push");
      }
      if (tok_win.count_key != "St") {
        throw ContractViolation(
            "decode_reader: second staging producer window must reserve/push "
            "'St' token-input tiles (found '" +
            tok_win.count_expr + "')");
      }
      if (tok_win.open_line <= seed_win.close_line) {
        throw ContractViolation(
            "decode_reader: token-input staging window must strictly follow "
            "the state-seed window");
      }
      if (tok_win.loop_scope.size() != 3 ||
          tok_win.loop_scope[0] !=
              "for (uint32_t inst = inst_start; inst < inst_end; inst++)" ||
          tok_win.loop_scope[1] != "for (uint32_t t = 0; t < L; t++)" ||
          tok_win.loop_scope[2] != "for (uint32_t inp = 0; inp < 6; inp++)") {
        throw ContractViolation(
            "decode_reader: token-input staging window loop scope drifted "
            "(got [" +
            format_scope(tok_win.loop_scope) + "])");
      }
      if (!contains_op(tok_win.access_ops, "get_write_ptr")) {
        throw ContractViolation(
            "decode_reader: token-input staging window lacks get_write_ptr "
            "inside reserve..push");
      }
    }
    const CbWindowState &reader_scratch = reader_phases.windows.at(scratch_idx);
    if (!reader_scratch.open_producer.has_value() ||
        reader_scratch.open_producer->count_key != "1" ||
        !contains_op(reader_scratch.open_producer->access_ops,
                     "get_write_ptr")) {
      throw ContractViolation(
          "decode_reader: scratch CB '" + scratch_name +
          "' must reserve 1 tile and acquire get_write_ptr inside that "
          "reservation");
    }

    // ---- Compute phase-order protocol on staging & output CBs ----
    const CbWindowState &compute_staging =
        compute_phases.windows.at(staging_idx);
    if (compute_staging.total_producer_events != 0) {
      throw ContractViolation(
          "decode_compute: unexpected producer operations on staging CB '" +
          staging_name + "'");
    }
    if (compute_staging.consumer_windows.size() != 2) {
      throw ContractViolation(
          "decode_compute: staging CB '" + staging_name +
          "' must have exactly 2 ordered consumer windows (state seed NS, "
          "then token inputs St), found " +
          std::to_string(compute_staging.consumer_windows.size()));
    }
    {
      const ClosedPhaseWindow &seed_win = compute_staging.consumer_windows[0];
      const ClosedPhaseWindow &tok_win = compute_staging.consumer_windows[1];
      if (seed_win.count_key != "NS") {
        throw ContractViolation(
            "decode_compute: first staging consumer window must wait/pop "
            "'NS' state-seed tiles (found '" +
            seed_win.count_expr + "')");
      }
      if (seed_win.loop_scope.size() != 1 ||
          seed_win.loop_scope[0] !=
              "for (uint32_t inst = inst_lo; inst < inst_hi; inst++)") {
        throw ContractViolation(
            "decode_compute: state-seed staging window loop scope drifted "
            "(got [" +
            format_scope(seed_win.loop_scope) + "])");
      }
      if (!contains_op(seed_win.access_ops, "copy_tile")) {
        throw ContractViolation(
            "decode_compute: state-seed staging window lacks copy_tile inside "
            "wait..pop");
      }
      if (tok_win.count_key != "St") {
        throw ContractViolation(
            "decode_compute: second staging consumer window must wait/pop "
            "'St' token-input tiles (found '" +
            tok_win.count_expr + "')");
      }
      if (tok_win.open_line <= seed_win.close_line) {
        throw ContractViolation(
            "decode_compute: token-input staging window must strictly follow "
            "the state-seed window");
      }
      if (tok_win.loop_scope.size() != 3 ||
          tok_win.loop_scope[0] !=
              "for (uint32_t inst = inst_lo; inst < inst_hi; inst++)" ||
          tok_win.loop_scope[1] != "for (uint32_t tok = 0; tok < L; tok++)" ||
          tok_win.loop_scope[2] != "for (uint32_t inp = 0; inp < 6; inp++)") {
        throw ContractViolation(
            "decode_compute: token-input staging window loop scope drifted "
            "(got [" +
            format_scope(tok_win.loop_scope) + "])");
      }
      if (!contains_op(tok_win.access_ops, "mul_tiles") ||
          !contains_op(tok_win.access_ops, "transpose_wh_tile") ||
          !contains_op(tok_win.access_ops, "copy_tile")) {
        throw ContractViolation(
            "decode_compute: token-input staging window must read tiles "
            "(mul_tiles/transpose_wh_tile/copy_tile) inside wait..pop");
      }
    }

    const CbWindowState &compute_output = compute_phases.windows.at(output_idx);
    if (compute_output.total_consumer_events != 0) {
      throw ContractViolation(
          "decode_compute: unexpected consumer operations on output CB '" +
          output_name + "'");
    }
    if (compute_output.producer_windows.size() != 1) {
      throw ContractViolation(
          "decode_compute: compute output tile-count mismatch (expected "
          "exactly 1 producer window on '" +
          output_name + "', found " +
          std::to_string(compute_output.producer_windows.size()) + ")");
    }
    {
      const ClosedPhaseWindow &out_win = compute_output.producer_windows[0];
      if (out_win.count_key != "St+NS") {
        throw ContractViolation(
            "decode_compute: compute output tile-count mismatch (expected "
            "'St + NS', found '" +
            out_win.count_expr + "')");
      }
      if (out_win.loop_scope.size() != 2 ||
          out_win.loop_scope[0] !=
              "for (uint32_t inst = inst_lo; inst < inst_hi; inst++)" ||
          out_win.loop_scope[1] != "for (uint32_t tok = 0; tok < L; tok++)") {
        throw ContractViolation(
            "decode_compute: output producer window loop scope drifted (got [" +
            format_scope(out_win.loop_scope) + "])");
      }
      const std::string expected_r1 = "pack_tile(it, " + output_name + ", it)";
      const std::string expected_r2 =
          "pack_tile(t, " + output_name + ", St + t)";
      std::optional<std::size_t> r1_pos;
      std::optional<std::size_t> r2_pos;
      for (std::size_t idx = 0; idx < out_win.access_calls.size(); ++idx) {
        if (!r1_pos.has_value() && out_win.access_calls[idx] == expected_r1) {
          r1_pos = idx;
        }
        if (!r2_pos.has_value() && out_win.access_calls[idx] == expected_r2) {
          r2_pos = idx;
        }
      }
      if (!r1_pos.has_value()) {
        throw ContractViolation(
            "decode_compute: region-1 output layout drifted (missing '" +
            expected_r1 + "' inside reserve..push window)");
      }
      if (!r2_pos.has_value()) {
        throw ContractViolation(
            "decode_compute: region-2 output layout drifted (missing '" +
            expected_r2 + "' inside reserve..push window)");
      }
      if (*r1_pos >= *r2_pos) {
        throw ContractViolation(
            "decode_compute: region-1/region-2 output pack_tile ordering "
            "drifted (region-1 must precede region-2 inside reserve..push)");
      }
    }

    // ---- Writer phase-order protocol on output CB ----
    const CbWindowState &writer_output = writer_phases.windows.at(output_idx);
    if (writer_output.total_producer_events != 0) {
      throw ContractViolation(
          "decode_writer: unexpected producer operations on output CB '" +
          output_name + "'");
    }
    if (writer_output.consumer_windows.size() != 2) {
      throw ContractViolation(
          "decode_writer: output CB '" + output_name +
          "' must have exactly 2 ordered consumer windows (region-1 St, "
          "then region-2 NS), found " +
          std::to_string(writer_output.consumer_windows.size()));
    }
    {
      const ClosedPhaseWindow &r1_win = writer_output.consumer_windows[0];
      const ClosedPhaseWindow &r2_win = writer_output.consumer_windows[1];
      const std::vector<std::string> expected_writer_scope = {
          "for (uint32_t inst0 = inst_lo; inst0 < inst_hi; inst0 += NB)",
          "for (uint32_t cc = 0; cc < nc; cc++)",
          "for (uint32_t g = 0; g < nb; g++)",
      };
      if (r1_win.loop_scope != expected_writer_scope) {
        throw ContractViolation(
            "decode_writer: region-1 consumer window loop scope drifted "
            "(got [" +
            format_scope(r1_win.loop_scope) + "])");
      }
      if (r2_win.loop_scope != expected_writer_scope) {
        throw ContractViolation(
            "decode_writer: region-2 consumer window loop scope drifted "
            "(got [" +
            format_scope(r2_win.loop_scope) + "])");
      }
      if (r1_win.count_key != "St" || r2_win.count_key != "NS") {
        throw ContractViolation(
            "decode_writer: writer region-1/region-2 phase ordering drifted "
            "(expected region-1 window 'St' before region-2 window 'NS', "
            "found '" +
            r1_win.count_expr + "' then '" + r2_win.count_expr + "')");
      }
      if (r2_win.open_line <= r1_win.close_line) {
        throw ContractViolation(
            "decode_writer: region-2 cb_wait_front must strictly follow "
            "region-1 cb_pop_front");
      }
      if (!contains_op(r1_win.access_ops, "get_read_ptr")) {
        throw ContractViolation(
            "decode_writer: region-1 consumer window lacks get_read_ptr "
            "inside wait..pop");
      }
      if (!contains_op(r2_win.access_ops, "get_read_ptr")) {
        throw ContractViolation(
            "decode_writer: region-2 consumer window lacks get_read_ptr "
            "inside wait..pop");
      }
    }

    const std::uint32_t state_tiles = tiles_per_head * tiles_per_head;
    const std::uint32_t reader_staging_tiles =
        state_tiles + ttwkv7::decode_abi::kInputCount * fixture_token_count *
                          tiles_per_head;
    const std::uint32_t compute_consumed_tiles = reader_staging_tiles;
    const std::uint32_t compute_output_tiles =
        fixture_token_count * (tiles_per_head + state_tiles);
    const std::uint32_t writer_consumed_tiles =
        fixture_token_count * (tiles_per_head + state_tiles);
    if (reader_staging_tiles != compute_consumed_tiles ||
        compute_output_tiles != writer_consumed_tiles) {
      throw ContractViolation(
          "bounded producer/consumer tile counts do not balance");
    }
    if (fixture_token_count >
        ttwkv7::decode_abi::kMaximumTokenCount) {
      throw ContractViolation(
          "fixture token count exceeds the bounded decode tile contract");
    }

    // ---- layout expectations ----
    const Json &layout = require_field(contract, "layout", "contract");
    for (const auto &stage : layout.items()) {
      const SourceFile *source = nullptr;
      if (stage.key() == "reader") {
        source = &reader;
      } else if (stage.key() == "compute") {
        source = &compute;
      } else if (stage.key() == "writer") {
        source = &writer;
      } else if (stage.key() == "host") {
        source = &host;
      } else {
        throw ContractViolation("contract.layout names unknown stage " +
                                stage.key());
      }
      if (!stage.value().is_array()) {
        throw ContractViolation("contract.layout." + stage.key() +
                                " must be an array");
      }
      for (const Json &item : stage.value()) {
        if (!item.is_string()) {
          throw ContractViolation("contract.layout." + stage.key() +
                                  " must contain only strings");
        }
        require_collapsed_contains(*source, item.get<std::string>(),
                                   "tile shape/format/layout expectation drifted");
      }
    }

    // ---- host construction binding ----
    const Json &host_contract = require_field(contract, "host", "contract");
    for (const std::string &fragment :
         require_string_array(host_contract, "required", "contract.host")) {
      require_contains(host, fragment, "host construction drifted");
    }
    for (const std::string &fragment :
         require_string_array(host_contract, "forbidden", "contract.host")) {
      require_absent(host, fragment, "host construction retained a stale path");
    }
    require_occurrence_count(
        host, "ttwkv7::decode_abi::make_decode_runtime_arguments(", 1,
        "host must construct the decode runtime arguments exactly once");
    require_collapsed_occurrence_count(
        host,
        "SetRuntimeArgs(prog, rd, cc2, std::vector<uint32_t>(arguments->reader.begin(), "
        "arguments->reader.end()))",
        1, "host reader SetRuntimeArgs drifted");
    require_collapsed_occurrence_count(
        host,
        "SetRuntimeArgs(prog, cp, cc2, std::vector<uint32_t>(arguments->compute.begin(), "
        "arguments->compute.end()))",
        1, "host compute SetRuntimeArgs drifted");
    require_collapsed_occurrence_count(
        host,
        "SetRuntimeArgs(prog, wrk, cc2, std::vector<uint32_t>(arguments->writer.begin(), "
        "arguments->writer.end()))",
        1, "host writer SetRuntimeArgs drifted");
    require_contains(host, "if (!arguments) throw std::logic_error(\"decode "
                           "runtime arguments are invalid\");",
                     "host lost the invalid-argument guard");
    require_contains(host, "kpath(kind, true)", "decode reader kernel path drifted");
    require_contains(host, "kpath(kind, false)",
                     "decode compute kernel path drifted");
    require_contains(host, "\"kernels/wkv7_writer.cpp\"",
                     "decode writer kernel path drifted");

    // ---- stage fingerprints: stale producer/consumer versions ----
    const Json &stage_array = require_field(contract, "stages", "contract");
    if (!stage_array.is_array()) {
      throw ContractViolation("contract.stages must be an array");
    }
    Json stage_receipt = Json::array();
    for (const Json &stage : stage_array) {
      const std::string name =
          require_string_field(stage, "name", "contract.stages");
      const std::string role =
          require_string_field(stage, "role", "contract.stages");
      const std::string expected =
          require_string_field(stage, "fingerprint", "contract.stages");
      const SourceFile *source = nullptr;
      if (name == "abi_header") {
        source = &abi;
      } else if (name == "host") {
        source = &host;
      } else if (name == "decode_reader") {
        source = &reader;
      } else if (name == "decode_compute") {
        source = &compute;
      } else if (name == "decode_writer") {
        source = &writer;
      } else {
        throw ContractViolation("contract.stages names unknown stage " + name);
      }
      const std::string actual = fingerprint(source->text);
      if (actual != expected) {
        throw ContractViolation("producer/consumer version is stale for " +
                                role + " stage '" + name + "' (has " + actual +
                                ", contract requires " + expected + ")");
      }
      stage_receipt.push_back(
          {{"name", name}, {"role", role}, {"fingerprint", actual}});
    }

    Json receipt;
    receipt["target"] = "ttwkv7-decode-composition";
    receipt["device_initialized"] = false;
    receipt["hardware_execution_claimed"] = false;
    receipt["fixture"] = {
        {"head_count", fixture_probe.head_count},
        {"head_size", fixture_probe.head_size},
        {"sequence_count", fixture_probe.sequence_count},
        {"token_count", fixture_probe.token_count},
        {"input_order", expected_input_order},
    };
    receipt["tile"] = {
        {"width", tile_width},
        {"height", tile_height},
        {"element_bytes", tile_element_bytes},
        {"page_bytes", tile_page_bytes},
        {"format", tile_format},
    };
    receipt["runtime_arguments"] = {
        {"fixture",
         {{"reader",
           std::vector<std::uint32_t>(fixture_arguments.reader.begin(),
                                      fixture_arguments.reader.end())},
          {"compute",
           std::vector<std::uint32_t>(fixture_arguments.compute.begin(),
                                      fixture_arguments.compute.end())},
          {"writer",
           std::vector<std::uint32_t>(fixture_arguments.writer.begin(),
                                      fixture_arguments.writer.end())}}},
        {"distinct_probe",
         {{"reader",
           std::vector<std::uint32_t>(distinct_arguments.reader.begin(),
                                      distinct_arguments.reader.end())},
          {"compute",
           std::vector<std::uint32_t>(distinct_arguments.compute.begin(),
                                      distinct_arguments.compute.end())},
          {"writer",
           std::vector<std::uint32_t>(distinct_arguments.writer.begin(),
                                      distinct_arguments.writer.end())}}},
    };
    receipt["tile_flow"] = {
        {"reader_staging_tiles", reader_staging_tiles},
        {"compute_consumed_tiles", compute_consumed_tiles},
        {"compute_output_tiles", compute_output_tiles},
        {"writer_consumed_tiles", writer_consumed_tiles},
        {"state_tiles_per_instance", state_tiles},
    };
    receipt["circular_buffers"] = {
        {"staging_index", reader_cbs.at(staging_name)},
        {"scratch_index", reader_cbs.at(
                              require_string_field(flow_reader, "scratch",
                                                   "tile_flow.reader"))},
        {"output_index", compute_cbs.at(output_name)},
        {"capacities", capacity_receipt},
    };
    receipt["stages"] = stage_receipt;
    receipt["checks"] = {
        "boundary_fixture_dimensions",
        "abi_ordered_argument_meaning",
        "runtime_argument_values",
        "bounded_rejection_probes",
        "circular_buffer_indices_capacities_aliases",
        "producer_consumer_tile_balance",
        "tile_shape_format_layout",
        "host_set_runtime_args_construction",
        "producer_consumer_version_fingerprints",
    };
    receipt["cb_phase_readback"] = {
        {"decode_reader", build_stage_readback_json(reader_phases.extraction)},
        {"decode_compute",
         build_stage_readback_json(compute_phases.extraction)},
        {"decode_writer", build_stage_readback_json(writer_phases.extraction)},
    };
    std::cout << receipt.dump(2) << '\n';
    return 0;
  } catch (const UsageError &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 2;
  } catch (const ContractViolation &error) {
    std::fprintf(stderr, "decode-composition contract violation: %s\n",
                 error.what());
    return 1;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "decode-composition validator failure: %s\n",
                 error.what());
    return 2;
  }
}
