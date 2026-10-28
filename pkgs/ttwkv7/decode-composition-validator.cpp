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
    const std::string output_name =
        require_string_field(flow_compute, "output", "tile_flow.compute");
    if (reader_cbs.at(staging_name) != compute_cbs.at(staging_name) ||
        compute_cbs.at(output_name) != writer_cbs.at(output_name)) {
      throw ContractViolation(
          "producer/consumer tile-flow endpoints do not share the contracted "
          "CB indices");
    }

    require_contains(reader,
                     "cb_reserve_back(" + staging_name + ", NS);",
                     "reader state-seed tile reservation drifted");
    require_occurrence_count(
        reader, "cb_reserve_back(" + staging_name + ", NS);", 1,
        "reader must reserve exactly one state-seed tile group");
    require_occurrence_count(
        reader, "cb_push_back(" + staging_name + ", NS);", 1,
        "reader must publish exactly one state-seed tile group");
    require_occurrence_count(
        reader, "cb_reserve_back(" + staging_name + ", St);", 1,
        "reader token-input reservation drifted");
    require_occurrence_count(
        reader, "cb_push_back(" + staging_name + ", St);", 1,
        "reader token-input publication drifted");
    require_contains(reader, "for (uint32_t t = 0; t < L; t++)",
                     "reader token loop drifted");
    require_contains(reader, "for (uint32_t inp = 0; inp < 6; inp++)",
                     "reader input loop drifted");
    require_contains(reader, "for (uint32_t inp = 0; inp < 6; inp++)",
                     "reader input ordering drifted");

    require_occurrence_count(
        compute, "cb_wait_front(" + staging_name + ", NS);", 1,
        "compute must consume exactly one state-seed tile group");
    require_occurrence_count(
        compute, "cb_pop_front(" + staging_name + ", NS);", 1,
        "compute state-seed consumption drifted");
    require_occurrence_count(
        compute, "cb_wait_front(" + staging_name + ", St);", 1,
        "compute token-input consumption drifted");
    require_occurrence_count(
        compute, "cb_pop_front(" + staging_name + ", St);", 1,
        "compute token-input consumption drifted");
    require_contains(compute, "for (uint32_t tok = 0; tok < L; tok++)",
                     "compute token loop drifted");
    require_contains(compute, "for (uint32_t inp = 0; inp < 6; inp++)",
                     "compute input loop drifted");
    require_occurrence_count(
        compute, "cb_reserve_back(" + output_name + ", St + NS);", 1,
        "compute output tile-count mismatch");
    require_occurrence_count(
        compute, "cb_push_back(" + output_name + ", St + NS);", 1,
        "compute output tile-count mismatch");
    require_contains(compute, "pack_tile(it, " + output_name + ", it);",
                     "compute region-1 output layout drifted");
    require_contains(compute, "pack_tile(t, " + output_name + ", St + t);",
                     "compute region-2 output layout drifted");

    require_contains(writer, "for (uint32_t cc = 0; cc < nc; cc++)",
                     "writer chunk loop drifted");
    require_occurrence_count(writer, "cb_wait_front(" + output_name + ", St);",
                             1, "writer region-1 consumption drifted");
    require_occurrence_count(writer, "cb_pop_front(" + output_name + ", St);",
                             1, "writer region-1 consumption drifted");
    require_occurrence_count(writer, "cb_wait_front(" + output_name + ", NS);",
                             1, "writer region-2 consumption drifted");
    require_occurrence_count(writer, "cb_pop_front(" + output_name + ", NS);",
                             1, "writer region-2 consumption drifted");
    require_contains(writer, "cb_wait_front(" + output_name + ", St);",
                     "writer region-1 ordering drifted");
    require_contains(writer, "cb_wait_front(" + output_name + ", NS);",
                     "writer region-2 ordering drifted");

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
