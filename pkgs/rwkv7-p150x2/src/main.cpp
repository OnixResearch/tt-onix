#include "config.hpp"
#include "fused_wkv_program.hpp"
#include "fused_wkv_reference.hpp"
#include "fused_wkv_reference_self_test.hpp"
#include "persistent_decode_self_test.hpp"
#ifdef RWKV7_FETCH_QUEUE_DIAGNOSTICS_AVAILABLE
#include "hardware_counter_diagnostics.hpp"
#endif

#include <blake3.h>
#include <nlohmann/json.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#ifdef RWKV7_FETCH_QUEUE_DIAGNOSTICS_AVAILABLE
#include <tt-metalium/experimental/fetch_queue_diagnostics.hpp>
#endif
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include <ttnn/distributed/api.hpp>
#include <ttnn/distributed/distributed_tensor.hpp>
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/copy/typecast/typecast.hpp>
#include <ttnn/operations/creation/creation.hpp>
#include <ttnn/operations/data_movement/copy/copy.hpp>
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
#include <ttnn/operations/trace.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/types.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sched.h>
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
constexpr double kNanosecondsPerSecond = 1.0e9;
constexpr std::uint64_t kMaximumFetchQueueWaitChecksPerReserve = 2U;
constexpr std::uint32_t kVocabularySize = RWKV7_PINNED_VOCABULARY_SIZE;
constexpr std::uint32_t kLayerZero = 0U;
constexpr std::uint32_t kMinimumGeneratedTokens = 2U;
constexpr std::uint32_t kLongGeneratedTokenWindow = 4U;
constexpr std::uint32_t kWindowEightGeneratedTokens = 8U;
constexpr std::uint32_t kMaximumGeneratedTokens =
    RWKV7_MAXIMUM_GENERATED_TOKENS;
static_assert(kMaximumGeneratedTokens >= kLongGeneratedTokenWindow);
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
constexpr std::string_view kRuntimeName = RWKV7_RUNTIME_NAME;
constexpr std::string_view kPersistentDecodeProfileBlake3 =
    RWKV7_PERSISTENT_DECODE_PROFILE_BLAKE3;
constexpr bool kFusedWkvBfloat16ReadoutBoundary =
    RWKV7_FUSED_WKV_BF16_READOUT_BOUNDARY != 0;
constexpr bool kFusedWkvFp32RecurrenceVectors = RWKV7_FUSED_WKV_FP32_RECURRENCE_VECTORS != 0;
constexpr std::string_view kSelfTestArgument = "--self-test";
constexpr std::string_view kHardwareCounterSmokeArgument =
    "--hardware-counter-smoke";
constexpr std::string_view kFastWorkerArgument = "--fast-worker";
constexpr std::string_view kSimulatorSmokeArgument = "--simulator-smoke";
constexpr std::string_view kFusedWkvLayerSmokeArgument =
    "--fused-wkv-layer-smoke";
constexpr std::string_view kFusedWkvProbeArgument = "--fused-wkv-probe";
constexpr std::string_view kFusedWkvAsymmetricProbeArgument =
    "--fused-wkv-asymmetric-probe";
constexpr std::string_view kPersistentDecodeSimulatorSmokeArgument =
    "--persistent-decode-simulator-smoke";
constexpr std::string_view kFastDeviceArgument = "--device";
constexpr std::string_view kFastRequestsArgument = "--requests";
constexpr std::string_view kFastOutputArgument = "--output";
constexpr std::string_view kWkvPolicyArgument = "--wkv-policy";
constexpr std::string_view kFastInitializationReadyBarrierArgument =
    "--initialization-ready-barrier";
constexpr std::string_view kFastWarmupStartBarrierArgument =
    "--warmup-start-barrier";
constexpr std::string_view kFastTimedReadyBarrierArgument =
    "--timed-ready-barrier";
constexpr std::string_view kFastTimedStartBarrierArgument =
    "--timed-start-barrier";
constexpr std::string_view kProductionWkvPolicy = "production";
constexpr std::string_view kFusedCandidateWkvPolicy = "fused-candidate";
constexpr std::string_view kGenericReferenceWkvPolicy = "generic-reference";
constexpr std::string_view kTokensArgument = "--tokens";
constexpr std::string_view kMaxNewArgument = "--max-new";
constexpr std::string_view kEvidenceRootArgument = "--evidence-root";
constexpr std::string_view kEnabledEnvironmentValue = "1";
constexpr std::string_view kDisabledEnvironmentValue = "0";
constexpr std::string_view kFetchQueueDiagnosticsEnvironment =
    "TT_METAL_FETCH_QUEUE_DIAGNOSTICS";
constexpr std::string_view kPrimaryThreadCpuEnvironment =
    "RWKV7_PRIMARY_THREAD_CPU";
constexpr std::string_view kPrimaryThreadFifoPriorityEnvironment =
    "RWKV7_PRIMARY_THREAD_FIFO_PRIORITY";
constexpr std::string_view kFusedWkvKernelRootEnvironment =
    "RWKV7_FUSED_WKV_KERNEL_ROOT";
constexpr std::string_view kPersistentDecodeCandidateEnvironment =
    "RWKV7_PERSISTENT_DECODE_CANDIDATE";
constexpr std::string_view kPersistentDecodeTerminalProbeEnvironment =
    "RWKV7_PERSISTENT_DECODE_TERMINAL_PROBE";
constexpr std::string_view kPersistentDecodeCanaryInterlockEnvironment =
    "RWKV7_PERSISTENT_DECODE_CANARY_INTERLOCK";
constexpr std::string_view kPersistentDecodeCanaryInterlockBlake3Environment =
    "RWKV7_PERSISTENT_DECODE_CANARY_INTERLOCK_BLAKE3";
constexpr std::string_view kPersistentDecodeCanaryBindingBlake3Environment =
    "RWKV7_PERSISTENT_DECODE_CANARY_BINDING_BLAKE3";
constexpr std::string_view kPersistentDecodeCanaryInterlockKind =
    "rwkv7_persistent_decode_canary_interlock";
constexpr std::string_view kPersistentDecodeCanaryHashAlgorithm = "blake3";
constexpr std::size_t kPersistentDecodeCanaryMaximumFileBytes = 65536U;
constexpr std::size_t kPersistentDecodeCanaryTopLevelFieldCount = 9U;
constexpr std::size_t kPersistentDecodeCanaryLimitFieldCount = 3U;
constexpr std::uint32_t kPersistentDecodeCanarySchemaVersion = 1U;
constexpr std::string_view kPersistentDecodeCancelProbe = "cancel";
constexpr std::string_view kPersistentDecodeTimeoutProbe = "timeout";
constexpr std::string_view kPersistentDecodeFailureProbe = "failure";
constexpr std::string_view kPersistentDecodeTelemetryEventKind =
    "rwkv7_persistent_decode_worker";
constexpr std::size_t kBytesPerKibibyte = 1024U;
constexpr std::size_t kKibibytesPerMebibyte = 1024U;
constexpr std::size_t kPersistentTraceRegionMebibytes = 256U;
constexpr std::size_t kPersistentTraceRegionBytes =
    kPersistentTraceRegionMebibytes * kKibibytesPerMebibyte *
    kBytesPerKibibyte;
constexpr bool kFusedWkvPhysicallyAdmitted = true;
constexpr std::uint32_t kAdmittedCardZeroPrimaryThreadCpu = 2U;
constexpr std::uint32_t kAdmittedCardOnePrimaryThreadCpu = 6U;
constexpr int kAdmittedPrimaryThreadFifoPriority = 1;
constexpr std::string_view kLogsDirectoryName = "logs";
constexpr std::string_view kProfilerDirectoryName = "profiler";
constexpr std::string_view kInspectorRelativeDirectory =
    "logs/generated/inspector";
constexpr std::string_view kProfilerCsvRelativePath =
    "profiler/.logs/profile_log_device.csv";
constexpr std::size_t kSelfTestArgumentCount = 2U;
constexpr std::size_t kRuntimeArgumentCount = 7U;
constexpr std::size_t kFastWorkerArgumentCount = 18U;
constexpr std::size_t kSimulatorSmokeArgumentCount = 4U;
constexpr std::size_t kSimulatorOutputNameIndex = 2U;
constexpr std::size_t kSimulatorOutputValueIndex = 3U;
constexpr std::size_t kFastDeviceNameIndex = 2U;
constexpr std::size_t kFastDeviceValueIndex = 3U;
constexpr std::size_t kFastRequestsNameIndex = 4U;
constexpr std::size_t kFastRequestsValueIndex = 5U;
constexpr std::size_t kFastOutputNameIndex = 6U;
constexpr std::size_t kFastOutputValueIndex = 7U;
constexpr std::size_t kWkvPolicyNameIndex = 8U;
constexpr std::size_t kWkvPolicyValueIndex = 9U;
constexpr std::size_t kFastInitializationReadyBarrierNameIndex = 10U;
constexpr std::size_t kFastInitializationReadyBarrierValueIndex = 11U;
constexpr std::size_t kFastWarmupStartBarrierNameIndex = 12U;
constexpr std::size_t kFastWarmupStartBarrierValueIndex = 13U;
constexpr std::size_t kFastTimedReadyBarrierNameIndex = 14U;
constexpr std::size_t kFastTimedReadyBarrierValueIndex = 15U;
constexpr std::size_t kFastTimedStartBarrierNameIndex = 16U;
constexpr std::size_t kFastTimedStartBarrierValueIndex = 17U;
constexpr auto kFastBarrierPollInterval = std::chrono::milliseconds(1);
constexpr auto kFastBarrierTimeout = std::chrono::seconds(120);
constexpr auto kProfilerCsvPollInterval = std::chrono::milliseconds(10);
constexpr auto kProfilerCsvWaitTimeout = std::chrono::seconds(60);
constexpr double kFastBoundaryTimingToleranceSeconds = 1.0e-4;
constexpr std::size_t kFastPerCardSlotCount = 4U;
constexpr std::uint32_t kFastWarmupTokenCount = 4U;
constexpr std::uint32_t kStallProbeWarmupTokenCount = 1U;
constexpr std::uint32_t kFastMaximumGeneratedTokens = 64U;
constexpr std::uint64_t kCapacityTokenShiftBufferCount = 2U;
constexpr std::uint64_t kCapacityFp32Bytes = 4U;
constexpr std::uint64_t kCapacityBf16Bytes = 2U;
constexpr std::size_t kProgramArgumentIndex = 0U;
constexpr std::size_t kModeArgumentIndex = 1U;
constexpr std::size_t kTokensValueIndex = 2U;
constexpr std::size_t kMaxNewNameIndex = 3U;
constexpr std::size_t kMaxNewValueIndex = 4U;
constexpr std::size_t kEvidenceNameIndex = 5U;
constexpr std::size_t kEvidenceValueIndex = 6U;
constexpr std::size_t kFirstIndex = 0U;
constexpr std::size_t kSecondIndex = 1U;
constexpr std::uint32_t kFirstSimulatorSmokeStep = 0U;
constexpr std::uint32_t kSecondSimulatorSmokeStep = 1U;
constexpr std::uint32_t kSelfTestTokenTwo = 2U;
constexpr std::uint32_t kSelfTestRows = 2U;
constexpr std::uint32_t kSelfTestColumns = 3U;
constexpr std::uint16_t kSelfTestBitOne = 1U;
constexpr std::uint16_t kSelfTestBitTwo = 2U;
constexpr std::uint16_t kSelfTestBitThree = 3U;
constexpr std::uint16_t kSelfTestBitFour = 4U;
constexpr std::uint16_t kSelfTestBitFive = 5U;
constexpr std::uint16_t kSelfTestBitSix = 6U;
constexpr std::uint32_t kFusedWkvProbeBatchSize =
    rwkv7::fused_wkv::kMaximumBatchSize;
constexpr float kFusedWkvProbeA = 0.125F;
constexpr float kFusedWkvProbeW = 0.875F;
constexpr float kFusedWkvProbeK = 0.25F;
constexpr float kFusedWkvProbeV = 0.5F;
constexpr float kFusedWkvProbeR = 0.375F;
constexpr float kFusedWkvProbeB = 0.125F;
constexpr float kFusedWkvProbeRk = 1.0F;
constexpr float kFusedWkvProbeGroupNormWeight = 0.5F;
constexpr float kFusedWkvProbeGroupNormBias = 0.25F;
constexpr float kFusedWkvProbeGate = 0.5F;
constexpr float kFusedWkvProbeFirstStateMagnitude = 0.125F;
constexpr float kFusedWkvProbeSecondStateMagnitude = 0.359375F;
constexpr float kFusedWkvProbeNegativeOutput = -1.625F;
constexpr float kFusedWkvProbePositiveOutput = 1.875F;
constexpr std::uint32_t kFusedWkvProbeNegativeWidth = kHeadSize / 2U;
constexpr float kFusedWkvProbeExactTolerance = 0.0F;
constexpr std::size_t kFusedWkvProbeComparisonCount = 4U;
constexpr std::uint32_t kFusedWkvAsymmetricProbeBatchSize = 1U;
constexpr std::size_t kFusedWkvAsymmetricComparisonCount = 2U;
constexpr float kFusedWkvAsymmetricExactTolerance = 0.0F;
struct AsymmetricVectorPattern {
  float base;
  float linear_scale;
  float modular_scale;
};
constexpr AsymmetricVectorPattern kFusedWkvAsymmetricA{0.03125F, 0.00001F,
                                                       0.001F};
constexpr AsymmetricVectorPattern kFusedWkvAsymmetricW{0.75F, -0.00001F,
                                                       0.002F};
constexpr AsymmetricVectorPattern kFusedWkvAsymmetricK{-0.125F, 0.00002F,
                                                       0.003F};
constexpr AsymmetricVectorPattern kFusedWkvAsymmetricV{0.25F, -0.00002F,
                                                       0.004F};
constexpr AsymmetricVectorPattern kFusedWkvAsymmetricR{0.375F, 0.00001F,
                                                       -0.003F};
constexpr AsymmetricVectorPattern kFusedWkvAsymmetricB{-0.0625F, 0.00001F,
                                                       0.002F};
constexpr AsymmetricVectorPattern kFusedWkvAsymmetricRk{0.5F, 0.00001F,
                                                        -0.001F};
constexpr AsymmetricVectorPattern kFusedWkvAsymmetricGroupNormWeight{
    0.75F, -0.00001F, 0.002F};
constexpr AsymmetricVectorPattern kFusedWkvAsymmetricGroupNormBias{
    -0.125F, 0.00001F, 0.001F};
constexpr AsymmetricVectorPattern kFusedWkvAsymmetricGate{0.625F, -0.00001F,
                                                          0.001F};

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

std::uint64_t checked_uint64_add(std::uint64_t left, std::uint64_t right) {
  require(right <= std::numeric_limits<std::uint64_t>::max() - left,
          "card-local memory addition overflows the address space");
  return left + right;
}

std::uint64_t checked_uint64_mul(std::uint64_t left, std::uint64_t right) {
  require(right != 0U &&
              left <= std::numeric_limits<std::uint64_t>::max() / right,
          "card-local memory product overflows the address space");
  return left * right;
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

void validate_primary_thread_cpu(std::uint32_t cpu) {
  require(cpu < static_cast<std::uint32_t>(CPU_SETSIZE),
          "primary thread CPU exceeds CPU_SETSIZE");
}

std::optional<std::uint32_t> parse_primary_thread_cpu_affinity() {
  const char *const value = std::getenv(kPrimaryThreadCpuEnvironment.data());
  if (value == nullptr) {
    return std::nullopt;
  }
  const std::uint32_t cpu = parse_u32(value, "primary thread CPU");
  validate_primary_thread_cpu(cpu);
  return cpu;
}

void validate_primary_thread_fifo_priority(std::uint32_t priority) {
  const int minimum_priority = sched_get_priority_min(SCHED_FIFO);
  const int maximum_priority = sched_get_priority_max(SCHED_FIFO);
  require(minimum_priority >= 0 && maximum_priority >= minimum_priority,
          "failed to read FIFO priority bounds");
  require(priority >= static_cast<std::uint32_t>(minimum_priority) &&
              priority <= static_cast<std::uint32_t>(maximum_priority),
          "primary thread FIFO priority is outside the scheduler bounds");
}

std::optional<int> parse_primary_thread_fifo_priority() {
  const char *const value =
      std::getenv(kPrimaryThreadFifoPriorityEnvironment.data());
  if (value == nullptr) {
    return std::nullopt;
  }
  const std::uint32_t priority =
      parse_positive_u32(value, "primary thread FIFO priority");
  validate_primary_thread_fifo_priority(priority);
  return static_cast<int>(priority);
}

void configure_current_thread_fifo(int priority) {
  const sched_param parameters{.sched_priority = priority};
  require(sched_setscheduler(0, SCHED_FIFO, &parameters) == 0,
          "failed to set primary thread FIFO scheduling");
  require(sched_getscheduler(0) == SCHED_FIFO,
          "primary thread did not enter FIFO scheduling");
}

void restore_current_thread_normal_scheduling() {
  const sched_param parameters{.sched_priority = 0};
  require(sched_setscheduler(0, SCHED_OTHER, &parameters) == 0,
          "failed to restore primary thread normal scheduling");
  require(sched_getscheduler(0) == SCHED_OTHER,
          "primary thread did not restore normal scheduling");
}

void pin_current_thread(std::uint32_t cpu) {
  cpu_set_t cpu_set;
  CPU_ZERO(&cpu_set);
  CPU_SET(static_cast<int>(cpu), &cpu_set);
  require(sched_setaffinity(0, sizeof(cpu_set), &cpu_set) == 0,
          "failed to set primary thread CPU affinity");
  require(sched_getcpu() == static_cast<int>(cpu),
          "primary thread did not move to its configured CPU");
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
      : DeviceContext(
            std::vector<int>{kPhysicalDeviceZero, kPhysicalDeviceOne}) {
    require(expected_device_count_ == kExpectedDeviceCount,
            "evidence context must open both physical cards");
  }

  explicit DeviceContext(int physical_device_id)
      : DeviceContext(physical_device_id, DEFAULT_TRACE_REGION_SIZE) {}

  DeviceContext(int physical_device_id, std::size_t trace_region_size)
      : DeviceContext(std::vector<int>{physical_device_id}, trace_region_size) {
    require(physical_device_id == kPhysicalDeviceZero ||
                physical_device_id == kPhysicalDeviceOne,
            "fast worker device must be physical device 0 or 1");
  }

  explicit DeviceContext(std::vector<int> physical_device_ids)
      : DeviceContext(std::move(physical_device_ids),
                      DEFAULT_TRACE_REGION_SIZE) {}

  DeviceContext(std::vector<int> physical_device_ids,
                std::size_t trace_region_size)
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
        DEFAULT_L1_SMALL_SIZE, trace_region_size, 1U,
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

  void flush_profiler(std::string stage,
                      bool close_device_after_read = false) {
    if (!profiler_flush_enabled_) {
      return;
    }
    require(mesh_ != nullptr, "cannot flush profiler without an open mesh");
    // The mesh-level read drains the profiler thread pool (wait_for_thread_pool)
    // before returning, so the processed device CSV exists synchronously. A
    // per-device detail::ReadDeviceProfilerResults only enqueues the
    // processor to that pool and returns before the file is written.
    tt::tt_metal::ReadMeshDeviceProfilerResults(*mesh_);
    if (close_device_after_read) {
      ttnn::distributed::close_mesh_device(mesh_);
      mesh_.reset();
    }
    const auto deadline =
        std::chrono::steady_clock::now() + kProfilerCsvWaitTimeout;
    std::uint64_t end_byte = profiler_cursor_;
    while (end_byte <= profiler_cursor_ &&
           std::chrono::steady_clock::now() < deadline) {
      if (std::filesystem::exists(profiler_csv_path_)) {
        end_byte = std::filesystem::file_size(profiler_csv_path_);
      }
      if (end_byte <= profiler_cursor_) {
        std::this_thread::sleep_for(kProfilerCsvPollInterval);
      }
    }
    require(end_byte > profiler_cursor_,
            "device profiler flush appended no raw CSV evidence before its bounded deadline");
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
                                 std::uint32_t batch_size,
                                 std::size_t max_slot_count) {
  require(batch_size > 0U &&
              static_cast<std::size_t>(batch_size) <= max_slot_count,
          "fast state batch size is outside the admitted card-local capacity");
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

void zero_tensor_in_place(DeviceContext &context, Tensor &tensor,
                          std::string_view operation) {
  const Tensor zeroed = ttnn::zeros_like(
      tensor, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
      tensor);
  tensor = context.checked(zeroed, operation);
}

void zero_layer_state_in_place(DeviceContext &context, LayerState &layer) {
  zero_tensor_in_place(context, layer.attention_previous,
                       "persistent-zero-attention-state");
  zero_tensor_in_place(context, layer.ffn_previous,
                       "persistent-zero-ffn-state");
  zero_tensor_in_place(context, layer.matrix,
                       "persistent-zero-recurrent-state");
}

void zero_persistent_model_state(DeviceContext &context, ModelState &state) {
  require(state.layers.size() == kLayerCount,
          "persistent decode cleanup has the wrong layer count");
  for (LayerState &layer : state.layers) {
    zero_layer_state_in_place(context, layer);
  }
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

enum class WkvExecutionPolicy {
  Production,
  ProductionPhysicallyAdmitted,
  RequireFusedCandidate,
  RequireGenericReference,
};

std::string_view wkv_execution_policy_name(WkvExecutionPolicy policy) {
  switch (policy) {
  case WkvExecutionPolicy::Production:
  case WkvExecutionPolicy::ProductionPhysicallyAdmitted:
    return kProductionWkvPolicy;
  case WkvExecutionPolicy::RequireFusedCandidate:
    return kFusedCandidateWkvPolicy;
  case WkvExecutionPolicy::RequireGenericReference:
    return kGenericReferenceWkvPolicy;
  }
  fail("unsupported WKV execution policy");
}

WkvExecutionPolicy parse_wkv_execution_policy(std::string_view policy) {
  if (policy == kProductionWkvPolicy) {
    return WkvExecutionPolicy::Production;
  }
  if (policy == kFusedCandidateWkvPolicy) {
    return WkvExecutionPolicy::RequireFusedCandidate;
  }
  if (policy == kGenericReferenceWkvPolicy) {
    return WkvExecutionPolicy::RequireGenericReference;
  }
  fail("WKV execution policy must be production, fused-candidate, or generic-reference");
}

bool production_scheduling_physically_admitted(
    int physical_device_id,
    const std::optional<std::uint32_t> &primary_thread_cpu_affinity,
    const std::optional<int> &primary_thread_fifo_priority) {
  if (!kFusedWkvPhysicallyAdmitted ||
      !primary_thread_fifo_priority.has_value() ||
      *primary_thread_fifo_priority != kAdmittedPrimaryThreadFifoPriority ||
      !primary_thread_cpu_affinity.has_value()) {
    return false;
  }
  if (physical_device_id == kPhysicalDeviceZero) {
    return *primary_thread_cpu_affinity ==
           kAdmittedCardZeroPrimaryThreadCpu;
  }
  if (physical_device_id == kPhysicalDeviceOne) {
    return *primary_thread_cpu_affinity ==
           kAdmittedCardOnePrimaryThreadCpu;
  }
  return false;
}

WkvExecutionPolicy effective_wkv_execution_policy(
    WkvExecutionPolicy requested_policy, bool production_physically_admitted) {
  if (requested_policy == WkvExecutionPolicy::Production &&
      production_physically_admitted) {
    return WkvExecutionPolicy::ProductionPhysicallyAdmitted;
  }
  return requested_policy;
}

bool execution_policy_is_production_physically_admitted(
    WkvExecutionPolicy policy) {
  return policy == WkvExecutionPolicy::ProductionPhysicallyAdmitted;
}

struct WkvOutput {
  Tensor state;
  Tensor output_for_hidden;
  Tensor r_rows;
  Tensor k_rows;
  Tensor v_rows;
  bool correction_complete;
  std::optional<Tensor> state_dot_a;
  std::optional<Tensor> decay;
  std::optional<Tensor> rank_update;
  std::optional<Tensor> outer_update;
  std::optional<Tensor> first_add;
};

WkvOutput generic_wkv_step(DeviceContext &context, const Tensor &state,
                           const Tensor &a_rows, const Tensor &b_rows,
                           const Tensor &w_rows, const Tensor &v_rows,
                           const Tensor &k_rows, const Tensor &r_rows) {
  const bool fp32_state = state.dtype() == ttnn::DataType::FLOAT32;
  const auto source_to_state = [&](const Tensor &rows) {
    return fp32_state
               ? op_typecast(context, rows, ttnn::DataType::FLOAT32,
                             "typecast-recurrence-fp32")
               : rows;
  };
  const Tensor a_state_rows = source_to_state(a_rows);
  const Tensor b_state_rows = source_to_state(b_rows);
  const Tensor w_state_rows = source_to_state(w_rows);
  const Tensor v_state_rows = source_to_state(v_rows);
  const Tensor k_state_rows = source_to_state(k_rows);
  const Tensor state_dot_a =
      op_matmul_transpose_right(context, state, a_state_rows);
  const Tensor rank_update =
      op_matmul(context, state_dot_a, b_state_rows);
  const Tensor decay = op_multiply(context, state, w_state_rows);
  const Tensor outer =
      op_matmul_transpose_left(context, v_state_rows, k_state_rows);
  const Tensor first_add = op_add(context, decay, rank_update);
  const Tensor next_state = op_add(context, first_add, outer);
  const Tensor output_for_hidden =
      fp32_state
          ? op_matmul_transpose_right_bf16(context, next_state, r_rows)
          : op_matmul_transpose_right(context, next_state, r_rows);
  return WkvOutput{
      next_state, output_for_hidden, r_rows, k_rows, v_rows, false, state_dot_a, decay, rank_update, outer, first_add};
}

WkvOutput wkv_step(DeviceContext &context, const LayerWeights &weights,
                   const Tensor &state, const TimePreparation &preparation,
                   WkvExecutionPolicy execution_policy) {
  const std::uint32_t batch_size = state.logical_shape()[0];
  const auto source_row = [&](const Tensor &tensor) {
    return op_reshape(context, tensor,
                      {batch_size, kHeadCount, 1U, kHeadSize});
  };
  const Tensor a_rows = source_row(preparation.a);
  const Tensor w_rows = source_row(preparation.w);
  const Tensor k_rows = source_row(preparation.k);
  const Tensor v_rows = source_row(preparation.v);
  const Tensor r_rows = source_row(preparation.r);
  const Tensor b_rows = source_row(preparation.b);
  const Tensor gate_rows = source_row(preparation.gate);
  const rwkv7::fused_wkv::Inputs fused_inputs{
      a_rows,
      w_rows,
      k_rows,
      v_rows,
      r_rows,
      b_rows,
      weights.at("r_k"),
      weights.at("group_norm_weight"),
      weights.at("group_norm_bias"),
      gate_rows,
      state,
  };
  const bool production_physically_admitted =
      execution_policy_is_production_physically_admitted(execution_policy);
  const rwkv7::fused_wkv::DispatchDecision decision =
      rwkv7::fused_wkv::select_dispatch(fused_inputs,
                                        production_physically_admitted);
  const auto run_generic_reference = [&]() {
    return generic_wkv_step(context, state, a_rows, b_rows, w_rows, v_rows,
                            k_rows, r_rows);
  };
  const auto run_fused_candidate = [&]() {
    const char *const kernel_root =
        std::getenv(kFusedWkvKernelRootEnvironment.data());
    require(kernel_root != nullptr,
            "fused WKV candidate has no kernel source root");
    rwkv7::fused_wkv::Outputs fused =
        rwkv7::fused_wkv::run(context.mesh(), fused_inputs, kernel_root);
    Tensor fused_state =
        context.checked(std::move(fused.next_state), "fused-wkv-state");
    Tensor fused_output =
        context.checked(std::move(fused.output), "fused-wkv-output");
    return WkvOutput{
        std::move(fused_state),
        std::move(fused_output),
        r_rows,
        k_rows,
        v_rows,
        fused.correction_complete,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt};
  };
  if (execution_policy == WkvExecutionPolicy::RequireFusedCandidate) {
    require(decision.capability_supported,
            std::string("fused WKV layer smoke rejected candidate: ") +
                decision.reason);
    return run_fused_candidate();
  }
  if (execution_policy == WkvExecutionPolicy::RequireGenericReference) {
    return run_generic_reference();
  }
  if (decision.use_fused_path) {
    return run_fused_candidate();
  }
  return run_generic_reference();
}

Tensor apply_attention_correction(
    DeviceContext& context,
    const Tensor& rows,
    const Tensor& r_rows,
    const Tensor& k_rows,
    const Tensor& v_rows,
    const Tensor& r_k,
    const Tensor& group_norm_weight,
    const Tensor& group_norm_bias,
    const Tensor& gate_rows) {
    const Tensor normalized_rows = op_layer_norm(context, rows, kGroupNormEpsilon);
    const Tensor scaled_rows =
        op_add(context, op_multiply(context, normalized_rows, group_norm_weight), group_norm_bias);
    const Tensor scalar = op_sum_last(context, op_multiply(context, op_multiply(context, r_rows, k_rows), r_k));
    return op_multiply(context, op_add(context, scaled_rows, op_multiply(context, scalar, v_rows)), gate_rows);
}

Tensor corrected_attention_rows(DeviceContext &context,
                                const LayerWeights &weights,
                                const TimePreparation &preparation,
                                const WkvOutput &wkv) {
  const std::uint32_t batch_size = wkv.output_for_hidden.logical_shape()[0];
  const Tensor rows = op_reshape(
      context, wkv.output_for_hidden,
      {batch_size, kHeadCount, 1U, kHeadSize});
  if (wkv.correction_complete) {
    return rows;
  }
  return apply_attention_correction(
      context,
      rows,
      wkv.r_rows,
      wkv.k_rows,
      wkv.v_rows,
      weights.at("r_k"),
      weights.at("group_norm_weight"),
      weights.at("group_norm_bias"),
      op_reshape(context, preparation.gate, {batch_size, kHeadCount, 1U, kHeadSize}));
}

Tensor finish_attention(DeviceContext &context, const LayerWeights &weights,
                        const TimePreparation &preparation,
                        const WkvOutput &wkv) {
  const std::uint32_t batch_size = wkv.output_for_hidden.logical_shape()[0];
  const Tensor corrected_rows =
      corrected_attention_rows(context, weights, preparation, wkv);
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
                     std::string_view stage_prefix,
                     WkvExecutionPolicy wkv_execution_policy) {
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
  const WkvOutput wkv =
      wkv_step(context, weights, state.matrix, preparation,
               wkv_execution_policy);
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

struct FastStepBoundaryTiming {
  bool selection_requested = false;
  double model_submission_seconds = 0.0;
  std::vector<double> layer_submission_seconds;
  double blocking_readback_seconds = 0.0;
  double host_materialization_seconds = 0.0;
  double readback_validation_seconds = 0.0;
  double total_seconds = 0.0;
  double model_submission_thread_cpu_seconds = 0.0;
  std::vector<double> layer_submission_thread_cpu_seconds;
  double model_submission_process_cpu_seconds = 0.0;
  std::vector<double> layer_submission_process_cpu_seconds;
};

struct FetchQueueDiagnostics {
  std::uint64_t reserve_calls;
  std::uint64_t wait_calls;
  std::uint64_t poll_iterations;
  std::uint64_t total_wait_nanoseconds;
  std::uint64_t maximum_wait_nanoseconds;
};

void reset_fetch_queue_diagnostics() {
#ifdef RWKV7_FETCH_QUEUE_DIAGNOSTICS_AVAILABLE
  tt::tt_metal::experimental::reset_fetch_queue_diagnostics();
#else
  throw std::runtime_error(
      "fetch queue diagnostics are unavailable in the production runtime");
#endif
}

FetchQueueDiagnostics read_fetch_queue_diagnostics() {
#ifdef RWKV7_FETCH_QUEUE_DIAGNOSTICS_AVAILABLE
  const auto diagnostics =
      tt::tt_metal::experimental::get_fetch_queue_diagnostics();
  return FetchQueueDiagnostics{
      diagnostics.reserve_calls, diagnostics.wait_calls,
      diagnostics.poll_iterations, diagnostics.total_wait_nanoseconds,
      diagnostics.maximum_wait_nanoseconds};
#else
  throw std::runtime_error(
      "fetch queue diagnostics are unavailable in the production runtime");
#endif
}

void validate_fetch_queue_diagnostics(
    const FetchQueueDiagnostics &diagnostics) {
  require(diagnostics.reserve_calls > 0U,
          "fetch queue diagnostics contain no reserve calls");
  require(diagnostics.wait_calls <=
              diagnostics.reserve_calls *
                  kMaximumFetchQueueWaitChecksPerReserve,
          "fetch queue diagnostics contain too many wait calls");
  require(diagnostics.poll_iterations >= diagnostics.wait_calls,
          "fetch queue diagnostics contain too few poll iterations");
  require(diagnostics.total_wait_nanoseconds >=
              diagnostics.maximum_wait_nanoseconds,
          "fetch queue diagnostics maximum exceeds the total wait");
  if (diagnostics.wait_calls == 0U) {
    require(diagnostics.poll_iterations == 0U &&
                diagnostics.total_wait_nanoseconds == 0U &&
                diagnostics.maximum_wait_nanoseconds == 0U,
            "fetch queue diagnostics contain wait data without a wait");
  } else {
    require(diagnostics.poll_iterations > 0U &&
                diagnostics.total_wait_nanoseconds > 0U &&
                diagnostics.maximum_wait_nanoseconds > 0U,
            "fetch queue diagnostics omit data for a recorded wait");
  }
}

#ifdef RWKV7_FETCH_QUEUE_DIAGNOSTICS_AVAILABLE
Json hardware_counter_reading_json(
    const rwkv7::hardware_counter::Reading &reading) {
  return Json{{"value", reading.value},
              {"time_enabled_nanoseconds",
               reading.time_enabled_nanoseconds},
              {"time_running_nanoseconds",
               reading.time_running_nanoseconds}};
}

Json hardware_counter_diagnostics_json(
    const rwkv7::hardware_counter::Diagnostics &diagnostics) {
  return Json{{"scope", "primary_worker_thread_timed_decode"},
              {"cycles", hardware_counter_reading_json(diagnostics.cycles)},
              {"instructions",
               hardware_counter_reading_json(diagnostics.instructions)},
              {"cache_references",
               hardware_counter_reading_json(diagnostics.cache_references)},
              {"cache_misses",
               hardware_counter_reading_json(diagnostics.cache_misses)},
              {"aperf", hardware_counter_reading_json(diagnostics.aperf)},
              {"mperf", hardware_counter_reading_json(diagnostics.mperf)},
              {"task_clock_nanoseconds",
               hardware_counter_reading_json(
                   diagnostics.task_clock_nanoseconds)},
              {"context_switches",
               hardware_counter_reading_json(diagnostics.context_switches)},
              {"cpu_migrations",
               hardware_counter_reading_json(diagnostics.cpu_migrations)}};
}
#endif

double elapsed_seconds_between(std::chrono::steady_clock::time_point begin,
                               std::chrono::steady_clock::time_point end) {
  require(end >= begin, "fast boundary timing range is negative");
  return std::chrono::duration<double>(end - begin).count();
}

double timespec_seconds(const timespec &value) {
  require(value.tv_sec >= 0 && value.tv_nsec >= 0,
          "CPU clock returned a negative time");
  return static_cast<double>(value.tv_sec) +
         static_cast<double>(value.tv_nsec) / kNanosecondsPerSecond;
}

double read_cpu_seconds(clockid_t clock_id) {
  timespec value{};
  require(clock_gettime(clock_id, &value) == 0,
          "failed to read a CPU clock");
  return timespec_seconds(value);
}

double elapsed_cpu_seconds(double begin, double end) {
  require(std::isfinite(begin) && std::isfinite(end) && end >= begin,
          "CPU timing range is not finite and non-negative");
  return end - begin;
}

void validate_fast_step_boundary_timing(
    const FastStepBoundaryTiming &timing) {
  require(timing.layer_submission_seconds.size() == kLayerCount &&
              timing.layer_submission_thread_cpu_seconds.size() ==
                  kLayerCount &&
              timing.layer_submission_process_cpu_seconds.size() ==
                  kLayerCount,
          "fast boundary timing layer count is invalid");
  const std::array scalar_timings = {
      timing.model_submission_seconds,
      timing.blocking_readback_seconds,
      timing.host_materialization_seconds,
      timing.readback_validation_seconds,
      timing.total_seconds,
      timing.model_submission_thread_cpu_seconds,
      timing.model_submission_process_cpu_seconds};
  for (const double seconds : scalar_timings) {
    require(std::isfinite(seconds) && seconds >= 0.0,
            "fast boundary timing is not finite and non-negative");
  }
  double summed_layer_seconds = 0.0;
  for (const double seconds : timing.layer_submission_seconds) {
    require(std::isfinite(seconds) && seconds >= 0.0,
            "fast layer boundary timing is not finite and non-negative");
    summed_layer_seconds += seconds;
  }
  for (const double seconds : timing.layer_submission_thread_cpu_seconds) {
    require(std::isfinite(seconds) && seconds >= 0.0,
            "fast layer thread CPU timing is not finite and non-negative");
  }
  for (const double seconds : timing.layer_submission_process_cpu_seconds) {
    require(std::isfinite(seconds) && seconds >= 0.0,
            "fast layer process CPU timing is not finite and non-negative");
  }
  require(summed_layer_seconds <= timing.model_submission_seconds +
                                      kFastBoundaryTimingToleranceSeconds,
          "fast layer timings exceed model submission time");
  const double classified_seconds =
      timing.model_submission_seconds + timing.blocking_readback_seconds +
      timing.host_materialization_seconds +
      timing.readback_validation_seconds;
  require(std::abs(classified_seconds - timing.total_seconds) <=
              kFastBoundaryTimingToleranceSeconds,
          "fast boundary timings do not cover the full step");
  if (!timing.selection_requested) {
    require(timing.blocking_readback_seconds == 0.0 &&
                timing.host_materialization_seconds == 0.0 &&
                timing.readback_validation_seconds == 0.0,
            "fast non-selection step contains readback timing");
  }
}

std::vector<std::uint32_t>
read_fast_u32(DeviceContext &context, const Tensor &input,
              std::size_t batch_size, std::size_t max_slot_count,
              FastStepBoundaryTiming *boundary_timing) {
  require(batch_size > 0U && batch_size <= max_slot_count,
          "fast UINT32 readback batch size exceeds the admitted card-local capacity");
  context.set_stage("fast-selection/readback-u32");
  const bool record_boundaries = boundary_timing != nullptr;
  const auto readback_begin = record_boundaries
                                  ? std::chrono::steady_clock::now()
                                  : std::chrono::steady_clock::time_point{};
  const Tensor host = ttnn::from_device(input, true);
  const auto readback_end = record_boundaries
                                ? std::chrono::steady_clock::now()
                                : std::chrono::steady_clock::time_point{};
  const std::vector<Tensor> shards =
      ttnn::distributed::get_device_tensors(host);
  require(shards.size() == 1U,
          "fast UINT32 readback did not contain one card shard");
  std::vector<std::uint32_t> values =
      shards[kFirstIndex].to_vector<std::uint32_t>();
  const auto materialization_end =
      record_boundaries ? std::chrono::steady_clock::now()
                        : std::chrono::steady_clock::time_point{};
  require(values.size() == batch_size,
          "fast device argmax did not return one value per batch row");
  for (const std::uint32_t value : values) {
    require(value < kVocabularySize,
            "fast device argmax selected an invalid token ID");
  }
  if (record_boundaries) {
    const auto validation_end = std::chrono::steady_clock::now();
    boundary_timing->blocking_readback_seconds =
        elapsed_seconds_between(readback_begin, readback_end);
    boundary_timing->host_materialization_seconds =
        elapsed_seconds_between(readback_end, materialization_end);
    boundary_timing->readback_validation_seconds =
        elapsed_seconds_between(materialization_end, validation_end);
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
    hidden = execute_layer(context, weights.layers[layer_index], hidden,
                           state.layers[layer_index], value_anchor,
                           stage_prefix, WkvExecutionPolicy::Production);
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

struct FastCardCapacity {
  std::string id;
  std::size_t per_card_slot_count;
  std::uint64_t temporary_tensor_bytes_per_slot;
  std::uint64_t device_budget_bytes;
  std::uint64_t safety_reserve_bytes;
};

struct FastWorkerRequest {
  std::string run_id;
  FastCardCapacity capacity;
  std::vector<FastRequest> requests;
};

struct FastWorkerArguments {
  int physical_device_id;
  std::filesystem::path requests_path;
  std::filesystem::path output_path;
  WkvExecutionPolicy wkv_execution_policy;
  std::filesystem::path initialization_ready_barrier_path;
  std::filesystem::path warmup_start_barrier_path;
  std::filesystem::path timed_ready_barrier_path;
  std::filesystem::path timed_start_barrier_path;
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

bool supported_fast_capacity_id(std::string_view id,
                                std::size_t slot_count) {
  return (id == "B4" && slot_count == 4U) ||
         (id == "B16" && slot_count == 16U) ||
         (id == "B32" && slot_count == 32U) ||
         (id == "B64" && slot_count == 64U);
}

struct FastMemoryPlan {
  std::uint64_t recurrent_state_bytes;
  std::uint64_t token_shift_bytes;
  std::uint64_t temporary_tensor_bytes;
  std::uint64_t safety_reserve_bytes;
  std::uint64_t total_required_bytes;
  std::uint64_t device_budget_bytes;
};

// r[impl tenstorrent.rwkv7_p150x2.scalable_card_local_batching.invalid_memory]
// Pure, checked card-local memory derivation. It must run before any device
// opens. Every product uses checked arithmetic; overflow and budget
// exhaustion fail with the rejected bound.
FastMemoryPlan plan_fast_card_memory(const FastCardCapacity &capacity) {
  const std::uint64_t heads_area =
      static_cast<std::uint64_t>(kHeadCount) *
      static_cast<std::uint64_t>(kHeadSize) *
      static_cast<std::uint64_t>(kHeadSize);
  const std::uint64_t recurrent_bytes =
      checked_uint64_mul(
          checked_uint64_mul(heads_area, static_cast<std::uint64_t>(kLayerCount)),
          kCapacityFp32Bytes);
  const std::uint64_t recurrent_total = checked_uint64_mul(
      recurrent_bytes, static_cast<std::uint64_t>(capacity.per_card_slot_count));
  const std::uint64_t shift_bytes = checked_uint64_mul(
      checked_uint64_mul(checked_uint64_mul(
                             static_cast<std::uint64_t>(kHiddenSize),
                             kCapacityTokenShiftBufferCount),
                         static_cast<std::uint64_t>(kLayerCount)),
      kCapacityBf16Bytes);
  const std::uint64_t shift_total =
      checked_uint64_mul(shift_bytes,
                         static_cast<std::uint64_t>(capacity.per_card_slot_count));
  const std::uint64_t temporary_total = checked_uint64_mul(
      static_cast<std::uint64_t>(capacity.per_card_slot_count),
      capacity.temporary_tensor_bytes_per_slot);
  const std::uint64_t total = checked_uint64_add(
      checked_uint64_add(checked_uint64_add(recurrent_total, shift_total),
                         temporary_total),
      capacity.safety_reserve_bytes);
  require(total <= capacity.device_budget_bytes,
          "admitted card-local capacity exceeds the device memory budget");
  return FastMemoryPlan{recurrent_total, shift_total, temporary_total,
                        capacity.safety_reserve_bytes, total,
                        capacity.device_budget_bytes};
}

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
          "fast worker invocation must place --output third");
  require(arguments[kWkvPolicyNameIndex] == kWkvPolicyArgument,
          "fast worker invocation must place --wkv-policy fourth");
  require(arguments[kFastInitializationReadyBarrierNameIndex] ==
              kFastInitializationReadyBarrierArgument,
          "fast worker invocation must place --initialization-ready-barrier fifth");
  require(arguments[kFastWarmupStartBarrierNameIndex] ==
              kFastWarmupStartBarrierArgument,
          "fast worker invocation must place --warmup-start-barrier sixth");
  require(arguments[kFastTimedReadyBarrierNameIndex] ==
              kFastTimedReadyBarrierArgument,
          "fast worker invocation must place --timed-ready-barrier seventh");
  require(arguments[kFastTimedStartBarrierNameIndex] ==
              kFastTimedStartBarrierArgument,
          "fast worker invocation must place --timed-start-barrier last");
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
  const WkvExecutionPolicy wkv_execution_policy =
      parse_wkv_execution_policy(arguments[kWkvPolicyValueIndex]);
  const std::filesystem::path initialization_ready_barrier_path(
      arguments[kFastInitializationReadyBarrierValueIndex]);
  const std::filesystem::path warmup_start_barrier_path(
      arguments[kFastWarmupStartBarrierValueIndex]);
  const std::filesystem::path timed_ready_barrier_path(
      arguments[kFastTimedReadyBarrierValueIndex]);
  const std::filesystem::path timed_start_barrier_path(
      arguments[kFastTimedStartBarrierValueIndex]);
  const std::array barrier_paths{
      initialization_ready_barrier_path, warmup_start_barrier_path,
      timed_ready_barrier_path, timed_start_barrier_path};
  for (const std::filesystem::path &barrier_path : barrier_paths) {
    require(barrier_path.is_absolute(),
            "fast worker barrier path must be absolute");
    require(!std::filesystem::exists(barrier_path),
            "fast worker barrier path already exists");
    require(barrier_path != output_path,
            "fast worker barrier path must differ from the output path");
    require(std::filesystem::is_directory(barrier_path.parent_path()),
            "fast worker barrier parent is not a directory");
    require(!barrier_path.string().starts_with("/nix/store/"),
            "fast worker barrier must be outside the Nix store");
  }
  for (std::size_t left = 0U; left < barrier_paths.size(); ++left) {
    for (std::size_t right = left + 1U; right < barrier_paths.size();
         ++right) {
      require(barrier_paths[left] != barrier_paths[right],
              "fast worker barrier paths must be distinct");
    }
  }
  return FastWorkerArguments{
      static_cast<int>(device_id), requests_path, output_path,
      wkv_execution_policy, initialization_ready_barrier_path,
      warmup_start_barrier_path, timed_ready_barrier_path,
      timed_start_barrier_path};
}

FastWorkerRequest read_fast_worker_request(
    const std::filesystem::path &path) {
  std::ifstream input(path);
  require(input.good(), "failed to open fast worker request");
  const Json document = Json::parse(input);
  require(document.is_object() && document.size() == 4U,
          "fast worker request must contain exactly four fields");
  require(document.at("schema_version").get<std::uint32_t>() ==
              kSchemaVersion,
          "fast worker request schema mismatch");
  FastWorkerRequest request{document.at("run_id").get<std::string>(),
                            FastCardCapacity{}, {}};
  require(!request.run_id.empty(), "fast worker run ID must not be empty");
  const Json &raw_capacity = document.at("capacity");
  require(raw_capacity.is_object() && raw_capacity.size() == 5U,
          "fast worker capacity must contain exactly five fields");
  request.capacity.id = raw_capacity.at("id").get<std::string>();
  request.capacity.per_card_slot_count =
      raw_capacity.at("per_card_slot_count").get<std::size_t>();
  request.capacity.temporary_tensor_bytes_per_slot =
      raw_capacity.at("temporary_tensor_bytes_per_slot").get<std::uint64_t>();
  request.capacity.device_budget_bytes =
      raw_capacity.at("device_budget_bytes").get<std::uint64_t>();
  request.capacity.safety_reserve_bytes =
      raw_capacity.at("safety_reserve_bytes").get<std::uint64_t>();
  require(supported_fast_capacity_id(request.capacity.id,
                                     request.capacity.per_card_slot_count),
          "fast worker capacity is not a supported grid capacity");
  require(request.capacity.temporary_tensor_bytes_per_slot > 0U &&
              request.capacity.device_budget_bytes > 0U &&
              request.capacity.safety_reserve_bytes > 0U,
          "fast worker capacity memory bounds must be positive");
  // The checked memory plan rejects overflow and budget exhaustion before
  // the worker opens a device. r[verify tenstorrent.rwkv7_p150x2.scalable_card_local_batching.invalid_memory]
  static_cast<void>(plan_fast_card_memory(request.capacity));
  const Json &raw_requests = document.at("requests");
  require(raw_requests.is_array() && !raw_requests.empty(),
          "fast worker request list must not be empty");
  require(raw_requests.size() <= request.capacity.per_card_slot_count,
          "fast worker request list exceeds the admitted card-local capacity");
  std::map<std::string, bool> request_ids;
  std::vector<bool> used_slots(request.capacity.per_card_slot_count, false);
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
    require(current.state_slot < request.capacity.per_card_slot_count,
            "fast request state slot is outside the admitted card-local capacity");
    require(!used_slots[current.state_slot],
            "fast worker received duplicate active state slots");
    used_slots[current.state_slot] = true;
    request.requests.push_back(std::move(current));
  }
  return request;
}

FastBatchStepPlan
plan_fast_batch_step(const std::vector<FastSlot> &slots,
                     const std::vector<bool> &completed,
                     std::size_t max_slot_count) {
  require(!slots.empty() && slots.size() <= max_slot_count,
          "fast batch planner slot count exceeds the admitted card-local capacity");
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
                    bool select_tokens, std::uint64_t step_index,
                    std::size_t max_slot_count,
                    WkvExecutionPolicy wkv_execution_policy,
                    FastStepBoundaryTiming *boundary_timing = nullptr) {
  require(!input_tokens.empty() && input_tokens.size() <= max_slot_count,
          "fast model batch size exceeds the admitted card-local capacity");
  require(state.layers.size() == kLayerCount,
          "fast model state has the wrong layer count");
  const bool record_boundaries = boundary_timing != nullptr;
  const auto step_begin = record_boundaries
                              ? std::chrono::steady_clock::now()
                              : std::chrono::steady_clock::time_point{};
  const double step_thread_cpu_begin =
      record_boundaries ? read_cpu_seconds(CLOCK_THREAD_CPUTIME_ID) : 0.0;
  const double step_process_cpu_begin =
      record_boundaries ? read_cpu_seconds(CLOCK_PROCESS_CPUTIME_ID) : 0.0;
  if (record_boundaries) {
    *boundary_timing = FastStepBoundaryTiming{};
    boundary_timing->selection_requested = select_tokens;
    boundary_timing->layer_submission_seconds.reserve(kLayerCount);
    boundary_timing->layer_submission_thread_cpu_seconds.reserve(kLayerCount);
    boundary_timing->layer_submission_process_cpu_seconds.reserve(kLayerCount);
  }
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
    const auto layer_begin = record_boundaries
                                 ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point{};
    const double layer_thread_cpu_begin =
        record_boundaries ? read_cpu_seconds(CLOCK_THREAD_CPUTIME_ID) : 0.0;
    const double layer_process_cpu_begin =
        record_boundaries ? read_cpu_seconds(CLOCK_PROCESS_CPUTIME_ID) : 0.0;
    const std::string stage_prefix =
        step_name + "/layer-" + std::to_string(layer_index);
    hidden = execute_layer(context, weights.layers[layer_index], hidden,
                           state.layers[layer_index], value_anchor,
                           stage_prefix, wkv_execution_policy);
    if (record_boundaries) {
      const auto layer_end = std::chrono::steady_clock::now();
      const double layer_thread_cpu_end =
          read_cpu_seconds(CLOCK_THREAD_CPUTIME_ID);
      const double layer_process_cpu_end =
          read_cpu_seconds(CLOCK_PROCESS_CPUTIME_ID);
      boundary_timing->layer_submission_seconds.push_back(
          elapsed_seconds_between(layer_begin, layer_end));
      boundary_timing->layer_submission_thread_cpu_seconds.push_back(
          elapsed_cpu_seconds(layer_thread_cpu_begin, layer_thread_cpu_end));
      boundary_timing->layer_submission_process_cpu_seconds.push_back(
          elapsed_cpu_seconds(layer_process_cpu_begin,
                              layer_process_cpu_end));
    }
  }
  require(value_anchor.has_value(),
          "fast full model did not establish the token-local value anchor");
  if (!select_tokens) {
    if (record_boundaries) {
      const auto step_end = std::chrono::steady_clock::now();
      boundary_timing->model_submission_seconds =
          elapsed_seconds_between(step_begin, step_end);
      boundary_timing->model_submission_thread_cpu_seconds =
          elapsed_cpu_seconds(
              step_thread_cpu_begin,
              read_cpu_seconds(CLOCK_THREAD_CPUTIME_ID));
      boundary_timing->model_submission_process_cpu_seconds =
          elapsed_cpu_seconds(
              step_process_cpu_begin,
              read_cpu_seconds(CLOCK_PROCESS_CPUTIME_ID));
      boundary_timing->total_seconds =
          boundary_timing->model_submission_seconds;
      validate_fast_step_boundary_timing(*boundary_timing);
    }
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
  if (record_boundaries) {
    const auto model_submission_end = std::chrono::steady_clock::now();
    boundary_timing->model_submission_seconds =
        elapsed_seconds_between(step_begin, model_submission_end);
    boundary_timing->model_submission_thread_cpu_seconds =
        elapsed_cpu_seconds(step_thread_cpu_begin,
                            read_cpu_seconds(CLOCK_THREAD_CPUTIME_ID));
    boundary_timing->model_submission_process_cpu_seconds =
        elapsed_cpu_seconds(step_process_cpu_begin,
                            read_cpu_seconds(CLOCK_PROCESS_CPUTIME_ID));
  }
  std::vector<std::uint32_t> values =
      read_fast_u32(context, selected, input_tokens.size(),
                    input_tokens.size(), boundary_timing);
  if (record_boundaries) {
    boundary_timing->total_seconds = elapsed_seconds_between(
        step_begin, std::chrono::steady_clock::now());
    validate_fast_step_boundary_timing(*boundary_timing);
  }
  return values;
}

void submit_persistent_decode_step(
    DeviceContext &context, const ModelWeights &weights, ModelState &state,
    const Tensor &input_token, std::uint32_t batch_size,
    std::uint64_t step_index, WkvExecutionPolicy wkv_execution_policy,
    std::vector<Tensor> &retained_intermediates, Tensor &selected_output) {
  require(batch_size > 0U,
          "persistent decode batch size must be positive");
  require(state.layers.size() == kLayerCount,
          "persistent decode state has the wrong layer count");
  if (step_index == kFirstIndex) {
    require(state.layers[kFirstIndex].matrix.dtype() ==
                ttnn::DataType::FLOAT32,
            "persistent decode recurrent state must remain FP32");
    require(input_token.dtype() == ttnn::DataType::UINT32,
            "persistent decode token input must be UINT32");
    require(input_token.logical_shape()[0] == batch_size,
            "persistent decode token count differs from its batch size");
  }
  const std::string step_name =
      "persistent-decode-step-" + std::to_string(step_index);
  context.set_stage(step_name + "/embedding");
  Tensor hidden = context.checked(
      ttnn::embedding(input_token, weights.embeddings, std::nullopt,
                      ttnn::Layout::TILE),
      "persistent-embedding");
  hidden = op_reshape(context, hidden, {1U, 1U, batch_size, kHiddenSize});
  std::optional<Tensor> value_anchor;
  for (std::uint32_t layer_index = 0U; layer_index < kLayerCount;
       ++layer_index) {
    const std::string stage_prefix =
        step_name + "/layer-" + std::to_string(layer_index);
    hidden = execute_layer(context, weights.layers[layer_index], hidden,
                           state.layers[layer_index], value_anchor,
                           stage_prefix, wkv_execution_policy);
  }
  require(value_anchor.has_value(),
          "persistent decode did not establish its value anchor");
  context.set_stage(step_name + "/final-normalization");
  const Tensor final_hidden =
      op_layer_norm(context, hidden, weights.final_norm_weight,
                    weights.final_norm_bias, kLayerNormEpsilon);
  context.set_stage(step_name + "/language-model-head");
  const Tensor logits = op_matmul(context, final_hidden, weights.head);
  const Tensor logits_row_major =
      context.checked(ttnn::untilize(logits), "persistent-untilize-logits");
  context.set_stage(step_name + "/greedy-selection");
  const Tensor selected_keepdim = context.checked(
      ttnn::argmax(logits_row_major, kLastDimension, true),
      "persistent-argmax");
  selected_output =
      op_reshape(context, selected_keepdim, {batch_size, 1U});
  retained_intermediates.push_back(hidden);
  retained_intermediates.push_back(value_anchor.value());
  retained_intermediates.push_back(final_hidden);
  retained_intermediates.push_back(logits);
  retained_intermediates.push_back(logits_row_major);
  retained_intermediates.push_back(selected_keepdim);
}

class PersistentTraceLease {
public:
  explicit PersistentTraceLease(MeshDevice &device) : device_(&device) {
    trace_id_ =
        ttnn::operations::trace::begin_trace_capture(device_, std::nullopt);
  }

  PersistentTraceLease(const PersistentTraceLease &) = delete;
  PersistentTraceLease &operator=(const PersistentTraceLease &) = delete;

  ~PersistentTraceLease() {
    try {
      release();
    } catch (...) {
    }
  }

  void finish_capture() {
    require(trace_id_.has_value(),
            "persistent decode trace capture did not start");
    require(!capture_finished_,
            "persistent decode trace capture finished more than once");
    ttnn::operations::trace::end_trace_capture(device_, *trace_id_,
                                               std::nullopt);
    capture_finished_ = true;
  }

  void execute() {
    require(trace_id_.has_value() && capture_finished_,
            "persistent decode trace executed before capture completed");
    ttnn::operations::trace::execute_trace(device_, *trace_id_, std::nullopt,
                                           true);
  }

  void release() {
    if (!trace_id_.has_value()) {
      return;
    }
    require(capture_finished_,
            "persistent decode released an unfinished trace");
    ttnn::operations::trace::release_trace(device_, *trace_id_);
    trace_id_.reset();
  }

private:
  MeshDevice *device_;
  std::optional<ttnn::MeshTraceId> trace_id_;
  bool capture_finished_ = false;
};

struct PreparedPersistentTrace {
  std::unique_ptr<PersistentTraceLease> trace;
  Tensor initial_token;
  ModelState initial_state;
  ModelState captured_state;
  std::vector<Tensor> selected_outputs;
  std::vector<Tensor> retained_intermediates;
  std::size_t batch_size;
};

struct PersistentTraceResult {
  std::vector<std::vector<std::uint32_t>> selected_tokens_by_step;
  ModelState final_state;
  double execution_seconds;
  std::chrono::steady_clock::time_point execution_end;
  std::chrono::system_clock::time_point execution_end_wall_clock;
  std::uint64_t host_trace_submission_count;
};

PreparedPersistentTrace prepare_persistent_decode_trace(
    DeviceContext &context, const ModelWeights &weights,
    const ModelState &initial_state,
    const std::vector<std::uint32_t> &initial_tokens,
    std::uint32_t step_count, WkvExecutionPolicy wkv_execution_policy) {
  require(!initial_tokens.empty(),
          "persistent decode trace needs an initial token");
  require(step_count > 0U,
          "persistent decode trace step count must be positive");
  const auto batch_size =
      static_cast<std::uint32_t>(initial_tokens.size());
  require(initial_state.layers[kFirstIndex].matrix.logical_shape()[0] ==
              batch_size,
          "persistent decode initial state differs from its batch size");

  const Tensor initial_token =
      context.uint32_tensor(initial_tokens, {batch_size, 1U});
  Tensor current_token = initial_token;
  ModelState retained_initial_state = initial_state;
  ModelState captured_state = retained_initial_state;
  std::vector<Tensor> selected_outputs;
  selected_outputs.reserve(step_count);
  constexpr std::size_t kRetainedIntermediatesPerStep = 6U;
  std::vector<Tensor> retained_intermediates;
  retained_intermediates.reserve(step_count * kRetainedIntermediatesPerStep);
  auto trace = std::make_unique<PersistentTraceLease>(context.mesh());
  for (std::uint32_t step = 0U; step < step_count; ++step) {
    selected_outputs.emplace_back();
    submit_persistent_decode_step(
        context, weights, captured_state, current_token, batch_size, step,
        wkv_execution_policy, retained_intermediates,
        selected_outputs.back());
    current_token = selected_outputs.back();
  }
  trace->finish_capture();
  return PreparedPersistentTrace{
      std::move(trace), initial_token, std::move(retained_initial_state),
      std::move(captured_state), std::move(selected_outputs),
      std::move(retained_intermediates), initial_tokens.size()};
}

PersistentTraceResult execute_persistent_decode_trace(
    DeviceContext &context, PreparedPersistentTrace &prepared) {
  require(prepared.trace != nullptr,
          "persistent decode lost its prepared trace");
  require(!prepared.selected_outputs.empty() && prepared.batch_size > 0U,
          "persistent decode prepared an empty trace");
  const auto execute_begin = std::chrono::steady_clock::now();
  prepared.trace->execute();
  const auto execute_end = std::chrono::steady_clock::now();
  const auto execute_end_wall_clock = std::chrono::system_clock::now();
  prepared.trace->release();
  const double execution_seconds =
      elapsed_seconds_between(execute_begin, execute_end);
  require(std::isfinite(execution_seconds) && execution_seconds > 0.0,
          "persistent decode trace time is not finite and positive");

  std::vector<std::vector<std::uint32_t>> selected_tokens_by_step;
  selected_tokens_by_step.reserve(prepared.selected_outputs.size());
  for (const Tensor &selected : prepared.selected_outputs) {
    selected_tokens_by_step.push_back(read_fast_u32(
        context, selected, prepared.batch_size, prepared.batch_size,
        nullptr));
  }
  constexpr std::uint64_t kSingleTraceSubmission = 1U;
  return PersistentTraceResult{
      std::move(selected_tokens_by_step),
      std::move(prepared.captured_state), execution_seconds, execute_end,
      execute_end_wall_clock, kSingleTraceSubmission};
}

void write_json(const std::filesystem::path &path, const Json &value) {
  std::ofstream output(path, std::ios::trunc);
  require(output.good(), "failed to create runtime receipt");
  output << value.dump(2) << '\n';
  output.flush();
  require(output.good(), "failed to write runtime receipt");
}

void signal_fast_barrier(const std::filesystem::path &path,
                         std::string_view name) {
  std::error_code error;
  const bool created = std::filesystem::create_directory(path, error);
  require(created && !error,
          "failed to signal fast worker " + std::string(name) +
              " barrier: " + error.message());
}

void wait_for_fast_barrier(const std::filesystem::path &path,
                           std::string_view name) {
  const auto deadline = std::chrono::steady_clock::now() + kFastBarrierTimeout;
  while (true) {
    std::error_code error;
    const bool exists = std::filesystem::exists(path, error);
    require(!error,
            "failed to inspect fast worker " + std::string(name) +
                " barrier: " + error.message());
    if (exists) {
      require(std::filesystem::is_directory(path),
              "fast worker " + std::string(name) +
                  " barrier is not a directory");
      return;
    }
    require(std::chrono::steady_clock::now() < deadline,
            "fast worker timed out waiting for " + std::string(name) +
                " barrier");
    std::this_thread::sleep_for(kFastBarrierPollInterval);
  }
}

std::uint64_t monotonic_nanoseconds(
    std::chrono::steady_clock::time_point time_point) {
  const auto count = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         time_point.time_since_epoch())
                         .count();
  require(count >= 0, "monotonic clock returned a negative timestamp");
  return static_cast<std::uint64_t>(count);
}

std::uint64_t unix_nanoseconds(
    std::chrono::system_clock::time_point time_point) {
  const auto count = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         time_point.time_since_epoch())
                         .count();
  require(count >= 0, "wall clock returned a negative Unix timestamp");
  return static_cast<std::uint64_t>(count);
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

std::filesystem::path parse_new_output(
    const std::vector<std::string> &arguments, std::string_view mode,
    std::string_view invocation_name) {
  require(arguments.size() == kSimulatorSmokeArgumentCount,
          std::string(invocation_name) +
              " invocation has the wrong argument count");
  require(arguments[kModeArgumentIndex] == mode,
          std::string(invocation_name) + " invocation has the wrong mode");
  require(arguments[kSimulatorOutputNameIndex] == kFastOutputArgument,
          std::string(invocation_name) +
              " invocation must place --output last");
  const std::filesystem::path output(arguments[kSimulatorOutputValueIndex]);
  require(output.is_absolute() &&
              !output.string().starts_with("/nix/store/") &&
              !std::filesystem::exists(output),
          std::string(invocation_name) +
              " output must be a new absolute non-store path");
  return output;
}

std::filesystem::path parse_simulator_smoke_output(
    const std::vector<std::string> &arguments) {
  return parse_new_output(arguments, kSimulatorSmokeArgument,
                          "simulator smoke");
}

std::filesystem::path parse_fused_wkv_layer_smoke_output(
    const std::vector<std::string> &arguments) {
  return parse_new_output(arguments, kFusedWkvLayerSmokeArgument,
                          "fused WKV layer smoke");
}

std::filesystem::path parse_fused_wkv_probe_output(
    const std::vector<std::string> &arguments) {
  return parse_new_output(arguments, kFusedWkvProbeArgument,
                          "fused WKV probe");
}

std::filesystem::path parse_fused_wkv_asymmetric_probe_output(
    const std::vector<std::string> &arguments) {
  return parse_new_output(arguments, kFusedWkvAsymmetricProbeArgument,
                          "fused WKV asymmetric probe");
}

std::filesystem::path parse_persistent_decode_simulator_smoke_output(
    const std::vector<std::string> &arguments) {
  return parse_new_output(arguments,
                          kPersistentDecodeSimulatorSmokeArgument,
                          "persistent decode simulator smoke");
}

template <typename Value>
std::vector<Value> read_fused_wkv_probe_values(DeviceContext &context,
                                                const Tensor &input,
                                                bool untilize);

void run_simulator_smoke(const std::filesystem::path &output_path,
                         WkvExecutionPolicy wkv_execution_policy) {
  static_assert(kMinimumGeneratedTokens == kSecondSimulatorSmokeStep + 1U);
  select_single_card_device(kPhysicalDeviceZero);
  const MappedFile checkpoint_file(RWKV7_PINNED_CHECKPOINT_PATH);
  const CheckpointIndex checkpoint(checkpoint_file);
  DeviceContext context(kPhysicalDeviceZero);
  context.set_profiler_flush_enabled(false);
  const ModelWeights weights = load_model(context, checkpoint);
  ModelState state = zero_fast_model_state(context, 1U, 1U);
  require(state.layers[kFirstIndex].matrix.dtype() == ttnn::DataType::FLOAT32,
          "simulator smoke recurrent matrix is not FP32");

  const auto execute_smoke_step = [&](std::uint32_t step,
                                      std::string_view carry_mode,
                                      LayerState &layer_state) {
    context.set_stage("simulator-smoke/embedding");
    const Tensor token = context.uint32_tensor({kGenerationEosToken}, {1U, 1U});
    Tensor hidden = context.checked(
        ttnn::embedding(token, weights.embeddings, std::nullopt,
                        ttnn::Layout::TILE),
        "embedding");
    hidden = op_reshape(context, hidden, {1U, 1U, 1U, kHiddenSize});
    std::optional<Tensor> value_anchor;
    Tensor layer_output = execute_layer(
        context, weights.layers[kFirstIndex], hidden, layer_state,
        value_anchor,
        "simulator-smoke/layer-0/" + std::string(carry_mode) + "/step-" +
            std::to_string(step),
        wkv_execution_policy);
    require(value_anchor.has_value(),
            "simulator smoke layer did not establish its value anchor");
    require(layer_output.dtype() == ttnn::DataType::BFLOAT16,
            "simulator smoke layer output is not BF16");
    return layer_output;
  };

  static_cast<void>(execute_smoke_step(
      kFirstSimulatorSmokeStep, "carried", state.layers[kFirstIndex]));
  const std::vector<float> first_state =
      read_fused_wkv_probe_values<float>(
          context, state.layers[kFirstIndex].matrix, false);
  LayerState reset_control_state = state.layers[kFirstIndex];
  reset_control_state.matrix = context.zeros_fp32(
      {1U, kHeadCount, kHeadSize, kHeadSize});
  static_cast<void>(execute_smoke_step(
      kSecondSimulatorSmokeStep, "carried", state.layers[kFirstIndex]));
  const std::vector<float> second_state =
      read_fused_wkv_probe_values<float>(
          context, state.layers[kFirstIndex].matrix, false);
  static_cast<void>(execute_smoke_step(
      kSecondSimulatorSmokeStep, "reset-control", reset_control_state));
  const std::vector<float> reset_control_second_state =
      read_fused_wkv_probe_values<float>(context, reset_control_state.matrix,
                                         false);
  const bool first_state_finite =
      std::all_of(first_state.begin(), first_state.end(),
                  [](float value) { return std::isfinite(value); });
  const bool second_state_finite =
      std::all_of(second_state.begin(), second_state.end(),
                  [](float value) { return std::isfinite(value); });
  const bool reset_control_state_finite =
      std::all_of(reset_control_second_state.begin(),
                  reset_control_second_state.end(),
                  [](float value) { return std::isfinite(value); });
  const bool first_state_nonzero =
      std::any_of(first_state.begin(), first_state.end(),
                  [](float value) { return value != 0.0F; });
  const bool state_changed_after_carry =
      first_state.size() == second_state.size() &&
      !std::equal(first_state.begin(), first_state.end(), second_state.begin());
  const bool carried_state_differs_from_reset_control =
      second_state.size() == reset_control_second_state.size() &&
      !std::equal(second_state.begin(), second_state.end(),
                  reset_control_second_state.begin());
  require(first_state_finite && second_state_finite &&
              reset_control_state_finite,
          "simulator smoke produced a non-finite recurrent state");
  require(first_state_nonzero,
          "simulator smoke first recurrent state remained zero");
  require(state_changed_after_carry,
          "simulator smoke second recurrent state did not change");
  require(carried_state_differs_from_reset_control,
          "simulator smoke carried state matches its reset-matrix control");
  require(!context.operation_counts().empty(),
          "simulator smoke executed no TT-NN operations");

  const bool fused_candidate_requested =
      wkv_execution_policy == WkvExecutionPolicy::RequireFusedCandidate;
  const Json receipt{
      {"schema_version", kSchemaVersion},
      {"outcome",
       fused_candidate_requested
           ? "simulated_single_card_fused_wkv_layer_state_carry"
           : "simulated_single_card_layer_state_carry"},
      {"simulated", true},
      {"physical_hardware", false},
      {"architecture", "blackhole"},
      {"mesh_shape", {kMeshRows, kMeshRows}},
      {"selection_ordinal", kPhysicalDeviceZero},
      {"layer_count_executed", 1U},
      {"recurrent_steps", kMinimumGeneratedTokens},
      {"recurrent_matrix_residency", "device_fp32"},
      {"first_state_finite", first_state_finite},
      {"second_state_finite", second_state_finite},
      {"reset_control_state_finite", reset_control_state_finite},
      {"first_state_nonzero", first_state_nonzero},
      {"state_changed_after_carry", state_changed_after_carry},
      {"reset_control_replays", 1U},
      {"carried_state_differs_from_reset_control",
       carried_state_differs_from_reset_control},
      {"fused_wkv_candidate_requested", fused_candidate_requested},
      {"production_physically_admitted", false},
      {"output_projection_outside_fused_wkv", true},
      {"performance_claim", false},
      {"operation_counts", map_to_json(context.operation_counts())}};
  write_json(output_path, receipt);
  std::cout << output_path << '\n';
}

void run_persistent_decode_simulator_smoke(
    const std::filesystem::path &output_path) {
  select_single_card_device(kPhysicalDeviceZero);
  const MappedFile checkpoint_file(RWKV7_PINNED_CHECKPOINT_PATH);
  const CheckpointIndex checkpoint(checkpoint_file);
  DeviceContext context(kPhysicalDeviceZero, kPersistentTraceRegionBytes);
  context.set_profiler_flush_enabled(false);
  const ModelWeights weights = load_model(context, checkpoint);
  const auto make_hidden = [&context, &weights]() {
    const Tensor token =
        context.uint32_tensor({kGenerationEosToken}, {1U, 1U});
    Tensor hidden = context.checked(
        ttnn::embedding(token, weights.embeddings, std::nullopt,
                        ttnn::Layout::TILE),
        "persistent-simulator-embedding");
    return op_reshape(context, hidden, {1U, 1U, 1U, kHiddenSize});
  };
  const auto execute_layer_step =
      [&context, &weights](const Tensor &input, LayerState &layer_state,
                           std::string_view stage) {
        std::optional<Tensor> value_anchor;
        Tensor output = execute_layer(
            context, weights.layers[kFirstIndex], input, layer_state,
            value_anchor, stage,
            WkvExecutionPolicy::RequireFusedCandidate);
        require(value_anchor.has_value(),
                "persistent simulator layer omitted its value anchor");
        require(output.dtype() == ttnn::DataType::BFLOAT16,
                "persistent simulator layer output is not BF16");
        return output;
      };

  LayerState warmup_state =
      zero_fast_model_state(context, 1U, 1U).layers[kFirstIndex];
  Tensor warmup_output = execute_layer_step(
      make_hidden(), warmup_state, "persistent-simulator-warmup/step-0");
  static_cast<void>(execute_layer_step(
      warmup_output, warmup_state, "persistent-simulator-warmup/step-1"));
  const std::size_t cache_after_warmup = context.program_cache_entries();
  require(cache_after_warmup > 0U,
          "persistent simulator warmup did not populate the program cache");

  const Tensor initial_hidden = make_hidden();
  const LayerState initial_state =
      zero_fast_model_state(context, 1U, 1U).layers[kFirstIndex];
  LayerState captured_state = initial_state;
  PersistentTraceLease trace(context.mesh());
  std::vector<Tensor> traced_outputs;
  traced_outputs.reserve(kMaximumGeneratedTokens);
  Tensor traced_hidden = initial_hidden;
  for (std::uint32_t step = 0U; step < kMaximumGeneratedTokens; ++step) {
    traced_hidden = execute_layer_step(
        traced_hidden, captured_state,
        "persistent-simulator-trace/step-" + std::to_string(step));
    traced_outputs.push_back(traced_hidden);
  }
  trace.finish_capture();
  trace.execute();
  trace.release();
  require(context.program_cache_entries() == cache_after_warmup,
          "persistent simulator trace compiled a new program");
  const bool outputs_are_bf16 =
      std::ranges::all_of(traced_outputs, [](const Tensor &output) {
        return output.dtype() == ttnn::DataType::BFLOAT16;
      });
  require(outputs_are_bf16 &&
              captured_state.matrix.dtype() == ttnn::DataType::FLOAT32,
          "persistent simulator trace changed a precision boundary");
  const std::vector<float> final_state =
      read_fused_wkv_probe_values<float>(context, captured_state.matrix,
                                         false);
  const bool state_finite =
      std::ranges::all_of(final_state,
                          [](float value) { return std::isfinite(value); });
  const bool state_nonzero =
      std::ranges::any_of(final_state,
                          [](float value) { return value != 0.0F; });
  require(state_finite && state_nonzero,
          "persistent simulator trace did not carry recurrent state");
  zero_layer_state_in_place(context, captured_state);
  const std::vector<float> zeroed_state =
      read_fused_wkv_probe_values<float>(context, captured_state.matrix,
                                         false);
  const bool state_zeroed =
      std::ranges::all_of(zeroed_state,
                          [](float value) { return value == 0.0F; });
  require(state_zeroed,
          "persistent simulator trace did not zero recurrent state");
  constexpr std::uint64_t kTraceReplayHostSubmissions = 1U;
  std::vector<std::uint32_t> evaluated_windows = {
      kMinimumGeneratedTokens, kLongGeneratedTokenWindow};
  if (kMaximumGeneratedTokens >= kWindowEightGeneratedTokens) {
    evaluated_windows.push_back(kWindowEightGeneratedTokens);
  }
  if (kMaximumGeneratedTokens > kWindowEightGeneratedTokens) {
    evaluated_windows.push_back(kMaximumGeneratedTokens);
  }
  const Json receipt{
      {"schema_version", kSchemaVersion},
      {"outcome", "simulated_persistent_decode_layer_trace"},
      {"simulated", true},
      {"physical_hardware", false},
      {"architecture", "blackhole"},
      {"mesh_shape", {kMeshRows, kMeshRows}},
      {"selection_ordinal", kPhysicalDeviceZero},
      {"dispatch_mode", "fast_trace_required"},
      {"simulator_determinism_claim", false},
      {"layer_count_executed", 1U},
      {"admitted_decode_windows", evaluated_windows},
      {"recurrent_steps", kMaximumGeneratedTokens},
      {"trace_replay_host_submissions", kTraceReplayHostSubmissions},
      {"per_step_host_model_submissions_inside_trace", 0U},
      {"recurrent_matrix_residency", "device_fp32"},
      {"trace_output_dtype", "bf16"},
      {"state_finite", state_finite},
      {"state_nonzero", state_nonzero},
      {"state_zeroed_before_release", state_zeroed},
      {"program_cache_stable", true},
      {"production_physically_admitted", false},
      {"performance_claim", false},
      {"operation_counts", map_to_json(context.operation_counts())}};
  write_json(output_path, receipt);
  std::cout << output_path << '\n';
}

template <typename Value>
std::vector<Value> read_fused_wkv_probe_values(DeviceContext &context,
                                                const Tensor &input,
                                                bool untilize) {
  Tensor readable = input;
  if (untilize) {
    readable = context.checked(ttnn::untilize(input),
                               "untilize-fused-wkv-probe");
  }
  const Tensor host = ttnn::from_device(readable, true);
  const std::vector<Tensor> shards =
      ttnn::distributed::get_device_tensors(host);
  require(shards.size() == 1U,
          "fused WKV probe readback must contain one device shard");
  return shards[kFirstIndex].to_vector<Value>();
}

std::string persistent_model_state_blake3(DeviceContext &context,
                                          const ModelState &state) {
  require(state.layers.size() == kLayerCount,
          "persistent state hash has the wrong layer count");
  constexpr std::string_view kStateHashDomain =
      "rwkv7-persistent-device-state-v1";
  blake3_hasher hasher;
  blake3_hasher_init(&hasher);
  blake3_hasher_update(&hasher, kStateHashDomain.data(),
                       kStateHashDomain.size());
  const auto hash_values = [&hasher]<typename Value>(
                               const std::vector<Value> &values) {
    require(!values.empty(),
            "persistent state hash received an empty tensor");
    blake3_hasher_update(&hasher, values.data(),
                         values.size() * sizeof(Value));
  };
  for (const LayerState &layer : state.layers) {
    require(layer.attention_previous.dtype() == ttnn::DataType::BFLOAT16 &&
                layer.ffn_previous.dtype() == ttnn::DataType::BFLOAT16 &&
                layer.matrix.dtype() == ttnn::DataType::FLOAT32,
            "persistent state hash found a precision mismatch");
    hash_values(read_fused_wkv_probe_values<bfloat16>(
        context, layer.attention_previous, true));
    hash_values(read_fused_wkv_probe_values<bfloat16>(
        context, layer.ffn_previous, true));
    hash_values(read_fused_wkv_probe_values<float>(context, layer.matrix,
                                                   false));
  }
  std::array<std::uint8_t, kBlake3DigestBytes> digest{};
  blake3_hasher_finalize(&hasher, digest.data(), digest.size());
  return digest_hex(digest);
}

template <typename Value>
float patterned_fixture_max_error(const std::vector<Value> &values,
                                  std::size_t expected_size,
                                  std::size_t coordinate_stride,
                                  float negative_expected,
                                  float positive_expected) {
  require(values.size() == expected_size && coordinate_stride > 0U,
          "fused WKV probe readback has the wrong geometry");
  const std::size_t negative_width = kFusedWkvProbeNegativeWidth;
  float maximum_error = 0.0F;
  for (std::size_t index = 0U; index < values.size(); ++index) {
    const float converted = static_cast<float>(values[index]);
    require(std::isfinite(converted),
            "fused WKV probe readback contains a non-finite value");
    const std::size_t coordinate =
        (index / coordinate_stride) % kHeadSize;
    const float expected = coordinate < negative_width
                               ? negative_expected
                               : positive_expected;
    maximum_error =
        std::max(maximum_error, std::abs(converted - expected));
  }
  return maximum_error;
}

template <typename Value>
Json fixture_summary(const std::vector<Value> &values,
                     std::size_t expected_size) {
  require(values.size() == expected_size && !values.empty(),
          "fused WKV probe summary has the wrong element count");
  float minimum = std::numeric_limits<float>::infinity();
  float maximum = -std::numeric_limits<float>::infinity();
  std::size_t nonzero_count = 0U;
  for (const Value value : values) {
    const float converted = static_cast<float>(value);
    require(std::isfinite(converted),
            "fused WKV probe summary contains a non-finite value");
    minimum = std::min(minimum, converted);
    maximum = std::max(maximum, converted);
    if (converted != 0.0F) {
      ++nonzero_count;
    }
  }
  return Json{{"first", static_cast<float>(values[kFirstIndex])},
              {"minimum", minimum},
              {"maximum", maximum},
              {"nonzero_count", nonzero_count}};
}

template <typename Value>
float compared_max_error(const std::vector<Value> &expected,
                         const std::vector<Value> &actual) {
  require(expected.size() == actual.size() && !expected.empty(),
          "fused WKV comparison has the wrong geometry");
  float maximum_error = 0.0F;
  for (std::size_t index = 0U; index < expected.size(); ++index) {
    const float expected_value = static_cast<float>(expected[index]);
    const float actual_value = static_cast<float>(actual[index]);
    require(std::isfinite(expected_value) && std::isfinite(actual_value),
            "fused WKV comparison contains a non-finite value");
    maximum_error =
        std::max(maximum_error, std::abs(expected_value - actual_value));
  }
  return maximum_error;
}

std::vector<bfloat16> asymmetric_bf16_values(std::size_t element_count,
                                             float base, float linear_scale,
                                             float modular_scale) {
  std::vector<bfloat16> values;
  values.reserve(element_count);
  constexpr std::size_t kPatternModulus = 17U;
  constexpr std::size_t kPatternCenter = kPatternModulus / 2U;
  for (std::size_t index = 0U; index < element_count; ++index) {
    const auto centered = static_cast<std::int64_t>(index % kPatternModulus) -
                          static_cast<std::int64_t>(kPatternCenter);
    const float value =
        base + linear_scale * static_cast<float>(index) +
        modular_scale * static_cast<float>(centered);
    values.emplace_back(value);
  }
  return values;
}

std::vector<float> asymmetric_fp32_values(std::size_t element_count) {
  std::vector<float> values;
  values.reserve(element_count);
  constexpr std::size_t kPatternModulus = 23U;
  constexpr std::size_t kPatternCenter = kPatternModulus / 2U;
  constexpr float kBase = -0.03125F;
  constexpr float kLinearScale = 0.000003F;
  constexpr float kModularScale = 0.0007F;
  for (std::size_t index = 0U; index < element_count; ++index) {
    const auto centered = static_cast<std::int64_t>(index % kPatternModulus) -
                          static_cast<std::int64_t>(kPatternCenter);
    values.push_back(kBase + kLinearScale * static_cast<float>(index) +
                     kModularScale * static_cast<float>(centered));
  }
  return values;
}

void run_fused_wkv_asymmetric_probe(
    const std::filesystem::path &output_path) {
  using ReferenceInputs = rwkv7::fused_wkv_reference::Inputs;
  const auto build_decision = rwkv7::fused_wkv_reference::validate_diagnostic_build(
      kRuntimeName, kFusedWkvBfloat16ReadoutBoundary, kFusedWkvFp32RecurrenceVectors);
  require(build_decision.valid,
          std::string("fused WKV asymmetric probe rejected its build: ") +
              build_decision.reason);

  const std::size_t vector_elements =
      static_cast<std::size_t>(kFusedWkvAsymmetricProbeBatchSize) *
      kHeadCount * kHeadSize;
  const std::size_t state_elements = vector_elements * kHeadSize;
  const auto pattern = [vector_elements](
                           const AsymmetricVectorPattern &definition) {
    return asymmetric_bf16_values(vector_elements, definition.base,
                                  definition.linear_scale,
                                  definition.modular_scale);
  };
  const std::vector<bfloat16> a_values = pattern(kFusedWkvAsymmetricA);
  const std::vector<bfloat16> w_values = pattern(kFusedWkvAsymmetricW);
  const std::vector<bfloat16> k_values = pattern(kFusedWkvAsymmetricK);
  const std::vector<bfloat16> v_values = pattern(kFusedWkvAsymmetricV);
  const std::vector<bfloat16> r_values = pattern(kFusedWkvAsymmetricR);
  const std::vector<bfloat16> b_values = pattern(kFusedWkvAsymmetricB);
  const std::vector<bfloat16> r_k_values = pattern(kFusedWkvAsymmetricRk);
  const std::vector<bfloat16> group_norm_weight_values =
      pattern(kFusedWkvAsymmetricGroupNormWeight);
  const std::vector<bfloat16> group_norm_bias_values =
      pattern(kFusedWkvAsymmetricGroupNormBias);
  const std::vector<bfloat16> gate_values =
      pattern(kFusedWkvAsymmetricGate);
  const std::vector<float> state_values =
      asymmetric_fp32_values(state_elements);
  const auto as_float = [](const std::vector<bfloat16> &values) {
    std::vector<float> converted;
    converted.reserve(values.size());
    for (const bfloat16 value : values) {
      converted.push_back(static_cast<float>(value));
    }
    return converted;
  };
  const auto reference = rwkv7::fused_wkv_reference::evaluate(ReferenceInputs{
      kHeadCount,
      kHeadSize,
      as_float(a_values),
      as_float(w_values),
      as_float(k_values),
      as_float(v_values),
      as_float(r_values),
      as_float(b_values),
      as_float(r_k_values),
      as_float(group_norm_weight_values),
      as_float(group_norm_bias_values),
      as_float(gate_values),
      state_values,
      kGroupNormEpsilon,
  });

  std::cout << "fused-wkv-asymmetric-stage: host-reference-ready\n"
            << std::flush;
  select_single_card_device(kPhysicalDeviceZero);
  DeviceContext context(kPhysicalDeviceZero);
  context.set_profiler_flush_enabled(false);
  std::cout << "fused-wkv-asymmetric-stage: device-opened\n" << std::flush;

  const std::vector<std::uint32_t> vector_shape = {
      kFusedWkvAsymmetricProbeBatchSize, kHeadCount, 1U, kHeadSize};
  const std::vector<std::uint32_t> shared_vector_shape = {
      1U, kHeadCount, 1U, kHeadSize};
  const std::vector<std::uint32_t> state_shape = {
      kFusedWkvAsymmetricProbeBatchSize, kHeadCount, kHeadSize, kHeadSize};
  const Tensor a =
      context.bf16_tensor(a_values, vector_shape, ttnn::Layout::TILE);
  const Tensor w =
      context.bf16_tensor(w_values, vector_shape, ttnn::Layout::TILE);
  const Tensor k =
      context.bf16_tensor(k_values, vector_shape, ttnn::Layout::TILE);
  const Tensor v =
      context.bf16_tensor(v_values, vector_shape, ttnn::Layout::TILE);
  const Tensor r =
      context.bf16_tensor(r_values, vector_shape, ttnn::Layout::TILE);
  const Tensor b =
      context.bf16_tensor(b_values, vector_shape, ttnn::Layout::TILE);
  const Tensor r_k = context.bf16_tensor(
      r_k_values, shared_vector_shape, ttnn::Layout::TILE);
  const Tensor group_norm_weight = context.bf16_tensor(
      group_norm_weight_values, shared_vector_shape, ttnn::Layout::TILE);
  const Tensor group_norm_bias = context.bf16_tensor(
      group_norm_bias_values, shared_vector_shape, ttnn::Layout::TILE);
  const Tensor gate =
      context.bf16_tensor(gate_values, vector_shape, ttnn::Layout::TILE);
  const Tensor state = context.float32_tensor(
      state_values, state_shape, ttnn::Layout::TILE);
  std::cout << "fused-wkv-asymmetric-stage: device-inputs-ready\n"
            << std::flush;

  const char *const kernel_root =
      std::getenv(kFusedWkvKernelRootEnvironment.data());
  require(kernel_root != nullptr,
          "fused WKV asymmetric probe has no kernel source root");
  const rwkv7::fused_wkv::Inputs fused_inputs{
      a, w, k, v, r, b, r_k, group_norm_weight, group_norm_bias, gate,
      state};
  std::cout << "fused-wkv-asymmetric-stage: fused-launch-start\n"
            << std::flush;
  std::cout << "fused-wkv-asymmetric-stage: state-dot-a-launch-start\n" << std::flush;
  const Tensor fused_state_dot_a = context.checked(
      rwkv7::fused_wkv::run_state_dot_a_diagnostic(context.mesh(), fused_inputs, kernel_root),
      "fused-wkv-asymmetric-state-dot-a");
  std::cout << "fused-wkv-asymmetric-stage: state-dot-a-launch-finished\n" << std::flush;
  const std::vector<float> fused_state_dot_a_values =
      read_fused_wkv_probe_values<float>(context, fused_state_dot_a, false);
  std::cout << "fused-wkv-asymmetric-stage: state-dot-a-readback-finished\n" << std::flush;

  std::cout << "fused-wkv-asymmetric-stage: decay-launch-start\n" << std::flush;
  const Tensor fused_decay = context.checked(
      rwkv7::fused_wkv::run_decay_diagnostic(context.mesh(), fused_inputs, kernel_root), "fused-wkv-asymmetric-decay");
  std::cout << "fused-wkv-asymmetric-stage: decay-launch-finished\n" << std::flush;
  const std::vector<float> fused_decay_values = read_fused_wkv_probe_values<float>(context, fused_decay, false);
  std::cout << "fused-wkv-asymmetric-stage: decay-readback-finished\n" << std::flush;

  std::cout << "fused-wkv-asymmetric-stage: rank-update-launch-start\n" << std::flush;
  const Tensor fused_rank_update = context.checked(
      rwkv7::fused_wkv::run_rank_update_diagnostic(context.mesh(), fused_inputs, kernel_root),
      "fused-wkv-asymmetric-rank-update");
  std::cout << "fused-wkv-asymmetric-stage: rank-update-launch-finished\n" << std::flush;
  const std::vector<float> fused_rank_update_values =
      read_fused_wkv_probe_values<float>(context, fused_rank_update, false);
  std::cout << "fused-wkv-asymmetric-stage: rank-update-readback-finished\n" << std::flush;

  std::cout << "fused-wkv-asymmetric-stage: outer-update-launch-start\n" << std::flush;
  const Tensor fused_outer_update = context.checked(
      rwkv7::fused_wkv::run_outer_update_diagnostic(context.mesh(), fused_inputs, kernel_root),
      "fused-wkv-asymmetric-outer-update");
  std::cout << "fused-wkv-asymmetric-stage: outer-update-launch-finished\n" << std::flush;
  const std::vector<float> fused_outer_update_values =
      read_fused_wkv_probe_values<float>(context, fused_outer_update, false);
  std::cout << "fused-wkv-asymmetric-stage: outer-update-readback-finished\n" << std::flush;

  std::cout << "fused-wkv-asymmetric-stage: first-add-launch-start\n" << std::flush;
  const Tensor fused_first_add = context.checked(
      rwkv7::fused_wkv::run_first_add_diagnostic(
          context.mesh(), fused_inputs, fused_decay, fused_rank_update, kernel_root),
      "fused-wkv-asymmetric-first-add");
  std::cout << "fused-wkv-asymmetric-stage: first-add-launch-finished\n" << std::flush;
  const std::vector<float> fused_first_add_values = read_fused_wkv_probe_values<float>(context, fused_first_add, false);
  std::cout << "fused-wkv-asymmetric-stage: first-add-readback-finished\n" << std::flush;

  std::cout << "fused-wkv-asymmetric-stage: final-state-launch-start\n" << std::flush;
  const Tensor fused_final_state = context.checked(
      rwkv7::fused_wkv::run_first_add_diagnostic(
          context.mesh(), fused_inputs, fused_first_add, fused_outer_update, kernel_root),
      "fused-wkv-asymmetric-final-state");
  std::cout << "fused-wkv-asymmetric-stage: final-state-launch-finished\n" << std::flush;
  const std::vector<float> fused_final_state_values =
      read_fused_wkv_probe_values<float>(context, fused_final_state, false);
  std::cout << "fused-wkv-asymmetric-stage: final-state-readback-finished\n" << std::flush;

  std::cout << "fused-wkv-asymmetric-stage: bf16-readout-launch-start\n" << std::flush;
  const Tensor fused_bf16_readout = context.checked(
      rwkv7::fused_wkv::run_bf16_readout_diagnostic(context.mesh(), fused_inputs, fused_final_state, kernel_root),
      "fused-wkv-asymmetric-bf16-readout");
  std::cout << "fused-wkv-asymmetric-stage: bf16-readout-launch-finished\n" << std::flush;
  const std::vector<bfloat16> fused_bf16_readout_values =
      read_fused_wkv_probe_values<bfloat16>(context, fused_bf16_readout, false);
  std::cout << "fused-wkv-asymmetric-stage: bf16-readout-readback-finished\n" << std::flush;

  std::cout << "fused-wkv-asymmetric-stage: full-split-launch-start\n" << std::flush;
  rwkv7::fused_wkv::Outputs full_split = rwkv7::fused_wkv::run(context.mesh(), fused_inputs, kernel_root);
  const bool full_split_correction_complete = full_split.correction_complete;
  const Tensor full_split_state =
      context.checked(std::move(full_split.next_state), "fused-wkv-asymmetric-full-split-state");
  const Tensor full_split_readout =
      context.checked(std::move(full_split.output), "fused-wkv-asymmetric-full-split-readout");
  std::cout << "fused-wkv-asymmetric-stage: full-split-launch-finished\n" << std::flush;
  const std::vector<float> full_split_state_values =
      read_fused_wkv_probe_values<float>(context, full_split_state, false);
  const std::vector<bfloat16> full_split_readout_values =
      read_fused_wkv_probe_values<bfloat16>(context, full_split_readout, false);
  std::cout << "fused-wkv-asymmetric-stage: full-split-readback-finished\n" << std::flush;
  const Tensor full_split_readout_rows = op_reshape(context, full_split_readout, vector_shape);
  const Tensor full_split_corrected = apply_attention_correction(
      context, full_split_readout_rows, r, k, v, r_k, group_norm_weight, group_norm_bias, gate);
  const std::vector<bfloat16> full_split_corrected_values =
      read_fused_wkv_probe_values<bfloat16>(context, full_split_corrected, false);
  std::cout << "fused-wkv-asymmetric-stage: full-split-correction-finished\n" << std::flush;

  const WkvOutput generic = generic_wkv_step(context, state, a, b, w, v, k, r);
  require(generic.state_dot_a.has_value(), "generic WKV control omitted state_dot_a");
  require(generic.decay.has_value(), "generic WKV control omitted decay");
  require(generic.rank_update.has_value(), "generic WKV control omitted rank_update");
  require(generic.outer_update.has_value(), "generic WKV control omitted outer_update");
  require(generic.first_add.has_value(), "generic WKV control omitted first_add");
  const std::vector<float> generic_state_dot_a_values =
      read_fused_wkv_probe_values<float>(context, *generic.state_dot_a, false);
  const std::vector<float> generic_decay_values = read_fused_wkv_probe_values<float>(context, *generic.decay, false);
  const std::vector<float> generic_rank_update_values =
      read_fused_wkv_probe_values<float>(context, *generic.rank_update, false);
  const std::vector<float> generic_outer_update_values =
      read_fused_wkv_probe_values<float>(context, *generic.outer_update, false);
  const std::vector<float> generic_first_add_values =
      read_fused_wkv_probe_values<float>(context, *generic.first_add, false);
  const std::vector<float> generic_final_state_values =
      read_fused_wkv_probe_values<float>(context, generic.state, false);
  const std::vector<bfloat16> generic_bf16_readout_values =
      read_fused_wkv_probe_values<bfloat16>(context, generic.output_for_hidden, false);
  const Tensor generic_readout_rows = op_reshape(context, generic.output_for_hidden, vector_shape);
  const Tensor generic_corrected =
      apply_attention_correction(context, generic_readout_rows, r, k, v, r_k, group_norm_weight, group_norm_bias, gate);
  const std::vector<bfloat16> generic_corrected_values =
      read_fused_wkv_probe_values<bfloat16>(context, generic_corrected, false);
  std::cout << "fused-wkv-asymmetric-stage: generic-control-finished\n" << std::flush;

  const float state_dot_a_error = compared_max_error(generic_state_dot_a_values, fused_state_dot_a_values);
  const bool exact_state_dot_a = generic_state_dot_a_values == fused_state_dot_a_values;
  const float decay_error = compared_max_error(generic_decay_values, fused_decay_values);
  const bool exact_decay = generic_decay_values == fused_decay_values;
  const float rank_update_error = compared_max_error(generic_rank_update_values, fused_rank_update_values);
  const bool exact_rank_update = generic_rank_update_values == fused_rank_update_values;
  const float outer_update_error = compared_max_error(generic_outer_update_values, fused_outer_update_values);
  const bool exact_outer_update = generic_outer_update_values == fused_outer_update_values;
  const float first_add_error = compared_max_error(generic_first_add_values, fused_first_add_values);
  const bool exact_first_add = generic_first_add_values == fused_first_add_values;
  const float final_state_error = compared_max_error(generic_final_state_values, fused_final_state_values);
  const bool exact_final_state = generic_final_state_values == fused_final_state_values;
  const float bf16_readout_error = compared_max_error(generic_bf16_readout_values, fused_bf16_readout_values);
  const bool exact_bf16_readout = generic_bf16_readout_values == fused_bf16_readout_values;
  const float full_split_state_error = compared_max_error(generic_final_state_values, full_split_state_values);
  const bool exact_full_split_state = generic_final_state_values == full_split_state_values;
  const float full_split_readout_error = compared_max_error(generic_bf16_readout_values, full_split_readout_values);
  const bool exact_full_split_readout = generic_bf16_readout_values == full_split_readout_values;
  const float full_split_corrected_error = compared_max_error(generic_corrected_values, full_split_corrected_values);
  const bool exact_full_split_corrected = generic_corrected_values == full_split_corrected_values;
  const bool exact_full_split = exact_full_split_state && exact_full_split_readout && exact_full_split_corrected &&
                                !full_split_correction_complete;
  const bool exact_recurrence_prefix = exact_state_dot_a && exact_decay && exact_rank_update && exact_outer_update &&
                                       exact_first_add && exact_final_state && exact_bf16_readout;
  const bool exact_full_path = exact_recurrence_prefix && exact_full_split;
  const bool simulated = std::getenv("TT_METAL_SIMULATOR") != nullptr;
  const Json receipt{
      {"schema_version", kSchemaVersion},
      {"outcome",
       !exact_state_dot_a               ? "fused_wkv_state_dot_a_generic_mismatch"
       : !exact_decay                   ? "fused_wkv_decay_generic_mismatch"
       : !exact_rank_update             ? "fused_wkv_rank_update_generic_mismatch"
       : !exact_outer_update            ? "fused_wkv_outer_update_generic_mismatch"
       : !exact_first_add               ? "fused_wkv_first_add_generic_mismatch"
       : !exact_final_state             ? "fused_wkv_final_state_generic_mismatch"
       : !exact_bf16_readout            ? "fused_wkv_bf16_readout_generic_mismatch"
       : !exact_full_split_state        ? "fused_wkv_full_split_state_generic_mismatch"
       : !exact_full_split_readout      ? "fused_wkv_full_split_readout_generic_mismatch"
       : full_split_correction_complete ? "fused_wkv_full_split_correction_ownership_mismatch"
       : exact_full_split_corrected     ? "fused_wkv_full_split_generic_parity"
                                        : "fused_wkv_full_split_corrected_generic_mismatch"},
      {"simulated", simulated},
      {"physical_hardware", !simulated},
      {"architecture", "blackhole"},
      {"selection_ordinal", kPhysicalDeviceZero},
      {"batch_size", kFusedWkvAsymmetricProbeBatchSize},
      {"fixture", "asymmetric_indexed_values_v1"},
      {"runtime_name", kRuntimeName},
      {"bfloat16_readout_boundary", kFusedWkvBfloat16ReadoutBoundary},
      {"fp32_recurrence_vectors", kFusedWkvFp32RecurrenceVectors},
      {"reference", "pure_host_fused_wkv_v1"},
      {"generic_device_operations_before_fused_launch", 0U},
      {"generic_device_operations_after_fused_readback", true},
      {"exact_fused_generic_state_dot_a_parity", exact_state_dot_a},
      {"exact_fused_generic_decay_parity", exact_decay},
      {"exact_fused_generic_rank_update_parity", exact_rank_update},
      {"exact_fused_generic_outer_update_parity", exact_outer_update},
      {"exact_fused_generic_first_add_parity", exact_first_add},
      {"exact_fused_generic_final_state_parity", exact_final_state},
      {"exact_fused_generic_bf16_readout_parity", exact_bf16_readout},
      {"exact_fused_generic_recurrence_prefix_parity", exact_recurrence_prefix},
      {"full_split_correction_complete", full_split_correction_complete},
      {"full_split_caller_owns_correction", !full_split_correction_complete},
      {"exact_full_split_generic_state_parity", exact_full_split_state},
      {"exact_full_split_generic_bf16_readout_parity", exact_full_split_readout},
      {"exact_full_split_generic_corrected_row_parity", exact_full_split_corrected},
      {"exact_full_split_generic_parity", exact_full_split},
      {"exact_full_path_generic_parity", exact_full_path},
      {"maximum_absolute_error",
       {{"fused_generic_state_dot_a", state_dot_a_error},
        {"fused_generic_decay", decay_error},
        {"fused_generic_rank_update", rank_update_error},
        {"fused_generic_outer_update", outer_update_error},
        {"fused_generic_first_add", first_add_error},
        {"fused_generic_final_state", final_state_error},
        {"fused_generic_bf16_readout", bf16_readout_error},
        {"full_split_generic_final_state", full_split_state_error},
        {"full_split_generic_bf16_readout", full_split_readout_error},
        {"full_split_generic_corrected_row", full_split_corrected_error}}},
      {"host_reference_value_summary",
       {{"state", fixture_summary(reference.next_state, state_elements)},
        {"output_bits", fixture_summary(reference.corrected_rows_bfloat16, vector_elements)}}},
      {"recurrence_stage_value_summary",
       {{"state_dot_a",
         {{"fused", fixture_summary(fused_state_dot_a_values, vector_elements)},
          {"generic", fixture_summary(generic_state_dot_a_values, vector_elements)}}},
        {"decay",
         {{"fused", fixture_summary(fused_decay_values, state_elements)},
          {"generic", fixture_summary(generic_decay_values, state_elements)}}},
        {"rank_update",
         {{"fused", fixture_summary(fused_rank_update_values, state_elements)},
          {"generic", fixture_summary(generic_rank_update_values, state_elements)}}},
        {"outer_update",
         {{"fused", fixture_summary(fused_outer_update_values, state_elements)},
          {"generic", fixture_summary(generic_outer_update_values, state_elements)}}},
        {"first_add",
         {{"fused", fixture_summary(fused_first_add_values, state_elements)},
          {"generic", fixture_summary(generic_first_add_values, state_elements)}}},
        {"final_state",
         {{"fused", fixture_summary(fused_final_state_values, state_elements)},
          {"generic", fixture_summary(generic_final_state_values, state_elements)}}},
        {"bf16_readout",
         {{"fused", fixture_summary(fused_bf16_readout_values, vector_elements)},
          {"generic", fixture_summary(generic_bf16_readout_values, vector_elements)}}},
        {"full_split",
         {{"state", fixture_summary(full_split_state_values, state_elements)},
          {"readout", fixture_summary(full_split_readout_values, vector_elements)},
          {"corrected_rows", fixture_summary(full_split_corrected_values, vector_elements)}}}}},
      {"candidate_dispatch", true},
      {"production_physically_admitted", false},
      {"performance_claim", false},
      {"operation_counts", map_to_json(context.operation_counts())}};
  write_json(output_path, receipt);
  require(exact_state_dot_a, "dedicated fused state_dot_a differs from generic TT-NN");
  require(exact_decay, "dedicated fused decay differs from generic TT-NN");
  require(exact_rank_update, "dedicated fused rank_update differs from generic TT-NN");
  require(exact_outer_update, "dedicated fused outer_update differs from generic TT-NN");
  require(exact_first_add, "dedicated fused first_add differs from generic TT-NN");
  require(exact_final_state, "dedicated fused final_state differs from generic TT-NN");
  require(exact_bf16_readout, "dedicated fused BF16 readout differs from generic TT-NN");
  require(exact_full_split_state, "full split fused state differs from generic TT-NN");
  require(exact_full_split_readout, "full split fused BF16 readout differs from generic TT-NN");
  require(!full_split_correction_complete, "full split fused output claimed correction completion");
  require(exact_full_split_corrected, "full split corrected rows differ from generic TT-NN");
  std::cout << "fused-wkv-asymmetric-stage: receipt-written\n"
            << output_path << '\n'
            << std::flush;
}

void run_fused_wkv_probe(const std::filesystem::path &output_path) {
  select_single_card_device(kPhysicalDeviceZero);
  DeviceContext context(kPhysicalDeviceZero);
  context.set_profiler_flush_enabled(false);

  const std::vector<std::uint32_t> vector_shape = {
      kFusedWkvProbeBatchSize, kHeadCount, 1U, kHeadSize};
  const std::vector<std::uint32_t> shared_vector_shape = {
      1U, kHeadCount, 1U, kHeadSize};
  const std::vector<std::uint32_t> state_shape = {
      kFusedWkvProbeBatchSize, kHeadCount, kHeadSize, kHeadSize};
  const std::size_t vector_elements =
      static_cast<std::size_t>(kFusedWkvProbeBatchSize) * kHeadCount *
      kHeadSize;
  const std::size_t state_elements = vector_elements * kHeadSize;
  const auto bf16_constant = [&](const std::vector<std::uint32_t> &shape,
                                 float value) {
    return op_add_scalar(context, context.zeros(shape), value);
  };

  const Tensor a = bf16_constant(vector_shape, kFusedWkvProbeA);
  const Tensor w = bf16_constant(vector_shape, kFusedWkvProbeW);
  const Tensor k = bf16_constant(vector_shape, kFusedWkvProbeK);
  std::vector<bfloat16> v_values;
  v_values.reserve(vector_elements);
  const std::size_t negative_width = kFusedWkvProbeNegativeWidth;
  for (std::size_t index = 0U; index < vector_elements; ++index) {
    const float value = index % kHeadSize < negative_width
                            ? -kFusedWkvProbeV
                            : kFusedWkvProbeV;
    v_values.emplace_back(value);
  }
  const Tensor v =
      context.bf16_tensor(std::move(v_values), vector_shape,
                          ttnn::Layout::TILE);
  const Tensor r = bf16_constant(vector_shape, kFusedWkvProbeR);
  const Tensor b = bf16_constant(vector_shape, kFusedWkvProbeB);
  const Tensor r_k =
      bf16_constant(shared_vector_shape, kFusedWkvProbeRk);
  const Tensor group_norm_weight = bf16_constant(
      shared_vector_shape, kFusedWkvProbeGroupNormWeight);
  const Tensor group_norm_bias = bf16_constant(
      shared_vector_shape, kFusedWkvProbeGroupNormBias);
  const Tensor gate =
      bf16_constant(vector_shape, kFusedWkvProbeGate);
  const Tensor zero_state = context.float32_tensor(
      std::vector<float>(state_elements, 0.0F), state_shape,
      ttnn::Layout::TILE);
  const char *const kernel_root =
      std::getenv(kFusedWkvKernelRootEnvironment.data());
  require(kernel_root != nullptr,
          "fused WKV probe has no kernel source root");

  const rwkv7::fused_wkv::Inputs first_inputs{
      a, w, k, v, r, b, r_k, group_norm_weight, group_norm_bias, gate,
      zero_state};
  rwkv7::fused_wkv::Outputs first = rwkv7::fused_wkv::run(
      context.mesh(), first_inputs, kernel_root);
  Tensor first_state =
      context.checked(std::move(first.next_state), "fused-wkv-probe-state-1");
  Tensor first_output =
      context.checked(std::move(first.output), "fused-wkv-probe-output-1");

  const rwkv7::fused_wkv::Inputs second_inputs{
      a, w, k, v, r, b, r_k, group_norm_weight, group_norm_bias, gate,
      first_state};
  rwkv7::fused_wkv::Outputs second = rwkv7::fused_wkv::run(
      context.mesh(), second_inputs, kernel_root);
  Tensor second_state = context.checked(std::move(second.next_state),
                                        "fused-wkv-probe-state-2");
  Tensor second_output = context.checked(std::move(second.output),
                                         "fused-wkv-probe-output-2");

  const std::vector<float> first_state_values =
      read_fused_wkv_probe_values<float>(context, first_state, false);
  const std::vector<float> second_state_values =
      read_fused_wkv_probe_values<float>(context, second_state, false);
  const std::vector<bfloat16> first_output_values =
      read_fused_wkv_probe_values<bfloat16>(context, first_output, true);
  const std::vector<bfloat16> second_output_values =
      read_fused_wkv_probe_values<bfloat16>(context, second_output, true);
  const float first_state_error = patterned_fixture_max_error(
      first_state_values, state_elements, kHeadSize,
      -kFusedWkvProbeFirstStateMagnitude,
      kFusedWkvProbeFirstStateMagnitude);
  const float second_state_error = patterned_fixture_max_error(
      second_state_values, state_elements, kHeadSize,
      -kFusedWkvProbeSecondStateMagnitude,
      kFusedWkvProbeSecondStateMagnitude);
  const float first_output_error = patterned_fixture_max_error(
      first_output_values, vector_elements, 1U,
      kFusedWkvProbeNegativeOutput, kFusedWkvProbePositiveOutput);
  const float second_output_error = patterned_fixture_max_error(
      second_output_values, vector_elements, 1U,
      kFusedWkvProbeNegativeOutput, kFusedWkvProbePositiveOutput);
  const std::array<float, kFusedWkvProbeComparisonCount> fixture_errors = {
      first_state_error, second_state_error, first_output_error,
      second_output_error};
  const bool fixture_exact =
      std::all_of(fixture_errors.begin(), fixture_errors.end(),
                  [](float error) {
                    return error <= kFusedWkvProbeExactTolerance;
                  });

  const bool simulated = std::getenv("TT_METAL_SIMULATOR") != nullptr;
  const Json receipt{
      {"schema_version", kSchemaVersion},
      {"outcome", fixture_exact ? "fused_wkv_corrected_fixture_pass"
                                  : "fused_wkv_corrected_fixture_mismatch"},
      {"simulated", simulated},
      {"physical_hardware", !simulated},
      {"architecture", "blackhole"},
      {"mesh_shape", {kMeshRows, kMeshRows}},
      {"selection_ordinal", kPhysicalDeviceZero},
      {"batch_size", kFusedWkvProbeBatchSize},
      {"head_count", kHeadCount},
      {"head_size", kHeadSize},
      {"recurrent_steps", kMinimumGeneratedTokens},
      {"state_dtype", "float32"},
      {"vector_dtype", "bfloat16"},
      {"output_dtype", "bfloat16"},
      {"corrected_fixture_exact", fixture_exact},
      {"fused_attention_boundary",
       {{"group_normalization", true},
        {"r_k_correction", true},
        {"gate", true},
        {"bf16_output", true},
        {"output_projection", false}}},
      {"maximum_absolute_error",
       {{"first_state", first_state_error},
        {"second_state", second_state_error},
        {"first_output", first_output_error},
        {"second_output", second_output_error}}},
      {"value_summary",
       {{"first_state",
         fixture_summary(first_state_values, state_elements)},
        {"second_state",
         fixture_summary(second_state_values, state_elements)},
        {"first_output",
         fixture_summary(first_output_values, vector_elements)},
        {"second_output",
         fixture_summary(second_output_values, vector_elements)}}},
      {"candidate_dispatch", true},
      {"production_physically_admitted", false},
      {"performance_claim", false},
      {"operation_counts", map_to_json(context.operation_counts())}};
  write_json(output_path, receipt);
  require(fixture_exact,
          "fused WKV corrected fixture differs from the exact reference");
  std::cout << output_path << '\n';
}

struct LoadedPersistentDecodeProfile {
  rwkv7::persistent_decode::Profile profile;
  std::string bytes;
};

LoadedPersistentDecodeProfile load_persistent_decode_profile() {
  std::ifstream input(RWKV7_PERSISTENT_DECODE_PROFILE_PATH);
  require(input.good(), "failed to open the persistent decode profile");
  std::ostringstream buffer;
  buffer << input.rdbuf();
  require(input.good() || input.eof(),
          "failed to read the persistent decode profile");
  std::string bytes = buffer.str();
  require(!bytes.empty(), "persistent decode profile is empty");
  const Json document = Json::parse(bytes);
  return LoadedPersistentDecodeProfile{
      rwkv7::persistent_decode::parse_profile(document), std::move(bytes)};
}

struct LoadedCanaryInterlock {
  bool present;
  rwkv7::persistent_decode::CanaryInterlockState state;
  rwkv7::persistent_decode::CanaryInterlockDecision decision;
  std::string file_blake3;
  std::string canary_binding_blake3;
};

rwkv7::persistent_decode::CanaryInterlockState
parse_canary_interlock_state(std::string_view state) {
  using rwkv7::persistent_decode::CanaryInterlockState;
  if (state == "armed") {
    return CanaryInterlockState::Armed;
  }
  if (state == "retain") {
    return CanaryInterlockState::Retain;
  }
  if (state == "hold") {
    return CanaryInterlockState::Hold;
  }
  if (state == "disabled") {
    return CanaryInterlockState::Disabled;
  }
  return CanaryInterlockState::Invalid;
}

// r[impl tenstorrent.rwkv7_p150x2.persistent_decode_canary.runtime]
LoadedCanaryInterlock load_canary_interlock(
    const LoadedPersistentDecodeProfile &profile) {
  using rwkv7::persistent_decode::CanaryInterlockEvidence;
  using rwkv7::persistent_decode::CanaryInterlockState;
  const char *const path_value =
      std::getenv(kPersistentDecodeCanaryInterlockEnvironment.data());
  const char *const expected_file_blake3_value = std::getenv(
      kPersistentDecodeCanaryInterlockBlake3Environment.data());
  const char *const expected_binding_value = std::getenv(
      kPersistentDecodeCanaryBindingBlake3Environment.data());
  const bool any_present = path_value != nullptr ||
                           expected_file_blake3_value != nullptr ||
                           expected_binding_value != nullptr;
  const std::string expected_profile_blake3 =
      rwkv7::persistent_decode::blake3_hex(profile.bytes);
  if (!any_present) {
    const CanaryInterlockEvidence evidence{
        false, CanaryInterlockState::NotPresent, {}, {}, {}, {}, {}, {}, {},
        0U, 0U, 0.0};
    const auto decision = rwkv7::persistent_decode::validate_canary_interlock(
        evidence, profile.profile.profile_id, expected_profile_blake3);
    return LoadedCanaryInterlock{false, CanaryInterlockState::NotPresent,
                                 decision, {}, {}};
  }
  const auto invalid_interlock = [&]() {
    const CanaryInterlockEvidence evidence{
        true,
        CanaryInterlockState::Invalid,
        {},
        expected_file_blake3_value == nullptr
            ? std::string{}
            : std::string(expected_file_blake3_value),
        {},
        expected_binding_value == nullptr ? std::string{}
                                          : std::string(expected_binding_value),
        {},
        {},
        {},
        0U,
        0U,
        0.0};
    const auto decision = rwkv7::persistent_decode::validate_canary_interlock(
        evidence, profile.profile.profile_id, expected_profile_blake3);
    return LoadedCanaryInterlock{true, CanaryInterlockState::Invalid,
                                 decision, {}, {}};
  };
  if (path_value == nullptr || expected_file_blake3_value == nullptr ||
      expected_binding_value == nullptr) {
    return invalid_interlock();
  }
  try {
    std::ifstream input(path_value, std::ios::binary | std::ios::ate);
    if (!input.good()) {
      return invalid_interlock();
    }
    const std::streampos end = input.tellg();
    if (end <= 0 ||
        end > static_cast<std::streampos>(
                  kPersistentDecodeCanaryMaximumFileBytes)) {
      return invalid_interlock();
    }
    const std::size_t byte_count = static_cast<std::size_t>(end);
    std::string bytes(byte_count, '\0');
    input.seekg(0, std::ios::beg);
    input.read(bytes.data(), static_cast<std::streamsize>(byte_count));
    if (!input.good()) {
      return invalid_interlock();
    }
    const std::string actual_file_blake3 =
        rwkv7::persistent_decode::blake3_hex(bytes);
    const Json document = Json::parse(bytes);
    if (!document.is_object() ||
        document.size() != kPersistentDecodeCanaryTopLevelFieldCount ||
        document.at("schema_version").get<std::uint32_t>() !=
            kPersistentDecodeCanarySchemaVersion ||
        document.at("event_kind").get<std::string>() !=
            kPersistentDecodeCanaryInterlockKind ||
        document.at("hash_algorithm").get<std::string>() !=
            kPersistentDecodeCanaryHashAlgorithm) {
      return invalid_interlock();
    }
    const Json &limits = document.at("limits");
    if (!limits.is_object() ||
        limits.size() != kPersistentDecodeCanaryLimitFieldCount) {
      return invalid_interlock();
    }
    const CanaryInterlockState state = parse_canary_interlock_state(
        document.at("state").get<std::string>());
    const std::string canary_binding_blake3 =
        document.at("canary_binding_blake3").get<std::string>();
    const CanaryInterlockEvidence evidence{
        true,
        state,
        document.at("reason").get<std::string>(),
        std::string(expected_file_blake3_value),
        actual_file_blake3,
        std::string(expected_binding_value),
        canary_binding_blake3,
        document.at("profile_id").get<std::string>(),
        document.at("profile_blake3").get<std::string>(),
        limits.at("maximum_request_executions").get<std::uint64_t>(),
        limits.at("maximum_duration_seconds").get<std::uint64_t>(),
        limits.at("candidate_traffic_fraction").get<double>()};
    const auto decision = rwkv7::persistent_decode::validate_canary_interlock(
        evidence, profile.profile.profile_id, expected_profile_blake3);
    return LoadedCanaryInterlock{true, state, decision, actual_file_blake3,
                                 canary_binding_blake3};
  } catch (const std::exception &) {
    return invalid_interlock();
  }
}

std::vector<std::string_view>
binding_component_views(const std::vector<std::string> &components) {
  std::vector<std::string_view> views;
  views.reserve(components.size());
  for (const std::string &component : components) {
    views.push_back(component);
  }
  return views;
}

std::string binding_identity(
    rwkv7::persistent_decode::BindingKind kind,
    const std::vector<std::string> &components) {
  const std::vector<std::string_view> views =
      binding_component_views(components);
  return rwkv7::persistent_decode::derive_binding_identity(kind, views);
}

rwkv7::persistent_decode::BindingSet derive_fast_worker_bindings(
    const rwkv7::persistent_decode::Profile &profile,
    std::string_view profile_bytes, const FastWorkerRequest &request,
    int physical_device_id, WkvExecutionPolicy execution_policy) {
  using rwkv7::persistent_decode::BindingKind;
  rwkv7::persistent_decode::BindingSet bindings{};

  std::vector<std::string> token_components = {"token-ring-v1",
                                               request.run_id};
  std::vector<std::string> state_components = {
      "recurrent-state-v1", "fp32", std::to_string(kLayerCount),
      std::to_string(kHeadCount), std::to_string(kHeadSize)};
  std::vector<std::string> slot_components = {
      "slot-map-v1", request.capacity.id,
      std::to_string(request.capacity.per_card_slot_count)};
  std::vector<std::string> output_components = {
      "output-ring-v1",
      std::to_string(profile.limits.output_capacity_tokens)};
  for (const FastRequest &current : request.requests) {
    token_components.push_back(current.request_id);
    token_components.push_back(std::to_string(current.prompt_tokens.size()));
    for (const std::uint32_t token : current.prompt_tokens) {
      token_components.push_back(std::to_string(token));
    }
    state_components.push_back(current.request_id);
    state_components.push_back(std::to_string(current.state_slot));
    slot_components.push_back(current.request_id);
    slot_components.push_back(std::to_string(current.state_slot));
    output_components.push_back(current.request_id);
    output_components.push_back(std::to_string(current.max_new_tokens));
  }

  bindings.identities[static_cast<std::size_t>(BindingKind::Token)] =
      binding_identity(BindingKind::Token, token_components);
  bindings.identities[static_cast<std::size_t>(BindingKind::State)] =
      binding_identity(BindingKind::State, state_components);
  bindings.identities[static_cast<std::size_t>(BindingKind::Slot)] =
      binding_identity(BindingKind::Slot, slot_components);
  bindings.identities[static_cast<std::size_t>(BindingKind::Weight)] =
      binding_identity(BindingKind::Weight,
                       {RWKV7_PINNED_ARTIFACT_BLAKE3_JSON,
                        RWKV7_PINNED_MODEL_REVISION});
  bindings.identities[static_cast<std::size_t>(BindingKind::Program)] =
      binding_identity(BindingKind::Program,
                       {profile.mechanism, request.capacity.id,
                        std::string(wkv_execution_policy_name(execution_policy))});
  bindings.identities[static_cast<std::size_t>(BindingKind::Runtime)] =
      binding_identity(BindingKind::Runtime,
                       {RWKV7_PINNED_TARGET_NAME, RWKV7_PINNED_METALIUM_ROOT,
                        std::to_string(physical_device_id)});
  bindings.identities[static_cast<std::size_t>(BindingKind::Checkpoint)] =
      binding_identity(BindingKind::Checkpoint,
                       {RWKV7_PINNED_CHECKPOINT_BLAKE3,
                        RWKV7_PINNED_MODEL_REVISION});
  bindings.identities[static_cast<std::size_t>(BindingKind::Output)] =
      binding_identity(BindingKind::Output, output_components);
  bindings.identities[static_cast<std::size_t>(BindingKind::Profile)] =
      binding_identity(BindingKind::Profile,
                       {rwkv7::persistent_decode::blake3_hex(profile_bytes),
                        profile.model_profile_blake3});
  return bindings;
}

enum class PersistentDecodeControl {
  ProfileDefault,
  ForceNormal,
  ForceCandidate,
};

PersistentDecodeControl parse_persistent_decode_control(
    const std::optional<std::string_view> value) {
  if (!value.has_value()) {
    return PersistentDecodeControl::ProfileDefault;
  }
  require(*value == kEnabledEnvironmentValue ||
              *value == kDisabledEnvironmentValue,
          "RWKV7_PERSISTENT_DECODE_CANDIDATE must be 0 or 1");
  return *value == kEnabledEnvironmentValue
             ? PersistentDecodeControl::ForceCandidate
             : PersistentDecodeControl::ForceNormal;
}

PersistentDecodeControl persistent_decode_control() {
  const char *const value =
      std::getenv(kPersistentDecodeCandidateEnvironment.data());
  return parse_persistent_decode_control(
      value == nullptr
          ? std::nullopt
          : std::optional<std::string_view>(std::string_view(value)));
}

enum class PersistentDecodeTerminalProbe {
  None,
  Cancel,
  Timeout,
  Failure,
};

PersistentDecodeTerminalProbe parse_persistent_decode_terminal_probe(
    const std::optional<std::string_view> value) {
  if (!value.has_value()) {
    return PersistentDecodeTerminalProbe::None;
  }
  if (*value == kPersistentDecodeCancelProbe) {
    return PersistentDecodeTerminalProbe::Cancel;
  }
  if (*value == kPersistentDecodeTimeoutProbe) {
    return PersistentDecodeTerminalProbe::Timeout;
  }
  if (*value == kPersistentDecodeFailureProbe) {
    return PersistentDecodeTerminalProbe::Failure;
  }
  throw std::invalid_argument(
      "RWKV7_PERSISTENT_DECODE_TERMINAL_PROBE must be cancel, timeout, or failure");
}

PersistentDecodeTerminalProbe persistent_decode_terminal_probe() {
  const char *const value =
      std::getenv(kPersistentDecodeTerminalProbeEnvironment.data());
  return parse_persistent_decode_terminal_probe(
      value == nullptr
          ? std::nullopt
          : std::optional<std::string_view>(std::string_view(value)));
}

std::string_view persistent_decode_terminal_probe_name(
    PersistentDecodeTerminalProbe probe) {
  switch (probe) {
  case PersistentDecodeTerminalProbe::None:
    return "none";
  case PersistentDecodeTerminalProbe::Cancel:
    return kPersistentDecodeCancelProbe;
  case PersistentDecodeTerminalProbe::Timeout:
    return kPersistentDecodeTimeoutProbe;
  case PersistentDecodeTerminalProbe::Failure:
    return kPersistentDecodeFailureProbe;
  }
  return "invalid";
}

rwkv7::persistent_decode::Event persistent_decode_terminal_probe_event(
    PersistentDecodeTerminalProbe probe,
    const rwkv7::persistent_decode::Limits &limits,
    std::uint64_t elapsed_milliseconds) {
  switch (probe) {
  case PersistentDecodeTerminalProbe::Cancel:
    return rwkv7::persistent_decode::Event{
        rwkv7::persistent_decode::EventKind::Cancel, 0U,
        elapsed_milliseconds, 0U, {}};
  case PersistentDecodeTerminalProbe::Timeout:
    return rwkv7::persistent_decode::Event{
        rwkv7::persistent_decode::EventKind::Timeout, 0U,
        limits.maximum_elapsed_milliseconds, 0U, {}};
  case PersistentDecodeTerminalProbe::Failure:
    return rwkv7::persistent_decode::Event{
        rwkv7::persistent_decode::EventKind::Fail, 0U,
        elapsed_milliseconds, 0U, "physical_terminal_probe"};
  case PersistentDecodeTerminalProbe::None:
    break;
  }
  throw std::invalid_argument(
      "persistent decode terminal probe event requires a terminal probe");
}

std::optional<rwkv7::persistent_decode::Limits>
persistent_workload_limits(
    const FastWorkerRequest &request,
    const rwkv7::persistent_decode::Profile &profile) {
  if (request.requests.empty() ||
      request.requests.size() > profile.limits.queue_depth) {
    return std::nullopt;
  }
  const std::size_t prompt_length =
      request.requests[kFirstIndex].prompt_tokens.size();
  const std::uint32_t decode_steps =
      request.requests[kFirstIndex].max_new_tokens;
  for (const FastRequest &current : request.requests) {
    if (current.prompt_tokens.size() != prompt_length ||
        current.max_new_tokens != decode_steps ||
        current.state_slot >= profile.limits.queue_depth) {
      return std::nullopt;
    }
  }
  const rwkv7::persistent_decode::WindowLimitsResult selected =
      rwkv7::persistent_decode::select_window_limits(profile, decode_steps);
  if (!selected.accepted) {
    return std::nullopt;
  }
  return selected.limits;
}

bool persistent_workload_eligible(
    const FastWorkerRequest &request,
    const rwkv7::persistent_decode::Profile &profile) {
  return persistent_workload_limits(request, profile).has_value();
}

void prefill_homogeneous_prompt_prefix(
    DeviceContext &context, const ModelWeights &weights, ModelState &state,
    const FastWorkerRequest &request, std::size_t max_slot_count,
    WkvExecutionPolicy execution_policy) {
  const std::size_t prompt_length =
      request.requests[kFirstIndex].prompt_tokens.size();
  require(prompt_length > 0U,
          "persistent decode prompt must not be empty");
  for (std::size_t prompt_index = 0U;
       prompt_index + 1U < prompt_length; ++prompt_index) {
    std::vector<std::uint32_t> prompt_tokens;
    prompt_tokens.reserve(request.requests.size());
    for (const FastRequest &current : request.requests) {
      prompt_tokens.push_back(current.prompt_tokens[prompt_index]);
    }
    const auto ignored_selection = advance_tokens_fast(
        context, weights, state, prompt_tokens, false,
        static_cast<std::uint64_t>(prompt_index), max_slot_count,
        execution_policy);
    require(!ignored_selection.has_value(),
            "persistent decode prefill selected an early token");
  }
}

Json persistent_binding_json(
    const rwkv7::persistent_decode::BindingSet &bindings) {
  Json document = Json::object();
  for (std::size_t index = 0U;
       index < rwkv7::persistent_decode::kBindingKindCount; ++index) {
    const auto kind =
        static_cast<rwkv7::persistent_decode::BindingKind>(index);
    document[std::string(rwkv7::persistent_decode::binding_kind_name(kind))] =
        bindings.identities[index];
  }
  return document;
}

// r[impl tenstorrent.rwkv7_p150x2.persistent_decode_telemetry.aggregate_only]
// r[impl tenstorrent.rwkv7_p150x2.persistent_decode_telemetry.fallback]
// r[impl tenstorrent.rwkv7_p150x2.persistent_decode_telemetry.terminal]
Json persistent_telemetry_json(
    const rwkv7::persistent_decode::PersistentTelemetry &telemetry) {
  return Json{
      {"schema_version", telemetry.schema_version},
      {"event_kind", std::string(kPersistentDecodeTelemetryEventKind)},
      {"eligibility",
       {{"window_admitted", telemetry.window_admitted},
        {"workload_eligible", telemetry.workload_eligible},
        {"bindings_validated", telemetry.bindings_validated},
        {"profile_physically_admitted",
         telemetry.profile_physically_admitted},
        {"candidate_requested", telemetry.candidate_requested}}},
      {"dispatch",
       {{"selected_mode",
         std::string(rwkv7::persistent_decode::execution_mode_name(
             telemetry.selected_mode))},
        {"selected_window_steps",
         telemetry.selected_window_steps.has_value()
             ? Json(*telemetry.selected_window_steps)
             : Json(nullptr)},
        {"fallback_reason",
         std::string(rwkv7::persistent_decode::fallback_reason_name(
             telemetry.fallback_reason))}}},
      {"terminal_outcomes",
       {{"completed", telemetry.terminal_outcomes.completed},
        {"cancelled", telemetry.terminal_outcomes.cancelled},
        {"timed_out", telemetry.terminal_outcomes.timed_out},
        {"failed", telemetry.terminal_outcomes.failed}}},
      {"timing",
       {{"clock", "steady_monotonic"},
        {"unit", "seconds"},
        {"trace_setup_seconds", telemetry.trace_setup_seconds},
        {"trace_replay_seconds", telemetry.trace_replay_seconds},
        {"cleanup_seconds", telemetry.cleanup_seconds}}},
      {"parity",
       {{"generated_tokens",
         std::string(
             rwkv7::persistent_decode::telemetry_parity_status_name(
                 telemetry.generated_token_parity))},
        {"fp32_recurrent_state",
         std::string(
             rwkv7::persistent_decode::telemetry_parity_status_name(
                 telemetry.fp32_recurrent_state_parity))}}},
      {"cleanup_status",
       std::string(rwkv7::persistent_decode::telemetry_cleanup_status_name(
           telemetry.cleanup_status))}};
}

void run_fast_worker(const FastWorkerArguments &arguments) {
  const char *const inspector_value = std::getenv("TT_METAL_INSPECTOR");
  require(inspector_value != nullptr &&
              std::string_view(inspector_value) == kDisabledEnvironmentValue,
          "fast worker requires TT_METAL_INSPECTOR=0");
  select_single_card_device(arguments.physical_device_id);
  const char *const simulator_value = std::getenv("TT_METAL_SIMULATOR");
  const bool simulated = simulator_value != nullptr;
  const char *const profiler_value = std::getenv("TT_METAL_DEVICE_PROFILER");
  const bool profiler_attributed =
      profiler_value != nullptr &&
      std::string_view(profiler_value) == kEnabledEnvironmentValue;
  const char *const trace_profiler_value =
      std::getenv("TT_METAL_TRACE_PROFILER");
  require(trace_profiler_value == nullptr ||
              std::string_view(trace_profiler_value) ==
                  kEnabledEnvironmentValue ||
              std::string_view(trace_profiler_value) ==
                  kDisabledEnvironmentValue,
          "TT_METAL_TRACE_PROFILER must be 0 or 1");
  const bool profiler_trace_only =
      profiler_attributed && trace_profiler_value != nullptr &&
      std::string_view(trace_profiler_value) == kEnabledEnvironmentValue;
  const char *const stall_probe_value =
      std::getenv("RWKV7_STALL_PROBE_STEPS");
  const bool stall_probe_enabled =
      stall_probe_value != nullptr &&
      std::string_view(stall_probe_value) == kEnabledEnvironmentValue;
  const char *const fetch_queue_diagnostics_value =
      std::getenv(kFetchQueueDiagnosticsEnvironment.data());
  const bool fetch_queue_diagnostics_enabled =
      fetch_queue_diagnostics_value != nullptr &&
      std::string_view(fetch_queue_diagnostics_value) ==
          kEnabledEnvironmentValue;
  require(fetch_queue_diagnostics_enabled == stall_probe_enabled,
          "fetch queue diagnostics must match stall-probe mode");
  const std::optional<std::uint32_t> primary_thread_cpu_affinity =
      parse_primary_thread_cpu_affinity();
  const std::optional<int> primary_thread_fifo_priority =
      parse_primary_thread_fifo_priority();
  require(!primary_thread_fifo_priority.has_value() ||
              primary_thread_cpu_affinity.has_value(),
          "primary thread FIFO scheduling requires CPU affinity");
  const bool scheduling_physically_admitted =
      production_scheduling_physically_admitted(
          arguments.physical_device_id, primary_thread_cpu_affinity,
          primary_thread_fifo_priority);
  const WkvExecutionPolicy effective_execution_policy =
      effective_wkv_execution_policy(arguments.wkv_execution_policy,
                                     scheduling_physically_admitted);
  const bool production_physically_admitted =
      execution_policy_is_production_physically_admitted(
          effective_execution_policy);
  const std::uint32_t warmup_token_count =
      stall_probe_enabled ? kStallProbeWarmupTokenCount
                          : kFastWarmupTokenCount;
  const FastWorkerRequest request =
      read_fast_worker_request(arguments.requests_path);
  const LoadedPersistentDecodeProfile persistent_profile =
      load_persistent_decode_profile();
  require(persistent_profile.profile.model_profile_blake3 ==
              RWKV7_PINNED_TARGET_PROFILE_BLAKE3,
          "persistent decode profile targets a different model profile");
  const rwkv7::persistent_decode::ValidationResult precision_binding =
      rwkv7::persistent_decode::validate_precision_binding(
          persistent_profile.profile, kFusedWkvFp32RecurrenceVectors,
          kFusedWkvBfloat16ReadoutBoundary);
  require(precision_binding.valid, precision_binding.reason);
  const std::string persistent_profile_identity =
      rwkv7::persistent_decode::blake3_hex(persistent_profile.bytes);
  require(kPersistentDecodeProfileBlake3.empty() ||
              persistent_profile_identity == kPersistentDecodeProfileBlake3,
          "persistent decode profile identity differs from the immutable runtime");
  const LoadedCanaryInterlock canary_interlock =
      load_canary_interlock(persistent_profile);
  const rwkv7::persistent_decode::BindingSet expected_persistent_bindings =
      derive_fast_worker_bindings(
          persistent_profile.profile, persistent_profile.bytes, request,
          arguments.physical_device_id, effective_execution_policy);
  const rwkv7::persistent_decode::BindingSet actual_persistent_bindings =
      derive_fast_worker_bindings(
          persistent_profile.profile, persistent_profile.bytes, request,
          arguments.physical_device_id, effective_execution_policy);
  const rwkv7::persistent_decode::ValidationResult binding_validation =
      rwkv7::persistent_decode::validate_bindings(
          expected_persistent_bindings, actual_persistent_bindings);
  require(binding_validation.valid,
          "persistent decode bindings changed before device admission");
  const PersistentDecodeControl persistent_control =
      persistent_decode_control();
  const PersistentDecodeTerminalProbe terminal_probe =
      persistent_decode_terminal_probe();
  const bool terminal_probe_enabled =
      terminal_probe != PersistentDecodeTerminalProbe::None;
  const bool candidate_requested =
      persistent_control == PersistentDecodeControl::ForceCandidate;
  const bool force_normal =
      persistent_control == PersistentDecodeControl::ForceNormal;
  const bool candidate_enabled =
      !force_normal &&
      (candidate_requested || persistent_profile.profile.physically_admitted);
  const std::optional<rwkv7::persistent_decode::Limits>
      selected_persistent_limits =
          persistent_workload_limits(request, persistent_profile.profile);
  const bool window_admitted = selected_persistent_limits.has_value();
  const bool candidate_eligible = window_admitted && !stall_probe_enabled;
  const bool normal_ab_reference = force_normal && candidate_eligible;
  rwkv7::persistent_decode::Profile admission_profile =
      persistent_profile.profile;
  if (candidate_requested) {
    admission_profile.physically_admitted = true;
  }
  rwkv7::persistent_decode::Profile persistent_request_profile =
      persistent_profile.profile;
  if (selected_persistent_limits.has_value()) {
    persistent_request_profile.limits = *selected_persistent_limits;
  }
  const rwkv7::persistent_decode::CapabilityEvidence candidate_evidence{
      window_admitted,
      canary_interlock.decision.allows_persistent,
      candidate_enabled && candidate_eligible,
      candidate_enabled,
      candidate_enabled,
      candidate_enabled,
      candidate_enabled};
  const rwkv7::persistent_decode::ExecutionDecision execution_decision =
      rwkv7::persistent_decode::select_execution_mode(
          admission_profile, binding_validation, candidate_evidence);
  const bool use_persistent_trace =
      execution_decision.mode ==
      rwkv7::persistent_decode::ExecutionMode::PersistentTrace;
  require(!terminal_probe_enabled ||
              (candidate_requested && use_persistent_trace),
          "persistent decode terminal probes require an explicit trace candidate");
  require(!terminal_probe_enabled ||
              (selected_persistent_limits.has_value() &&
               selected_persistent_limits->maximum_decode_steps >
                   selected_persistent_limits
                       ->cancellation_poll_interval_steps),
          "persistent decode terminal probes require a window beyond the polling boundary");
  // The checked memory plan already ran inside the request read, before any
  // device opened. Recompute it here for the receipt so the reported numbers
  // are the exact same checked values the worker enforced.
  const FastMemoryPlan memory_plan =
      plan_fast_card_memory(request.capacity);
  const MappedFile checkpoint_file(RWKV7_PINNED_CHECKPOINT_PATH);
  const CheckpointIndex checkpoint(checkpoint_file);
  const std::size_t trace_region_size =
      use_persistent_trace ? kPersistentTraceRegionBytes
                           : DEFAULT_TRACE_REGION_SIZE;
  DeviceContext context(kFilteredLogicalDevice, trace_region_size);
  context.set_profiler_flush_enabled(false);
  context.set_operation_recording_enabled(false);
  const ModelWeights weights = load_model(context, checkpoint);
  const auto batch_size =
      static_cast<std::uint32_t>(request.requests.size());
  const std::size_t max_slot_count =
      request.capacity.per_card_slot_count;

  signal_fast_barrier(arguments.initialization_ready_barrier_path,
                      "initialization ready");
  wait_for_fast_barrier(arguments.warmup_start_barrier_path,
                        "warmup start");
  if (profiler_attributed && use_persistent_trace &&
      !profiler_trace_only) {
    context.set_profiler_flush_enabled(true);
  }
  ModelState warmup_state =
      zero_fast_model_state(context, batch_size, max_slot_count);
  std::vector<std::uint32_t> warmup_tokens(
      request.requests.size(), kGenerationEosToken);
  for (std::uint32_t index = 0U; index < warmup_token_count; ++index) {
    const auto selected = advance_tokens_fast(
        context, weights, warmup_state, warmup_tokens, true, index,
        max_slot_count, effective_execution_policy);
    require(selected.has_value(),
            "fast worker warmup did not select tokens");
    warmup_tokens = *selected;
  }
  ModelState state = zero_fast_model_state(context, batch_size, max_slot_count);
  const std::size_t cache_after_warmup = context.program_cache_entries();
  require(cache_after_warmup > 0U,
          "fast worker warmup did not populate the program cache");
  if (primary_thread_cpu_affinity.has_value()) {
    pin_current_thread(*primary_thread_cpu_affinity);
  }
  if (primary_thread_fifo_priority.has_value()) {
    configure_current_thread_fifo(*primary_thread_fifo_priority);
  }

  std::vector<FastSlot> slots;
  slots.reserve(request.requests.size());
  for (const FastRequest &current : request.requests) {
    slots.push_back(FastSlot{&current, 0U, {}, 0U, {}});
  }
  std::vector<bool> completed(slots.size(), false);
  std::vector<std::uint32_t> persistent_initial_tokens;
  std::vector<rwkv7::persistent_decode::RequestState>
      persistent_lifecycle_states;
  std::vector<std::vector<std::uint32_t>>
      persistent_reference_tokens_by_step;
  std::string persistent_reference_state_blake3;
  bool persistent_reference_state_zeroed = false;
  std::optional<PreparedPersistentTrace> prepared_persistent_trace;
  std::optional<std::chrono::steady_clock::time_point>
      persistent_trace_execution_end;
  std::optional<std::chrono::system_clock::time_point>
      persistent_trace_execution_end_wall_clock;
  double persistent_trace_setup_seconds = 0.0;
  double persistent_cleanup_seconds = 0.0;
  const bool prompt_prefill_outside_timed_region =
      use_persistent_trace || normal_ab_reference;
  if (prompt_prefill_outside_timed_region) {
    prefill_homogeneous_prompt_prefix(
        context, weights, state, request, max_slot_count,
        effective_execution_policy);
  }
  if (use_persistent_trace) {
    persistent_initial_tokens.reserve(request.requests.size());
    persistent_lifecycle_states.reserve(request.requests.size());
    for (const FastRequest &current : request.requests) {
      persistent_initial_tokens.push_back(current.prompt_tokens.back());
      auto lifecycle = rwkv7::persistent_decode::initial_state(
          static_cast<std::uint32_t>(current.state_slot));
      auto admitted = rwkv7::persistent_decode::transition(
          lifecycle,
          rwkv7::persistent_decode::Event{
              rwkv7::persistent_decode::EventKind::Admit, 0U, 0U, 0U, {}},
          persistent_request_profile);
      require(admitted.accepted,
              "persistent decode lifecycle rejected admission");
      auto started = rwkv7::persistent_decode::transition(
          admitted.state,
          rwkv7::persistent_decode::Event{
              rwkv7::persistent_decode::EventKind::Start, 0U, 0U, 0U, {}},
          persistent_request_profile);
      require(started.accepted,
              "persistent decode lifecycle rejected start");
      persistent_lifecycle_states.push_back(std::move(started.state));
    }
    std::cout << "persistent-decode-setup: reference-start\n"
              << std::flush;
    ModelState reference_state = state;
    std::vector<std::uint32_t> reference_tokens =
        persistent_initial_tokens;
    persistent_reference_tokens_by_step.reserve(
        persistent_request_profile.limits.maximum_decode_steps);
    for (std::uint32_t step = 0U;
         step < persistent_request_profile.limits.maximum_decode_steps;
         ++step) {
      const auto selected = advance_tokens_fast(
          context, weights, reference_state, reference_tokens, true, step,
          max_slot_count, effective_execution_policy);
      require(selected.has_value(),
              "persistent decode reference did not select a token");
      persistent_reference_tokens_by_step.push_back(*selected);
      reference_tokens = *selected;
    }
    persistent_reference_state_blake3 =
        persistent_model_state_blake3(context, reference_state);
    zero_persistent_model_state(context, reference_state);
    persistent_reference_state_zeroed = true;
    std::cout << "persistent-decode-setup: reference-finished\n"
              << std::flush;
    std::cout << "persistent-decode-setup: trace-capture-start\n"
              << std::flush;
    context.set_profiler_flush_enabled(false);
    const auto persistent_trace_setup_begin =
        std::chrono::steady_clock::now();
    prepared_persistent_trace.emplace(prepare_persistent_decode_trace(
        context, weights, state, persistent_initial_tokens,
        persistent_request_profile.limits.maximum_decode_steps,
        effective_execution_policy));
    const auto persistent_trace_setup_end = std::chrono::steady_clock::now();
    persistent_trace_setup_seconds = elapsed_seconds_between(
        persistent_trace_setup_begin, persistent_trace_setup_end);
    require(std::isfinite(persistent_trace_setup_seconds) &&
                persistent_trace_setup_seconds >= 0.0,
            "persistent decode trace setup time is invalid");
    std::cout << "persistent-decode-setup: trace-capture-finished\n"
              << std::flush;
    if (profiler_attributed && !profiler_trace_only) {
      context.set_profiler_flush_enabled(true);
      context.flush_profiler("persistent-decode-setup");
      context.set_profiler_flush_enabled(false);
    }
  } else if (normal_ab_reference) {
    const std::size_t consumed_prefix_tokens =
        request.requests[kFirstIndex].prompt_tokens.size() - 1U;
    for (FastSlot &slot : slots) {
      slot.prompt_index = consumed_prefix_tokens;
      slot.processed_token_count =
          static_cast<std::uint64_t>(consumed_prefix_tokens);
    }
  }
  const std::size_t cache_before_timed = context.program_cache_entries();
  require(cache_before_timed >= cache_after_warmup,
          "persistent decode preflight lost cached programs");
  Json request_metrics = Json::array();
#ifdef RWKV7_FETCH_QUEUE_DIAGNOSTICS_AVAILABLE
  std::unique_ptr<rwkv7::hardware_counter::Session> hardware_counter_session;
#endif
  if (stall_probe_enabled) {
    reset_fetch_queue_diagnostics();
#ifdef RWKV7_FETCH_QUEUE_DIAGNOSTICS_AVAILABLE
    hardware_counter_session =
        std::make_unique<rwkv7::hardware_counter::Session>();
#endif
  }
  signal_fast_barrier(arguments.timed_ready_barrier_path, "timed ready");
  wait_for_fast_barrier(arguments.timed_start_barrier_path, "timed start");
#ifdef RWKV7_FETCH_QUEUE_DIAGNOSTICS_AVAILABLE
  if (stall_probe_enabled) {
    require(hardware_counter_session != nullptr,
            "stall probe omitted its hardware counter session");
    hardware_counter_session->start();
  }
#endif
  const auto timed_start_wall_clock = std::chrono::system_clock::now();
  const auto timed_start = std::chrono::steady_clock::now();
  const std::uint64_t timed_start_monotonic_ns =
      monotonic_nanoseconds(timed_start);
  const std::uint64_t timed_start_unix_ns =
      unix_nanoseconds(timed_start_wall_clock);
  for (FastSlot &slot : slots) {
    slot.started_at = timed_start;
  }
  std::size_t completed_count = 0U;
  std::uint64_t batch_step_count = 0U;
  // r[impl tenstorrent.rwkv7_p150x2.scalable_card_local_batching.valid]
  // Host gap is scheduler overhead: the pure host time spent planning and
  // applying one batch step, excluding the device `advance_tokens_fast` call.
  double host_gap_accumulated_seconds = 0.0;
  // B16 stall probe: when RWKV7_STALL_PROBE_STEPS is set, record each decode
  // step's wall time so an intermittent slow step becomes attributable.
  std::vector<double> step_wall_seconds;
  std::vector<FastStepBoundaryTiming> step_boundary_timings;
  double persistent_trace_execution_seconds = 0.0;
  std::uint64_t persistent_host_trace_submission_count = 0U;
  bool persistent_state_zeroed = false;
  std::string final_state_blake3;
  if (stall_probe_enabled) {
    step_wall_seconds.reserve(kFastMaximumGeneratedTokens);
    step_boundary_timings.reserve(kFastMaximumGeneratedTokens);
  }
  if (use_persistent_trace) {
    const auto apply_lifecycle_event =
        [&persistent_request_profile](
            rwkv7::persistent_decode::RequestState &lifecycle,
            const rwkv7::persistent_decode::Event &event) {
          auto result = rwkv7::persistent_decode::transition(
              lifecycle, event, persistent_request_profile);
          require(result.accepted,
                  "persistent decode lifecycle rejected a runtime event");
          lifecycle = std::move(result.state);
        };
    if (terminal_probe_enabled) {
      require(prepared_persistent_trace.has_value(),
              "persistent decode trace was not prepared before its terminal poll");
      const rwkv7::persistent_decode::Event terminal_event =
          persistent_decode_terminal_probe_event(
              terminal_probe, persistent_request_profile.limits, 0U);
      for (auto &lifecycle : persistent_lifecycle_states) {
        apply_lifecycle_event(lifecycle, terminal_event);
      }
      const auto persistent_cleanup_begin = std::chrono::steady_clock::now();
      for (auto &lifecycle : persistent_lifecycle_states) {
        apply_lifecycle_event(
            lifecycle,
            rwkv7::persistent_decode::Event{
                rwkv7::persistent_decode::EventKind::Acknowledge, 0U, 0U,
                0U, {}});
      }
      prepared_persistent_trace->trace->release();
      const auto terminal_poll_end = std::chrono::steady_clock::now();
      persistent_trace_execution_end = terminal_poll_end;
      persistent_trace_execution_end_wall_clock =
          std::chrono::system_clock::now();
      zero_persistent_model_state(
          context, prepared_persistent_trace->initial_state);
      zero_persistent_model_state(
          context, prepared_persistent_trace->captured_state);
      zero_persistent_model_state(context, state);
      persistent_state_zeroed = true;
      for (auto &lifecycle : persistent_lifecycle_states) {
        apply_lifecycle_event(
            lifecycle,
            rwkv7::persistent_decode::Event{
                rwkv7::persistent_decode::EventKind::StateZeroed, 0U, 0U,
                0U, {}});
        require(rwkv7::persistent_decode::slot_releasable(lifecycle),
                "persistent terminal probe did not release a zeroed slot");
      }
      persistent_cleanup_seconds = elapsed_seconds_between(
          persistent_cleanup_begin, std::chrono::steady_clock::now());
      require(std::isfinite(persistent_cleanup_seconds) &&
                  persistent_cleanup_seconds >= 0.0,
              "persistent terminal cleanup time is invalid");
      for (std::size_t slot_index = 0U; slot_index < slots.size();
           ++slot_index) {
        FastSlot &slot = slots[slot_index];
        slot.prompt_index = slot.request->prompt_tokens.size();
        slot.processed_token_count =
            static_cast<std::uint64_t>(slot.prompt_index);
        completed[slot_index] = true;
        request_metrics.push_back(
            Json{{"request_id", slot.request->request_id},
                 {"card_id", arguments.physical_device_id},
                 {"state_slot", slot.request->state_slot},
                 {"token_count", slot.processed_token_count},
                 {"latency_seconds",
                  std::chrono::duration<double>(terminal_poll_end -
                                                slot.started_at)
                      .count()},
                 {"generated_token_ids", Json::array()}});
        ++completed_count;
      }
    } else {
    try {
      require(prepared_persistent_trace.has_value(),
              "persistent decode trace was not prepared before timing");
      PersistentTraceResult trace_result = execute_persistent_decode_trace(
          context, *prepared_persistent_trace);
      state = std::move(trace_result.final_state);
      final_state_blake3 = persistent_model_state_blake3(context, state);
      require(trace_result.selected_tokens_by_step ==
                  persistent_reference_tokens_by_step,
              "persistent decode token sequence differs from normal dispatch");
      require(final_state_blake3 == persistent_reference_state_blake3,
              "persistent decode FP32 recurrent state differs from normal dispatch");
      persistent_trace_execution_seconds = trace_result.execution_seconds;
      persistent_trace_execution_end = trace_result.execution_end;
      persistent_trace_execution_end_wall_clock =
          trace_result.execution_end_wall_clock;
      persistent_host_trace_submission_count =
          trace_result.host_trace_submission_count;
      require(trace_result.selected_tokens_by_step.size() ==
                  persistent_request_profile.limits.maximum_decode_steps,
              "persistent decode output step count changed");
      const auto elapsed_milliseconds_signed =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::duration<double>(
                  persistent_trace_execution_seconds))
              .count();
      require(elapsed_milliseconds_signed >= 0,
              "persistent decode elapsed time became negative");
      const std::uint64_t elapsed_milliseconds = std::max<std::uint64_t>(
          1U, static_cast<std::uint64_t>(elapsed_milliseconds_signed));
      require(elapsed_milliseconds <
                  persistent_request_profile.limits
                      .maximum_elapsed_milliseconds,
              "persistent decode trace exceeded its time limit");
      const std::uint32_t lifecycle_step_limit =
          terminal_probe_enabled
              ? persistent_request_profile.limits
                    .cancellation_poll_interval_steps
              : persistent_request_profile.limits.maximum_decode_steps;
      for (std::uint32_t step = 1U; step <= lifecycle_step_limit; ++step) {
        const std::uint64_t step_elapsed =
            elapsed_milliseconds * step /
            persistent_request_profile.limits.maximum_decode_steps;
        for (auto &lifecycle : persistent_lifecycle_states) {
          apply_lifecycle_event(
              lifecycle,
              rwkv7::persistent_decode::Event{
                  rwkv7::persistent_decode::EventKind::StepCompleted, step,
                  step_elapsed, step, {}});
        }
      }
      if (terminal_probe_enabled) {
        const rwkv7::persistent_decode::Event terminal_event =
            persistent_decode_terminal_probe_event(
                terminal_probe, persistent_request_profile.limits,
                elapsed_milliseconds);
        for (auto &lifecycle : persistent_lifecycle_states) {
          apply_lifecycle_event(lifecycle, terminal_event);
        }
      }
      const auto persistent_cleanup_begin = std::chrono::steady_clock::now();
      for (auto &lifecycle : persistent_lifecycle_states) {
        apply_lifecycle_event(
            lifecycle,
            rwkv7::persistent_decode::Event{
                rwkv7::persistent_decode::EventKind::Acknowledge, 0U, 0U,
                0U, {}});
      }
      zero_persistent_model_state(context, state);
      persistent_state_zeroed = true;
      for (auto &lifecycle : persistent_lifecycle_states) {
        apply_lifecycle_event(
            lifecycle,
            rwkv7::persistent_decode::Event{
                rwkv7::persistent_decode::EventKind::StateZeroed, 0U, 0U,
                0U, {}});
        require(rwkv7::persistent_decode::slot_releasable(lifecycle),
                "persistent decode did not release a zeroed slot");
      }
      persistent_cleanup_seconds = elapsed_seconds_between(
          persistent_cleanup_begin, std::chrono::steady_clock::now());
      require(std::isfinite(persistent_cleanup_seconds) &&
                  persistent_cleanup_seconds >= 0.0,
              "persistent decode cleanup time is invalid");
      const auto completed_at = std::chrono::steady_clock::now();
      for (std::size_t slot_index = 0U; slot_index < slots.size();
           ++slot_index) {
        FastSlot &slot = slots[slot_index];
        for (const auto &step_tokens :
             trace_result.selected_tokens_by_step) {
          require(step_tokens.size() == slots.size(),
                  "persistent decode output batch size changed");
          if (!terminal_probe_enabled) {
            slot.generated_tokens.push_back(step_tokens[slot_index]);
          }
        }
        slot.prompt_index = slot.request->prompt_tokens.size();
        slot.processed_token_count =
            static_cast<std::uint64_t>(slot.prompt_index) +
            slot.generated_tokens.size();
        completed[slot_index] = true;
        const double latency_seconds = std::chrono::duration<double>(
                                           completed_at - slot.started_at)
                                           .count();
        require(std::isfinite(latency_seconds) && latency_seconds > 0.0,
                "persistent request latency is not finite and positive");
        request_metrics.push_back(
            Json{{"request_id", slot.request->request_id},
                 {"card_id", arguments.physical_device_id},
                 {"state_slot", slot.request->state_slot},
                 {"token_count", slot.processed_token_count},
                 {"latency_seconds", latency_seconds},
                 {"generated_token_ids", slot.generated_tokens}});
        ++completed_count;
      }
      batch_step_count =
          persistent_request_profile.limits.maximum_decode_steps;
    } catch (...) {
      for (auto &lifecycle : persistent_lifecycle_states) {
        if (lifecycle.phase ==
            rwkv7::persistent_decode::RequestPhase::DeviceActive) {
          apply_lifecycle_event(
              lifecycle,
              rwkv7::persistent_decode::Event{
                  rwkv7::persistent_decode::EventKind::Fail, 0U, 0U, 0U,
                  "persistent_trace_failure"});
        }
        if (lifecycle.phase ==
                rwkv7::persistent_decode::RequestPhase::CompletionPending ||
            lifecycle.phase ==
                rwkv7::persistent_decode::RequestPhase::FailurePending ||
            lifecycle.phase ==
                rwkv7::persistent_decode::RequestPhase::TimeoutPending ||
            lifecycle.phase ==
                rwkv7::persistent_decode::RequestPhase::CancellationPending) {
          apply_lifecycle_event(
              lifecycle,
              rwkv7::persistent_decode::Event{
                  rwkv7::persistent_decode::EventKind::Acknowledge, 0U, 0U,
                  0U, {}});
        }
      }
      if (prepared_persistent_trace.has_value() &&
          prepared_persistent_trace->captured_state.layers.size() ==
              kLayerCount) {
        zero_persistent_model_state(
            context, prepared_persistent_trace->captured_state);
      }
      zero_persistent_model_state(context, state);
      for (auto &lifecycle : persistent_lifecycle_states) {
        if (lifecycle.phase ==
            rwkv7::persistent_decode::RequestPhase::Zeroing) {
          apply_lifecycle_event(
              lifecycle,
              rwkv7::persistent_decode::Event{
                  rwkv7::persistent_decode::EventKind::StateZeroed, 0U, 0U,
                  0U, {}});
        }
      }
      throw;
    }
    }
  } else {
    while (completed_count < slots.size()) {
      const std::vector<bool> completed_before = completed;
      const auto plan_begin = std::chrono::steady_clock::now();
      const FastBatchStepPlan plan =
          plan_fast_batch_step(slots, completed, max_slot_count);
      const auto plan_end = std::chrono::steady_clock::now();
      const auto device_begin = plan_end;
      FastStepBoundaryTiming step_boundary_timing;
      FastStepBoundaryTiming *const step_boundary_timing_output =
          stall_probe_enabled ? &step_boundary_timing : nullptr;
      const auto selected = advance_tokens_fast(
          context, weights, state, plan.input_tokens,
          plan.any_selection_required, batch_step_count, max_slot_count,
          effective_execution_policy, step_boundary_timing_output);
      const auto device_end = std::chrono::steady_clock::now();
      const auto apply_begin = device_end;
      apply_fast_batch_step(slots, completed, plan, selected);
      ++batch_step_count;
      const auto apply_end = std::chrono::steady_clock::now();
      if (stall_probe_enabled) {
        step_wall_seconds.push_back(std::chrono::duration<double>(
            device_end - device_begin)
                                        .count());
        require(std::isfinite(step_wall_seconds.back()) &&
                    step_wall_seconds.back() >= 0.0,
                "stall probe step wall time is not finite and non-negative");
        step_boundary_timings.push_back(std::move(step_boundary_timing));
      }
      const double host_gap_seconds =
          std::chrono::duration<double>(plan_end - plan_begin).count() +
          std::chrono::duration<double>(apply_end - apply_begin).count();
      require(std::isfinite(host_gap_seconds) && host_gap_seconds >= 0.0,
              "fast host gap is not finite and non-negative");
      host_gap_accumulated_seconds += host_gap_seconds;
      const auto completed_at = apply_end;
      for (std::size_t slot_index = 0U; slot_index < slots.size();
           ++slot_index) {
        if (completed_before[slot_index] || !completed[slot_index]) {
          continue;
        }
        FastSlot &slot = slots[slot_index];
        const double latency_seconds = std::chrono::duration<double>(
                                           completed_at - slot.started_at)
                                           .count();
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
  }
  const auto timed_end = use_persistent_trace
                             ? persistent_trace_execution_end.value()
                             : std::chrono::steady_clock::now();
  const auto timed_end_wall_clock =
      use_persistent_trace
          ? persistent_trace_execution_end_wall_clock.value()
          : std::chrono::system_clock::now();
#ifdef RWKV7_FETCH_QUEUE_DIAGNOSTICS_AVAILABLE
  std::optional<rwkv7::hardware_counter::Diagnostics>
      hardware_counter_diagnostics;
  if (stall_probe_enabled) {
    require(hardware_counter_session != nullptr,
            "stall probe lost its hardware counter session");
    hardware_counter_diagnostics = hardware_counter_session->stop();
    if (primary_thread_cpu_affinity.has_value()) {
      require(hardware_counter_diagnostics->cpu_migrations.value == 0U,
              "pinned primary thread migrated during timed decode");
    }
  }
#endif
  if (primary_thread_fifo_priority.has_value()) {
    restore_current_thread_normal_scheduling();
  }
  std::optional<FetchQueueDiagnostics> fetch_queue_diagnostics;
  if (stall_probe_enabled) {
    fetch_queue_diagnostics = read_fetch_queue_diagnostics();
    validate_fetch_queue_diagnostics(*fetch_queue_diagnostics);
  }
  const std::uint64_t timed_end_monotonic_ns = monotonic_nanoseconds(timed_end);
  const std::uint64_t timed_end_unix_ns =
      unix_nanoseconds(timed_end_wall_clock);
  require(timed_end_monotonic_ns > timed_start_monotonic_ns,
          "fast worker monotonic timing range is not positive");
  require(timed_end_unix_ns > timed_start_unix_ns,
          "fast worker wall-clock timing range is not positive");
  const double elapsed_seconds =
      std::chrono::duration<double>(timed_end - timed_start).count();
  require(std::isfinite(elapsed_seconds) && elapsed_seconds > 0.0,
          "fast worker elapsed time is not finite and positive");
  require(context.program_cache_entries() == cache_before_timed,
          "fast worker compiled a new program during timed decode");
  if (stall_probe_enabled) {
    require(step_wall_seconds.size() == batch_step_count,
            "stall probe step timing count differs from batch step count");
    require(step_boundary_timings.size() == batch_step_count,
            "stall probe boundary timing count differs from batch step count");
  }
  const double queue_occupancy_fraction =
      static_cast<double>(request.requests.size()) /
      static_cast<double>(max_slot_count);
  require(std::isfinite(queue_occupancy_fraction) &&
              queue_occupancy_fraction > 0.0 &&
              queue_occupancy_fraction <= 1.0 + 1.0e-9,
          "fast queue occupancy is not within (0, 1]");

  context.set_profiler_flush_enabled(profiler_attributed);
  context.flush_profiler("fast-worker-post-timing-attribution",
                         profiler_trace_only);
  Json receipt = Json::object();
  receipt["schema_version"] = kSchemaVersion;
  receipt["run_id"] = request.run_id;
  receipt["physical_device_ids"] = {arguments.physical_device_id};
  receipt["visible_physical_device_ids"] = {arguments.physical_device_id};
  receipt["logical_device_ids"] = {kFilteredLogicalDevice};
  receipt["physical_hardware"] = !simulated;
  receipt["profiler_attributed"] = profiler_attributed;
  receipt["profiler_trace_only"] = profiler_trace_only;
  receipt["elapsed_seconds"] = elapsed_seconds;
  receipt["timed_start_monotonic_ns"] = timed_start_monotonic_ns;
  receipt["timed_end_monotonic_ns"] = timed_end_monotonic_ns;
  receipt["timed_start_unix_ns"] = timed_start_unix_ns;
  receipt["timed_end_unix_ns"] = timed_end_unix_ns;
  receipt["capacity"] = {
      {"id", request.capacity.id},
      {"per_card_slot_count", max_slot_count},
      {"memory_plan_bytes",
       {{"recurrent_state_bytes", memory_plan.recurrent_state_bytes},
        {"token_shift_bytes", memory_plan.token_shift_bytes},
        {"temporary_tensor_bytes", memory_plan.temporary_tensor_bytes},
        {"safety_reserve_bytes", memory_plan.safety_reserve_bytes},
        {"total_required_bytes", memory_plan.total_required_bytes},
        {"device_budget_bytes", memory_plan.device_budget_bytes}}}};
  receipt["effective_batch_size"] = request.requests.size();
  receipt["queue_occupancy_fraction"] = queue_occupancy_fraction;
  receipt["batch_step_count"] = batch_step_count;
  receipt["host_gap_accumulated_seconds"] =
      host_gap_accumulated_seconds;
  receipt["primary_thread_cpu_affinity"] =
      primary_thread_cpu_affinity.has_value()
          ? Json(*primary_thread_cpu_affinity)
          : Json(nullptr);
  receipt["primary_thread_fifo_priority"] =
      primary_thread_fifo_priority.has_value()
          ? Json(*primary_thread_fifo_priority)
          : Json(nullptr);
  if (stall_probe_enabled) {
    receipt["stall_probe_step_wall_seconds"] = step_wall_seconds;
    Json boundary_timings = Json::array();
    for (const FastStepBoundaryTiming &timing : step_boundary_timings) {
      boundary_timings.push_back(
          Json{{"selection_requested", timing.selection_requested},
               {"model_submission_seconds", timing.model_submission_seconds},
               {"layer_submission_seconds",
                timing.layer_submission_seconds},
               {"model_submission_thread_cpu_seconds",
                timing.model_submission_thread_cpu_seconds},
               {"layer_submission_thread_cpu_seconds",
                timing.layer_submission_thread_cpu_seconds},
               {"model_submission_process_cpu_seconds",
                timing.model_submission_process_cpu_seconds},
               {"layer_submission_process_cpu_seconds",
                timing.layer_submission_process_cpu_seconds},
               {"blocking_readback_seconds",
                timing.blocking_readback_seconds},
               {"host_materialization_seconds",
                timing.host_materialization_seconds},
               {"readback_validation_seconds",
                timing.readback_validation_seconds},
               {"total_seconds", timing.total_seconds}});
    }
    receipt["stall_probe_step_boundary_timings"] =
        std::move(boundary_timings);
    receipt["fetch_queue_diagnostics"] = {
        {"reserve_calls", fetch_queue_diagnostics->reserve_calls},
        {"wait_calls", fetch_queue_diagnostics->wait_calls},
        {"poll_iterations", fetch_queue_diagnostics->poll_iterations},
        {"total_wait_nanoseconds",
         fetch_queue_diagnostics->total_wait_nanoseconds},
        {"maximum_wait_nanoseconds",
         fetch_queue_diagnostics->maximum_wait_nanoseconds}};
#ifdef RWKV7_FETCH_QUEUE_DIAGNOSTICS_AVAILABLE
    require(hardware_counter_diagnostics.has_value(),
            "stall probe omitted hardware counter diagnostics");
    receipt["hardware_counter_diagnostics"] =
        hardware_counter_diagnostics_json(*hardware_counter_diagnostics);
#endif
  }
  receipt["requests"] = std::move(request_metrics);
  Json persistent_lifecycle_receipts = Json::array();
  for (const auto &lifecycle : persistent_lifecycle_states) {
    persistent_lifecycle_receipts.push_back(
        Json{{"slot", lifecycle.slot},
             {"phase",
              std::string(rwkv7::persistent_decode::request_phase_name(
                  lifecycle.phase))},
             {"terminal_reason",
              std::string(rwkv7::persistent_decode::terminal_reason_name(
                  lifecycle.terminal_reason))},
             {"completed_steps", lifecycle.completed_steps},
             {"failure_code", lifecycle.failure_code},
             {"slot_releasable",
              rwkv7::persistent_decode::slot_releasable(lifecycle)},
             {"receipt_blake3",
              rwkv7::persistent_decode::derive_receipt_identity(
                  persistent_profile_identity, actual_persistent_bindings,
                  lifecycle)}});
  }
  const std::uint64_t timed_host_model_submissions =
      use_persistent_trace ? persistent_host_trace_submission_count
                           : batch_step_count;
  const bool zero_per_token_host_submissions_proven =
      use_persistent_trace &&
      persistent_host_trace_submission_count == 1U &&
      batch_step_count > persistent_host_trace_submission_count;
  const bool persistent_trace_replayed =
      use_persistent_trace && !terminal_probe_enabled &&
      persistent_host_trace_submission_count == 1U;
  const bool exact_generated_token_parity = persistent_trace_replayed;
  const bool exact_fp32_recurrent_state_parity = persistent_trace_replayed;
  const rwkv7::persistent_decode::PersistentTelemetry persistent_telemetry =
      rwkv7::persistent_decode::summarize_persistent_telemetry(
          rwkv7::persistent_decode::PersistentTelemetryInput{
              window_admitted,
              candidate_eligible,
              binding_validation.valid,
              persistent_profile.profile.physically_admitted,
              candidate_requested,
              selected_persistent_limits.has_value()
                  ? std::optional<std::uint32_t>(
                        selected_persistent_limits->maximum_decode_steps)
                  : std::nullopt,
              execution_decision,
              std::span<const rwkv7::persistent_decode::RequestState>(
                  persistent_lifecycle_states.data(),
                  persistent_lifecycle_states.size()),
              persistent_trace_replayed,
              exact_generated_token_parity,
              exact_fp32_recurrent_state_parity,
              persistent_trace_setup_seconds,
              persistent_trace_execution_seconds,
              persistent_cleanup_seconds});
  const rwkv7::persistent_decode::ValidationResult telemetry_validation =
      rwkv7::persistent_decode::validate_persistent_telemetry(
          persistent_telemetry);
  require(telemetry_validation.valid, telemetry_validation.reason);
  receipt["persistent_decode_telemetry"] =
      persistent_telemetry_json(persistent_telemetry);
  receipt["persistent_decode"] = {
      {"profile_id", persistent_profile.profile.profile_id},
      {"profile_blake3", persistent_profile_identity},
      {"admitted_decode_windows",
       persistent_profile.profile.limits.admitted_decode_windows},
      {"selected_decode_window_steps",
       selected_persistent_limits.has_value()
           ? Json(selected_persistent_limits->maximum_decode_steps)
           : Json(nullptr)},
      {"model_profile_blake3",
       persistent_profile.profile.model_profile_blake3},
      {"hash_algorithm", "blake3"},
      {"mechanism", persistent_profile.profile.mechanism},
      {"simulated", simulated},
      {"physically_admitted",
       persistent_profile.profile.physically_admitted},
      {"candidate_requested", candidate_requested},
      {"canary_interlock_present", canary_interlock.present},
      {"canary_interlock_valid", canary_interlock.decision.valid},
      {"canary_interlock_state",
       std::string(rwkv7::persistent_decode::canary_interlock_state_name(
           canary_interlock.state))},
      {"canary_interlock_allows_persistent",
       canary_interlock.decision.allows_persistent},
      {"canary_interlock_file_blake3",
       canary_interlock.file_blake3.empty()
           ? Json(nullptr)
           : Json(canary_interlock.file_blake3)},
      {"canary_binding_blake3",
       canary_interlock.canary_binding_blake3.empty()
           ? Json(nullptr)
           : Json(canary_interlock.canary_binding_blake3)},
      {"terminal_probe",
       std::string(persistent_decode_terminal_probe_name(terminal_probe))},
      {"terminal_event_poll_step",
       terminal_probe_enabled ? Json(0U) : Json(nullptr)},
      {"cancellation_poll_interval_steps",
       persistent_request_profile.limits
           .cancellation_poll_interval_steps},
      {"pre_replay_terminal_event", terminal_probe_enabled},
      {"window_admitted", window_admitted},
      {"workload_eligible", candidate_eligible},
      {"prompt_prefill_outside_timed_region",
       prompt_prefill_outside_timed_region},
      {"binding_identities",
       persistent_binding_json(actual_persistent_bindings)},
      {"bindings_validated", binding_validation.valid},
      {"selected_mode",
       std::string(rwkv7::persistent_decode::execution_mode_name(
           execution_decision.mode))},
      {"fallback_reason",
       std::string(rwkv7::persistent_decode::fallback_reason_name(
           execution_decision.fallback_reason))},
      {"fallback_detail", execution_decision.detail},
      {"persistent_region_entered",
       use_persistent_trace && !terminal_probe_enabled},
      {"trace_region_bytes", trace_region_size},
      {"trace_capture_outside_timed_region", use_persistent_trace},
      {"trace_execution_seconds", persistent_trace_execution_seconds},
      {"timed_host_model_submissions", timed_host_model_submissions},
      {"zero_per_token_host_submissions_proven",
       zero_per_token_host_submissions_proven},
      {"selected_token_device_resident_inside_region",
       use_persistent_trace && !terminal_probe_enabled},
      {"selected_token_readback_inside_region", false},
      {"output_readback_after_region",
       use_persistent_trace && !terminal_probe_enabled},
      {"recurrent_state_dtype", "fp32"},
      {"final_state_blake3",
       final_state_blake3.empty() ? Json(nullptr) : Json(final_state_blake3)},
      {"reference_state_blake3",
       persistent_reference_state_blake3.empty()
           ? Json(nullptr)
           : Json(persistent_reference_state_blake3)},
      {"exact_generated_token_parity", exact_generated_token_parity},
      {"exact_fp32_recurrent_state_parity",
       exact_fp32_recurrent_state_parity},
      {"reference_state_zeroed", persistent_reference_state_zeroed},
      {"state_zeroed_before_slot_release", persistent_state_zeroed},
      {"lifecycle_receipts", std::move(persistent_lifecycle_receipts)}};
  receipt["execution"] = {
      {"scheduler_mode", "deterministic_continuous_batching"},
      {"scheduler_tie_break",
       "least_loaded_card_then_lowest_card_then_lowest_slot"},
      {"replica_mode", "independent_card_replica"},
      {"recurrent_state_residency", "device_fp32"},
      {"timed_region", "decode_only"},
      {"warmup_token_count", warmup_token_count},
      {"program_cache_stable", true},
      {"dynamic_token_and_state_bindings_validated", true},
      {"state_readback_during_timing", false},
      {"profiler_flush_during_timing", false},
      {"control_replay_during_timing", false},
      {"artifact_writes_during_timing", false},
      {"inspector_during_timing", false},
      {"fetch_queue_diagnostics_during_timing", stall_probe_enabled},
      {"hardware_counters_during_timing", stall_probe_enabled},
      {"primary_thread_only_affinity",
       primary_thread_cpu_affinity.has_value()},
      {"primary_thread_fifo_scheduling",
       primary_thread_fifo_priority.has_value()},
      {"synchronized_timed_start", true},
      {"wkv_execution_policy",
       wkv_execution_policy_name(effective_execution_policy)},
      {"production_physically_admitted", production_physically_admitted}};
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
  require(parse_wkv_execution_policy(kProductionWkvPolicy) ==
              WkvExecutionPolicy::Production,
          "production WKV policy parser changed");
  require(parse_wkv_execution_policy(kFusedCandidateWkvPolicy) ==
              WkvExecutionPolicy::RequireFusedCandidate,
          "fused candidate WKV policy parser changed");
  require(parse_wkv_execution_policy(kGenericReferenceWkvPolicy) ==
              WkvExecutionPolicy::RequireGenericReference,
          "generic reference WKV policy parser changed");
  expect_operation_failure(
      []() { static_cast<void>(parse_wkv_execution_policy("invalid")); },
      "must be production, fused-candidate, or generic-reference");
  require(production_scheduling_physically_admitted(
              kPhysicalDeviceZero, kAdmittedCardZeroPrimaryThreadCpu,
              kAdmittedPrimaryThreadFifoPriority),
          "admitted card-zero scheduling contract changed");
  require(production_scheduling_physically_admitted(
              kPhysicalDeviceOne, kAdmittedCardOnePrimaryThreadCpu,
              kAdmittedPrimaryThreadFifoPriority),
          "admitted card-one scheduling contract changed");
  require(!production_scheduling_physically_admitted(
              kPhysicalDeviceZero, kAdmittedCardOnePrimaryThreadCpu,
              kAdmittedPrimaryThreadFifoPriority),
          "wrong primary CPU entered production admission");
  require(!production_scheduling_physically_admitted(
              kPhysicalDeviceZero, kAdmittedCardZeroPrimaryThreadCpu,
              std::nullopt),
          "missing FIFO priority entered production admission");
  require(effective_wkv_execution_policy(WkvExecutionPolicy::Production,
                                         true) ==
              WkvExecutionPolicy::ProductionPhysicallyAdmitted,
          "production scheduling did not open admitted dispatch");
  require(effective_wkv_execution_policy(
              WkvExecutionPolicy::RequireFusedCandidate, true) ==
              WkvExecutionPolicy::RequireFusedCandidate,
          "candidate policy changed under production admission");
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
  const std::size_t fast_self_test_capacity = fast_slots.size();
  const FastBatchStepPlan initial_plan =
      plan_fast_batch_step(fast_slots, fast_completed, fast_self_test_capacity);
  require(initial_plan.input_tokens ==
              std::vector<std::uint32_t>{kFirstPromptToken,
                                         kFirstPromptToken} &&
              !initial_plan.any_selection_required,
          "positive fast batch planner self-test changed initial prompt plan");

  fast_slots[kFirstIndex].prompt_index = 1U;
  fast_slots[kSecondIndex].prompt_index = 2U;
  const FastBatchStepPlan final_prompt_plan =
      plan_fast_batch_step(fast_slots, fast_completed, fast_self_test_capacity);
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
      plan_fast_batch_step(fast_slots, fast_completed, fast_self_test_capacity);
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
        static_cast<void>(plan_fast_batch_step(
            fast_slots, std::vector<bool>{false}, fast_slots.size()));
      },
      "completion count differs from slot count");
  expect_operation_failure(
      [&fast_slots]() {
        std::vector<bool> incomplete(fast_slots.size(), false);
        const FastBatchStepPlan plan = plan_fast_batch_step(
            fast_slots, incomplete, fast_slots.size());
        apply_fast_batch_step(fast_slots, incomplete, plan, std::nullopt);
      },
      "selection result does not match the plan");
  expect_operation_failure(
      [&fast_slots]() {
        // A plan with more slots than the admitted capacity must fail before
        // any decode dispatch.
        std::vector<bool> incomplete(fast_slots.size(), false);
        static_cast<void>(plan_fast_batch_step(
            fast_slots, incomplete, fast_slots.size() - 1U));
      },
      "exceeds the admitted card-local capacity");

  constexpr double kBoundaryTestModelSeconds = 0.3;
  constexpr double kBoundaryTestLayerSeconds = 0.01;
  constexpr double kBoundaryTestReadbackSeconds = 0.02;
  constexpr double kBoundaryTestReadbackComponentCount = 3.0;
  FastStepBoundaryTiming valid_boundary_timing{
      true,
      kBoundaryTestModelSeconds,
      std::vector<double>(kLayerCount, kBoundaryTestLayerSeconds),
      kBoundaryTestReadbackSeconds,
      kBoundaryTestReadbackSeconds,
      kBoundaryTestReadbackSeconds,
      kBoundaryTestModelSeconds +
          kBoundaryTestReadbackComponentCount * kBoundaryTestReadbackSeconds,
      kBoundaryTestModelSeconds,
      std::vector<double>(kLayerCount, kBoundaryTestLayerSeconds),
      kBoundaryTestModelSeconds,
      std::vector<double>(kLayerCount, kBoundaryTestLayerSeconds)};
  validate_fast_step_boundary_timing(valid_boundary_timing);
  FastStepBoundaryTiming invalid_boundary_timing = valid_boundary_timing;
  invalid_boundary_timing.layer_submission_seconds.pop_back();
  expect_operation_failure(
      [&invalid_boundary_timing]() {
        validate_fast_step_boundary_timing(invalid_boundary_timing);
      },
      "layer count is invalid");
  invalid_boundary_timing = valid_boundary_timing;
  invalid_boundary_timing.total_seconds =
      std::numeric_limits<double>::quiet_NaN();
  expect_operation_failure(
      [&invalid_boundary_timing]() {
        validate_fast_step_boundary_timing(invalid_boundary_timing);
      },
      "not finite and non-negative");

  constexpr std::uint64_t kDiagnosticsTestReserveCalls = 4U;
  constexpr std::uint64_t kDiagnosticsTestWaitCalls = 2U;
  constexpr std::uint64_t kDiagnosticsTestPollIterations = 8U;
  constexpr std::uint64_t kDiagnosticsTestTotalWaitNanoseconds = 100U;
  constexpr std::uint64_t kDiagnosticsTestMaximumWaitNanoseconds = 75U;
  const FetchQueueDiagnostics valid_fetch_queue_diagnostics{
          kDiagnosticsTestReserveCalls, kDiagnosticsTestWaitCalls,
          kDiagnosticsTestPollIterations,
          kDiagnosticsTestTotalWaitNanoseconds,
          kDiagnosticsTestMaximumWaitNanoseconds};
  validate_fetch_queue_diagnostics(valid_fetch_queue_diagnostics);
  auto invalid_fetch_queue_diagnostics = valid_fetch_queue_diagnostics;
  invalid_fetch_queue_diagnostics.poll_iterations = 0U;
  expect_operation_failure(
      [&invalid_fetch_queue_diagnostics]() {
        validate_fetch_queue_diagnostics(invalid_fetch_queue_diagnostics);
      },
      "too few poll iterations");

#ifdef RWKV7_FETCH_QUEUE_DIAGNOSTICS_AVAILABLE
  constexpr std::uint64_t kCounterTestValue = 100U;
  constexpr std::uint64_t kCounterTestMissValue = 25U;
  constexpr std::uint64_t kCounterTestTimeNanoseconds = 1'000U;
  const rwkv7::hardware_counter::Reading counter_test_reading{
      kCounterTestValue, kCounterTestTimeNanoseconds,
      kCounterTestTimeNanoseconds};
  rwkv7::hardware_counter::Diagnostics valid_hardware_counters{
      counter_test_reading,
      counter_test_reading,
      counter_test_reading,
      rwkv7::hardware_counter::Reading{kCounterTestMissValue,
                                       kCounterTestTimeNanoseconds,
                                       kCounterTestTimeNanoseconds},
      counter_test_reading,
      counter_test_reading,
      counter_test_reading,
      counter_test_reading,
      counter_test_reading};
  rwkv7::hardware_counter::validate(valid_hardware_counters);
  auto invalid_hardware_counters = valid_hardware_counters;
  invalid_hardware_counters.instructions.value = 0U;
  expect_operation_failure(
      [&invalid_hardware_counters]() {
        rwkv7::hardware_counter::validate(invalid_hardware_counters);
      },
      "instructions counter is zero");
  invalid_hardware_counters = valid_hardware_counters;
  invalid_hardware_counters.cycles.time_running_nanoseconds = 0U;
  expect_operation_failure(
      [&invalid_hardware_counters]() {
        rwkv7::hardware_counter::validate(invalid_hardware_counters);
      },
      "cycles counter has invalid running time");
#endif

  validate_primary_thread_cpu(0U);
  expect_operation_failure(
      []() {
        validate_primary_thread_cpu(
            static_cast<std::uint32_t>(CPU_SETSIZE));
      },
      "primary thread CPU exceeds CPU_SETSIZE");
  const int minimum_fifo_priority = sched_get_priority_min(SCHED_FIFO);
  const int maximum_fifo_priority = sched_get_priority_max(SCHED_FIFO);
  require(minimum_fifo_priority > 0 &&
              maximum_fifo_priority >= minimum_fifo_priority,
          "FIFO priority self-test bounds are invalid");
  validate_primary_thread_fifo_priority(
      static_cast<std::uint32_t>(minimum_fifo_priority));
  expect_operation_failure(
      [maximum_fifo_priority]() {
        validate_primary_thread_fifo_priority(
            static_cast<std::uint32_t>(maximum_fifo_priority) + 1U);
      },
      "primary thread FIFO priority is outside the scheduler bounds");

  // r[verify tenstorrent.rwkv7_p150x2.scalable_card_local_batching.valid]
  // All four supported candidate capacities plan without a device.
  for (const std::string_view capacity_id :
       {std::string_view("B4"), std::string_view("B16"),
        std::string_view("B32"), std::string_view("B64")}) {
    require(supported_fast_capacity_id(capacity_id, 4U) ||
                supported_fast_capacity_id(capacity_id, 16U) ||
                supported_fast_capacity_id(capacity_id, 32U) ||
                supported_fast_capacity_id(capacity_id, 64U),
            "supported capacity id self-test is incomplete");
  }
  const FastCardCapacity b64_capacity{"B64", 64U, 5U * 1024U * 1024U,
                                      3U * 1024U * 1024U * 1024U,
                                      256U * 1024U * 1024U};
  const FastMemoryPlan b64_plan = plan_fast_card_memory(b64_capacity);
  require(b64_plan.total_required_bytes <= b64_plan.device_budget_bytes,
          "B64 memory plan must fit the pinned budget");
  require(b64_plan.recurrent_state_bytes > 0U &&
              b64_plan.token_shift_bytes > 0U &&
              b64_plan.temporary_tensor_bytes > 0U,
          "B64 memory plan has zero contributions");

  // r[verify tenstorrent.rwkv7_p150x2.scalable_card_local_batching.invalid_memory]
  expect_operation_failure(
      []() {
        static_cast<void>(supported_fast_capacity_id("B8", 8U) ||
                          supported_fast_capacity_id("B4", 8U));
        // Force the unsupported branch so the negative is explicit.
        require(supported_fast_capacity_id("B8", 8U),
                "unsupported capacity must not be admitted");
      },
      "must not be admitted");
  expect_operation_failure(
      []() {
        FastCardCapacity over_budget{"B64", 64U, 5U * 1024U * 1024U,
                                     static_cast<std::uint64_t>(1U) << 20U,
                                     256U * 1024U * 1024U};
        static_cast<void>(plan_fast_card_memory(over_budget));
      },
      "exceeds the device memory budget");
  expect_operation_failure(
      []() {
        // A safety reserve beyond the address space must overflow.
        FastCardCapacity overflowing{"B64", 64U, 5U * 1024U * 1024U,
                                     std::numeric_limits<std::uint64_t>::max(),
                                     std::numeric_limits<std::uint64_t>::max()};
        static_cast<void>(plan_fast_card_memory(overflowing));
      },
      "overflows the address space");

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
  const LoadedPersistentDecodeProfile persistent_profile =
      load_persistent_decode_profile();
  const rwkv7::persistent_decode::ValidationResult precision_binding =
      rwkv7::persistent_decode::validate_precision_binding(
          persistent_profile.profile, kFusedWkvFp32RecurrenceVectors,
          kFusedWkvBfloat16ReadoutBoundary);
  require(precision_binding.valid, precision_binding.reason);
  const std::string persistent_profile_identity =
      rwkv7::persistent_decode::blake3_hex(persistent_profile.bytes);
  require(kPersistentDecodeProfileBlake3.empty() ||
              persistent_profile_identity == kPersistentDecodeProfileBlake3,
          "persistent decode self-test profile identity differs from the runtime");
  require(parse_persistent_decode_control(std::nullopt) ==
              PersistentDecodeControl::ProfileDefault,
          "persistent decode omitted the profile-default control");
  require(parse_persistent_decode_control(kDisabledEnvironmentValue) ==
              PersistentDecodeControl::ForceNormal,
          "persistent decode did not accept the normal A/B control");
  require(parse_persistent_decode_control(kEnabledEnvironmentValue) ==
              PersistentDecodeControl::ForceCandidate,
          "persistent decode did not accept the candidate control");
  expect_operation_failure(
      []() {
        static_cast<void>(parse_persistent_decode_control("invalid"));
      },
      "must be 0 or 1");
  require(parse_persistent_decode_terminal_probe(std::nullopt) ==
              PersistentDecodeTerminalProbe::None &&
              parse_persistent_decode_terminal_probe(
                  kPersistentDecodeCancelProbe) ==
                  PersistentDecodeTerminalProbe::Cancel &&
              parse_persistent_decode_terminal_probe(
                  kPersistentDecodeTimeoutProbe) ==
                  PersistentDecodeTerminalProbe::Timeout &&
              parse_persistent_decode_terminal_probe(
                  kPersistentDecodeFailureProbe) ==
                  PersistentDecodeTerminalProbe::Failure,
          "persistent decode terminal probe parsing changed");
  expect_operation_failure(
      []() {
        static_cast<void>(
            parse_persistent_decode_terminal_probe("invalid"));
      },
      "must be cancel, timeout, or failure");
  const FastWorkerRequest eligible_persistent_request{
      "persistent-self-test",
      FastCardCapacity{"B4", kFastPerCardSlotCount, 1U, 1U, 1U},
      {FastRequest{"persistent-first",
                   {kFirstPromptToken, kSecondPromptToken},
                   persistent_profile.profile.limits.maximum_decode_steps,
                   kFirstIndex},
       FastRequest{"persistent-second",
                   {kFirstPromptToken, kSecondPromptToken},
                   persistent_profile.profile.limits.maximum_decode_steps,
                   kSecondIndex}}};
  require(persistent_workload_eligible(eligible_persistent_request,
                                       persistent_profile.profile),
          "persistent decode rejected an eligible homogeneous workload");
  FastWorkerRequest short_persistent_request = eligible_persistent_request;
  for (FastRequest &request : short_persistent_request.requests) {
    request.max_new_tokens = kMinimumGeneratedTokens;
  }
  require(persistent_workload_eligible(short_persistent_request,
                                       persistent_profile.profile),
          "persistent decode rejected the shorter admitted window");
  FastWorkerRequest unlisted_persistent_request = eligible_persistent_request;
  for (FastRequest &request : unlisted_persistent_request.requests) {
    request.max_new_tokens = kMaximumGeneratedTokens - 1U;
  }
  require(!persistent_workload_eligible(unlisted_persistent_request,
                                        persistent_profile.profile),
          "persistent decode accepted an unlisted window");
  FastWorkerRequest ineligible_persistent_request =
      eligible_persistent_request;
  ineligible_persistent_request.requests[kSecondIndex]
      .prompt_tokens.push_back(kThirdPromptToken);
  require(!persistent_workload_eligible(ineligible_persistent_request,
                                        persistent_profile.profile),
          "persistent decode accepted mixed prompt lengths");
  ineligible_persistent_request = eligible_persistent_request;
  ineligible_persistent_request.requests[kSecondIndex].max_new_tokens =
      persistent_profile.profile.limits.maximum_decode_steps - 1U;
  require(!persistent_workload_eligible(ineligible_persistent_request,
                                        persistent_profile.profile),
          "persistent decode accepted a mismatched output bound");
  rwkv7::fused_wkv_reference::run_fused_wkv_reference_self_tests();
  rwkv7::persistent_decode::run_persistent_decode_self_test(
      persistent_profile.profile, persistent_profile.bytes);
  std::cout << "rwkv7 P150x2 runtime self-test: PASS\n";
}

void run_hardware_counter_smoke() {
#ifdef RWKV7_FETCH_QUEUE_DIAGNOSTICS_AVAILABLE
  constexpr std::size_t kSmokeIterationCount = 10'000U;
  rwkv7::hardware_counter::Session session;
  session.start();
  double clock_sum = 0.0;
  for (std::size_t index = 0U; index < kSmokeIterationCount; ++index) {
    clock_sum += read_cpu_seconds(CLOCK_MONOTONIC);
  }
  const rwkv7::hardware_counter::Diagnostics diagnostics = session.stop();
  require(std::isfinite(clock_sum) && clock_sum > 0.0,
          "hardware counter smoke workload is invalid");
  std::cout << hardware_counter_diagnostics_json(diagnostics).dump(2) << '\n';
#else
  throw std::runtime_error(
      "hardware counter smoke is unavailable in the production runtime");
#endif
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
    if (arguments.size() == kSelfTestArgumentCount &&
        arguments[kModeArgumentIndex] == kHardwareCounterSmokeArgument) {
      run_hardware_counter_smoke();
      return EXIT_SUCCESS;
    }
    if (arguments.size() == kFastWorkerArgumentCount &&
        arguments[kModeArgumentIndex] == kFastWorkerArgument) {
      run_fast_worker(parse_fast_worker_arguments(arguments));
      return EXIT_SUCCESS;
    }
    if (arguments.size() == kSimulatorSmokeArgumentCount &&
        arguments[kModeArgumentIndex] == kSimulatorSmokeArgument) {
      run_simulator_smoke(parse_simulator_smoke_output(arguments),
                          WkvExecutionPolicy::Production);
      return EXIT_SUCCESS;
    }
    if (arguments.size() == kSimulatorSmokeArgumentCount &&
        arguments[kModeArgumentIndex] == kFusedWkvLayerSmokeArgument) {
      run_simulator_smoke(parse_fused_wkv_layer_smoke_output(arguments),
                          WkvExecutionPolicy::RequireFusedCandidate);
      return EXIT_SUCCESS;
    }
    if (arguments.size() == kSimulatorSmokeArgumentCount &&
        arguments[kModeArgumentIndex] == kFusedWkvProbeArgument) {
      run_fused_wkv_probe(parse_fused_wkv_probe_output(arguments));
      return EXIT_SUCCESS;
    }
    if (arguments.size() == kSimulatorSmokeArgumentCount &&
        arguments[kModeArgumentIndex] ==
            kFusedWkvAsymmetricProbeArgument) {
      run_fused_wkv_asymmetric_probe(
          parse_fused_wkv_asymmetric_probe_output(arguments));
      return EXIT_SUCCESS;
    }
    if (arguments.size() == kSimulatorSmokeArgumentCount &&
        arguments[kModeArgumentIndex] ==
            kPersistentDecodeSimulatorSmokeArgument) {
      run_persistent_decode_simulator_smoke(
          parse_persistent_decode_simulator_smoke_output(arguments));
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
