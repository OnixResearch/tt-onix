#include "hardware_counter_diagnostics.hpp"

#include <linux/perf_event.h>

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <sys/ioctl.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace rwkv7::hardware_counter {
namespace {

constexpr pid_t kCurrentThread = 0;
constexpr int kAnyCpu = -1;
constexpr int kNoGroup = -1;
constexpr int kInvalidFileDescriptor = -1;
constexpr unsigned long kGroupIoctlFlag = PERF_IOC_FLAG_GROUP;
constexpr std::uint64_t kReadFormat =
    PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
constexpr double kMinimumRunningFraction = 0.99;
constexpr std::string_view kMsrPmuTypePath =
    "/sys/bus/event_source/devices/msr/type";
constexpr std::uint64_t kAperfEventConfig = 0x01U;
constexpr std::uint64_t kMperfEventConfig = 0x02U;

struct EventSpec {
  std::string_view name;
  std::uint32_t type;
  std::uint64_t config;
  bool inherit;
  bool exclude_hypervisor;
  bool pinned;
};

constexpr std::array<EventSpec, 4U> kHardwareEventSpecs{{
    {"cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, false, true,
     true},
    {"instructions", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, false,
     true, false},
    {"cache references", PERF_TYPE_HARDWARE,
     PERF_COUNT_HW_CACHE_REFERENCES, false, true, false},
    {"cache misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES, false,
     true, false},
}};

constexpr std::array<EventSpec, 3U> kSoftwareEventSpecs{{
    {"task clock", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK, false, true,
     false},
    {"context switches", PERF_TYPE_SOFTWARE,
     PERF_COUNT_SW_CONTEXT_SWITCHES, false, true, false},
    {"CPU migrations", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CPU_MIGRATIONS,
     false, true, false},
}};

struct RawReading {
  std::uint64_t value;
  std::uint64_t time_enabled_nanoseconds;
  std::uint64_t time_running_nanoseconds;
};

[[noreturn]] void throw_system_error(std::string_view operation) {
  throw std::system_error(errno, std::generic_category(),
                          std::string(operation));
}

std::uint32_t read_msr_pmu_type() {
  std::ifstream input{std::string(kMsrPmuTypePath)};
  if (!input) {
    throw std::runtime_error("cannot open the Linux MSR PMU type");
  }
  std::uint64_t value = 0U;
  input >> value;
  if (!input || value > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("Linux MSR PMU type is invalid");
  }
  return static_cast<std::uint32_t>(value);
}

int open_event(const EventSpec &spec, int group_file_descriptor) {
  perf_event_attr attributes{};
  attributes.type = spec.type;
  attributes.size = sizeof(attributes);
  attributes.config = spec.config;
  attributes.disabled = group_file_descriptor == kNoGroup ? 1U : 0U;
  attributes.inherit = spec.inherit ? 1U : 0U;
  attributes.pinned = spec.pinned ? 1U : 0U;
  attributes.exclude_hv = spec.exclude_hypervisor ? 1U : 0U;
  attributes.sample_type = PERF_SAMPLE_IDENTIFIER;
  attributes.read_format = kReadFormat;
  const long result = ::syscall(SYS_perf_event_open, &attributes,
                                kCurrentThread, kAnyCpu,
                                group_file_descriptor,
                                PERF_FLAG_FD_CLOEXEC);
  if (result < 0L) {
    throw_system_error(std::string("perf_event_open ") +
                       std::string(spec.name));
  }
  if (result > std::numeric_limits<int>::max()) {
    throw std::runtime_error("perf_event_open returned an invalid descriptor");
  }
  return static_cast<int>(result);
}

class EventFile {
 public:
  EventFile(const EventSpec &spec, int group_file_descriptor)
      : file_descriptor_(open_event(spec, group_file_descriptor)) {}

  ~EventFile() {
    if (file_descriptor_ != kInvalidFileDescriptor) {
      ::close(file_descriptor_);
    }
  }

  EventFile(const EventFile &) = delete;
  EventFile &operator=(const EventFile &) = delete;

  EventFile(EventFile &&other) noexcept
      : file_descriptor_(std::exchange(other.file_descriptor_,
                                       kInvalidFileDescriptor)) {}

  EventFile &operator=(EventFile &&other) noexcept {
    if (this == &other) {
      return *this;
    }
    if (file_descriptor_ != kInvalidFileDescriptor) {
      ::close(file_descriptor_);
    }
    file_descriptor_ =
        std::exchange(other.file_descriptor_, kInvalidFileDescriptor);
    return *this;
  }

  int descriptor() const {
    if (file_descriptor_ == kInvalidFileDescriptor) {
      throw std::runtime_error("hardware counter descriptor is invalid");
    }
    return file_descriptor_;
  }

  Reading read() const {
    RawReading raw{};
    const ssize_t byte_count = ::read(descriptor(), &raw, sizeof(raw));
    if (byte_count < 0) {
      throw_system_error("read hardware counter");
    }
    if (byte_count != static_cast<ssize_t>(sizeof(raw))) {
      throw std::runtime_error("hardware counter read was incomplete");
    }
    return Reading{raw.value, raw.time_enabled_nanoseconds,
                   raw.time_running_nanoseconds};
  }

 private:
  int file_descriptor_ = kInvalidFileDescriptor;
};

class EventGroup {
 public:
  explicit EventGroup(std::span<const EventSpec> specs) {
    if (specs.empty()) {
      throw std::runtime_error("hardware counter group cannot be empty");
    }
    files_.reserve(specs.size());
    files_.emplace_back(specs.front(), kNoGroup);
    const int leader = files_.front().descriptor();
    for (const EventSpec &spec : specs.subspan(1U)) {
      files_.emplace_back(spec, leader);
    }
  }

  void reset() const {
    if (::ioctl(files_.front().descriptor(), PERF_EVENT_IOC_RESET,
                kGroupIoctlFlag) != 0) {
      throw_system_error("reset hardware counter group");
    }
  }

  Reading read(std::size_t index) const {
    if (index >= files_.size()) {
      throw std::runtime_error("hardware counter index is outside its group");
    }
    return files_[index].read();
  }

 private:
  std::vector<EventFile> files_;
};

void validate_reading(const Reading &reading, std::string_view name,
                      bool require_nonzero_value) {
  if (require_nonzero_value && reading.value == 0U) {
    throw std::runtime_error(std::string(name) + " counter is zero");
  }
  if (reading.time_enabled_nanoseconds == 0U ||
      reading.time_running_nanoseconds == 0U ||
      reading.time_running_nanoseconds > reading.time_enabled_nanoseconds) {
    throw std::runtime_error(std::string(name) +
                             " counter has invalid running time");
  }
  const double running_fraction =
      static_cast<double>(reading.time_running_nanoseconds) /
      static_cast<double>(reading.time_enabled_nanoseconds);
  if (running_fraction < kMinimumRunningFraction) {
    throw std::runtime_error(std::string(name) +
                             " counter was multiplexed below the bound");
  }
}

}  // namespace

void validate(const Diagnostics &diagnostics) {
  validate_reading(diagnostics.cycles, "cycles", true);
  validate_reading(diagnostics.instructions, "instructions", true);
  validate_reading(diagnostics.cache_references, "cache references", true);
  validate_reading(diagnostics.cache_misses, "cache misses", false);
  validate_reading(diagnostics.aperf, "APERF", true);
  validate_reading(diagnostics.mperf, "MPERF", true);
  validate_reading(diagnostics.task_clock_nanoseconds, "task clock", true);
  validate_reading(diagnostics.context_switches, "context switches", false);
  validate_reading(diagnostics.cpu_migrations, "CPU migrations", false);
  if (diagnostics.cache_misses.value > diagnostics.cache_references.value) {
    throw std::runtime_error("cache misses exceed cache references");
  }
}

class Session::Implementation {
 public:
  Implementation()
      : hardware_group_(kHardwareEventSpecs),
        msr_group_(make_msr_event_specs()),
        software_group_(kSoftwareEventSpecs) {}

  ~Implementation() {
    if (active_) {
      static_cast<void>(::prctl(PR_TASK_PERF_EVENTS_DISABLE));
    }
  }

  void start() {
    if (active_) {
      throw std::runtime_error("hardware counter session is already active");
    }
    hardware_group_.reset();
    msr_group_.reset();
    software_group_.reset();
    if (::prctl(PR_TASK_PERF_EVENTS_ENABLE) != 0) {
      throw_system_error("enable hardware counters");
    }
    active_ = true;
  }

  Diagnostics stop() {
    if (!active_) {
      throw std::runtime_error("hardware counter session is not active");
    }
    if (::prctl(PR_TASK_PERF_EVENTS_DISABLE) != 0) {
      throw_system_error("disable hardware counters");
    }
    active_ = false;
    Diagnostics diagnostics{
        hardware_group_.read(0U), hardware_group_.read(1U),
        hardware_group_.read(2U), hardware_group_.read(3U),
        msr_group_.read(0U),      msr_group_.read(1U),
        software_group_.read(0U), software_group_.read(1U),
        software_group_.read(2U)};
    validate(diagnostics);
    return diagnostics;
  }

 private:
  static std::array<EventSpec, 2U> make_msr_event_specs() {
    const std::uint32_t msr_type = read_msr_pmu_type();
    return {{{"APERF", msr_type, kAperfEventConfig, true, false, false},
             {"MPERF", msr_type, kMperfEventConfig, true, false, false}}};
  }

  EventGroup hardware_group_;
  EventGroup msr_group_;
  EventGroup software_group_;
  bool active_ = false;
};

Session::Session() : implementation_(std::make_unique<Implementation>()) {}

Session::~Session() = default;

void Session::start() { implementation_->start(); }

Diagnostics Session::stop() { return implementation_->stop(); }

}  // namespace rwkv7::hardware_counter
