#pragma once

#include <cstdint>
#include <memory>

namespace rwkv7::hardware_counter {

struct Reading {
  std::uint64_t value;
  std::uint64_t time_enabled_nanoseconds;
  std::uint64_t time_running_nanoseconds;
};

struct Diagnostics {
  Reading cycles;
  Reading instructions;
  Reading cache_references;
  Reading cache_misses;
  Reading aperf;
  Reading mperf;
  Reading task_clock_nanoseconds;
  Reading context_switches;
  Reading cpu_migrations;
};

void validate(const Diagnostics &diagnostics);

class Session {
 public:
  Session();
  ~Session();

  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;
  Session(Session &&) = delete;
  Session &operator=(Session &&) = delete;

  void start();
  Diagnostics stop();

 private:
  class Implementation;
  std::unique_ptr<Implementation> implementation_;
};

}  // namespace rwkv7::hardware_counter
