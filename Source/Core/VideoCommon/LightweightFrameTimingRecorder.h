// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

class LightweightFrameTimingRecorder
{
public:
  using ThreadCpuClock = std::uint64_t (*)();

  explicit LightweightFrameTimingRecorder(std::optional<std::string> output_path);
  LightweightFrameTimingRecorder(std::optional<std::string> output_path,
                                 ThreadCpuClock thread_cpu_clock);
  ~LightweightFrameTimingRecorder();

  LightweightFrameTimingRecorder(const LightweightFrameTimingRecorder&) = delete;
  LightweightFrameTimingRecorder& operator=(const LightweightFrameTimingRecorder&) = delete;
  LightweightFrameTimingRecorder(LightweightFrameTimingRecorder&&) = delete;
  LightweightFrameTimingRecorder& operator=(LightweightFrameTimingRecorder&&) = delete;

  void Record(std::chrono::steady_clock::time_point wall_time, std::uint64_t emulated_frame);

private:
  struct TimingRecord
  {
    std::uint64_t emulated_frame;
    std::uint64_t host_frame_end_unix_ns;
    std::uint64_t wall_ns;
    std::uint64_t thread_cpu_ns;
  };

  static std::uint64_t ReadThreadCpuTimeNs();
  void Flush() const;

  ThreadCpuClock m_thread_cpu_clock;
  std::optional<std::string> m_output_path;
  std::chrono::steady_clock::time_point m_wall_base;
  std::uint64_t m_unix_base_ns = 0;
  std::optional<std::chrono::steady_clock::time_point> m_last_wall_time;
  std::uint64_t m_last_thread_cpu_ns = 0;
  std::vector<TimingRecord> m_records;
};
