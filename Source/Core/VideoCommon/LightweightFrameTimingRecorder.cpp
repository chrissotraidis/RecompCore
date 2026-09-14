// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "VideoCommon/LightweightFrameTimingRecorder.h"

#include <ctime>
#include <fstream>
#include <iomanip>
#include <utility>

LightweightFrameTimingRecorder::LightweightFrameTimingRecorder(
    std::optional<std::string> output_path)
    : LightweightFrameTimingRecorder(std::move(output_path), ReadThreadCpuTimeNs)
{
}

LightweightFrameTimingRecorder::LightweightFrameTimingRecorder(
    std::optional<std::string> output_path, ThreadCpuClock thread_cpu_clock)
    : m_thread_cpu_clock{thread_cpu_clock}
{
  if (!output_path || output_path->empty())
    return;

  m_output_path = std::move(*output_path);
  m_wall_base = std::chrono::steady_clock::now();
  m_unix_base_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
  m_records.reserve(262144);
}

LightweightFrameTimingRecorder::~LightweightFrameTimingRecorder()
{
  Flush();
}

void LightweightFrameTimingRecorder::Record(std::chrono::steady_clock::time_point wall_time,
                                            std::uint64_t emulated_frame)
{
  if (!m_output_path)
    return;

  const std::uint64_t thread_cpu_ns = m_thread_cpu_clock();
  if (!m_last_wall_time)
  {
    m_last_wall_time = wall_time;
    m_last_thread_cpu_ns = thread_cpu_ns;
    return;
  }

  const auto wall_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(wall_time - *m_last_wall_time).count();
  const std::uint64_t thread_ns = thread_cpu_ns >= m_last_thread_cpu_ns ?
                                      thread_cpu_ns - m_last_thread_cpu_ns :
                                      0;
  if (wall_ns >= 0)
  {
    const auto unix_offset_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(wall_time - m_wall_base).count();
    m_records.push_back({emulated_frame, m_unix_base_ns + unix_offset_ns,
                         static_cast<std::uint64_t>(wall_ns), thread_ns});
  }

  m_last_wall_time = wall_time;
  m_last_thread_cpu_ns = thread_cpu_ns;
}

std::uint64_t LightweightFrameTimingRecorder::ReadThreadCpuTimeNs()
{
  timespec time{};
  if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &time) != 0)
    return 0;
  return static_cast<std::uint64_t>(time.tv_sec) * 1000000000ULL + time.tv_nsec;
}

void LightweightFrameTimingRecorder::Flush() const
{
  if (!m_output_path)
    return;

  std::ofstream stream{*m_output_path, std::ios::out | std::ios::trunc};
  if (!stream)
    return;

  stream << "index,emulated_frame,host_frame_end_unix_ns,total_ms,thread_cpu_ms,"
            "wall_minus_thread_ms\n";
  stream << std::fixed << std::setprecision(6);
  for (std::size_t index = 0; index < m_records.size(); ++index)
  {
    const TimingRecord& record = m_records[index];
    const std::uint64_t remainder_ns =
        record.wall_ns > record.thread_cpu_ns ? record.wall_ns - record.thread_cpu_ns : 0;
    stream << index + 1 << ',' << record.emulated_frame << ',' << record.host_frame_end_unix_ns
           << ','
           << record.wall_ns / 1000000.0 << ',' << record.thread_cpu_ns / 1000000.0 << ','
           << remainder_ns / 1000000.0 << '\n';
  }
}
