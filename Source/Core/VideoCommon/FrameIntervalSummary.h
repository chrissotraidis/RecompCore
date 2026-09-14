// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <mutex>

// Reuses existing frame timestamps; no clock reads or allocation per frame.
// Percentiles use 1 ms buckets. The overflow bucket uses the observed maximum.
class FrameIntervalSummary
{
public:
  struct Snapshot
  {
    std::uint64_t frames = 0;
    double average_ms = 0;
    double p95_upper_ms = 0;
    double maximum_ms = 0;
    std::uint64_t over_20_ms = 0;
    std::uint64_t over_33_ms = 0;
    std::uint64_t over_50_ms = 0;
  };

  void Record(double milliseconds)
  {
    if (!std::isfinite(milliseconds) || milliseconds <= 0)
      return;
    const auto bucket = static_cast<std::size_t>(std::min(250.0, std::ceil(milliseconds)));
    std::lock_guard lock(m_mutex);
    ++m_bins[bucket];
    ++m_snapshot.frames;
    m_total_ms += milliseconds;
    m_snapshot.maximum_ms = std::max(m_snapshot.maximum_ms, milliseconds);
    m_snapshot.over_20_ms += milliseconds > 20.0;
    m_snapshot.over_33_ms += milliseconds > 1000.0 / 30.0;
    m_snapshot.over_50_ms += milliseconds > 50.0;
  }

  Snapshot Take()
  {
    std::lock_guard lock(m_mutex);
    Snapshot result = m_snapshot;
    if (result.frames)
    {
      result.average_ms = m_total_ms / result.frames;
      const auto rank = result.frames - result.frames / 20; // ceil(0.95 * n)
      std::uint64_t count = 0;
      for (std::size_t i = 0; i < m_bins.size(); ++i)
      {
        count += m_bins[i];
        if (count >= rank)
        {
          result.p95_upper_ms = i == 250 ? result.maximum_ms : static_cast<double>(i);
          break;
        }
      }
    }
    m_snapshot = {};
    m_total_ms = 0;
    m_bins.fill(0);
    return result;
  }

private:
  std::mutex m_mutex;
  Snapshot m_snapshot;
  double m_total_ms = 0;
  std::array<std::uint64_t, 251> m_bins{};
};
