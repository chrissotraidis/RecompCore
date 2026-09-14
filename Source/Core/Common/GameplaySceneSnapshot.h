// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>

// One CPU-frame producer. UI reads copies only; it never accesses guest RAM.
// The producer skips a contended sample instead of waiting for the UI.
namespace Common::GameplayScene
{
struct Snapshot
{
  std::uint64_t session = 0;
  std::uint64_t frames = 0;
  std::uint64_t transitions = 0;
  std::uint32_t routing = 0;
  int revision = -1;
  bool valid = false;
};

class Recorder
{
public:
  void Configure(int revision)
  {
    m_enabled.store(false, std::memory_order_relaxed);
    std::lock_guard lock(m_mutex);
    const auto session = m_snapshot.session + 1;
    m_snapshot = {};
    m_snapshot.session = session;
    m_address = revision == 0 ? 0x80477D68u : revision == 2 ? 0x80479D30u : 0;
    m_snapshot.revision = m_address ? revision : -1;
    m_enabled.store(m_address != 0, std::memory_order_relaxed);
  }

  template <typename ReadWord>
  void RecordFrame(ReadWord&& read_word)
  {
    if (!m_enabled.load(std::memory_order_relaxed))
      return;
    std::unique_lock lock(m_mutex, std::try_to_lock);
    if (!lock.owns_lock() || !m_address)
      return;
    const auto word = read_word(m_address);
    ++m_snapshot.frames;
    if (!word)
    {
      if (m_snapshot.valid)
        ++m_snapshot.transitions;
      m_snapshot.valid = false;
      return;
    }
    // Ignore pending/previous mode bytes when counting actual scene changes.
    constexpr std::uint32_t scene_mask = 0xff0000ffu;
    if ((!m_snapshot.valid && m_snapshot.frames > 1) ||
        (m_snapshot.valid && ((*word ^ m_snapshot.routing) & scene_mask)))
      ++m_snapshot.transitions;
    m_snapshot.routing = *word;
    m_snapshot.valid = true;
  }

  Snapshot Read()
  {
    std::lock_guard lock(m_mutex);
    return m_snapshot;
  }

private:
  std::atomic<bool> m_enabled{false};
  std::mutex m_mutex;
  std::uint32_t m_address = 0;
  Snapshot m_snapshot;
};

inline Recorder recorder;

// Declare before the runtime owner so reset follows runtime destruction,
// including early boot failures and stop/restart paths.
class Session
{
public:
  explicit Session(int revision) { recorder.Configure(revision); }
  ~Session() { recorder.Configure(-1); }
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;
};

inline const char* Label(const Snapshot& scene)
{
  if (!scene.valid)
    return "unavailable";
  const auto mode = scene.routing >> 24;
  const auto state = scene.routing & 0xff;
  if (mode == 0x02)
  {
    switch (state)
    {
    case 0: return "vs-character-select";
    case 1: return "vs-stage-select";
    case 2: return "vs-combat";
    case 3: return "vs-sudden-death";
    case 4: return "vs-results";
    default: return "vs-other";
    }
  }
  if (mode == 0x00) return "title";
  if (mode == 0x01) return "main-menu";
  if (mode == 0x18) return "opening";
  return "other";
}
}  // namespace Common::GameplayScene
