#pragma once
#if !defined(_WIN32)
#include <pthread.h>
#endif

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <chrono>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

namespace aurora::gfx::render_worker {

#if !defined(_WIN32)
// The worker's thread (0 before it starts), for diagnostics.
pthread_t native_thread();
#endif

enum class ItemType : uint8_t {
  BeginFrame,
  EncodePass,
  EndFrame,
  Sync,
  Shutdown,
};

using WorkCallback = std::function<void()>;

struct SyncState {
  std::mutex mutex;
  std::condition_variable cv;
  bool complete = false;
};

struct QueueItem {
  ItemType type = ItemType::Sync;
  uint64_t frameId = 0;
  uint32_t passIndex = 0;
  WorkCallback work;
  std::shared_ptr<SyncState> sync;
};

class BoundedQueue {
public:
  explicit BoundedQueue(size_t capacity);

  bool push(QueueItem item);
  std::optional<QueueItem> pop_for(std::chrono::milliseconds timeout, bool& closed);
  void close();
  void reset();
  [[nodiscard]] size_t size() const;

private:
  size_t m_capacity = 0;
  mutable std::mutex m_mutex;
  std::condition_variable m_notEmpty;
  std::condition_variable m_notFull;
  std::deque<QueueItem> m_items;
  bool m_closed = false;
};

class FrameSlotPool {
public:
  explicit FrameSlotPool(size_t slotCount);

  size_t acquire();
  std::optional<size_t> try_acquire();
  void release(size_t slot);
  void reset();
  [[nodiscard]] size_t free_count() const;

private:
  mutable std::mutex m_mutex;
  std::condition_variable m_cv;
  std::vector<bool> m_freeSlots;
};

void initialize();
void shutdown();

void enqueue_begin_frame(uint64_t frameId, WorkCallback work);
void enqueue_encode_pass(uint64_t frameId, uint32_t passIndex, WorkCallback work);
void enqueue_end_frame(uint64_t frameId, WorkCallback work);
void enqueue_work(WorkCallback work);
void synchronize();

bool is_worker_thread() noexcept;
// Runs on the worker between queue items and at least every millisecond while
// it waits: timed work such as a deferred present. Set before initialize().
void set_idle_hook(WorkCallback hook);
bool is_idle() noexcept;

// Holds the worker still while it lives: the worker finishes everything queued
// before it, runs `before` (on the worker), then waits, running neither queue
// items nor the idle hook, until the Pause is destroyed. synchronize() only
// drains the queue; the idle hook (Smooth Motion's deferred presents) can run
// right after it. Nothing on the pausing thread may wait for the worker while
// paused. On the worker itself, or before it starts, a Pause does nothing.
class Pause {
public:
  explicit Pause(WorkCallback before = {});
  ~Pause();
  Pause(const Pause&) = delete;
  Pause& operator=(const Pause&) = delete;

private:
  struct State;
  std::shared_ptr<State> m_state;
};

} // namespace aurora::gfx::render_worker
