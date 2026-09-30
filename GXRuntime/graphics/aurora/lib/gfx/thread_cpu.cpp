#include "thread_cpu.hpp"

#include <atomic>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <pthread.h>
#else
#include <pthread.h>
#include <time.h>
#endif

namespace aurora::gfx::thread_cpu {
namespace {
constexpr uint32_t kRoles = static_cast<uint32_t>(Role::Count);
#if defined(_WIN32)
std::atomic<HANDLE> g_threads[kRoles]{};
#elif defined(__APPLE__)
std::atomic<mach_port_t> g_threads[kRoles]{};
#else
std::atomic<bool> g_registered[kRoles]{};
clockid_t g_clocks[kRoles]{};
#endif
} // namespace

void register_current(Role role) {
  const uint32_t i = static_cast<uint32_t>(role);
  if (i >= kRoles)
    return;
#if defined(_WIN32)
  HANDLE handle = nullptr;
  if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &handle,
                       THREAD_QUERY_LIMITED_INFORMATION, FALSE, 0))
    return;
  if (HANDLE old = g_threads[i].exchange(handle, std::memory_order_acq_rel))
    CloseHandle(old);
#elif defined(__APPLE__)
  g_threads[i].store(pthread_mach_thread_np(pthread_self()), std::memory_order_release);
#else
  clockid_t clock;
  if (pthread_getcpuclockid(pthread_self(), &clock) != 0)
    return;
  g_clocks[i] = clock;
  g_registered[i].store(true, std::memory_order_release);
#endif
}

uint64_t cpu_us(Role role) {
  const uint32_t i = static_cast<uint32_t>(role);
  if (i >= kRoles)
    return 0;
#if defined(_WIN32)
  const HANDLE handle = g_threads[i].load(std::memory_order_acquire);
  FILETIME created, exited, kernel, user;
  if (handle == nullptr || !GetThreadTimes(handle, &created, &exited, &kernel, &user))
    return 0;
  const auto ticks = [](const FILETIME& t) {
    return (static_cast<uint64_t>(t.dwHighDateTime) << 32) | t.dwLowDateTime;
  };
  return (ticks(kernel) + ticks(user)) / 10u;
#elif defined(__APPLE__)
  const mach_port_t thread = g_threads[i].load(std::memory_order_acquire);
  if (thread == MACH_PORT_NULL)
    return 0;
  thread_basic_info_data_t info;
  mach_msg_type_number_t count = THREAD_BASIC_INFO_COUNT;
  if (thread_info(thread, THREAD_BASIC_INFO, reinterpret_cast<thread_info_t>(&info), &count) != KERN_SUCCESS)
    return 0;
  return static_cast<uint64_t>(info.user_time.seconds + info.system_time.seconds) * 1000000u +
         static_cast<uint64_t>(info.user_time.microseconds + info.system_time.microseconds);
#else
  if (!g_registered[i].load(std::memory_order_acquire))
    return 0;
  timespec ts{};
  if (clock_gettime(g_clocks[i], &ts) != 0)
    return 0;
  return static_cast<uint64_t>(ts.tv_sec) * 1000000u + static_cast<uint64_t>(ts.tv_nsec) / 1000u;
#endif
}

} // namespace aurora::gfx::thread_cpu
