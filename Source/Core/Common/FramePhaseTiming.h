// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <cstdlib>

#include "Common/CommonTypes.h"

namespace Common::FramePhaseTiming
{
inline std::atomic<u64> s_cpu_wall_ns{0};
inline std::atomic<u64> s_cpu_thread_ns{0};
inline std::atomic<u64> s_cpu_idle_ns{0};
inline std::atomic<u64> s_cpu_throttle_sleep_ns{0};
inline std::atomic<u64> s_cpu_throttle_requested_ns{0};
inline std::atomic<u64> s_cpu_throttle_lateness_ns{0};
inline std::atomic<u64> s_cpu_precision_throttle_calls{0};
inline std::atomic<u64> s_cpu_precision_throttle_coarse_ns{0};
inline std::atomic<u64> s_cpu_precision_throttle_spin_ns{0};
inline std::atomic<u64> s_cpu_precision_present_calls{0};
inline std::atomic<u64> s_cpu_precision_present_coarse_ns{0};
inline std::atomic<u64> s_cpu_precision_present_spin_ns{0};
inline std::atomic<u64> s_metal_bind_surface_ns{0};
inline std::atomic<u64> s_metal_next_drawable_ns{0};
inline std::atomic<u64> s_metal_update_backbuffer_ns{0};
inline std::atomic<u64> s_metal_set_framebuffer_ns{0};
inline std::atomic<u64> s_metal_pipeline_creates{0};
inline std::atomic<u64> s_metal_pipeline_create_ns{0};
inline std::atomic<u64> s_texture_pool_hits{0};
inline std::atomic<u64> s_texture_pool_empty_misses{0};
inline std::atomic<u64> s_texture_pool_same_frame_misses{0};
inline std::atomic<u64> s_texture_pool_expirations{0};
inline std::atomic<u64> s_texture_pool_recent_expiry_misses{0};
inline std::atomic<u64> s_texture_create_calls{0};
inline std::atomic<u64> s_texture_create_ns{0};
inline std::atomic<u64> s_framebuffer_create_calls{0};
inline std::atomic<u64> s_framebuffer_create_ns{0};
inline std::atomic<u64> s_static_recomp_bursts{0};
inline std::atomic<u64> s_static_recomp_cycles{0};
inline std::atomic<u64> s_static_recomp_native_dispatches{0};
inline std::atomic<u64> s_static_recomp_fallback_steps{0};
inline std::atomic<u64> s_static_recomp_hook_fallbacks{0};
inline std::atomic<u64> s_static_recomp_fallback_mfspr{0};
inline std::atomic<u64> s_static_recomp_fallback_mtspr{0};
inline std::atomic<u64> s_static_recomp_fallback_cache{0};
inline std::atomic<u64> s_static_recomp_fallback_dcbst{0};
inline std::atomic<u64> s_static_recomp_fallback_dcbf{0};
inline std::atomic<u64> s_static_recomp_fallback_dcbi{0};
inline std::atomic<u64> s_static_recomp_fallback_icbi{0};
inline std::atomic<u64> s_static_recomp_fallback_other{0};
inline std::atomic<u64> s_static_recomp_cache_controls{0};
inline std::atomic<u64> s_static_recomp_cache_control_dcbst{0};
inline std::atomic<u64> s_static_recomp_cache_control_dcbf{0};
inline std::atomic<u64> s_static_recomp_cache_control_dcbi{0};
inline std::atomic<u64> s_static_recomp_cache_control_icbi{0};
inline std::atomic<u64> s_audio_mix_ns{0};
inline std::atomic<u64> s_efb_vram_pipeline_misses{0};
inline std::atomic<u64> s_efb_vram_shader_ns{0};
inline std::atomic<u64> s_efb_vram_pipeline_ns{0};
inline std::atomic<u64> s_efb_ram_pipeline_misses{0};
inline std::atomic<u64> s_efb_ram_shader_ns{0};
inline std::atomic<u64> s_efb_ram_pipeline_ns{0};
inline std::atomic<u64> s_present_frame_index{0};
inline std::atomic<u64> s_xfb_output_requests{0};
inline std::atomic<u64> s_xfb_swap_queued{0};
inline std::atomic<u64> s_xfb_swap_executed{0};
inline std::atomic<u64> s_xfb_duplicates{0};
inline std::atomic<u64> s_xfb_presents{0};
inline std::atomic<u64> s_emulated_frame_index{0};

inline bool IsEnabled()
{
  static const bool enabled = [] {
    const char* path = std::getenv("MELEEPAD_FRAME_PHASE_LOG");
    return path != nullptr && path[0] != '\0';
  }();
  return enabled;
}

inline bool IsEmulatedFrameIndexEnabled()
{
  static const bool enabled = [] {
    const char* path = std::getenv("MELEEPAD_LIGHTWEIGHT_FRAME_LOG");
    const char* burst_path = std::getenv("STATICRECOMP_DISPATCH_BURST_LOG");
    return IsEnabled() || (path != nullptr && path[0] != '\0') ||
           (burst_path != nullptr && burst_path[0] != '\0');
  }();
  return enabled;
}

inline u64 ToNanoseconds(DT duration)
{
  return static_cast<u64>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count());
}

inline void AddCpuWall(DT duration)
{
  s_cpu_wall_ns.fetch_add(ToNanoseconds(duration), std::memory_order_relaxed);
}

inline void AddCpuThreadNanoseconds(u64 nanoseconds)
{
  s_cpu_thread_ns.fetch_add(nanoseconds, std::memory_order_relaxed);
}

inline void AddCpuIdle(DT duration)
{
  s_cpu_idle_ns.fetch_add(ToNanoseconds(duration), std::memory_order_relaxed);
}

inline void AddCpuThrottleSleep(DT duration)
{
  s_cpu_throttle_sleep_ns.fetch_add(ToNanoseconds(duration), std::memory_order_relaxed);
}

inline void AddCpuThrottleRequested(DT duration)
{
  s_cpu_throttle_requested_ns.fetch_add(ToNanoseconds(duration), std::memory_order_relaxed);
}

inline void AddCpuThrottleLateness(DT duration)
{
  s_cpu_throttle_lateness_ns.fetch_add(ToNanoseconds(duration), std::memory_order_relaxed);
}

inline void AddCpuPrecisionTimer(bool presentation, DT coarse_sleep, DT final_spin)
{
  auto& calls = presentation ? s_cpu_precision_present_calls : s_cpu_precision_throttle_calls;
  auto& coarse =
      presentation ? s_cpu_precision_present_coarse_ns : s_cpu_precision_throttle_coarse_ns;
  auto& spin = presentation ? s_cpu_precision_present_spin_ns : s_cpu_precision_throttle_spin_ns;
  calls.fetch_add(1, std::memory_order_relaxed);
  coarse.fetch_add(ToNanoseconds(coarse_sleep), std::memory_order_relaxed);
  spin.fetch_add(ToNanoseconds(final_spin), std::memory_order_relaxed);
}

inline void AddMetalBindBackbuffer(DT surface, DT next_drawable, DT update_backbuffer,
                                   DT set_framebuffer)
{
  if (!IsEnabled())
    return;
  s_metal_bind_surface_ns.fetch_add(ToNanoseconds(surface), std::memory_order_relaxed);
  s_metal_next_drawable_ns.fetch_add(ToNanoseconds(next_drawable), std::memory_order_relaxed);
  s_metal_update_backbuffer_ns.fetch_add(ToNanoseconds(update_backbuffer),
                                         std::memory_order_relaxed);
  s_metal_set_framebuffer_ns.fetch_add(ToNanoseconds(set_framebuffer),
                                       std::memory_order_relaxed);
}

inline void AddMetalPipelineCreate(DT duration)
{
  if (!IsEnabled())
    return;
  s_metal_pipeline_creates.fetch_add(1, std::memory_order_relaxed);
  s_metal_pipeline_create_ns.fetch_add(ToNanoseconds(duration), std::memory_order_relaxed);
}

inline void AddTexturePoolLookup(bool hit, bool same_frame_miss)
{
  if (!IsEnabled())
    return;

  if (hit)
    s_texture_pool_hits.fetch_add(1, std::memory_order_relaxed);
  else if (same_frame_miss)
    s_texture_pool_same_frame_misses.fetch_add(1, std::memory_order_relaxed);
  else
    s_texture_pool_empty_misses.fetch_add(1, std::memory_order_relaxed);
}

inline void AddTexturePoolExpirations(u64 count)
{
  if (IsEnabled() && count != 0)
    s_texture_pool_expirations.fetch_add(count, std::memory_order_relaxed);
}

inline void AddTexturePoolRecentExpiryMiss()
{
  s_texture_pool_recent_expiry_misses.fetch_add(1, std::memory_order_relaxed);
}

inline void AddTextureCreate(DT duration)
{
  s_texture_create_calls.fetch_add(1, std::memory_order_relaxed);
  s_texture_create_ns.fetch_add(ToNanoseconds(duration), std::memory_order_relaxed);
}

inline void AddFramebufferCreate(DT duration)
{
  s_framebuffer_create_calls.fetch_add(1, std::memory_order_relaxed);
  s_framebuffer_create_ns.fetch_add(ToNanoseconds(duration), std::memory_order_relaxed);
}

inline void AddStaticRecompWork(u64 bursts, u64 cycles, u64 native_dispatches, u64 fallback_steps,
                                u64 hook_fallbacks)
{
  s_static_recomp_bursts.fetch_add(bursts, std::memory_order_relaxed);
  s_static_recomp_cycles.fetch_add(cycles, std::memory_order_relaxed);
  s_static_recomp_native_dispatches.fetch_add(native_dispatches, std::memory_order_relaxed);
  s_static_recomp_fallback_steps.fetch_add(fallback_steps, std::memory_order_relaxed);
  s_static_recomp_hook_fallbacks.fetch_add(hook_fallbacks, std::memory_order_relaxed);
}

inline void AddStaticRecompFallback(u32 raw)
{
  const u32 primary = raw >> 26;
  const u32 xo = (raw >> 1) & 0x3FFu;
  if (primary == 31u && xo == 339u)
    s_static_recomp_fallback_mfspr.fetch_add(1, std::memory_order_relaxed);
  else if (primary == 31u && xo == 467u)
    s_static_recomp_fallback_mtspr.fetch_add(1, std::memory_order_relaxed);
  else if (primary == 31u && xo == 54u)
  {
    s_static_recomp_fallback_cache.fetch_add(1, std::memory_order_relaxed);
    s_static_recomp_fallback_dcbst.fetch_add(1, std::memory_order_relaxed);
  }
  else if (primary == 31u && xo == 86u)
  {
    s_static_recomp_fallback_cache.fetch_add(1, std::memory_order_relaxed);
    s_static_recomp_fallback_dcbf.fetch_add(1, std::memory_order_relaxed);
  }
  else if (primary == 31u && xo == 470u)
  {
    s_static_recomp_fallback_cache.fetch_add(1, std::memory_order_relaxed);
    s_static_recomp_fallback_dcbi.fetch_add(1, std::memory_order_relaxed);
  }
  else if (primary == 31u && xo == 982u)
  {
    s_static_recomp_fallback_cache.fetch_add(1, std::memory_order_relaxed);
    s_static_recomp_fallback_icbi.fetch_add(1, std::memory_order_relaxed);
  }
  else
    s_static_recomp_fallback_other.fetch_add(1, std::memory_order_relaxed);
}

inline void AddStaticRecompCacheControl(u8 operation)
{
  s_static_recomp_cache_controls.fetch_add(1, std::memory_order_relaxed);
  switch (operation)
  {
  case 0:
    s_static_recomp_cache_control_dcbst.fetch_add(1, std::memory_order_relaxed);
    break;
  case 1:
    s_static_recomp_cache_control_dcbf.fetch_add(1, std::memory_order_relaxed);
    break;
  case 2:
    s_static_recomp_cache_control_dcbi.fetch_add(1, std::memory_order_relaxed);
    break;
  case 3:
    s_static_recomp_cache_control_icbi.fetch_add(1, std::memory_order_relaxed);
    break;
  default:
    break;
  }
}

inline void AddAudioMix(DT duration)
{
  s_audio_mix_ns.fetch_add(ToNanoseconds(duration), std::memory_order_relaxed);
}

inline void AddEfbPipelineMiss(bool to_vram, DT shader_duration, DT pipeline_duration)
{
  if (!IsEnabled())
    return;

  if (to_vram)
  {
    s_efb_vram_pipeline_misses.fetch_add(1, std::memory_order_relaxed);
    s_efb_vram_shader_ns.fetch_add(ToNanoseconds(shader_duration), std::memory_order_relaxed);
    s_efb_vram_pipeline_ns.fetch_add(ToNanoseconds(pipeline_duration), std::memory_order_relaxed);
  }
  else
  {
    s_efb_ram_pipeline_misses.fetch_add(1, std::memory_order_relaxed);
    s_efb_ram_shader_ns.fetch_add(ToNanoseconds(shader_duration), std::memory_order_relaxed);
    s_efb_ram_pipeline_ns.fetch_add(ToNanoseconds(pipeline_duration), std::memory_order_relaxed);
  }
}

inline void SetPresentFrameIndex(u64 frame)
{
  s_present_frame_index.store(frame, std::memory_order_release);
}

inline u64 GetPresentFrameIndex()
{
  return s_present_frame_index.load(std::memory_order_acquire);
}

inline void AddXfbOutputRequest()
{
  if (IsEnabled())
    s_xfb_output_requests.fetch_add(1, std::memory_order_relaxed);
}

inline void AddXfbSwapQueued()
{
  if (IsEnabled())
    s_xfb_swap_queued.fetch_add(1, std::memory_order_relaxed);
}

inline void AddXfbSwapExecuted()
{
  if (IsEnabled())
    s_xfb_swap_executed.fetch_add(1, std::memory_order_relaxed);
}

inline void AddXfbDuplicate()
{
  if (IsEnabled())
    s_xfb_duplicates.fetch_add(1, std::memory_order_relaxed);
}

inline void AddXfbPresent()
{
  if (IsEnabled())
    s_xfb_presents.fetch_add(1, std::memory_order_relaxed);
}

inline void SetEmulatedFrameIndex(u64 frame)
{
  s_emulated_frame_index.store(frame, std::memory_order_release);
}

inline u64 GetEmulatedFrameIndex()
{
  return s_emulated_frame_index.load(std::memory_order_acquire);
}

struct Totals
{
  u64 cpu_wall_ns;
  u64 cpu_thread_ns;
  u64 cpu_idle_ns;
  u64 cpu_throttle_sleep_ns;
  u64 cpu_throttle_requested_ns;
  u64 cpu_throttle_lateness_ns;
  u64 cpu_precision_throttle_calls;
  u64 cpu_precision_throttle_coarse_ns;
  u64 cpu_precision_throttle_spin_ns;
  u64 cpu_precision_present_calls;
  u64 cpu_precision_present_coarse_ns;
  u64 cpu_precision_present_spin_ns;
  u64 metal_bind_surface_ns;
  u64 metal_next_drawable_ns;
  u64 metal_update_backbuffer_ns;
  u64 metal_set_framebuffer_ns;
  u64 metal_pipeline_creates;
  u64 metal_pipeline_create_ns;
  u64 texture_pool_hits;
  u64 texture_pool_empty_misses;
  u64 texture_pool_same_frame_misses;
  u64 texture_pool_expirations;
  u64 texture_pool_recent_expiry_misses;
  u64 texture_create_calls;
  u64 texture_create_ns;
  u64 framebuffer_create_calls;
  u64 framebuffer_create_ns;
  u64 static_recomp_bursts;
  u64 static_recomp_cycles;
  u64 static_recomp_native_dispatches;
  u64 static_recomp_fallback_steps;
  u64 static_recomp_hook_fallbacks;
  u64 static_recomp_fallback_mfspr;
  u64 static_recomp_fallback_mtspr;
  u64 static_recomp_fallback_cache;
  u64 static_recomp_fallback_dcbst;
  u64 static_recomp_fallback_dcbf;
  u64 static_recomp_fallback_dcbi;
  u64 static_recomp_fallback_icbi;
  u64 static_recomp_fallback_other;
  u64 static_recomp_cache_controls;
  u64 static_recomp_cache_control_dcbst;
  u64 static_recomp_cache_control_dcbf;
  u64 static_recomp_cache_control_dcbi;
  u64 static_recomp_cache_control_icbi;
  u64 audio_mix_ns;
  u64 efb_vram_pipeline_misses;
  u64 efb_vram_shader_ns;
  u64 efb_vram_pipeline_ns;
  u64 efb_ram_pipeline_misses;
  u64 efb_ram_shader_ns;
  u64 efb_ram_pipeline_ns;
  u64 xfb_output_requests;
  u64 xfb_swap_queued;
  u64 xfb_swap_executed;
  u64 xfb_duplicates;
  u64 xfb_presents;
};

inline Totals GetTotals()
{
  return {
      s_cpu_wall_ns.load(std::memory_order_relaxed),
      s_cpu_thread_ns.load(std::memory_order_relaxed),
      s_cpu_idle_ns.load(std::memory_order_relaxed),
      s_cpu_throttle_sleep_ns.load(std::memory_order_relaxed),
      s_cpu_throttle_requested_ns.load(std::memory_order_relaxed),
      s_cpu_throttle_lateness_ns.load(std::memory_order_relaxed),
      s_cpu_precision_throttle_calls.load(std::memory_order_relaxed),
      s_cpu_precision_throttle_coarse_ns.load(std::memory_order_relaxed),
      s_cpu_precision_throttle_spin_ns.load(std::memory_order_relaxed),
      s_cpu_precision_present_calls.load(std::memory_order_relaxed),
      s_cpu_precision_present_coarse_ns.load(std::memory_order_relaxed),
      s_cpu_precision_present_spin_ns.load(std::memory_order_relaxed),
      s_metal_bind_surface_ns.load(std::memory_order_relaxed),
      s_metal_next_drawable_ns.load(std::memory_order_relaxed),
      s_metal_update_backbuffer_ns.load(std::memory_order_relaxed),
      s_metal_set_framebuffer_ns.load(std::memory_order_relaxed),
      s_metal_pipeline_creates.load(std::memory_order_relaxed),
      s_metal_pipeline_create_ns.load(std::memory_order_relaxed),
      s_texture_pool_hits.load(std::memory_order_relaxed),
      s_texture_pool_empty_misses.load(std::memory_order_relaxed),
      s_texture_pool_same_frame_misses.load(std::memory_order_relaxed),
      s_texture_pool_expirations.load(std::memory_order_relaxed),
      s_texture_pool_recent_expiry_misses.load(std::memory_order_relaxed),
      s_texture_create_calls.load(std::memory_order_relaxed),
      s_texture_create_ns.load(std::memory_order_relaxed),
      s_framebuffer_create_calls.load(std::memory_order_relaxed),
      s_framebuffer_create_ns.load(std::memory_order_relaxed),
      s_static_recomp_bursts.load(std::memory_order_relaxed),
      s_static_recomp_cycles.load(std::memory_order_relaxed),
      s_static_recomp_native_dispatches.load(std::memory_order_relaxed),
      s_static_recomp_fallback_steps.load(std::memory_order_relaxed),
      s_static_recomp_hook_fallbacks.load(std::memory_order_relaxed),
      s_static_recomp_fallback_mfspr.load(std::memory_order_relaxed),
      s_static_recomp_fallback_mtspr.load(std::memory_order_relaxed),
      s_static_recomp_fallback_cache.load(std::memory_order_relaxed),
      s_static_recomp_fallback_dcbst.load(std::memory_order_relaxed),
      s_static_recomp_fallback_dcbf.load(std::memory_order_relaxed),
      s_static_recomp_fallback_dcbi.load(std::memory_order_relaxed),
      s_static_recomp_fallback_icbi.load(std::memory_order_relaxed),
      s_static_recomp_fallback_other.load(std::memory_order_relaxed),
      s_static_recomp_cache_controls.load(std::memory_order_relaxed),
      s_static_recomp_cache_control_dcbst.load(std::memory_order_relaxed),
      s_static_recomp_cache_control_dcbf.load(std::memory_order_relaxed),
      s_static_recomp_cache_control_dcbi.load(std::memory_order_relaxed),
      s_static_recomp_cache_control_icbi.load(std::memory_order_relaxed),
      s_audio_mix_ns.load(std::memory_order_relaxed),
      s_efb_vram_pipeline_misses.load(std::memory_order_relaxed),
      s_efb_vram_shader_ns.load(std::memory_order_relaxed),
      s_efb_vram_pipeline_ns.load(std::memory_order_relaxed),
      s_efb_ram_pipeline_misses.load(std::memory_order_relaxed),
      s_efb_ram_shader_ns.load(std::memory_order_relaxed),
      s_efb_ram_pipeline_ns.load(std::memory_order_relaxed),
      s_xfb_output_requests.load(std::memory_order_relaxed),
      s_xfb_swap_queued.load(std::memory_order_relaxed),
      s_xfb_swap_executed.load(std::memory_order_relaxed),
      s_xfb_duplicates.load(std::memory_order_relaxed),
      s_xfb_presents.load(std::memory_order_relaxed),
  };
}
}  // namespace Common::FramePhaseTiming
