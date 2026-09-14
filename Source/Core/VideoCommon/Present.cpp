// Copyright 2023 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "VideoCommon/Present.h"

#include "Common/ChunkFile.h"
#include "Common/FileUtil.h"
#include "Common/FramePhaseTiming.h"
#include "Core/Config/GraphicsSettings.h"
#include "Core/Config/MainSettings.h"
#include "Core/CoreTiming.h"
#include "Core/HW/VideoInterface.h"
#include "Core/Host.h"
#include "Core/System.h"

#include "InputCommon/ControllerInterface/ControllerInterface.h"

#include "Present.h"
#include "VideoCommon/AbstractGfx.h"
#include "VideoCommon/FrameDumper.h"
#include "VideoCommon/FramebufferManager.h"
#include "VideoCommon/OnScreenUI.h"
#include "VideoCommon/PostProcessing.h"
#include "VideoCommon/Statistics.h"
#include "VideoCommon/VertexManagerBase.h"
#include "VideoCommon/VideoConfig.h"
#include "VideoCommon/VideoEvents.h"
#include "VideoCommon/Widescreen.h"

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#if defined(__APPLE__)
#include <mach/mach.h>
#endif

std::unique_ptr<VideoCommon::Presenter> g_presenter;

namespace VideoCommon
{
namespace
{
struct TaskEventCounts
{
  u64 context_switches;
  u64 mach_syscalls;
  u64 unix_syscalls;
};

TaskEventCounts GetTaskEventCounts()
{
#if defined(__APPLE__)
  task_events_info_data_t events{};
  mach_msg_type_number_t count = TASK_EVENTS_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_EVENTS_INFO, reinterpret_cast<task_info_t>(&events),
                &count) != KERN_SUCCESS)
    return {};
  return {static_cast<u64>(events.csw), static_cast<u64>(events.syscalls_mach),
          static_cast<u64>(events.syscalls_unix)};
#else
  return {};
#endif
}

class FramePhaseLogger
{
public:
  void Log(TimePoint frame_start, TimePoint present_start, TimePoint frame_end,
           DT flush_and_rect, DT bind_backbuffer, DT xfb_blit, DT onscreen_ui)
  {
    if (!Common::FramePhaseTiming::IsEnabled())
      return;

    if (!m_file.is_open())
    {
      const char* path = std::getenv("MELEEPAD_FRAME_PHASE_LOG");
      std::filesystem::path resolved_path(path);
      if (resolved_path.is_relative())
        resolved_path = std::filesystem::path(File::GetUserPath(D_LOGS_IDX)) / resolved_path;
      m_file.open(resolved_path, std::ios_base::out);
      if (!m_file.is_open())
        return;
      m_file << "frame,emulated_frame,host_frame_end_unix_ns,total_ms,cpu_wall_ms,cpu_thread_ms,task_context_switches,task_mach_syscalls,task_unix_syscalls,cpu_idle_ms,cpu_throttle_sleep_ms,"
                "cpu_throttle_requested_ms,cpu_throttle_lateness_ms,"
                "precision_throttle_calls,precision_throttle_coarse_ms,precision_throttle_spin_ms,"
                "precision_present_calls,precision_present_coarse_ms,precision_present_spin_ms,"
                "video_build_ms,present_ms,present_flush_rect_ms,present_bind_ms,"
                "present_xfb_ms,present_ui_ms,metal_bind_surface_ms,metal_next_drawable_ms,"
                "metal_update_backbuffer_ms,metal_set_framebuffer_ms,"
                "audio_mix_ms,draw_calls,primitives,vertex_shaders_created,pixel_shaders_created,textures_created,metal_pipeline_creates,metal_pipeline_ms,"
                "texture_pool_hits,texture_pool_empty_misses,texture_pool_same_frame_misses,texture_pool_expirations,texture_pool_recent_expiry_misses,texture_create_calls,texture_create_ms,framebuffer_create_calls,framebuffer_create_ms,"
                "static_bursts,static_cycles,static_native_dispatches,"
                "static_fallback_steps,static_hook_fallbacks,fallback_mfspr,fallback_mtspr,"
                "fallback_cache,fallback_dcbst,fallback_dcbf,fallback_dcbi,fallback_icbi,"
                "fallback_other,cache_controls,cache_control_dcbst,cache_control_dcbf,"
                "cache_control_dcbi,cache_control_icbi,efb_vram_pipeline_misses,"
                "efb_vram_shader_ms,efb_vram_pipeline_ms,efb_ram_pipeline_misses,"
                "efb_ram_shader_ms,efb_ram_pipeline_ms,xfb_output_requests,"
                "xfb_swap_queued,xfb_swap_executed,xfb_duplicates,xfb_presents\n";
      if (const char* start_frame = std::getenv("MELEEPAD_FRAME_PHASE_SLOW_START_FRAME"))
        m_slow_start_frame = std::strtoull(start_frame, nullptr, 10);
      if (const char* arm_path = std::getenv("MELEEPAD_FRAME_PHASE_SLOW_ARM_FILE"))
      {
        m_slow_arm_path = arm_path;
        m_slow_armed = false;
      }
      m_previous_frame_start = frame_start;
      m_previous_totals = Common::FramePhaseTiming::GetTotals();
      m_previous_task_events = GetTaskEventCounts();
      m_previous_vertex_shaders_created = g_stats.num_vertex_shaders_created;
      m_previous_pixel_shaders_created = g_stats.num_pixel_shaders_created;
      m_previous_textures_created = g_stats.num_textures_created;
    }

    const Common::FramePhaseTiming::Totals totals = Common::FramePhaseTiming::GetTotals();
    const TaskEventCounts task_events = GetTaskEventCounts();
    const auto milliseconds = [](DT duration) { return DT_ms(duration).count(); };
    const auto counter_ms = [](u64 current, u64 previous) {
      return static_cast<double>(current - previous) / 1'000'000.0;
    };
    const auto stat_delta = [](int current, int previous) {
      return current >= previous ? current - previous : current;
    };

    const double total_ms = milliseconds(frame_start - m_previous_frame_start);
    const u64 frame = m_frame++;
    const u64 emulated_frame = Common::FramePhaseTiming::GetEmulatedFrameIndex();
    const TimePoint steady_now = Clock::now();
    const auto system_now = std::chrono::system_clock::now();
    const auto frame_end_unix_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(system_now.time_since_epoch()).count() -
        std::chrono::duration_cast<std::chrono::nanoseconds>(steady_now - frame_end).count();
    Common::FramePhaseTiming::SetPresentFrameIndex(m_frame);
    m_file << frame << ',' << emulated_frame << ',' << frame_end_unix_ns << ','
           << std::fixed << std::setprecision(6) << total_ms << ','
           << counter_ms(totals.cpu_wall_ns, m_previous_totals.cpu_wall_ns) << ','
           << counter_ms(totals.cpu_thread_ns, m_previous_totals.cpu_thread_ns) << ','
           << task_events.context_switches - m_previous_task_events.context_switches << ','
           << task_events.mach_syscalls - m_previous_task_events.mach_syscalls << ','
           << task_events.unix_syscalls - m_previous_task_events.unix_syscalls << ','
           << counter_ms(totals.cpu_idle_ns, m_previous_totals.cpu_idle_ns) << ','
           << counter_ms(totals.cpu_throttle_sleep_ns,
                         m_previous_totals.cpu_throttle_sleep_ns)
           << ','
           << counter_ms(totals.cpu_throttle_requested_ns,
                         m_previous_totals.cpu_throttle_requested_ns)
           << ','
           << counter_ms(totals.cpu_throttle_lateness_ns,
                         m_previous_totals.cpu_throttle_lateness_ns)
           << ','
           << totals.cpu_precision_throttle_calls -
                  m_previous_totals.cpu_precision_throttle_calls
           << ','
           << counter_ms(totals.cpu_precision_throttle_coarse_ns,
                         m_previous_totals.cpu_precision_throttle_coarse_ns)
           << ','
           << counter_ms(totals.cpu_precision_throttle_spin_ns,
                         m_previous_totals.cpu_precision_throttle_spin_ns)
           << ','
           << totals.cpu_precision_present_calls - m_previous_totals.cpu_precision_present_calls
           << ','
           << counter_ms(totals.cpu_precision_present_coarse_ns,
                         m_previous_totals.cpu_precision_present_coarse_ns)
           << ','
           << counter_ms(totals.cpu_precision_present_spin_ns,
                         m_previous_totals.cpu_precision_present_spin_ns)
           << ','
           << milliseconds(present_start - frame_start) << ','
           << milliseconds(frame_end - present_start) << ','
           << milliseconds(flush_and_rect) << ',' << milliseconds(bind_backbuffer) << ','
           << milliseconds(xfb_blit) << ',' << milliseconds(onscreen_ui) << ','
           << counter_ms(totals.metal_bind_surface_ns, m_previous_totals.metal_bind_surface_ns)
           << ','
           << counter_ms(totals.metal_next_drawable_ns,
                         m_previous_totals.metal_next_drawable_ns)
           << ','
           << counter_ms(totals.metal_update_backbuffer_ns,
                         m_previous_totals.metal_update_backbuffer_ns)
           << ','
           << counter_ms(totals.metal_set_framebuffer_ns,
                         m_previous_totals.metal_set_framebuffer_ns)
           << ','
           << counter_ms(totals.audio_mix_ns, m_previous_totals.audio_mix_ns) << ','
           << g_stats.this_frame.num_draw_calls << ','
           << g_stats.this_frame.num_prims + g_stats.this_frame.num_dl_prims << ','
           << stat_delta(g_stats.num_vertex_shaders_created,
                         m_previous_vertex_shaders_created)
           << ','
           << stat_delta(g_stats.num_pixel_shaders_created,
                         m_previous_pixel_shaders_created)
           << ',' << stat_delta(g_stats.num_textures_created, m_previous_textures_created) << ','
           << totals.metal_pipeline_creates - m_previous_totals.metal_pipeline_creates << ','
           << counter_ms(totals.metal_pipeline_create_ns,
                         m_previous_totals.metal_pipeline_create_ns)
           << ','
           << totals.texture_pool_hits - m_previous_totals.texture_pool_hits << ','
           << totals.texture_pool_empty_misses - m_previous_totals.texture_pool_empty_misses
           << ','
           << totals.texture_pool_same_frame_misses -
                  m_previous_totals.texture_pool_same_frame_misses
           << ','
           << totals.texture_pool_expirations - m_previous_totals.texture_pool_expirations << ','
           << totals.texture_pool_recent_expiry_misses -
                  m_previous_totals.texture_pool_recent_expiry_misses
           << ','
           << totals.texture_create_calls - m_previous_totals.texture_create_calls << ','
           << counter_ms(totals.texture_create_ns, m_previous_totals.texture_create_ns) << ','
           << totals.framebuffer_create_calls - m_previous_totals.framebuffer_create_calls << ','
           << counter_ms(totals.framebuffer_create_ns, m_previous_totals.framebuffer_create_ns)
           << ','
           << totals.static_recomp_bursts - m_previous_totals.static_recomp_bursts << ','
           << totals.static_recomp_cycles - m_previous_totals.static_recomp_cycles << ','
           << totals.static_recomp_native_dispatches -
                  m_previous_totals.static_recomp_native_dispatches
           << ','
           << totals.static_recomp_fallback_steps - m_previous_totals.static_recomp_fallback_steps
           << ','
           << totals.static_recomp_hook_fallbacks -
                  m_previous_totals.static_recomp_hook_fallbacks
           << ','
           << totals.static_recomp_fallback_mfspr -
                  m_previous_totals.static_recomp_fallback_mfspr
           << ','
           << totals.static_recomp_fallback_mtspr -
                  m_previous_totals.static_recomp_fallback_mtspr
           << ','
           << totals.static_recomp_fallback_cache -
                  m_previous_totals.static_recomp_fallback_cache
           << ','
           << totals.static_recomp_fallback_dcbst -
                  m_previous_totals.static_recomp_fallback_dcbst
           << ','
           << totals.static_recomp_fallback_dcbf -
                  m_previous_totals.static_recomp_fallback_dcbf
           << ','
           << totals.static_recomp_fallback_dcbi -
                  m_previous_totals.static_recomp_fallback_dcbi
           << ','
           << totals.static_recomp_fallback_icbi -
                  m_previous_totals.static_recomp_fallback_icbi
           << ','
           << totals.static_recomp_fallback_other -
                  m_previous_totals.static_recomp_fallback_other
           << ','
           << totals.static_recomp_cache_controls -
                  m_previous_totals.static_recomp_cache_controls
           << ','
           << totals.static_recomp_cache_control_dcbst -
                  m_previous_totals.static_recomp_cache_control_dcbst
           << ','
           << totals.static_recomp_cache_control_dcbf -
                  m_previous_totals.static_recomp_cache_control_dcbf
           << ','
           << totals.static_recomp_cache_control_dcbi -
                  m_previous_totals.static_recomp_cache_control_dcbi
           << ','
           << totals.static_recomp_cache_control_icbi -
                  m_previous_totals.static_recomp_cache_control_icbi
           << ','
           << totals.efb_vram_pipeline_misses - m_previous_totals.efb_vram_pipeline_misses
           << ',' << counter_ms(totals.efb_vram_shader_ns, m_previous_totals.efb_vram_shader_ns)
           << ','
           << counter_ms(totals.efb_vram_pipeline_ns, m_previous_totals.efb_vram_pipeline_ns)
           << ',' << totals.efb_ram_pipeline_misses - m_previous_totals.efb_ram_pipeline_misses
           << ',' << counter_ms(totals.efb_ram_shader_ns, m_previous_totals.efb_ram_shader_ns)
           << ',' << counter_ms(totals.efb_ram_pipeline_ns, m_previous_totals.efb_ram_pipeline_ns)
           << ',' << totals.xfb_output_requests - m_previous_totals.xfb_output_requests
           << ',' << totals.xfb_swap_queued - m_previous_totals.xfb_swap_queued
           << ',' << totals.xfb_swap_executed - m_previous_totals.xfb_swap_executed
           << ',' << totals.xfb_duplicates - m_previous_totals.xfb_duplicates
           << ',' << totals.xfb_presents - m_previous_totals.xfb_presents << '\n';

    if (!m_spike_captured)
    {
      const char* marker_path = std::getenv("MELEEPAD_FRAME_PHASE_SPIKE_MARKER");
      const char* threshold_text = std::getenv("MELEEPAD_FRAME_PHASE_SPIKE_MS");
      const char* min_frame_text = std::getenv("MELEEPAD_FRAME_PHASE_SPIKE_MIN_EMULATED_FRAME");
      const double threshold_ms = threshold_text ? std::strtod(threshold_text, nullptr) : 33.0;
      const u64 min_emulated_frame = min_frame_text ? std::strtoull(min_frame_text, nullptr, 10) : 0;
      if (marker_path && emulated_frame >= min_emulated_frame && total_ms > threshold_ms)
      {
        m_file.flush();
        std::ofstream marker(marker_path, std::ios_base::out);
        marker << frame << ',' << emulated_frame << ',' << total_ms << ",spike\n";
        m_spike_captured = true;
      }
    }

    if (!m_slow_armed && m_frame % m_rolling_frame_ms.size() == 0)
    {
      std::ifstream arm_file(m_slow_arm_path);
      m_slow_armed = arm_file.good();
    }

    if (!m_slow_window_captured && m_slow_armed && m_frame > m_slow_start_frame)
    {
      m_rolling_total_ms -= m_rolling_frame_ms[m_rolling_index];
      m_rolling_frame_ms[m_rolling_index] = total_ms;
      m_rolling_total_ms += total_ms;
      m_rolling_index = (m_rolling_index + 1) % m_rolling_frame_ms.size();
      m_rolling_count = std::min(m_rolling_count + 1, m_rolling_frame_ms.size());

      constexpr double slow_window_ms = 60.0 * 1000.0 / 55.0;
      const char* marker_path = std::getenv("MELEEPAD_FRAME_PHASE_SLOW_MARKER");
      if (marker_path && m_rolling_count == m_rolling_frame_ms.size() &&
          m_rolling_total_ms > slow_window_ms)
      {
        m_file.flush();
        std::ofstream marker(marker_path, std::ios_base::out);
        marker << "frame=" << (m_frame - 1) << '\n'
               << "rolling_total_ms=" << m_rolling_total_ms << '\n'
               << "rolling_fps=" << 60000.0 / m_rolling_total_ms << '\n';
        m_slow_window_captured = true;
      }
    }

    m_previous_frame_start = frame_start;
    m_previous_totals = totals;
    m_previous_task_events = task_events;
    m_previous_vertex_shaders_created = g_stats.num_vertex_shaders_created;
    m_previous_pixel_shaders_created = g_stats.num_pixel_shaders_created;
    m_previous_textures_created = g_stats.num_textures_created;
  }

private:
  std::ofstream m_file;
  TimePoint m_previous_frame_start{};
  Common::FramePhaseTiming::Totals m_previous_totals{};
  TaskEventCounts m_previous_task_events{};
  int m_previous_vertex_shaders_created = 0;
  int m_previous_pixel_shaders_created = 0;
  int m_previous_textures_created = 0;
  u64 m_frame = 0;
  std::array<double, 60> m_rolling_frame_ms{};
  std::size_t m_rolling_index = 0;
  std::size_t m_rolling_count = 0;
  double m_rolling_total_ms = 0.0;
  bool m_slow_window_captured = false;
  bool m_spike_captured = false;
  u64 m_slow_start_frame = 0;
  const char* m_slow_arm_path = nullptr;
  bool m_slow_armed = true;
};

FramePhaseLogger s_frame_phase_logger;
}  // namespace

// Stretches the native/internal analog resolution aspect ratio from ~4:3 to ~16:9
static float SourceAspectRatioToWidescreen(float source_aspect)
{
  return source_aspect * ((16.0f / 9.0f) / (4.0f / 3.0f));
}

static std::tuple<int, int> FindClosestIntegerResolution(float width, float height,
                                                         float aspect_ratio)
{
  // We can't round both the x and y resolution as that might generate an aspect ratio
  // further away from the target one, we also can't either ceil or floor both sides,
  // so we find the combination or flooring and ceiling that is closest to the target ar.
  const int ceiled_width = static_cast<int>(std::ceil(width));
  const int ceiled_height = static_cast<int>(std::ceil(height));
  const int floored_width = static_cast<int>(std::floor(width));
  const int floored_height = static_cast<int>(std::floor(height));

  int int_width = floored_width;
  int int_height = floored_height;

  float min_aspect_ratio_distance = std::numeric_limits<float>::max();
  for (const int new_width : std::array<int, 2>{ceiled_width, floored_width})
  {
    for (const int new_height : std::array<int, 2>{ceiled_height, floored_height})
    {
      const float new_aspect_ratio = static_cast<float>(new_width) / new_height;
      const float aspect_ratio_distance = std::abs((new_aspect_ratio / aspect_ratio) - 1.f);
      if (aspect_ratio_distance < min_aspect_ratio_distance)
      {
        min_aspect_ratio_distance = aspect_ratio_distance;
        int_width = new_width;
        int_height = new_height;
      }
    }
  }

  return std::make_tuple(int_width, int_height);
}

static void TryToSnapToXFBSize(int& width, int& height, int xfb_width, int xfb_height)
{
  // Screen is blanking (e.g. game booting up), nothing to do here
  if (xfb_width == 0 || xfb_height == 0)
    return;

  // If there's only 1 pixel of either horizontal or vertical resolution difference,
  // make the output size match a multiple of the XFB native resolution,
  // to achieve the highest quality (least scaling).
  // The reason why the threshold is 1 pixel (per internal resolution multiplier) is because of
  // minor inaccuracies of the VI aspect ratio (and because some resolutions are rounded
  // while other are floored).
  const unsigned int efb_scale = g_framebuffer_manager->GetEFBScale();
  const unsigned int pixel_difference_width = std::abs(width - xfb_width);
  const unsigned int pixel_difference_height = std::abs(height - xfb_height);
  // We ignore this if there's an offset on both hor and ver size,
  // as then we'd be changing the aspect ratio too much and would need to
  // re-calculate a lot of stuff (like black bars).
  if ((pixel_difference_width <= efb_scale && pixel_difference_height == 0) ||
      (pixel_difference_height <= efb_scale && pixel_difference_width == 0))
  {
    width = xfb_width;
    height = xfb_height;
  }
}

Presenter::Presenter()
{
  auto& video_events = GetVideoEvents();

  m_config_changed =
      video_events.config_changed_event.Register([this](u32 bits) { ConfigChanged(bits); });

  m_end_field_hook = video_events.vi_end_field_event.Register(
      [this] { m_immediate_swap_happened_this_field.store(false, std::memory_order_relaxed); });
}

Presenter::~Presenter()
{
  // Disable ControllerInterface's aspect ratio adjustments so mapping dialog behaves normally.
  g_controller_interface.SetAspectRatioAdjustment(1);
}

bool Presenter::Initialize()
{
  UpdateDrawRectangle();

  m_immediate_swap_happened_this_field.store(false, std::memory_order_relaxed);

  if (!g_gfx->IsHeadless())
  {
    SetBackbuffer(g_gfx->GetSurfaceInfo());

    m_post_processor = std::make_unique<VideoCommon::PostProcessing>();
    if (!m_post_processor->Initialize(m_backbuffer_format))
      return false;

    m_onscreen_ui = std::make_unique<OnScreenUI>();
    if (!m_onscreen_ui->Initialize(m_backbuffer_width, m_backbuffer_height, m_backbuffer_scale))
      return false;

    // Draw a blank frame (and complete OnScreenUI initialization)
    g_gfx->BindBackbuffer({{0.0f, 0.0f, 0.0f, 1.0f}});
    g_gfx->PresentBackbuffer();
  }

  return true;
}

bool Presenter::FetchXFB(u32 xfb_addr, u32 fb_width, u32 fb_stride, u32 fb_height, u64 ticks)
{
  ReleaseXFBContentLock();
  u64 old_xfb_id = m_last_xfb_id;

  if (fb_width == 0 || fb_height == 0)
  {
    // Game is blanking the screen
    m_xfb_entry.reset();
    m_xfb_rect = MathUtil::Rectangle<int>();
    m_last_xfb_id = std::numeric_limits<u64>::max();
  }
  else
  {
    m_xfb_entry =
        g_texture_cache->GetXFBTexture(xfb_addr, fb_width, fb_height, fb_stride, &m_xfb_rect);
    m_last_xfb_id = m_xfb_entry->id;

    m_xfb_entry->AcquireContentLock();
  }
  m_last_xfb_addr = xfb_addr;
  m_last_xfb_ticks = ticks;
  m_last_xfb_width = fb_width;
  m_last_xfb_stride = fb_stride;
  m_last_xfb_height = fb_height;

  return old_xfb_id == m_last_xfb_id;
}

void Presenter::ViSwap(u32 xfb_addr, u32 fb_width, u32 fb_stride, u32 fb_height, u64 ticks,
                       TimePoint presentation_time)
{
  bool is_duplicate = FetchXFB(xfb_addr, fb_width, fb_stride, fb_height, ticks);
  if (is_duplicate)
    Common::FramePhaseTiming::AddXfbDuplicate();

  PresentInfo present_info{
      .present_count = m_present_count++,
      .emulated_timestamp = ticks,
      .intended_present_time = presentation_time,
  };

  if (is_duplicate)
  {
    present_info.frame_count = m_frame_count - 1;  // Previous frame
    present_info.reason = PresentInfo::PresentReason::VideoInterfaceDuplicate;
  }
  else
  {
    present_info.frame_count = m_frame_count++;
    present_info.reason = PresentInfo::PresentReason::VideoInterface;
  }

  if (m_xfb_entry)
  {
    // With no references, this XFB copy wasn't stitched together
    // so just use its name directly
    if (m_xfb_entry->references.empty())
    {
      if (!m_xfb_entry->texture_info_name.empty())
        present_info.xfb_copy_hashes.push_back(m_xfb_entry->texture_info_name);
    }
    else
    {
      for (const auto& reference : m_xfb_entry->references)
      {
        if (!reference->texture_info_name.empty())
          present_info.xfb_copy_hashes.push_back(reference->texture_info_name);
      }
    }
  }

  auto& video_events = GetVideoEvents();

  video_events.before_present_event.Trigger(present_info);

  if (!is_duplicate || !g_ActiveConfig.bSkipPresentingDuplicateXFBs)
  {
    Common::FramePhaseTiming::AddXfbPresent();
    Present(&present_info);
    ProcessFrameDumping(ticks);

    video_events.after_present_event.Trigger(present_info);
  }
}

void Presenter::ImmediateSwap(u32 xfb_addr, u32 fb_width, u32 fb_stride, u32 fb_height)
{
  if (m_immediate_swap_happened_this_field.exchange(true, std::memory_order_relaxed) &&
      Config::Get(Config::GFX_HACK_CAP_IMMEDIATE_XFB))
  {
    return;
  }

  const u64 ticks = m_next_swap_estimated_ticks;

  FetchXFB(xfb_addr, fb_width, fb_stride, fb_height, ticks);

  PresentInfo present_info{
      .frame_count = m_frame_count++,
      .present_count = m_present_count++,
      .reason = PresentInfo::PresentReason::Immediate,
      .emulated_timestamp = ticks,
      .intended_present_time = m_next_swap_estimated_time,
  };

  auto& video_events = GetVideoEvents();

  video_events.before_present_event.Trigger(present_info);

  Present(&present_info);
  ProcessFrameDumping(ticks);

  video_events.after_present_event.Trigger(present_info);
}

void Presenter::SetNextSwapEstimatedTime(u64 ticks, TimePoint host_time)
{
  m_next_swap_estimated_ticks = ticks;
  m_next_swap_estimated_time = host_time;
}

void Presenter::ProcessFrameDumping(u64 ticks) const
{
  if (g_frame_dumper->IsFrameDumping() && m_xfb_entry)
  {
    MathUtil::Rectangle<int> target_rect;
    switch (Config::Get(Config::GFX_FRAME_DUMPS_RESOLUTION_TYPE))
    {
    default:
    case FrameDumpResolutionType::WindowResolution:
    {
      if (!g_gfx->IsHeadless())
      {
        target_rect = GetTargetRectangle();
        break;
      }
      [[fallthrough]];
    }
    case FrameDumpResolutionType::XFBAspectRatioCorrectedResolution:
    {
      target_rect = m_xfb_rect;
      const bool allow_stretch = false;
      auto [float_width, float_height] =
          ScaleToDisplayAspectRatio(m_xfb_rect.GetWidth(), m_xfb_rect.GetHeight(), allow_stretch);
      const float draw_aspect_ratio = CalculateDrawAspectRatio(allow_stretch);
      auto [int_width, int_height] =
          FindClosestIntegerResolution(float_width, float_height, draw_aspect_ratio);
      target_rect = MathUtil::Rectangle<int>(0, 0, int_width, int_height);
      break;
    }
    case FrameDumpResolutionType::XFBRawResolution:
    {
      target_rect = m_xfb_rect;
      break;
    }
    }

    int width = target_rect.GetWidth();
    int height = target_rect.GetHeight();

    const int resolution_lcm = g_frame_dumper->GetRequiredResolutionLeastCommonMultiple();

    // Ensure divisibility by the dumper LCM and a min of 1 to make it compatible with all the
    // video encoders. Note that this is theoretically only necessary when recording videos and not
    // screenshots.
    // We always scale positively to make sure the least amount of information is lost.
    //
    // TODO: this should be added as black padding on the edges by the frame dumper.
    if ((width % resolution_lcm) != 0 || width == 0)
      width += resolution_lcm - (width % resolution_lcm);
    if ((height % resolution_lcm) != 0 || height == 0)
      height += resolution_lcm - (height % resolution_lcm);

    // Remove any black borders, there would be no point in including them in the recording
    target_rect.left = 0;
    target_rect.top = 0;
    target_rect.right = width;
    target_rect.bottom = height;

    // TODO: any scaling done by this won't be gamma corrected,
    // we should either apply post processing as well, or port its gamma correction code
    g_frame_dumper->DumpCurrentFrame(m_xfb_entry->texture.get(), m_xfb_rect, target_rect, ticks,
                                     m_frame_count);
  }
}

void Presenter::SetBackbuffer(int backbuffer_width, int backbuffer_height)
{
  const bool is_first = m_backbuffer_width == 0 && m_backbuffer_height == 0;
  const bool size_changed =
      (m_backbuffer_width != backbuffer_width || m_backbuffer_height != backbuffer_height);
  m_backbuffer_width = backbuffer_width;
  m_backbuffer_height = backbuffer_height;
  UpdateDrawRectangle();

  OnBackbufferSet(size_changed, is_first);
}

void Presenter::SetBackbuffer(SurfaceInfo info)
{
  const bool is_first = m_backbuffer_width == 0 && m_backbuffer_height == 0;
  const bool size_changed =
      (m_backbuffer_width != (int)info.width || m_backbuffer_height != (int)info.height);
  m_backbuffer_width = info.width;
  m_backbuffer_height = info.height;
  m_backbuffer_scale = info.scale;
  m_backbuffer_format = info.format;
  if (m_onscreen_ui)
    m_onscreen_ui->SetScale(info.scale);

  OnBackbufferSet(size_changed, is_first);
}

void Presenter::OnBackbufferSet(bool size_changed, bool is_first_set)
{
  UpdateDrawRectangle();

  // Automatically update the resolution scale if the window size changed,
  // or if the game XFB resolution changed.
  if (size_changed && !is_first_set && g_ActiveConfig.iEFBScale == EFB_SCALE_AUTO_INTEGRAL &&
      m_auto_resolution_scale != AutoIntegralScale())
  {
    g_framebuffer_manager->RecreateEFBFramebuffer(g_ActiveConfig.iEFBScale);
  }
  if (size_changed || is_first_set)
  {
    m_auto_resolution_scale = AutoIntegralScale();
  }
}

void Presenter::ConfigChanged(u32 changed_bits)
{
  // Check for post-processing shader changes. Done up here as it doesn't affect anything outside
  // the post-processor. Note that options are applied every frame, so no need to check those.
  if (changed_bits & ConfigChangeBits::CONFIG_CHANGE_BIT_POST_PROCESSING_SHADER && m_post_processor)
  {
    // The existing shader must not be in use when it's destroyed
    g_gfx->WaitForGPUIdle();

    m_post_processor->RecompileShader();
  }

  // Stereo mode change requires recompiling our post processing pipeline and imgui pipelines for
  // rendering the UI.
  if (changed_bits & ConfigChangeBits::CONFIG_CHANGE_BIT_STEREO_MODE)
  {
    if (m_onscreen_ui)
      m_onscreen_ui->RecompileImGuiPipeline();
    if (m_post_processor)
      m_post_processor->RecompilePipeline();
  }
}

std::tuple<MathUtil::Rectangle<int>, MathUtil::Rectangle<int>>
Presenter::ConvertStereoRectangle(const MathUtil::Rectangle<int>& rc) const
{
  // Resize target to half its original size
  auto draw_rc = rc;
  if (g_ActiveConfig.stereo_mode == StereoMode::TopAndBottom)
  {
    // The height may be negative due to flipped rectangles
    int height = rc.bottom - rc.top;
    draw_rc.top += height / 4;
    draw_rc.bottom -= height / 4;
  }
  else
  {
    int width = rc.right - rc.left;
    draw_rc.left += width / 4;
    draw_rc.right -= width / 4;
  }

  // Create two target rectangle offset to the sides of the backbuffer
  auto left_rc = draw_rc;
  auto right_rc = draw_rc;
  if (g_ActiveConfig.stereo_mode == StereoMode::TopAndBottom)
  {
    left_rc.top -= m_backbuffer_height / 4;
    left_rc.bottom -= m_backbuffer_height / 4;
    right_rc.top += m_backbuffer_height / 4;
    right_rc.bottom += m_backbuffer_height / 4;
  }
  else
  {
    left_rc.left -= m_backbuffer_width / 4;
    left_rc.right -= m_backbuffer_width / 4;
    right_rc.left += m_backbuffer_width / 4;
    right_rc.right += m_backbuffer_width / 4;
  }

  return std::make_tuple(left_rc, right_rc);
}

MathUtil::Rectangle<int> Presenter::GetCustomCrop(const MathUtil::Rectangle<int>& rect) const
{
  if (!g_ActiveConfig.bCropCustom)
    return MathUtil::Rectangle<int>(0, 0, 0, 0);

  const int uncropped_source_width = rect.GetWidth();
  const int uncropped_source_height = rect.GetHeight();
  const int efb_scale = g_framebuffer_manager->GetEFBScale();

  // Determine amount of pixels to crop from the source rect.
  const int source_crop_left =
      std::min(g_ActiveConfig.iCropCustomLeft * efb_scale, uncropped_source_width);
  const int source_crop_right = std::min(g_ActiveConfig.iCropCustomRight * efb_scale,
                                         uncropped_source_width - source_crop_left);
  const int source_crop_top =
      std::min(g_ActiveConfig.iCropCustomTop * efb_scale, uncropped_source_height);
  const int source_crop_bottom = std::min(g_ActiveConfig.iCropCustomBottom * efb_scale,
                                          uncropped_source_height - source_crop_top);

  return MathUtil::Rectangle<int>(source_crop_left, source_crop_top, source_crop_right,
                                  source_crop_bottom);
}

MathUtil::Rectangle<int> Presenter::AdjustForCustomCrop(const MathUtil::Rectangle<int>& rect) const
{
  const MathUtil::Rectangle<int> crop = GetCustomCrop(rect);
  const MathUtil::Rectangle<int> cropped(rect.left + crop.left, rect.top + crop.top,
                                         rect.right - crop.right, rect.bottom - crop.bottom);
  return cropped;
}

float Presenter::AdjustAspectRatioForCustomCrop(float input_aspect_ratio) const
{
  if (!g_ActiveConfig.bCropCustom)
    return input_aspect_ratio;

  const MathUtil::Rectangle<int> rect = m_xfb_rect;
  if (rect.GetWidth() <= 0 || rect.GetHeight() <= 0)
    return input_aspect_ratio;

  const MathUtil::Rectangle<int> cropped = AdjustForCustomCrop(rect);
  const float relative_width_difference =
      static_cast<float>(cropped.GetWidth()) / static_cast<float>(rect.GetWidth());
  const float relative_height_difference =
      static_cast<float>(cropped.GetHeight()) / static_cast<float>(rect.GetHeight());
  return input_aspect_ratio * (relative_width_difference / relative_height_difference);
}

float Presenter::CalculateDrawAspectRatio(bool allow_stretch) const
{
  auto aspect_mode = g_ActiveConfig.aspect_mode;
  float resulting_aspect_ratio;

  if (!allow_stretch && aspect_mode == AspectMode::Stretch)
    aspect_mode = AspectMode::Auto;

  // If stretch is enabled, we prefer the aspect ratio of the window.
  if (aspect_mode == AspectMode::Stretch)
  {
    resulting_aspect_ratio =
        (static_cast<float>(m_backbuffer_width) / static_cast<float>(m_backbuffer_height));
  }
  else
  {
    // The actual aspect ratio of the XFB texture is irrelevant, the VI one is the one that matters
    const auto& vi = Core::System::GetInstance().GetVideoInterface();
    const float vi_aspect_ratio = vi.GetAspectRatio();
    const float source_aspect_ratio = AdjustAspectRatioForCustomCrop(vi_aspect_ratio);

    // This will scale up the source ~4:3 resolution to its equivalent ~16:9 resolution
    if (aspect_mode == AspectMode::ForceWide ||
        (aspect_mode == AspectMode::Auto && g_widescreen->IsGameWidescreen()))
    {
      resulting_aspect_ratio = SourceAspectRatioToWidescreen(source_aspect_ratio);
    }
    else if (aspect_mode == AspectMode::Custom)
    {
      resulting_aspect_ratio =
          source_aspect_ratio * (g_ActiveConfig.GetCustomAspectRatio() / (4.0f / 3.0f));
    }
    // For the "custom stretch" mode, we force the exact target aspect ratio, without
    // acknowledging the difference between the source aspect ratio and 4:3.
    else if (aspect_mode == AspectMode::CustomStretch)
    {
      resulting_aspect_ratio = g_ActiveConfig.GetCustomAspectRatio();
    }
    else if (aspect_mode == AspectMode::Raw)
    {
      resulting_aspect_ratio =
          m_xfb_entry ? (static_cast<float>(m_last_xfb_width) / m_last_xfb_height) : 1.f;
    }
    else
    {
      resulting_aspect_ratio = source_aspect_ratio;
    }
  }

  if (g_ActiveConfig.stereo_per_eye_resolution_full)
  {
    if (g_ActiveConfig.stereo_mode == StereoMode::SideBySide)
    {
      // Render twice as wide if using side-by-side 3D, since the 3D will halve the horizontal
      // resolution
      resulting_aspect_ratio *= 2.0;
    }
    else if (g_ActiveConfig.stereo_mode == StereoMode::TopAndBottom)
    {
      // Render twice as tall if using top-and-bottom 3D, since the 3D will halve the vertical
      // resolution
      resulting_aspect_ratio /= 2.0;
    }
  }

  return resulting_aspect_ratio;
}

void Presenter::AdjustRectanglesToFitBounds(MathUtil::Rectangle<int>* target_rect,
                                            MathUtil::Rectangle<int>* source_rect, int fb_width,
                                            int fb_height)
{
  const int orig_target_width = target_rect->GetWidth();
  const int orig_target_height = target_rect->GetHeight();
  const int orig_source_width = source_rect->GetWidth();
  const int orig_source_height = source_rect->GetHeight();
  if (target_rect->left < 0)
  {
    const int offset = -target_rect->left;
    target_rect->left = 0;
    source_rect->left += offset * orig_source_width / orig_target_width;
  }
  if (target_rect->right > fb_width)
  {
    const int offset = target_rect->right - fb_width;
    target_rect->right -= offset;
    source_rect->right -= offset * orig_source_width / orig_target_width;
  }
  if (target_rect->top < 0)
  {
    const int offset = -target_rect->top;
    target_rect->top = 0;
    source_rect->top += offset * orig_source_height / orig_target_height;
  }
  if (target_rect->bottom > fb_height)
  {
    const int offset = target_rect->bottom - fb_height;
    target_rect->bottom -= offset;
    source_rect->bottom -= offset * orig_source_height / orig_target_height;
  }
}

void Presenter::ReleaseXFBContentLock()
{
  if (m_xfb_entry)
    m_xfb_entry->ReleaseContentLock();
}

void Presenter::ChangeSurface(void* new_surface_handle)
{
  std::lock_guard<std::mutex> lock(m_swap_mutex);
  m_new_surface_handle = new_surface_handle;
  m_surface_changed.Set();
}

void Presenter::ResizeSurface()
{
  std::lock_guard<std::mutex> lock(m_swap_mutex);
  m_surface_resized.Set();
}

void* Presenter::GetNewSurfaceHandle()
{
  void* handle = m_new_surface_handle;
  m_new_surface_handle = nullptr;
  return handle;
}

u32 Presenter::AutoIntegralScale() const
{
  // Take the source/native resolution (XFB) and stretch it on the target (window) aspect ratio.
  // If the target resolution is larger (on either x or y), we scale the source
  // by a integer multiplier until it won't have to be scaled up anymore.
  // NOTE: this might conflict with "Config::MAIN_RENDER_WINDOW_AUTOSIZE",
  // as they mutually influence each other.
  u32 source_width = m_last_xfb_width;
  u32 source_height = m_last_xfb_height;
  const u32 target_width = m_target_rectangle.GetWidth();
  const u32 target_height = m_target_rectangle.GetHeight();
  const float source_aspect_ratio = (float)source_width / source_height;
  const float target_aspect_ratio = (float)target_width / target_height;
  if (source_aspect_ratio >= target_aspect_ratio)
    source_width = std::round(source_height * target_aspect_ratio);
  else
    source_height = std::round(source_width / target_aspect_ratio);
  const u32 width_scale =
      source_width > 0 ? ((target_width + (source_width - 1)) / source_width) : 1;
  const u32 height_scale =
      source_height > 0 ? ((target_height + (source_height - 1)) / source_height) : 1;
  // Limit to the max to avoid creating textures larger than their max supported resolution.
  return std::min(std::max(width_scale, height_scale),
                  static_cast<u32>(Config::Get(Config::GFX_MAX_EFB_SCALE)));
}

void Presenter::SetSuggestedWindowSize(int width, int height)
{
  // While trying to guess the best window resolution, we can't allow it to use the
  // "AspectMode::Stretch" setting because that would self influence the output result,
  // given it would be based on the previous frame resolution
  const bool allow_stretch = false;
  const auto [out_width, out_height] = CalculateOutputDimensions(width, height, allow_stretch);

  // Track the last values of width/height to avoid sending a window resize event every frame.
  if (out_width == m_last_window_request_width && out_height == m_last_window_request_height)
    return;

  m_last_window_request_width = out_width;
  m_last_window_request_height = out_height;
  // Pass in the suggested window size. This might not always be acknowledged.
  Host_RequestRenderWindowSize(out_width, out_height);
}

// Crop to exact forced aspect ratios if enabled and not AspectMode::Stretch.
std::tuple<float, float> Presenter::ApplyStandardAspectCrop(float width, float height,
                                                            bool allow_stretch) const
{
  auto aspect_mode = g_ActiveConfig.aspect_mode;

  if (!allow_stretch && aspect_mode == AspectMode::Stretch)
    aspect_mode = AspectMode::Auto;

  if (!g_ActiveConfig.bCropToAspectRatio || aspect_mode == AspectMode::Stretch ||
      aspect_mode == AspectMode::Raw)
  {
    return {width, height};
  }

  // Force aspect ratios by cropping the image.
  const float current_aspect = width / height;
  float expected_aspect;
  switch (aspect_mode)
  {
  default:
  case AspectMode::Auto:
    expected_aspect = g_widescreen->IsGameWidescreen() ? (16.0f / 9.0f) : (4.0f / 3.0f);
    break;
  case AspectMode::ForceWide:
    expected_aspect = 16.0f / 9.0f;
    break;
  case AspectMode::ForceStandard:
    expected_aspect = 4.0f / 3.0f;
    break;
  // For the custom (relative) case, we want to crop from the native aspect ratio
  // to the specific target one, as they likely have a small difference
  case AspectMode::Custom:
  // There should be no cropping needed in the custom stretch case,
  // as output should always exactly match the target aspect ratio
  case AspectMode::CustomStretch:
    expected_aspect = g_ActiveConfig.GetCustomAspectRatio();
    break;
  }

  if (current_aspect > expected_aspect)
  {
    // keep height, crop width
    width = height * expected_aspect;
  }
  else
  {
    // keep width, crop height
    height = width / expected_aspect;
  }

  return {width, height};
}

void Presenter::UpdateDrawRectangle()
{
  const float draw_aspect_ratio = CalculateDrawAspectRatio();

  // Update aspect ratio hack values
  // Won't take effect until next frame
  // Don't know if there is a better place for this code so there isn't a 1 frame delay
  if (g_ActiveConfig.bWidescreenHack)
  {
    const auto& vi = Core::System::GetInstance().GetVideoInterface();
    float source_aspect_ratio = vi.GetAspectRatio();
    // If the game is meant to be in widescreen (or forced to),
    // scale the source aspect ratio to it.
    if (g_widescreen->IsGameWidescreen())
      source_aspect_ratio = SourceAspectRatioToWidescreen(source_aspect_ratio);

    const float adjust = source_aspect_ratio / draw_aspect_ratio;
    if (adjust > 1)
    {
      // Vert+
      g_Config.fAspectRatioHackW = 1;
      g_Config.fAspectRatioHackH = 1 / adjust;
    }
    else
    {
      // Hor+
      g_Config.fAspectRatioHackW = adjust;
      g_Config.fAspectRatioHackH = 1;
    }
  }
  else
  {
    // Hack is disabled.
    g_Config.fAspectRatioHackW = 1;
    g_Config.fAspectRatioHackH = 1;
  }

  // The rendering window size
  const float win_width = static_cast<float>(m_backbuffer_width);
  const float win_height = static_cast<float>(m_backbuffer_height);
  const float win_aspect_ratio = win_width / win_height;

  // FIXME: this breaks at very low widget sizes
  // Make ControllerInterface aware of the render window region actually being used
  // to adjust mouse cursor inputs.
  // This also doesn't handle the image cropping settings.
  g_controller_interface.SetAspectRatioAdjustment(draw_aspect_ratio / win_aspect_ratio);

  float draw_width = draw_aspect_ratio;
  float draw_height = 1;

  // Crop the picture to a standard aspect ratio. (if enabled)
  auto [crop_width, crop_height] = ApplyStandardAspectCrop(draw_width, draw_height);
  const float crop_aspect_ratio = crop_width / crop_height;

  // scale the picture to fit the rendering window
  if (win_aspect_ratio >= crop_aspect_ratio)
  {
    // the window is flatter than the picture
    draw_width *= win_height / crop_height;
    crop_width *= win_height / crop_height;
    draw_height *= win_height / crop_height;
    crop_height = win_height;
  }
  else
  {
    // the window is skinnier than the picture
    draw_width *= win_width / crop_width;
    draw_height *= win_width / crop_width;
    crop_height *= win_width / crop_width;
    crop_width = win_width;
  }

  int int_draw_width;
  int int_draw_height;

  if (g_ActiveConfig.aspect_mode != AspectMode::Raw || !m_xfb_entry)
  {
    // Find the best integer resolution: the closest aspect ratio with the least black bars.
    // This should have no influence if "AspectMode::Stretch" is active.
    const float updated_draw_aspect_ratio = draw_width / draw_height;
    const auto int_draw_res =
        FindClosestIntegerResolution(draw_width, draw_height, updated_draw_aspect_ratio);
    int_draw_width = std::get<0>(int_draw_res);
    int_draw_height = std::get<1>(int_draw_res);
    if (!g_ActiveConfig.bCropToAspectRatio)
    {
      if (g_ActiveConfig.aspect_mode != AspectMode::Stretch)
      {
        const MathUtil::Rectangle<int> rect = AdjustForCustomCrop(m_xfb_rect);
        TryToSnapToXFBSize(int_draw_width, int_draw_height, rect.GetWidth(), rect.GetHeight());
      }
      // We can't draw something bigger than the window, it will crop
      int_draw_width = std::min(int_draw_width, static_cast<int>(win_width));
      int_draw_height = std::min(int_draw_height, static_cast<int>(win_height));
    }
  }
  else
  {
    const MathUtil::Rectangle<int> rect = AdjustForCustomCrop(m_xfb_rect);
    int_draw_width = rect.GetWidth();
    int_draw_height = rect.GetHeight();
  }

  m_target_rectangle.left = static_cast<int>(std::round(win_width / 2.0 - int_draw_width / 2.0));
  m_target_rectangle.top = static_cast<int>(std::round(win_height / 2.0 - int_draw_height / 2.0));
  m_target_rectangle.right = m_target_rectangle.left + int_draw_width;
  m_target_rectangle.bottom = m_target_rectangle.top + int_draw_height;
}

std::tuple<float, float> Presenter::ScaleToDisplayAspectRatio(const int width, const int height,
                                                              bool allow_stretch) const
{
  // Scale either the width or height depending the content aspect ratio.
  // This way we preserve as much resolution as possible when scaling.
  float scaled_width = static_cast<float>(width);
  float scaled_height = static_cast<float>(height);
  const float draw_aspect = CalculateDrawAspectRatio(allow_stretch);
  if (scaled_width / scaled_height >= draw_aspect)
    scaled_height = scaled_width / draw_aspect;
  else
    scaled_width = scaled_height * draw_aspect;
  return std::make_tuple(scaled_width, scaled_height);
}

std::tuple<int, int> Presenter::CalculateOutputDimensions(int width, int height,
                                                          bool allow_stretch) const
{
  // Protect against zero width and height, a minimum of 1 will do
  width = std::max(width, 1);
  height = std::max(height, 1);

  auto [scaled_width, scaled_height] = ScaleToDisplayAspectRatio(width, height, allow_stretch);

  // Apply crop if enabled.
  std::tie(scaled_width, scaled_height) =
      ApplyStandardAspectCrop(scaled_width, scaled_height, allow_stretch);

  auto aspect_mode = g_ActiveConfig.aspect_mode;

  if (!allow_stretch && aspect_mode == AspectMode::Stretch)
    aspect_mode = AspectMode::Auto;

  if (!g_ActiveConfig.bCropToAspectRatio && aspect_mode != AspectMode::Stretch)
  {
    // Find the closest integer resolution for the aspect ratio,
    // this avoids a small black line from being drawn on one of the four edges
    const float draw_aspect_ratio = CalculateDrawAspectRatio(allow_stretch);
    auto [int_width, int_height] =
        FindClosestIntegerResolution(scaled_width, scaled_height, draw_aspect_ratio);
    if (aspect_mode != AspectMode::Raw)
    {
      TryToSnapToXFBSize(int_width, int_height, m_xfb_rect.GetWidth(), m_xfb_rect.GetHeight());
    }
    width = int_width;
    height = int_height;
  }
  else
  {
    width = static_cast<int>(std::ceil(scaled_width));
    height = static_cast<int>(std::ceil(scaled_height));
  }

  return std::make_tuple(width, height);
}

void Presenter::RenderXFBToScreen(const MathUtil::Rectangle<int>& target_rc,
                                  const AbstractTexture* source_texture,
                                  const MathUtil::Rectangle<int>& source_rc)
{
  if (g_ActiveConfig.stereo_mode == StereoMode::QuadBuffer &&
      g_backend_info.bUsesExplictQuadBuffering)
  {
    // Quad-buffered stereo is annoying on GL.
    g_gfx->SelectLeftBuffer();
    m_post_processor->BlitFromTexture(target_rc, source_rc, source_texture, 0);

    g_gfx->SelectRightBuffer();
    m_post_processor->BlitFromTexture(target_rc, source_rc, source_texture, 1);

    g_gfx->SelectMainBuffer();
  }
  else if (g_ActiveConfig.stereo_mode == StereoMode::SideBySide ||
           g_ActiveConfig.stereo_mode == StereoMode::TopAndBottom)
  {
    const auto [left_rc, right_rc] = ConvertStereoRectangle(target_rc);

    m_post_processor->BlitFromTexture(left_rc, source_rc, source_texture, 0);
    m_post_processor->BlitFromTexture(right_rc, source_rc, source_texture, 1);
  }
  // Every other case will be treated the same (stereo or not).
  // If there's multiple source layers, they should all be copied.
  else
  {
    m_post_processor->BlitFromTexture(target_rc, source_rc, source_texture);
  }
}

void Presenter::Present(PresentInfo* present_info)
{
  const TimePoint frame_phase_start = Clock::now();
  m_present_count++;

  if (g_gfx->IsHeadless() || (!m_onscreen_ui && !m_xfb_entry))
    return;

  if (!g_gfx->SupportsUtilityDrawing())
  {
    // Video Software doesn't support drawing a UI or doing post-processing
    // So just show the XFB
    if (m_xfb_entry)
    {
      const MathUtil::Rectangle<int> rect = AdjustForCustomCrop(m_xfb_rect);
      g_gfx->ShowImage(m_xfb_entry->texture.get(), rect);

      // Update the window size based on the frame that was just rendered.
      // Due to depending on guest state, we need to call this every frame.
      SetSuggestedWindowSize(rect.GetWidth(), rect.GetHeight());
    }
    return;
  }

  // Since we use the common pipelines here and draw vertices if a batch is currently being
  // built by the vertex loader, we end up trampling over its pointer, as we share the buffer
  // with the loader, and it has not been unmapped yet. Force a pipeline flush to avoid this.
  g_vertex_manager->Flush();

  UpdateDrawRectangle();
  const TimePoint flush_and_rect_end = Clock::now();

  g_gfx->BeginUtilityDrawing();
  const bool backbuffer_bound = g_gfx->BindBackbuffer({{0.0f, 0.0f, 0.0f, 1.0f}});
  const TimePoint bind_backbuffer_end = Clock::now();

  // Render the XFB to the screen.
  if (backbuffer_bound && m_xfb_entry)
  {
    // Adjust the source rectangle instead of using an oversized viewport to render the XFB.
    MathUtil::Rectangle<int> render_target_rc = GetTargetRectangle();
    MathUtil::Rectangle<int> render_source_rc = AdjustForCustomCrop(m_xfb_rect);
    AdjustRectanglesToFitBounds(&render_target_rc, &render_source_rc, m_backbuffer_width,
                                m_backbuffer_height);
    RenderXFBToScreen(render_target_rc, m_xfb_entry->texture.get(), render_source_rc);
  }
  const TimePoint xfb_blit_end = Clock::now();

  if (m_onscreen_ui)
  {
    m_onscreen_ui->Finalize();
    if (backbuffer_bound)
      m_onscreen_ui->DrawImGui();
  }
  const TimePoint onscreen_ui_end = Clock::now();

  // Present to the window system.
  const TimePoint present_phase_start = Clock::now();
  {
    std::lock_guard<std::mutex> guard(m_swap_mutex);

    if (present_info != nullptr)
    {
      const auto present_time = GetUpdatedPresentationTime(present_info->intended_present_time);

      Core::System::GetInstance().GetCoreTiming().SleepUntil(
          present_time, CoreTiming::SleepReason::Presentation);

      // Perhaps in the future a more accurate time can be acquired from the various backends.
      present_info->actual_present_time = Clock::now();
      present_info->present_time_accuracy = PresentInfo::PresentTimeAccuracy::PresentInProgress;
    }

    g_gfx->PresentBackbuffer();
  }
  const TimePoint frame_phase_end = Clock::now();
  s_frame_phase_logger.Log(frame_phase_start, present_phase_start, frame_phase_end,
                           flush_and_rect_end - frame_phase_start,
                           bind_backbuffer_end - flush_and_rect_end,
                           xfb_blit_end - bind_backbuffer_end,
                           onscreen_ui_end - xfb_blit_end);

  if (m_xfb_entry)
  {
    // Update the window size based on the frame that was just rendered.
    // Due to depending on guest state, we need to call this every frame.
    const MathUtil::Rectangle<int> rect = AdjustForCustomCrop(m_xfb_rect);
    SetSuggestedWindowSize(rect.GetWidth(), rect.GetHeight());
  }

  if (m_onscreen_ui)
    m_onscreen_ui->BeginImGuiFrame(m_backbuffer_width, m_backbuffer_height);

  g_gfx->EndUtilityDrawing();
}

TimePoint Presenter::GetUpdatedPresentationTime(TimePoint intended_presentation_time)
{
  const auto now = Clock::now();
  const auto arrival_offset = std::min(now - intended_presentation_time, DT{});

  if (!Config::Get(Config::MAIN_SMOOTH_EARLY_PRESENTATION))
  {
    m_presentation_time_offset = arrival_offset;

    // When SmoothEarlyPresentation is off and ImmediateXFB or RushFramePresentation are on,
    //  present as soon as possible as the goal is to achieve low input latency.
    if (g_ActiveConfig.bImmediateXFB || Config::Get(Config::MAIN_RUSH_FRAME_PRESENTATION))
      return now;

    return intended_presentation_time;
  }

  // Adjust slowly backward in time but quickly forward in time.
  // This keeps the pacing moderately smooth even if games produce regular sporadic bumps.
  // This was tuned to handle the terrible pacing in Brawl with "Immediate XFB".
  // Super Mario Galaxy 1 + 2 still perform poorly here in SingleCore mode.
  const auto adjustment_divisor = (arrival_offset < m_presentation_time_offset) ? 100 : 2;

  m_presentation_time_offset += (arrival_offset - m_presentation_time_offset) / adjustment_divisor;

  return intended_presentation_time + m_presentation_time_offset;
}

void Presenter::SetKeyMap(const DolphinKeyMap& key_map)
{
  if (m_onscreen_ui)
    m_onscreen_ui->SetKeyMap(key_map);
}

void Presenter::SetKey(u32 key, bool is_down, const char* chars)
{
  if (m_onscreen_ui)
    m_onscreen_ui->SetKey(key, is_down, chars);
}

void Presenter::SetMousePos(float x, float y)
{
  if (m_onscreen_ui)
    m_onscreen_ui->SetMousePos(x, y);
}

void Presenter::SetMousePress(u32 button_mask)
{
  if (m_onscreen_ui)
    m_onscreen_ui->SetMousePress(button_mask);
}

void Presenter::DoState(PointerWrap& p)
{
  p.Do(m_frame_count);
  p.Do(m_last_xfb_ticks);
  p.Do(m_last_xfb_addr);
  p.Do(m_last_xfb_width);
  p.Do(m_last_xfb_stride);
  p.Do(m_last_xfb_height);

  // If we're loading and there is a last XFB, re-display it.
  if (p.IsReadMode() && m_last_xfb_stride != 0)
  {
    // This technically counts as the end of the frame
    GetVideoEvents().after_frame_event.Trigger(Core::System::GetInstance());

    m_next_swap_estimated_ticks = m_last_xfb_ticks;
    m_next_swap_estimated_time = Clock::now();

    m_immediate_swap_happened_this_field.store(false, std::memory_order_relaxed);

    ImmediateSwap(m_last_xfb_addr, m_last_xfb_width, m_last_xfb_stride, m_last_xfb_height);
  }
}

}  // namespace VideoCommon
