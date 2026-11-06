#include <aurora/aurora.h>

#ifdef AURORA_ENABLE_GX
#include "gfx/common.hpp"
#include "gfx/render_worker.hpp"
#include "gx/fifo.hpp"
#include "imgui.hpp"
#include <imgui.h>

#include <atomic>
#include <cstdlib>
#include "webgpu/gpu.hpp"
#include "webgpu/gpu_prof.hpp"
#include <webgpu/webgpu_cpp.h>
#endif

#ifdef AURORA_ENABLE_RMLUI
#include "rmlui.hpp"
#endif

#include "input.hpp"
#include "internal.hpp"
#include "window.hpp"

#include <SDL3/SDL_filesystem.h>
#include <SDL3/SDL_timer.h>
#include <magic_enum.hpp>
#include <string_view>

#include "system_info.hpp"
#include "tracy/Tracy.hpp"

namespace aurora {
AuroraConfig g_config;
uint32_t g_sdlCustomEventsStart;
char g_gameName[4];

AuroraFpsOverlayPosition fps_overlay_position(std::string_view name) noexcept {
  return name == "top-left"       ? FPS_OVERLAY_TOP_LEFT
         : name == "top-right"    ? FPS_OVERLAY_TOP_RIGHT
         : name == "bottom-left"  ? FPS_OVERLAY_BOTTOM_LEFT
         : name == "bottom-right" ? FPS_OVERLAY_BOTTOM_RIGHT
                                  : FPS_OVERLAY_TOP_CENTER;
}

namespace {
Module Log("aurora");

#ifdef AURORA_ENABLE_GX
// GPU
using webgpu::g_device;
using webgpu::g_queue;
using webgpu::g_surface;

uint32_t clamp_scissor_coord(double value, uint32_t maximum) noexcept {
  if (!std::isfinite(value)) {
    return 0;
  }
  return static_cast<uint32_t>(std::clamp(value, 0.0, static_cast<double>(maximum)));
}

void set_present_viewport(const wgpu::RenderPassEncoder& pass, const gfx::Viewport& viewport, uint32_t surfaceWidth,
                          uint32_t surfaceHeight) noexcept {
  pass.SetViewport(viewport.left, viewport.top, viewport.width, viewport.height, viewport.znear, viewport.zfar);
  const auto scissorX = clamp_scissor_coord(std::floor(viewport.left), surfaceWidth);
  const auto scissorY = clamp_scissor_coord(std::floor(viewport.top), surfaceHeight);
  const auto scissorRight = clamp_scissor_coord(std::ceil(viewport.left + viewport.width), surfaceWidth);
  const auto scissorBottom = clamp_scissor_coord(std::ceil(viewport.top + viewport.height), surfaceHeight);
  pass.SetScissorRect(scissorX, scissorY, scissorRight - scissorX, scissorBottom - scissorY);
}
#endif

#ifdef AURORA_ENABLE_GX
constexpr std::array PreferredBackendOrder{
#ifdef ENABLE_BACKEND_WEBGPU
    BACKEND_WEBGPU,
#endif
#ifdef DAWN_ENABLE_BACKEND_D3D12
    BACKEND_D3D12,
#endif
#ifdef DAWN_ENABLE_BACKEND_METAL
    BACKEND_METAL,
#endif
#ifdef DAWN_ENABLE_BACKEND_VULKAN
    BACKEND_VULKAN,
#endif
#ifdef DAWN_ENABLE_BACKEND_D3D11
    BACKEND_D3D11,
#endif
// #ifdef DAWN_ENABLE_BACKEND_DESKTOP_GL
//     BACKEND_OPENGL,
// #endif
#ifdef DAWN_ENABLE_BACKEND_OPENGLES
    BACKEND_OPENGLES,
#endif
#ifdef DAWN_ENABLE_BACKEND_NULL
    BACKEND_NULL,
#endif
};
#else
constexpr std::array<AuroraBackend, 0> PreferredBackendOrder{};
#endif

bool g_initialFrame = false;

AuroraInfo initialize(int argc, char* argv[], const AuroraConfig& config) noexcept {
  g_config = config;
  Log.info("Aurora initializing");
  log_system_information();
  if (g_config.appName == nullptr) {
    g_config.appName = "Aurora";
  } else {
    g_config.appName = strdup(g_config.appName);
  }
  if (g_config.userPath == nullptr) {
    g_config.userPath = SDL_GetPrefPath(nullptr, g_config.appName);
  } else {
    g_config.userPath = strdup(g_config.userPath);
  }
  if (g_config.cachePath == nullptr) {
    g_config.cachePath = SDL_GetPrefPath(nullptr, g_config.appName);
  } else {
    g_config.cachePath = strdup(g_config.cachePath);
  }
  if (g_config.resourcesPath == nullptr) {
    g_config.resourcesPath = SDL_GetBasePath();
  } else {
    g_config.resourcesPath = strdup(g_config.resourcesPath);
  }
  if (g_config.msaa == 0) {
    g_config.msaa = 1;
  }
  if (g_config.maxTextureAnisotropy == 0) {
    g_config.maxTextureAnisotropy = 16;
  }
  ASSERT(window::initialize(), "Error initializing window");

  g_sdlCustomEventsStart = SDL_RegisterEvents(2);
  ASSERT(g_sdlCustomEventsStart, "Failed to allocate user events: {}", SDL_GetError());
  ASSERT(window::initialize_event_watch(), "Error initializing SDL event watch");

#ifdef AURORA_ENABLE_GX
  /* Attempt to create a window using the calling application's desired backend */
  AuroraBackend selectedBackend = config.desiredBackend;
  bool windowCreated = false;
  if (selectedBackend != BACKEND_AUTO && window::create_window(selectedBackend)) {
    if (webgpu::initialize(selectedBackend, config.allowCpuAdapter)) {
      windowCreated = true;
    } else {
      window::destroy_window();
    }
  }

  if (!windowCreated) {
    for (const auto backendType : PreferredBackendOrder) {
      selectedBackend = backendType;
      if (!window::create_window(selectedBackend)) {
        continue;
      }
      if (webgpu::initialize(selectedBackend, config.allowCpuAdapter)) {
        windowCreated = true;
        break;
      } else {
        window::destroy_window();
      }
    }
  }

  ASSERT(windowCreated, "Error creating window: {}", SDL_GetError());

  // Initialize SDL_Renderer for ImGui when we can't use a Dawn backend
  if (webgpu::g_backendType == wgpu::BackendType::Null) {
    ASSERT(window::create_renderer(), "Failed to initialize SDL renderer: {}", SDL_GetError());
  }
#else
  AuroraBackend selectedBackend = BACKEND_NULL;
  ASSERT(window::create_window(BACKEND_NULL), "Error creating window: {}", SDL_GetError());
  ASSERT(window::create_renderer(), "Failed to initialize SDL renderer: {}", SDL_GetError());
#endif

  window::show_window();

#ifdef AURORA_ENABLE_GX
  gfx::initialize();

  imgui::create_context();
  // DOL_AURORA_FPS_POSITION sets where the FPS counter starts.
  if (const char* env = std::getenv("DOL_AURORA_FPS_POSITION"); env != nullptr) {
    aurora_set_fps_overlay_position(fps_overlay_position(env));
  }
#endif
  const auto size = window::get_window_size();
  Log.info("Using framebuffer size {}x{} scale {}", size.fb_width, size.fb_height, size.scale);
#ifdef AURORA_ENABLE_GX
  if (g_config.imGuiInitCallback != nullptr) {
    g_config.imGuiInitCallback(&size);
  }
  imgui::initialize();
#endif

#ifdef AURORA_ENABLE_RMLUI
  rmlui::initialize(size);
#endif

  g_initialFrame = true;
  g_config.desiredBackend = selectedBackend;
  return {
      .backend = selectedBackend,
      .userPath = g_config.userPath,
      .cachePath = g_config.cachePath,
      .window = window::get_sdl_window(),
      .windowSize = size,
  };
}

void shutdown() noexcept {
#ifdef AURORA_ENABLE_GX
  gfx::render_worker::synchronize();
#ifdef AURORA_ENABLE_RMLUI
  rmlui::shutdown();
#endif
  imgui::shutdown();
  gfx::shutdown();
  webgpu::shutdown();
#endif
  input::shutdown();
  window::shutdown();
}

const AuroraEvent* update() noexcept {
  ZoneScoped;
  if (g_initialFrame) {
    g_initialFrame = false;
    input::initialize();
  }
  return window::poll_events();
}

bool begin_frame() noexcept {
  ZoneScoped;
#ifdef AURORA_ENABLE_GX
  {
    if (!window::is_presentable()) {
      webgpu::release_surface();
      return false;
    }
    if (window::is_paused()) {
      return false;
    }
    if (!g_surface) {
      webgpu::refresh_surface(true);
      if (!g_surface) {
        return false;
      }
    }
  }

  imgui::new_frame(window::get_window_size());
  if (!gfx::begin_frame()) {
    return false;
  }
#endif
  return true;
}

#ifdef AURORA_ENABLE_GX
// Frames a second at the top of the window (DOL_AURORA_SHOW_FPS=1 or
// aurora_set_fps_overlay): every present, and the game's own frames under it
// when in-between frames make the two differ.
static std::atomic_bool g_showFps{[] {
  const char* env = std::getenv("DOL_AURORA_SHOW_FPS");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}()};
// Where (aurora_set_fps_overlay_position).
static std::atomic_int g_fpsPosition{FPS_OVERLAY_TOP_CENTER};

static void draw_fps_overlay() {
  if (!g_showFps.load(std::memory_order_relaxed)) {
    return;
  }
  const float shown = gfx::calculate_fps();
  const float game = gfx::calculate_game_fps();
  // With Smooth Motion's in-between frames paused (the game ran slow a moment
  // ago), each game frame is presented twice: the count says 60 while the
  // picture moves at 30. Over the last second, fewer than 90% of game frames
  // interpolated says so.
  static AuroraFrameInterpTotals last{};
  static Uint64 last_at = 0;
  static bool paused = false;
  const Uint64 now = SDL_GetTicks();
  if (now - last_at >= 1000) {
    AuroraFrameInterpTotals totals{};
    aurora_get_frame_interp_totals(&totals);
    const unsigned long long frames = totals.frames - last.frames;
    const unsigned long long interpolated = totals.interpolated - last.interpolated;
    paused = aurora_get_frame_interpolation() && frames > 0 && interpolated * 10 < frames * 9;
    last = totals;
    last_at = now;
  }
  // Top center unless a corner is chosen: the corners hold the game's HUD
  // (hearts, buttons, map, rupees). Anchored to the window as it is this frame.
  const int position = g_fpsPosition.load(std::memory_order_relaxed);
  const bool center = position == FPS_OVERLAY_TOP_CENTER;
  const bool right = position == FPS_OVERLAY_TOP_RIGHT || position == FPS_OVERLAY_BOTTOM_RIGHT;
  const bool bottom = position == FPS_OVERLAY_BOTTOM_LEFT || position == FPS_OVERLAY_BOTTOM_RIGHT;
  constexpr float margin = 12.f;
  const ImVec2 display = ImGui::GetIO().DisplaySize;
  const float x = center ? display.x * 0.5f : right ? display.x - margin : margin;
  ImGui::SetNextWindowPos(ImVec2{x, bottom ? display.y - margin : margin}, ImGuiCond_Always,
                          ImVec2{center ? 0.5f : right ? 1.f : 0.f, bottom ? 1.f : 0.f});
  ImGui::SetNextWindowBgAlpha(0.6f);
  constexpr ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                                     ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoSavedSettings |
                                     ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav;
  if (ImGui::Begin("##aurora-fps", nullptr, flags)) {
    ImGui::SetWindowFontScale(1.5f);
    if (paused && game > 0.f) {
      ImGui::Text("%.0f FPS (Smooth Motion paused)", game);
    } else if (game > 0.f && shown > game + 5.f) {
      ImGui::Text("%.0f FPS (game %.0f)", shown, game);
    } else {
      ImGui::Text("%.0f FPS", shown);
    }
  }
  ImGui::End();
}
#endif

static std::atomic_bool g_presentSuppressed{false};

// The surface configure count when its texture was last Lost (render worker
// only), or kNotLost, and how many times a Lost surface has been configured
// again since the last frame that had a texture.
constexpr uint32_t kNotLost = UINT32_MAX;
static uint32_t g_lostAtConfigure = kNotLost;
static uint32_t g_lostReconfigures = 0;
static uint32_t g_lostWaits = 0;
// A window dragged to a new size can lose the surface again before the frame
// after its reconfigure: each of those is a new loss, not a surface that stays
// lost, so it is only released after this many configures in a row without a
// frame (#325).
constexpr uint32_t kLostReconfigureLimit = 30;
// How many more times a Lost or Outdated texture is asked for in the same
// frame, and how many times that has happened (render worker only).
constexpr int kAcquireRetries = 2;
static uint64_t g_acquireRetries = 0;
static uint64_t g_acquireRecovered = 0;
// The configure count when a suboptimal texture last asked for a configure.
static uint32_t g_suboptimalAtConfigure = kNotLost;
// The configure count at the last texture acquired (render worker only).
static uint32_t g_configuresAtAcquire = 0;

void end_frame() noexcept {
  ZoneScoped;
#ifdef AURORA_ENABLE_GX
  gx::fifo::drain();
  gfx::finish();
  draw_fps_overlay();
  auto imguiDrawData = imgui::freeze();

  const auto& presentSource = webgpu::present_source();
  const auto viewport = webgpu::calculate_present_viewport(webgpu::g_graphicsConfig.surfaceConfiguration.width,
                                                           webgpu::g_graphicsConfig.surfaceConfiguration.height,
                                                           presentSource.size.width, presentSource.size.height);

  wgpu::BindGroup rmlBindGroup;
  bool rmlOverlay = false;
#if AURORA_ENABLE_RMLUI
  if (rmlui::is_initialized()) {
    auto rmlFrame = rmlui::record_frame(viewport);
    rmlBindGroup = std::move(rmlFrame.bindGroup);
    rmlOverlay = rmlFrame.overlay;
  }
#endif

  gfx::end_frame([rmlBindGroup = std::move(rmlBindGroup), rmlOverlay, viewport,
                  imguiDrawData = std::move(imguiDrawData)](wgpu::CommandEncoder& encoder) {
    auto presentViewport = viewport;
    wgpu::Texture currentTexture;
    wgpu::TextureView currentView;
    auto surfaceStatus = wgpu::SurfaceGetCurrentTextureStatus::Error;
    // Read once: a suppressed present takes no texture, and its status (left
    // at Error) must not be taken for the surface's.
    const bool suppressed = g_presentSuppressed.load(std::memory_order_relaxed);
    // Out of sight, no texture is asked for either (window::is_occluded).
    const bool occluded = window::is_occluded();
    {
      window::SurfaceLock surfaceLock;
      if (!suppressed && !occluded && window::is_presentable() && g_surface) {
        ZoneScopedN("Acquire texture");
        // The window's newest size first (Vulkan, window::follow_window_size):
        // the picture is placed in the surface at its new size.
        webgpu::follow_surface_size();
        presentViewport = webgpu::calculate_present_viewport(
            webgpu::g_graphicsConfig.surfaceConfiguration.width, webgpu::g_graphicsConfig.surfaceConfiguration.height,
            webgpu::present_source().size.width, webgpu::present_source().size.height);
        wgpu::SurfaceTexture surfaceTexture;
        const auto acquireStart = std::chrono::steady_clock::now();
        g_surface.GetCurrentTexture(&surfaceTexture);
        // While a window is being resized, Vulkan's swapchain goes out of date
        // with every step. Dawn rebuilds it at the window's size and acquires
        // again, and reports Lost when the window has changed again in
        // between. Asking again rebuilds it at the newest size, and the frame
        // is drawn (stretched by Dawn until the main thread resizes the frame
        // buffers) instead of skipped, which left the window's old contents
        // trailing behind its edges for the whole drag (#325).
        bool retried = false;
        for (int retry = 0; retry < kAcquireRetries &&
                            (surfaceTexture.status == wgpu::SurfaceGetCurrentTextureStatus::Lost ||
                             surfaceTexture.status == wgpu::SurfaceGetCurrentTextureStatus::Outdated);
             ++retry) {
          surfaceTexture = {};
          g_surface.GetCurrentTexture(&surfaceTexture);
          ++g_acquireRetries;
          retried = true;
          if (surfaceTexture.status == wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal ||
              surfaceTexture.status == wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal) {
            if (++g_acquireRecovered % 50 == 1)
              Log.info("Surface texture acquired on a retry ({} of {} retries)", g_acquireRecovered,
                       g_acquireRetries);
          }
        }
        const auto acquireMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::steady_clock::now() - acquireStart)
                                   .count();
        if (acquireMs >= 30)
          std::fprintf(stderr, "[render-slow] acquire-drawable ms=%lld\n", static_cast<long long>(acquireMs));
        // Every drawable still queued behind the GPU's work: fewer in-between
        // frames. Not the first texture of a surface configured since the last
        // one, or one asked for again: that wait is the driver making the new
        // swapchain, and it turned Smooth Motion's in-between frames off for 3 s
        // after every step of a window being resized (#325).
        const uint32_t configures = webgpu::g_surfaceConfigures.load(std::memory_order_relaxed);
        const bool newSwapchain = configures != g_configuresAtAcquire || retried;
        g_configuresAtAcquire = configures;
        if (acquireMs >= 20 && !newSwapchain)
          gfx::note_render_overload(static_cast<double>(acquireMs));
        surfaceStatus = surfaceTexture.status;
        if (surfaceStatus == wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal ||
            surfaceStatus == wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal) {
          g_lostAtConfigure = kNotLost;
          g_lostReconfigures = 0;
          g_lostWaits = 0;
        }
        // A suboptimal texture can still be presented: it is drawn, and the
        // surface is configured again after it (below).
        if (surfaceStatus == wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal ||
            surfaceStatus == wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal) {
          currentTexture = std::move(surfaceTexture.texture);
          currentView = currentTexture.CreateView();
        }
      }
    }

    const bool canPresent = currentTexture && currentView;
    if (canPresent) {
      wgpu::BindGroup presentBindGroup;
      if (rmlBindGroup && !rmlOverlay) {
        presentBindGroup = rmlBindGroup;
      } else {
        const auto& resampledSource = webgpu::resample_present_source(encoder, presentViewport);
        presentBindGroup = webgpu::create_copy_bind_group(resampledSource);
      }
      {
        const std::array attachments{
            wgpu::RenderPassColorAttachment{
                .view = currentView,
                .loadOp = wgpu::LoadOp::Clear,
                .storeOp = wgpu::StoreOp::Store,
            },
        };
        const wgpu::RenderPassDescriptor renderPassDescriptor{
            .label = "EFB copy render pass",
            .colorAttachmentCount = attachments.size(),
            .colorAttachments = attachments.data(),
            .timestampWrites = webgpu::gpu_prof::pass_writes("Present blit"),
        };
        const auto pass = encoder.BeginRenderPass(&renderPassDescriptor);
        // Copy EFB -> XFB (swapchain)
        pass.SetPipeline(webgpu::g_CopyPipeline);
        pass.SetBindGroup(0, presentBindGroup, 0, nullptr);
        set_present_viewport(pass, presentViewport, webgpu::g_graphicsConfig.surfaceConfiguration.width,
                             webgpu::g_graphicsConfig.surfaceConfiguration.height);

        pass.Draw(3);
        if (rmlBindGroup && rmlOverlay) {
          pass.SetPipeline(webgpu::g_CopyPremultipliedAlphaPipeline);
          pass.SetBindGroup(0, rmlBindGroup, 0, nullptr);
          pass.Draw(3);
        }
        pass.End();
      }
      {
        const std::array attachments{
            wgpu::RenderPassColorAttachment{
                .view = currentView,
                .loadOp = wgpu::LoadOp::Load,
                .storeOp = wgpu::StoreOp::Store,
            },
        };
        const wgpu::RenderPassDescriptor renderPassDescriptor{
            .label = "ImGui render pass",
            .colorAttachmentCount = attachments.size(),
            .colorAttachments = attachments.data(),
            .timestampWrites = webgpu::gpu_prof::pass_writes("ImGui"),
        };
        const auto pass = encoder.BeginRenderPass(&renderPassDescriptor);
        pass.SetViewport(0.f, 0.f, static_cast<float>(webgpu::g_graphicsConfig.surfaceConfiguration.width),
                         static_cast<float>(webgpu::g_graphicsConfig.surfaceConfiguration.height), 0.f, 1.f);
        imgui::render(pass, imguiDrawData);
        pass.End();
      }
    } else if (!suppressed) {
      Log.info("Skipping present; window not presentable");
    }
    webgpu::gpu_prof::frame_end(encoder);
    const wgpu::CommandBufferDescriptor cmdBufDescriptor{.label = "Redraw command buffer"};
    const auto buffer = encoder.Finish(&cmdBufDescriptor);
    {
      ZoneScopedN("Queue Submit");
      g_queue.Submit(1, &buffer);
    }
    webgpu::gpu_prof::after_submit();
    if (canPresent && g_surface) {
      ZoneScopedN("Present");
      wgpu::ConvertibleStatus status = wgpu::Status::Error;
      {
        window::SurfaceLock surfaceLock;
        if (window::is_presentable()) {
          const auto presentStart = std::chrono::steady_clock::now();
          status = g_surface.Present();
          const auto presentMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::steady_clock::now() - presentStart)
                                     .count();
          if (presentMs >= 30)
            std::fprintf(stderr, "[render-slow] present ms=%lld\n", static_cast<long long>(presentMs));
        }
      }
      if (status) {
        gfx::after_present();
        // Presented, but the swapchain no longer matches the window: configure
        // it again, once per configure (it stays suboptimal until then).
        if (surfaceStatus == wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal) {
          const uint32_t configures = webgpu::g_surfaceConfigures.load(std::memory_order_relaxed);
          if (g_suboptimalAtConfigure != configures) {
            g_suboptimalAtConfigure = configures;
            window::push_custom_event(window::CustomEvent::RefreshSurface);
          }
        }
      } else {
        Log.warn("Surface present failed");
        webgpu::release_surface();
      }
    } else if (g_surface && !suppressed && !occluded) {
      // With the present suppressed (or the window out of sight) the status is
      // not the surface's: taking it for Error dropped a working surface, and
      // on Direct3D 12 the swapchain made in its place for the same window
      // failed (E_ACCESSDENIED) and lost the device at the first fast-forwarded
      // scene change.
      switch (surfaceStatus) {
      case wgpu::SurfaceGetCurrentTextureStatus::Timeout:
        Log.warn("Surface texture acquisition timed out");
        break;
      case wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal:
      case wgpu::SurfaceGetCurrentTextureStatus::Outdated:
        Log.info("Surface texture is {}, reconfiguring swapchain", magic_enum::enum_name(surfaceStatus));
        window::push_custom_event(window::CustomEvent::RefreshSurface);
        break;
      case wgpu::SurfaceGetCurrentTextureStatus::Lost: {
        // Reconfigure the surface first, as for Outdated. Releasing it at
        // once made Vulkan on Linux (X11 and Wayland) lose it on a resize or
        // a fullscreen switch, and the surface made in its place for the same
        // window then failed with VK_ERROR_NATIVE_WINDOW_IN_USE_KHR while the
        // old swapchain still held the window (#325). A loss while the
        // reconfigure asked for is still waiting asks for nothing more; only a
        // surface lost through kLostReconfigureLimit configures in a row,
        // with no frame between them, is released and made anew.
        const uint32_t configures = webgpu::g_surfaceConfigures.load(std::memory_order_relaxed);
        if (g_lostAtConfigure == configures) {
          // A refresh can end without a configure (a window not presentable
          // for a moment): ask again now and then rather than wait for good.
          if (++g_lostWaits % 60 == 0)
            window::push_custom_event(window::CustomEvent::RefreshSurface);
          break;
        }
        g_lostWaits = 0;
        if (g_lostReconfigures < kLostReconfigureLimit) {
          ++g_lostReconfigures;
          Log.warn("Surface texture is {}, reconfiguring swapchain ({} in a row)",
                   magic_enum::enum_name(surfaceStatus), g_lostReconfigures);
          g_lostAtConfigure = configures;
          window::push_custom_event(window::CustomEvent::RefreshSurface);
        } else {
          Log.warn("Surface texture is still {} after {} reconfigures, releasing surface",
                   magic_enum::enum_name(surfaceStatus), g_lostReconfigures);
          g_lostAtConfigure = kNotLost;
          g_lostReconfigures = 0;
          webgpu::release_surface();
        }
        break;
      }
      case wgpu::SurfaceGetCurrentTextureStatus::Error:
        Log.warn("Surface texture is {}, dropping surface", magic_enum::enum_name(surfaceStatus));
        g_surface = {};
        break;
      default:
        if (!window::is_presentable()) {
          webgpu::release_surface();
        } else {
          Log.error("Failed to get surface texture: {}", magic_enum::enum_name(surfaceStatus));
        }
        break;
      }
    }
    gfx::after_submit();

    TracyPlotConfig("aurora: lastVertSize", tracy::PlotFormatType::Memory, false, true, 0);
    TracyPlotConfig("aurora: lastUniformSize", tracy::PlotFormatType::Memory, false, true, 0);
    TracyPlotConfig("aurora: lastIndexSize", tracy::PlotFormatType::Memory, false, true, 0);
    TracyPlotConfig("aurora: lastStorageSize", tracy::PlotFormatType::Memory, false, true, 0);
    TracyPlotConfig("aurora: lastTextureUploadSize", tracy::PlotFormatType::Memory, false, true, 0);

    TracyPlot("aurora: queuedPipelines", static_cast<int64_t>(gfx::g_stats.queuedPipelines));
    TracyPlot("aurora: createdPipelines", static_cast<int64_t>(gfx::g_stats.createdPipelines));
    TracyPlot("aurora: drawCallCount", static_cast<int64_t>(gfx::g_stats.drawCallCount));
    TracyPlot("aurora: mergedDrawCallCount", static_cast<int64_t>(gfx::g_stats.mergedDrawCallCount));
    TracyPlot("aurora: lastVertSize", static_cast<int64_t>(gfx::g_stats.lastVertSize));
    TracyPlot("aurora: lastUniformSize", static_cast<int64_t>(gfx::g_stats.lastUniformSize));
    TracyPlot("aurora: lastIndexSize", static_cast<int64_t>(gfx::g_stats.lastIndexSize));
    TracyPlot("aurora: lastStorageSize", static_cast<int64_t>(gfx::g_stats.lastStorageSize));
    TracyPlot("aurora: lastTextureUploadSize", static_cast<int64_t>(gfx::g_stats.lastTextureUploadSize));
  });

#endif
}
} // namespace
} // namespace aurora

// C API bindings
AuroraInfo aurora_initialize(int argc, char* argv[], const AuroraConfig* config) {
  return aurora::initialize(argc, argv, *config);
}
void aurora_shutdown() { aurora::shutdown(); }
const AuroraEvent* aurora_update() { return aurora::update(); }
bool aurora_begin_frame() { return aurora::begin_frame(); }
void aurora_set_fps_overlay(bool enabled) {
#ifdef AURORA_ENABLE_GX
  aurora::g_showFps.store(enabled, std::memory_order_relaxed);
#else
  (void)enabled;
#endif
}
void aurora_set_fps_overlay_position(AuroraFpsOverlayPosition position) {
#ifdef AURORA_ENABLE_GX
  aurora::g_fpsPosition.store(position, std::memory_order_relaxed);
#else
  (void)position;
#endif
}
void aurora_end_frame() { aurora::end_frame(); }
AuroraBackend aurora_get_backend() { return aurora::g_config.desiredBackend; }
const AuroraBackend* aurora_get_available_backends(size_t* count) {
  if (count != nullptr) {
    *count = aurora::PreferredBackendOrder.size();
  }
  return aurora::PreferredBackendOrder.data();
}
void aurora_set_log_level(AuroraLogLevel level) { aurora::g_config.logLevel = level; }
void aurora_set_pause_on_focus_lost(bool value) { aurora::g_config.pauseOnFocusLost = value; }
void aurora_set_background_input(bool value) {
  aurora::g_config.allowJoystickBackgroundEvents = value;
  aurora::window::set_background_input(value);
}
void aurora_set_frame_buffer_scale(float scale) { aurora::window::set_frame_buffer_scale(scale); }
void aurora_set_present_suppressed(bool suppressed) {
#ifdef AURORA_ENABLE_GX
  aurora::g_presentSuppressed.store(suppressed, std::memory_order_relaxed);
#else
  (void)suppressed;
#endif
}
void aurora_set_resampler(AuroraSampler sampler) {
#ifdef AURORA_ENABLE_GX
  aurora::webgpu::set_resampler(sampler);
#else
  (void)sampler;
#endif
}
