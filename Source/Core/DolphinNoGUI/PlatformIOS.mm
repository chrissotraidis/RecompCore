// Copyright 2026 SunPad project
// SPDX-License-Identifier: GPL-2.0-or-later

// iOS host platform for the ModernGekko runtime: hands the app-provided
// CAMetalLayer to the Metal video backend and pumps the runtime without any
// macOS windowing APIs.

#include "DolphinNoGUI/Platform.h"

#include <TargetConditionals.h>

#include <chrono>
#include <thread>

#include "Core/Core.h"
#include "Core/System.h"

namespace
{
void* g_render_surface = nullptr;
}

extern "C" void ModernGekkoSetIOSRenderSurface(void* surface)
{
  g_render_surface = surface;
}

namespace
{
class PlatformIOS : public Platform
{
public:
  ~PlatformIOS() override = default;

  bool Init() override { return true; }

  void SetTitle(const std::string& /*title*/) override {}

  void MainLoop() override
  {
    while (IsRunning())
    {
      UpdateRunningFlag();
      Core::HostDispatchJobs(Core::System::GetInstance());
      std::this_thread::sleep_for(std::chrono::milliseconds(8));
    }
  }

  WindowSystemInfo GetWindowSystemInfo() const override
  {
    WindowSystemInfo wsi;
    // Metal's WindowSystemTypeSupportsMetal accepts MacOS and treats
    // render_surface as the CAMetalLayer, which is what the app provides.
    wsi.type = WindowSystemType::MacOS;
    wsi.display_connection = nullptr;
    wsi.render_window = g_render_surface;
    wsi.render_surface = g_render_surface;
    return wsi;
  }
};
}  // namespace

std::unique_ptr<Platform> Platform::CreateIOSPlatform()
{
  return std::make_unique<PlatformIOS>();
}
