// Copyright 2010 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "Core/HW/GCPad.h"
#include <cstddef>

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#include "Common/Common.h"
#include "Core/HW/GCPadEmu.h"
#include "InputCommon/ControllerEmu/ControlGroup/ControlGroup.h"
#include "InputCommon/GCPadStatus.h"
#include "InputCommon/InputConfig.h"

#if defined(__APPLE__) && TARGET_OS_TV
namespace
{
struct SunPadTVPadSnapshot
{
  u16 buttons;
  u8 stick_x;
  u8 stick_y;
  u8 c_stick_x;
  u8 c_stick_y;
  u8 trigger_l;
  u8 trigger_r;
  u8 analog_a;
  u8 analog_b;
  s32 connected;
};
static_assert(sizeof(SunPadTVPadSnapshot) == 16);

extern "C" bool SunPadTVReadPadSnapshot(void* buffer, std::size_t size)
    __attribute__((weak_import));
extern "C" bool SunPadTVSetRumble(bool enabled) __attribute__((weak_import));
}  // namespace
#endif

namespace Pad
{
static InputConfig s_config("GCPadNew", _trans("Pad"), "GCPad", "Pad");
InputConfig* GetConfig()
{
  return &s_config;
}

void Shutdown()
{
  s_config.UnregisterHotplugCallback();

  s_config.ClearControllers();
}

void Initialize()
{
  if (s_config.ControllersNeedToBeCreated())
  {
    for (unsigned int i = 0; i < 4; ++i)
      s_config.CreateController<GCPad>(i);
  }

  s_config.RegisterHotplugCallback();

  // Load the saved controller config
  s_config.LoadConfig();
}

void LoadConfig()
{
  s_config.LoadConfig();
}

void GenerateDynamicInputTextures()
{
  s_config.GenerateControllerTextures();
}

bool IsInitialized()
{
  return !s_config.ControllersNeedToBeCreated();
}

GCPadStatus GetStatus(int pad_num)
{
#if defined(__APPLE__) && TARGET_OS_TV
  if (pad_num == 0 && SunPadTVReadPadSnapshot != nullptr)
  {
    SunPadTVPadSnapshot input{};
    if (SunPadTVReadPadSnapshot(&input, sizeof(input)))
    {
      GCPadStatus status{};
      status.button = input.buttons;
      status.stickX = input.stick_x;
      status.stickY = input.stick_y;
      status.substickX = input.c_stick_x;
      status.substickY = input.c_stick_y;
      status.triggerLeft = input.trigger_l;
      status.triggerRight = input.trigger_r;
      status.analogA = input.analog_a;
      status.analogB = input.analog_b;
      status.isConnected = input.connected != 0;
      return status;
    }
  }
#endif
  return static_cast<GCPad*>(s_config.GetController(pad_num))->GetInput();
}

ControllerEmu::ControlGroup* GetGroup(int pad_num, PadGroup group)
{
  return static_cast<GCPad*>(s_config.GetController(pad_num))->GetGroup(group);
}

void Rumble(const int pad_num, const ControlState strength)
{
#if defined(__APPLE__) && TARGET_OS_TV
  if (pad_num == 0 && SunPadTVSetRumble != nullptr && SunPadTVSetRumble(strength > 0.0))
    return;
#endif
  static_cast<GCPad*>(s_config.GetController(pad_num))->SetOutput(strength);
}

void ResetRumble(const int pad_num)
{
#if defined(__APPLE__) && TARGET_OS_TV
  if (pad_num == 0 && SunPadTVSetRumble != nullptr && SunPadTVSetRumble(false))
    return;
#endif
  static_cast<GCPad*>(s_config.GetController(pad_num))->SetOutput(0.0);
}

bool GetMicButton(const int pad_num)
{
  return static_cast<GCPad*>(s_config.GetController(pad_num))->GetMicButton();
}
}  // namespace Pad
