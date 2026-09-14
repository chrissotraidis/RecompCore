// Copyright 2018 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "DolphinNoGUI/Platform.h"

#include <cstdio>

#include "Core/Core.h"
#include "Core/HW/ProcessorInterface.h"
#include "Core/IOS/IOS.h"
#include "Core/IOS/STM/STM.h"
#include "Core/State.h"
#include "Core/System.h"

Platform::~Platform() = default;

bool Platform::Init()
{
  return true;
}

void Platform::SetTitle(const std::string& title)
{
}

void Platform::UpdateRunningFlag()
{
  auto& system = Core::System::GetInstance();
  const Core::State core_state = Core::GetState(system);
  const bool can_process_state_request =
      core_state == Core::State::Running || core_state == Core::State::Paused;

  if (can_process_state_request && m_save_state_requested.TestAndClear())
  {
    std::fprintf(stderr, "[nogui] SIGUSR1: saving state to slot 1\n");
    State::Save(system, 1);
  }
  if (can_process_state_request && m_load_state_requested.TestAndClear())
  {
    std::fprintf(stderr, "[nogui] SIGUSR2: loading state from slot 1\n");
    State::Load(system, 1);
  }
  if (m_shutdown_requested.TestAndClear())
  {
    const auto ios = system.GetIOS();
    const auto stm = ios ? ios->GetDeviceByName("/dev/stm/eventhook") : nullptr;
    if (!m_tried_graceful_shutdown.IsSet() && stm &&
        std::static_pointer_cast<IOS::HLE::STMEventHookDevice>(stm)->HasHookInstalled())
    {
      system.GetProcessorInterface().PowerButton_Tap();
      m_tried_graceful_shutdown.Set();
    }
    else
    {
      m_running.Clear();
    }
  }
}

void Platform::Stop()
{
  m_running.Clear();
}

void Platform::RequestShutdown()
{
  m_shutdown_requested.Set();
}

void Platform::RequestSaveState()
{
  m_save_state_requested.Set();
}

void Platform::RequestLoadState()
{
  m_load_state_requested.Set();
}
