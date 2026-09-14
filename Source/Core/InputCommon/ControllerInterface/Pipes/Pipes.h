// Copyright 2015 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <map>
#include <string>
#include <vector>

#include "InputCommon/ControllerInterface/ControllerInterface.h"

namespace ciface::Pipes
{
// To create a piped controller input, create a named pipe in the
// Pipes directory and write commands out to it. Commands are separated
// by a newline character, with spaces separating command tokens.
// Command syntax is as follows, where curly brackets are one-of and square
// brackets are inclusive numeric ranges. Cases are sensitive. Numeric inputs
// are clamped to [0, 1] and otherwise invalid commands are discarded.
// {PRESS, RELEASE} {A, B, X, Y, Z, START, L, R, D_UP, D_DOWN, D_LEFT, D_RIGHT}
// SET {L, R} [0, 1]
// SET {MAIN, C} [0, 1] [0, 1]

std::unique_ptr<ciface::InputBackend> CreateInputBackend(ControllerInterface* controller_interface);

class PipeDevice : public Core::Device
{
public:
  PipeDevice(int fd, std::string name);
  ~PipeDevice();

  Core::DeviceRemoval UpdateInput() override;
  std::string GetName() const override { return m_name; }
  std::string GetSource() const override { return "Pipe"; }

private:
  class PipeInput : public Input
  {
  public:
    PipeInput(const std::string& name, bool latch_short_press = false)
        : m_name(name), m_latch_short_press(latch_short_press)
    {
    }
    std::string GetName() const override { return m_name; }
    ControlState GetState() const override
    {
      const ControlState state = m_state;
      if (m_latch_short_press && state > 0.0)
      {
        m_press_observed = true;
        if (m_release_pending)
        {
          m_state = 0.0;
          m_release_pending = false;
        }
      }
      return state;
    }
    void SetState(ControlState state)
    {
      if (!m_latch_short_press)
      {
        m_state = state;
        return;
      }
      if (state > 0.0)
      {
        if (m_state <= 0.0)
          m_press_observed = false;
        m_release_pending = false;
        m_state = state;
        return;
      }
      if (m_state > 0.0 && !m_press_observed)
      {
        m_release_pending = true;
        return;
      }
      m_state = state;
    }

  private:
    const std::string m_name;
    const bool m_latch_short_press;
    mutable ControlState m_state = 0.0;
    mutable bool m_press_observed = false;
    mutable bool m_release_pending = false;
  };

  void AddAxis(const std::string& name, double value);
  void ParseCommand(const std::string& command);
  void SetAxis(const std::string& entry, double value);

  const int m_fd;
  const std::string m_name;
  std::string m_buf;
  std::map<std::string, PipeInput*> m_buttons;
  std::map<std::string, PipeInput*> m_axes;
};
}  // namespace ciface::Pipes
