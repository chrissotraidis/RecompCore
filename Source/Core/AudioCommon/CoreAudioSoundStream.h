// Copyright 2008 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <AudioUnit/AudioUnit.h>

#include <vector>

#include "AudioCommon/SoundStream.h"

class CoreAudioSound final : public SoundStream
{
public:
  ~CoreAudioSound() override;
  bool Init() override;
  bool SetRunning(bool running) override;
  void SetVolume(int volume) override;
  static bool IsValid() { return true; }

private:
  static OSStatus OutputCallback(void* ref_con, AudioUnitRenderActionFlags* action_flags,
                                 const AudioTimeStamp* timestamp, UInt32 bus_number,
                                 UInt32 number_frames, AudioBufferList* io_data);
  bool SetOutputFormat(UInt32 channels, AudioChannelLayoutTag layout_tag);
  bool InitializeOutput(UInt32 channels, AudioChannelLayoutTag layout_tag);

  AudioUnit m_audio_unit = nullptr;
  UInt32 m_channels = 2;
  std::vector<float> m_surround_scratch;
  int m_volume = 100;
  bool m_running = false;
};
