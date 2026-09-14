// Copyright 2008 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "AudioCommon/CoreAudioSoundStream.h"

#include <algorithm>
#include <cstdio>

#include "Common/Logging/Log.h"
#include "Core/Config/MainSettings.h"

OSStatus CoreAudioSound::OutputCallback(void* ref_con, AudioUnitRenderActionFlags* action_flags,
                                        const AudioTimeStamp* timestamp, UInt32 bus_number,
                                        UInt32 number_frames, AudioBufferList* io_data)
{
  auto* const sound = static_cast<CoreAudioSound*>(ref_con);
  if (!sound || !io_data || io_data->mNumberBuffers == 0)
    return noErr;

  AudioBuffer& buffer = io_data->mBuffers[0];
  const UInt32 bytes_per_sample = sound->m_channels == 2 ? sizeof(s16) : sizeof(float);
  const UInt32 available_frames =
      buffer.mDataByteSize / (sound->m_channels * bytes_per_sample);
  const UInt32 frames = std::min(number_frames, available_frames);
  if (!buffer.mData || frames == 0)
    return noErr;

  if (sound->m_channels == 6)
  {
    sound->GetMixer()->MixSurround(static_cast<float*>(buffer.mData), frames);
  }
  else if (sound->m_channels == 5)
  {
    const UInt32 scratch_frames =
        std::min<UInt32>(frames, sound->m_surround_scratch.size() / 6);
    if (scratch_frames != frames)
    {
      std::fill_n(static_cast<float*>(buffer.mData), frames * 5, 0.0f);
      return noErr;
    }
    sound->GetMixer()->MixSurround(sound->m_surround_scratch.data(), scratch_frames);
    float* output = static_cast<float*>(buffer.mData);
    for (UInt32 frame = 0; frame < scratch_frames; ++frame)
    {
      output[frame * 5 + 0] = sound->m_surround_scratch[frame * 6 + 0];
      output[frame * 5 + 1] = sound->m_surround_scratch[frame * 6 + 1];
      output[frame * 5 + 2] = sound->m_surround_scratch[frame * 6 + 2];
      output[frame * 5 + 3] = sound->m_surround_scratch[frame * 6 + 4];
      output[frame * 5 + 4] = sound->m_surround_scratch[frame * 6 + 5];
    }
  }
  else
  {
    sound->GetMixer()->Mix(static_cast<s16*>(buffer.mData), frames);
  }

  return noErr;
}

CoreAudioSound::~CoreAudioSound()
{
  if (!m_audio_unit)
    return;
  AudioOutputUnitStop(m_audio_unit);
  AudioUnitUninitialize(m_audio_unit);
  AudioComponentInstanceDispose(m_audio_unit);
}

bool CoreAudioSound::SetOutputFormat(UInt32 channels, AudioChannelLayoutTag layout_tag)
{
  AudioStreamBasicDescription format{};
  const bool surround = channels >= 5;
  const UInt32 bits = surround ? 32 : 16;
  FillOutASBDForLPCM(format, GetMixer()->GetSampleRate(), channels, bits, bits, surround, false,
                    false);
  if (AudioUnitSetProperty(m_audio_unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0,
                           &format, sizeof(format)) != noErr)
  {
    return false;
  }

  AudioChannelLayout layout{};
  layout.mChannelLayoutTag = layout_tag;
  return AudioUnitSetProperty(m_audio_unit, kAudioUnitProperty_AudioChannelLayout,
                              kAudioUnitScope_Input, 0, &layout, sizeof(layout)) == noErr;
}

bool CoreAudioSound::InitializeOutput(UInt32 channels, AudioChannelLayoutTag layout_tag)
{
  AudioUnitUninitialize(m_audio_unit);
  m_channels = channels;
  return SetOutputFormat(channels, layout_tag) && AudioUnitInitialize(m_audio_unit) == noErr;
}

bool CoreAudioSound::Init()
{
  AudioComponentDescription description{};
  description.componentType = kAudioUnitType_Output;
  description.componentSubType = kAudioUnitSubType_RemoteIO;
  description.componentManufacturer = kAudioUnitManufacturer_Apple;

  AudioComponent component = AudioComponentFindNext(nullptr, &description);
  if (!component || AudioComponentInstanceNew(component, &m_audio_unit) != noErr)
  {
    ERROR_LOG_FMT(AUDIO, "Could not create the iOS RemoteIO audio component");
    return false;
  }

  AURenderCallbackStruct callback{};
  callback.inputProc = OutputCallback;
  callback.inputProcRefCon = this;
  if (AudioUnitSetProperty(m_audio_unit, kAudioUnitProperty_SetRenderCallback,
                           kAudioUnitScope_Input, 0, &callback, sizeof(callback)) != noErr)
  {
    ERROR_LOG_FMT(AUDIO, "Could not set the iOS RemoteIO render callback");
    return false;
  }

  bool initialized = false;
  if (Config::Get(Config::MAIN_DPL2_DECODER))
  {
    initialized = InitializeOutput(6, kAudioChannelLayoutTag_MPEG_5_1_A);
    if (!initialized)
      initialized = InitializeOutput(5, kAudioChannelLayoutTag_MPEG_5_0_A);
  }
  if (!initialized)
    initialized = InitializeOutput(2, kAudioChannelLayoutTag_Stereo);
  if (!initialized)
  {
    ERROR_LOG_FMT(AUDIO, "Could not initialize the Apple RemoteIO PCM format");
    return false;
  }
  if (m_channels == 5)
  {
    UInt32 maximum_frames = 4096;
    UInt32 property_size = sizeof(maximum_frames);
    AudioUnitGetProperty(m_audio_unit, kAudioUnitProperty_MaximumFramesPerSlice,
                         kAudioUnitScope_Global, 0, &maximum_frames, &property_size);
    m_surround_scratch.resize(maximum_frames * 6);
  }

  AudioUnitSetParameter(m_audio_unit, kHALOutputParam_Volume, kAudioUnitScope_Output, 0,
                        m_volume / 100.0f, 0);
  std::fprintf(stderr, "[SunPad audio] RemoteIO initialized at %u Hz, channels=%u, DPL2=%d\n",
               GetMixer()->GetSampleRate(), m_channels, m_channels >= 5);
  return true;
}

bool CoreAudioSound::SetRunning(bool running)
{
  if (!m_audio_unit || m_running == running)
    return m_audio_unit != nullptr;
  const OSStatus status = running ? AudioOutputUnitStart(m_audio_unit) :
                                    AudioOutputUnitStop(m_audio_unit);
  if (status == noErr)
    m_running = running;
  return status == noErr;
}

void CoreAudioSound::SetVolume(int volume)
{
  m_volume = std::clamp(volume, 0, 100);
  if (m_audio_unit)
    AudioUnitSetParameter(m_audio_unit, kHALOutputParam_Volume, kAudioUnitScope_Output, 0,
                          m_volume / 100.0f, 0);
}
