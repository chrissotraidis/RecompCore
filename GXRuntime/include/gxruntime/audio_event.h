// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef GXRUNTIME_AUDIO_EVENT_H
#define GXRUNTIME_AUDIO_EVENT_H

#include "gxruntime/audio_voice.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DOL_AUDIO_MAX_SOUND_ENTRIES 64u

typedef struct DolAudioSoundEntry {
    u32 sound_id;
    const s16* samples;
    u32 sample_count;
    bool registered;
} DolAudioSoundEntry;

typedef struct DolAudioEventAdapter {
    DolAudioVoiceMixer mixer;
    DolAudioSoundEntry sounds[DOL_AUDIO_MAX_SOUND_ENTRIES];
} DolAudioEventAdapter;

void dol_audio_event_init(DolAudioEventAdapter* adapter);
bool dol_audio_event_register(DolAudioEventAdapter* adapter, u32 sound_id,
                              const s16* samples, u32 sample_count);
s32 dol_audio_event_start(DolAudioEventAdapter* adapter, u32 sound_id,
                          u16 volume, s16 pan);
bool dol_audio_event_stop(DolAudioEventAdapter* adapter, s32 voice_id);
u32 dol_audio_event_mix(DolAudioEventAdapter* adapter, s16* output,
                        u32 frame_count);

#ifdef __cplusplus
}
#endif

#endif
