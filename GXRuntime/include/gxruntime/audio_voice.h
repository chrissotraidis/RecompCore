// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef GXRUNTIME_AUDIO_VOICE_H
#define GXRUNTIME_AUDIO_VOICE_H

#include "core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DOL_AUDIO_MAX_VOICES 16u

typedef struct DolAudioVoice {
    const s16* samples;
    u32 sample_count;
    u32 position;
    u16 volume;
    s16 pan;
    bool active;
} DolAudioVoice;

typedef struct DolAudioVoiceMixer {
    DolAudioVoice voices[DOL_AUDIO_MAX_VOICES];
} DolAudioVoiceMixer;

void dol_audio_voice_init(DolAudioVoiceMixer* mixer);

// Starts one mono PCM voice. Volume is Q15 [0, 32767], pan is Q15
// [-32768, 32767] where zero is centered. Returns a voice id or -1.
s32 dol_audio_voice_start(DolAudioVoiceMixer* mixer, const s16* samples,
                          u32 sample_count, u16 volume, s16 pan);
bool dol_audio_voice_stop(DolAudioVoiceMixer* mixer, s32 voice_id);

// Mixes up to `frame_count` stereo PCM frames and returns the number of input
// frames consumed. Completed voices become inactive.
u32 dol_audio_voice_mix(DolAudioVoiceMixer* mixer, s16* output,
                        u32 frame_count);

#ifdef __cplusplus
}
#endif

#endif
