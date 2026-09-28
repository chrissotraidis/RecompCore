// SPDX-License-Identifier: GPL-3.0-or-later
#include "gxruntime/audio_voice.h"

#include <string.h>

static s16 clamp_s16(s32 value) {
    if (value < -32768)
        return -32768;
    if (value > 32767)
        return 32767;
    return (s16)value;
}

void dol_audio_voice_init(DolAudioVoiceMixer* mixer) {
    if (mixer != NULL)
        memset(mixer, 0, sizeof(*mixer));
}

s32 dol_audio_voice_start(DolAudioVoiceMixer* mixer, const s16* samples,
                          u32 sample_count, u16 volume, s16 pan) {
    if (mixer == NULL || samples == NULL || sample_count == 0u)
        return -1;
    for (u32 i = 0; i < DOL_AUDIO_MAX_VOICES; i++) {
        DolAudioVoice* voice = &mixer->voices[i];
        if (voice->active)
            continue;
        voice->samples = samples;
        voice->sample_count = sample_count;
        voice->position = 0u;
        voice->volume = volume;
        voice->pan = pan;
        voice->active = true;
        return (s32)i;
    }
    return -1;
}

bool dol_audio_voice_stop(DolAudioVoiceMixer* mixer, s32 voice_id) {
    if (mixer == NULL || voice_id < 0 || voice_id >= (s32)DOL_AUDIO_MAX_VOICES)
        return false;
    DolAudioVoice* voice = &mixer->voices[voice_id];
    if (!voice->active)
        return false;
    voice->active = false;
    return true;
}

u32 dol_audio_voice_mix(DolAudioVoiceMixer* mixer, s16* output,
                        u32 frame_count) {
    if (mixer == NULL || output == NULL || frame_count == 0u)
        return 0u;
    memset(output, 0, frame_count * 2u * sizeof(*output));
    for (u32 i = 0; i < DOL_AUDIO_MAX_VOICES; i++) {
        DolAudioVoice* voice = &mixer->voices[i];
        if (!voice->active)
            continue;
        const s32 left_pan = 32767 - (s32)voice->pan;
        const s32 right_pan = 32767 + (s32)voice->pan;
        const s32 left_gain = ((s32)voice->volume * left_pan) / 32767;
        const s32 right_gain = ((s32)voice->volume * right_pan) / 32767;
        for (u32 frame = 0; frame < frame_count; frame++) {
            if (voice->position >= voice->sample_count) {
                voice->active = false;
                break;
            }
            const s32 sample = voice->samples[voice->position++];
            const u32 output_index = frame * 2u;
            const s32 left = output[output_index] +
                             (sample * left_gain) / 32767;
            const s32 right = output[output_index + 1u] +
                              (sample * right_gain) / 32767;
            output[output_index] = clamp_s16(left);
            output[output_index + 1u] = clamp_s16(right);
            if (voice->position >= voice->sample_count)
                voice->active = false;
        }
    }
    return frame_count;
}
