// SPDX-License-Identifier: GPL-3.0-or-later
#include "gxruntime/audio_event.h"

#include <string.h>

void dol_audio_event_init(DolAudioEventAdapter* adapter) {
    if (adapter == NULL)
        return;
    memset(adapter, 0, sizeof(*adapter));
    dol_audio_voice_init(&adapter->mixer);
}

bool dol_audio_event_register(DolAudioEventAdapter* adapter, u32 sound_id,
                              const s16* samples, u32 sample_count) {
    if (adapter == NULL || samples == NULL || sample_count == 0u)
        return false;
    for (u32 i = 0; i < DOL_AUDIO_MAX_SOUND_ENTRIES; i++) {
        DolAudioSoundEntry* entry = &adapter->sounds[i];
        if (entry->registered && entry->sound_id == sound_id)
            return false;
    }
    for (u32 i = 0; i < DOL_AUDIO_MAX_SOUND_ENTRIES; i++) {
        DolAudioSoundEntry* entry = &adapter->sounds[i];
        if (entry->registered)
            continue;
        entry->sound_id = sound_id;
        entry->samples = samples;
        entry->sample_count = sample_count;
        entry->registered = true;
        return true;
    }
    return false;
}

s32 dol_audio_event_start(DolAudioEventAdapter* adapter, u32 sound_id,
                          u16 volume, s16 pan) {
    if (adapter == NULL)
        return -1;
    for (u32 i = 0; i < DOL_AUDIO_MAX_SOUND_ENTRIES; i++) {
        const DolAudioSoundEntry* entry = &adapter->sounds[i];
        if (entry->registered && entry->sound_id == sound_id)
            return dol_audio_voice_start(&adapter->mixer, entry->samples,
                                         entry->sample_count, volume, pan);
    }
    return -1;
}

bool dol_audio_event_stop(DolAudioEventAdapter* adapter, s32 voice_id) {
    return adapter != NULL && dol_audio_voice_stop(&adapter->mixer, voice_id);
}

u32 dol_audio_event_mix(DolAudioEventAdapter* adapter, s16* output,
                        u32 frame_count) {
    return adapter != NULL ? dol_audio_voice_mix(&adapter->mixer, output,
                                                  frame_count)
                            : 0u;
}
