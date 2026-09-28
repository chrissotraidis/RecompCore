// SPDX-License-Identifier: GPL-3.0-or-later
#include "gxruntime/audio_adpcm.h"

#include <string.h>

static s16 clamp_s16(s32 value) {
    if (value < -32768)
        return -32768;
    if (value > 32767)
        return 32767;
    return (s16)value;
}

u32 dol_dsp_adpcm_decode(const u8* data, u32 data_size, u32 sample_count,
                         const DolDspAdpcmInfo* info, s16* output) {
    if (data == NULL || info == NULL || output == NULL || sample_count == 0u)
        return 0u;
    const u32 frame_count = (sample_count + 13u) / 14u;
    if (data_size < frame_count * 8u ||
        (sample_count % 14u) != 0u)
        return 0u;

    s16 yn1 = info->yn1;
    s16 yn2 = info->yn2;
    u32 sample_index = 0u;
    for (u32 frame = 0u; frame < frame_count; frame++) {
        const u8 pred_scale = data[frame * 8u];
        const u32 predictor = (u32)(pred_scale >> 4);
        const u32 scale_shift = (u32)(pred_scale & 0x0Fu);
        if (predictor >= 8u || scale_shift >= 15u)
            return 0u;
        const s32 scale = (s32)1 << scale_shift;
        const s16 coef1 = info->coef[predictor * 2u];
        const s16 coef2 = info->coef[predictor * 2u + 1u];

        for (u32 nibble_index = 0u; nibble_index < 14u; nibble_index++) {
            const u8 packed = data[frame * 8u + 1u + nibble_index / 2u];
            s32 nibble = (nibble_index & 1u) == 0u ? packed >> 4u : packed & 0x0Fu;
            if (nibble >= 8)
                nibble -= 16;
            const s32 predicted =
                ((s32)coef1 * (s32)yn1 + (s32)coef2 * (s32)yn2 + 1024) >> 11;
            const s16 sample = clamp_s16(scale * nibble + predicted);
            output[sample_index++] = sample;
            yn2 = yn1;
            yn1 = sample;
        }
    }
    return sample_index;
}
