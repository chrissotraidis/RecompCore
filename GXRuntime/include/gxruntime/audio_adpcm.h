// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef GXRUNTIME_AUDIO_ADPCM_H
#define GXRUNTIME_AUDIO_ADPCM_H

#include "core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Nintendo DSP ADPCM predictor state. The coefficient layout matches the
// 16-s16 DSPADPCMInfo coefficient array used by licensed GameCube references.
typedef struct DolDspAdpcmInfo {
    s16 coef[16];
    s16 yn1;
    s16 yn2;
} DolDspAdpcmInfo;

// Decode complete 8-byte DSP ADPCM frames (one predictor/scale byte plus
// fourteen nibbles) into signed PCM16 samples. Returns the number of samples
// written; partial final frames are rejected by returning zero.
u32 dol_dsp_adpcm_decode(const u8* data, u32 data_size, u32 sample_count,
                         const DolDspAdpcmInfo* info, s16* output);

#ifdef __cplusplus
}
#endif

#endif
