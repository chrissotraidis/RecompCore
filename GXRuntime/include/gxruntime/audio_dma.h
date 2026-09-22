// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef GXRUNTIME_AUDIO_DMA_H
#define GXRUNTIME_AUDIO_DMA_H

#include "core/types.h"

#define DOL_AUDIO_DMA_ENABLE 0x8000u
#define DOL_AUDIO_DMA_BLOCK_MASK 0x7FFFu
#define DOL_AUDIO_DMA_CHUNKS_PER_SECOND 4000u
#define DOL_AUDIO_DMA_FRAMES_PER_CHUNK 8u
#define DOL_AUDIO_DMA_32KHZ 32000u
#define DOL_AUDIO_DMA_48KHZ 48000u

// AI/DSP-AID device register geometry (Dolphin AudioInterface.h / DSP.h).
// The AI control register is at 0xCC006C00 and the DSP control block is at
// 0xCC005000; these offsets are register-file local, so the guest absolute
// bases stay owned by the game MMIO-bus registration.
#define DOL_AUDIO_DMA_AI_REG_BYTES 0x20u
#define DOL_AUDIO_DMA_DSP_REG_BYTES 0x40u

#define DOL_AUDIO_DMA_AI_CONTROL_OFF 0x00u
#define DOL_AUDIO_DMA_AI_SAMPLE_COUNTER_OFF 0x08u
// AICR bits (Dolphin AudioInterface.h). AIDFR is intentionally inverted w.r.t.
// the rate name: AIDFR set == 32 kHz AID, clear == 48 kHz AID.
#define DOL_AUDIO_DMA_AI_PSTAT_BIT 0x00000001u
#define DOL_AUDIO_DMA_AI_AISFR_BIT 0x00000002u
#define DOL_AUDIO_DMA_AI_SCRESET_BIT 0x00000020u
#define DOL_AUDIO_DMA_AI_AIDFR_BIT 0x00000040u
#define DOL_AUDIO_DMA_AI_CONTROL_INIT \
    (DOL_AUDIO_DMA_AI_AISFR_BIT | DOL_AUDIO_DMA_AI_AIDFR_BIT)

// UDSPControl bits (Dolphin DSP.h, DSP_CONTROL_MASK = 0x0C07). The AID bit is
// the DMA-to-AI/speakers interrupt status; AID_MSK gates it as a PI source.
#define DOL_AUDIO_DMA_DSP_CONTROL_OFF 0x0Au
#define DOL_AUDIO_DMA_DSP_AID_INT_BIT 0x0008u
#define DOL_AUDIO_DMA_DSP_AID_MSK_BIT 0x0010u

#define DOL_AUDIO_DMA_DSP_DMA_START_HI_OFF 0x30u
#define DOL_AUDIO_DMA_DSP_DMA_START_LO_OFF 0x32u
#define DOL_AUDIO_DMA_DSP_DMA_CONTROL_OFF 0x36u
#define DOL_AUDIO_DMA_DSP_DMA_BLOCKS_LEFT_OFF 0x3Au

typedef struct DolAudioDma {
    u16 control;
    u16 remaining_blocks;
    u32 source_address;
    u32 current_source_address;
    u32 sample_rate;
    u32 chunks_per_second;
    u32 refresh_hz;
    u64 work_counter;
    u64 work_units_per_second;
    u64 work_units_per_frame;
    u64 work_units_per_chunk;
    u64 sample_counter_remainder;
    u32 sample_counter;
    u32 stream_sample_rate;
    bool interrupt_pending;
    // Big-endian AI/DSP register files. AI CR DRIVES the AID sample rate; DSP
    // control carries the AID interrupt mask and DMA address/length state.
    u8 ai_regs[DOL_AUDIO_DMA_AI_REG_BYTES];
    u8 dsp_regs[DOL_AUDIO_DMA_DSP_REG_BYTES];
} DolAudioDma;

// Reads one hardware audio-DMA source chunk. The source is intentionally
// supplied by the caller so the runtime can use ARAM, a test fixture, or a
// future DSP-owned PCM store without baking storage ownership into this
// device primitive.
typedef bool (*DolAudioDmaReadFn)(void* user, u32 source_address, u8* data,
                                  u32 size);

void dol_audio_dma_init(DolAudioDma* dma);
void dol_audio_dma_set_work_rate(DolAudioDma* dma, u64 work_units_per_second);
void dol_audio_dma_set_vi_clock(DolAudioDma* dma, u64 work_units_per_frame,
                                u32 refresh_hz);
void dol_audio_dma_set_sample_rate(DolAudioDma* dma, u32 sample_rate);
u32 dol_audio_dma_sample_rate(const DolAudioDma* dma);
void dol_audio_dma_set_source(DolAudioDma* dma, u32 source_address);
void dol_audio_dma_write_control(DolAudioDma* dma, u16 control);
u16 dol_audio_dma_read_control(const DolAudioDma* dma);
u16 dol_audio_dma_blocks_left(const DolAudioDma* dma);
void dol_audio_dma_ack_interrupt(DolAudioDma* dma);
bool dol_audio_dma_interrupt_pending(const DolAudioDma* dma);

// Advance one guest-work unit through the compatibility API. New runtimes
// should use dol_audio_dma_advance with generated guest-cycle charges.
// Returns each 32-byte source chunk at the hardware's AID cadence.
bool dol_audio_dma_poll(DolAudioDma* dma, u32* source_address);

// Advance by an explicit number of units from the caller's guest clock.
// Unlike poll(), this preserves variable generated-cycle charges and their
// fractional progress toward the next 32-byte hardware chunk.
bool dol_audio_dma_advance(DolAudioDma* dma, u64 work_units,
                           u32* source_address);

// Consume one timed 32-byte big-endian stereo PCM16 chunk and send its eight
// frames to the installed platform audio sink. Returns false until the DMA
// cadence is ready or when the source callback fails.
bool dol_audio_dma_consume_pcm16_stereo(DolAudioDma* dma,
                                        DolAudioDmaReadFn read_source,
                                        void* user);
bool dol_audio_dma_consume_pcm16_stereo_work(DolAudioDma* dma,
                                             u64 work_units,
                                             DolAudioDmaReadFn read_source,
                                             void* user);

// Advance the AI streaming sample counter from the same guest-cycle domain as
// DMA. Reads do not advance time, and elapsed-work partitioning is invariant.
void dol_audio_dma_advance_stream(DolAudioDma* dma, u64 work_units);

// AI/DSP-AID device register MMIO. The game writes the AI control register to
// select the AID sample rate; DSP control carries the AID interrupt status/mask
// and the audio-DMA source address/control/blocks-left. These are reusable
// GameCube hardware semantics (Dolphin AudioInterface.cpp / DSP.cpp); title
// middleware (MusyX/AX) stays in the game client.
//
// Offsets are register-file local. Writes that change the AID rate also notify
// the platform HAL (dol_platform_audio_set_sample_rate), debounced against the
// currently applied rate. The game owns absolute MMIO base registration.
void dol_audio_dma_ai_mmio_read(DolAudioDma* dma, u32 offset, u8 size,
                                u64* value);
void dol_audio_dma_ai_mmio_write(DolAudioDma* dma, u32 offset, u8 size,
                                 u64 value);
void dol_audio_dma_dsp_mmio_read(DolAudioDma* dma, u32 offset, u8 size,
                                 u64* value);
void dol_audio_dma_dsp_mmio_write(DolAudioDma* dma, u32 offset, u8 size,
                                  u64 value);

// AID interrupt, gated by the guest's DSP AID mask, mapped into a PI source.
// True only while the DMA interrupt is pending AND the guest has enabled it.
bool dol_audio_dma_dsp_interrupt_pending(const DolAudioDma* dma);

#endif
