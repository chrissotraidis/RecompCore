// SPDX-License-Identifier: GPL-3.0-or-later
#include "aurora_backend_private.h"
#include "audio_stretch.hpp"
#include <cstdlib>
#include <vector>
#include <SDL3/SDL_audio.h>
#include <SDL3/SDL_timer.h>
#include <cstdio>

namespace {
// Stretch control (DOL_AUDIO_STRETCH=0 turns it off). The factor aims the
// queue at kStretchTargetMs: 1 at or above it, rising to kStretchMax as the
// queue empties, smoothed over about 100 ms of pushes. Nothing is stretched
// before playback starts, so the prebuffer fills with the game's own timing.
constexpr double kStretchTargetMs = 100.0;
constexpr double kStretchMax = 2.0;
gx_aurora::AudioStretcher g_stretcher;
std::vector<int16_t> g_stretch_out;
double g_stretch = 1.0;
bool stretch_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("DOL_AUDIO_STRETCH");
        return env == nullptr || env[0] != '0';
    }();
    return enabled;
}
}  // namespace

extern "C" {

void aurora_backend_audio_set_sample_rate(u32 sample_rate) {
    if (sample_rate != 48000)
        sample_rate = 32000;
    if (gx_aurora::g_audio_sample_rate == sample_rate)
        return;
    gx_aurora::g_audio_sample_rate = sample_rate;
    if (gx_aurora::g_audio_stream == nullptr)
        return;

    SDL_AudioSpec spec{};
    spec.format = SDL_AUDIO_S16;
    spec.channels = 2;
    spec.freq = static_cast<int>(gx_aurora::g_audio_sample_rate);
    if (!SDL_SetAudioStreamFormat(gx_aurora::g_audio_stream, &spec, nullptr))
        std::fprintf(stderr, "[audio] failed to set input rate %u: %s\n",
                     gx_aurora::g_audio_sample_rate, SDL_GetError());
    else if (gx_aurora::g_audio_queue_log)
        std::fprintf(stderr, "[audio-queue] input-rate=%u\n",
                     gx_aurora::g_audio_sample_rate);
}

void aurora_backend_audio_push(const s16* samples, u32 frames) {
    if (gx_aurora::g_audio_stream == nullptr || samples == nullptr || frames == 0)
        return;
    const int bytes_per_second =
        static_cast<int>(gx_aurora::g_audio_sample_rate) * 2 * static_cast<int>(sizeof(s16));
    const int prebuffer_bytes = bytes_per_second * gx_aurora::g_audio_prebuffer_ms / 1000;
    const int max_queued_bytes = bytes_per_second * gx_aurora::g_audio_max_queue_ms / 1000;
    const int bytes = static_cast<int>(frames * 2u * sizeof(s16));
    gx_aurora::g_audio_push_count++;
    u32 nonzero_samples = 0;
    s32 peak_sample = 0;
    u32 sample_hash = 2166136261u;
    for (u32 i = 0; i < frames * 2u; i++) {
        const s32 sample = samples[i];
        if (sample != 0)
            nonzero_samples++;
        const s32 magnitude = sample < 0 ? -sample : sample;
        if (magnitude > peak_sample)
            peak_sample = magnitude;
        sample_hash ^= static_cast<u16>(sample);
        sample_hash *= 16777619u;
    }

    int queued = SDL_GetAudioStreamQueued(gx_aurora::g_audio_stream);
    unsigned waited_ms = 0;
    while (queued > max_queued_bytes && waited_ms < 20u) {
        gx_aurora::g_audio_throttle_count++;
        if (gx_aurora::g_audio_queue_log &&
            (gx_aurora::g_audio_throttle_count <= 8 ||
             (gx_aurora::g_audio_throttle_count % 100) == 0))
            std::fprintf(stderr,
                         "[audio-queue] throttle push=%llu waits=%llu queued=%d "
                         "max=%d\n",
                         gx_aurora::g_audio_push_count, gx_aurora::g_audio_throttle_count, queued,
                         max_queued_bytes);
        SDL_Delay(1);
        waited_ms++;
        queued = SDL_GetAudioStreamQueued(gx_aurora::g_audio_stream);
    }

    const bool low_queue =
        queued >= 0 && queued < bytes_per_second / 100;
    // A push that finds less than 10 ms queued while playing means the device
    // ran dry or nearly so: the output played silence in between (stutter).
    if (low_queue && gx_aurora::g_audio_playing)
        gx_aurora::g_audio_starved_count++;
    if (gx_aurora::g_audio_queue_log &&
        (gx_aurora::g_audio_push_count <= 16 || (gx_aurora::g_audio_push_count % 4000) == 0 ||
         (low_queue && gx_aurora::g_audio_push_count - gx_aurora::g_audio_low_log_push >= 4000))) {
        if (low_queue)
            gx_aurora::g_audio_low_log_push = gx_aurora::g_audio_push_count;
        std::fprintf(stderr,
                     "[audio-queue] push=%llu queued_before=%d queued_after=%d "
                     "playing=%u nonzero=%u peak=%d hash=0x%08X throttles=%llu\n",
                     gx_aurora::g_audio_push_count, queued, queued + bytes,
                     gx_aurora::g_audio_playing ? 1u : 0u, nonzero_samples, peak_sample,
                     sample_hash, gx_aurora::g_audio_throttle_count);
    }
    const s16* out_samples = samples;
    int out_bytes = bytes;
    if (stretch_enabled() && queued >= 0) {
        double desired = 1.0;
        if (gx_aurora::g_audio_playing) {
            const double queued_ms = queued * 1000.0 / bytes_per_second;
            if (queued_ms < kStretchTargetMs)
                desired = 1.0 + (kStretchMax - 1.0) * (kStretchTargetMs - queued_ms) / kStretchTargetMs;
        }
        g_stretch += 0.1 * (desired - g_stretch);
        const double factor = g_stretch < 1.01 && !g_stretcher.active() ? 1.0
                              : (g_stretch < 1.002 ? 1.0 : g_stretch);
        if (factor > 1.0 || g_stretcher.active()) {
            g_stretch_out.clear();
            g_stretcher.process(samples, frames, factor, g_stretch_out);
            out_samples = g_stretch_out.data();
            out_bytes = static_cast<int>(g_stretch_out.size() * sizeof(s16));
            if (factor > 1.0)
                gx_aurora::g_audio_stretched_count++;
        }
    }
    if (out_bytes > 0 && !SDL_PutAudioStreamData(gx_aurora::g_audio_stream, out_samples, out_bytes))
        std::fprintf(stderr, "[audio] failed to queue samples: %s\n", SDL_GetError());
    else if (!gx_aurora::g_audio_playing && queued + out_bytes >= prebuffer_bytes) {
        if (SDL_ResumeAudioStreamDevice(gx_aurora::g_audio_stream))
        {
            gx_aurora::g_audio_playing = true;
            if (gx_aurora::g_audio_queue_log)
                std::fprintf(stderr,
                             "[audio-queue] playback-start push=%llu "
                             "queued_before=%d queued_after=%d prebuffer=%d "
                             "nonzero=%u peak=%d hash=0x%08X\n",
                             gx_aurora::g_audio_push_count, queued, queued + out_bytes,
                             prebuffer_bytes, nonzero_samples, peak_sample, sample_hash);
        }
        else
            std::fprintf(stderr, "[audio] failed to start playback: %s\n", SDL_GetError());
    }
}

} // extern "C"
