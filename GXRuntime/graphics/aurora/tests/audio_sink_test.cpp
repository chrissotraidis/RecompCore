// SPDX-License-Identifier: GPL-3.0-or-later
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "../../../backends/aurora/aurora_backend_private.h"
#include <SDL3/SDL.h>
#include <cassert>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
static void environment(const char* key, const char* value) {
#ifdef _WIN32
    assert(_putenv_s(key, value) == 0);
#else
    assert(setenv(key, value, 1) == 0);
#endif
}
int main() {
    environment("DOL_AUDIO_NO_THROTTLE", "1");
    environment("DOL_AUDIO_STRETCH", "0");
    assert(SDL_SetHint(SDL_HINT_AUDIO_DRIVER, "bluewake-nonexistent-driver"));
    assert(!gx_aurora::retry_audio_open(true));
    assert(!gx_aurora::retry_audio_open(false)); // timer backs off
    assert(SDL_SetHint(SDL_HINT_AUDIO_DRIVER, "dummy"));
    assert(gx_aurora::retry_audio_open(true)); // device-added style forced retry
    assert(gx_aurora::g_audio_stream != nullptr && !gx_aurora::g_audio_playing);
    auto* stream = gx_aurora::g_audio_stream;
    assert(gx_aurora::retry_audio_open(true) && stream == gx_aurora::g_audio_stream);
    auto capture = std::filesystem::temp_directory_path() / ("bluewake-sink-" + std::to_string(SDL_GetTicksNS()) + ".pcm");
    environment("DOL_AUDIO_SINK_CAPTURE", capture.string().c_str());
    gx_aurora::g_audio_prebuffer_ms = 10000; // dummy stays paused; queue deterministic
    gx_aurora::g_audio_max_queue_ms = 250;
    s16 samples[512];
    for (unsigned i = 0; i < 256; ++i) samples[i * 2] = samples[i * 2 + 1] = static_cast<s16>(10000 * std::sin(i * 0.1));
    aurora_backend_audio_push(samples, 256);
    gx_aurora::g_audio_discard = true;
    assert(SDL_PutAudioStreamData(stream, samples, sizeof samples));
    // Fill beyond 100ms for fast-forward discard, then beyond max queue.
    for (unsigned i = 0; i < 40; ++i) assert(SDL_PutAudioStreamData(stream, samples, sizeof samples));
    auto dropped = gx_aurora::g_audio_dropped_count;
    aurora_backend_audio_push(samples, 256);
    assert(gx_aurora::g_audio_dropped_count == dropped + 1 && gx_aurora::g_audio_dropped_frames == 256);
    gx_aurora::g_audio_discard = false;
    aurora_backend_audio_push(samples, 256);
    assert(gx_aurora::g_audio_dropped_count == dropped + 2 && gx_aurora::g_audio_dropped_frames == 512);
    gx_aurora::close_audio_capture();
    std::ifstream input(capture, std::ios::binary);
    s16 actual[512]; input.read(reinterpret_cast<char*>(actual), sizeof actual);
    assert(input.gcount() == sizeof actual && std::memcmp(actual, samples, sizeof actual) == 0);
    assert(input.peek() == std::ifstream::traits_type::eof());
    input.close(); std::filesystem::remove(capture);
    gx_aurora::g_audio_playing = true;
    assert(SDL_PauseAudioStreamDevice(stream));
    dol_aurora_audio_resume();
    gx_aurora::recover_audio_output();
    assert(!gx_aurora::g_audio_playing);
    gx_aurora::g_audio_prebuffer_ms = 1;
    assert(SDL_GetAudioStreamQueued(stream) > static_cast<int>(gx_aurora::g_audio_sample_rate) * 4 * gx_aurora::g_audio_max_queue_ms / 1000);
    dol_aurora_audio_resume();
    aurora_backend_audio_push(samples, 256);
    assert(gx_aurora::g_audio_playing && !SDL_AudioDevicePaused(SDL_GetAudioStreamDevice(stream)));
    assert(SDL_PauseAudioStreamDevice(stream));
    assert(SDL_ClearAudioStream(stream));
    gx_aurora::g_audio_prebuffer_ms = 1;
    dol_aurora_audio_resume();
    aurora_backend_audio_push(samples, 256);
    assert(gx_aurora::g_audio_playing && !SDL_AudioDevicePaused(SDL_GetAudioStreamDevice(stream)));
    gx_aurora::close_audio_capture();
    std::filesystem::remove(capture);
    SDL_DestroyAudioStream(stream); gx_aurora::g_audio_stream = nullptr;
    SDL_QuitSubSystem(SDL_INIT_AUDIO);
}
