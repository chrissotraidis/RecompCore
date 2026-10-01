// SPDX-License-Identifier: GPL-3.0-or-later
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "../../../backends/aurora/aurora_backend_private.h"
#include <SDL3/SDL.h>
#include <cassert>
int main() {
    assert(SDL_SetHint(SDL_HINT_AUDIO_DRIVER, "bluewake-nonexistent-driver"));
    assert(!gx_aurora::retry_audio_open(true));
    assert(!gx_aurora::retry_audio_open(false)); // timer backs off
    assert(SDL_SetHint(SDL_HINT_AUDIO_DRIVER, "dummy"));
    assert(gx_aurora::retry_audio_open(true)); // device-added style forced retry
    assert(gx_aurora::g_audio_stream != nullptr && !gx_aurora::g_audio_playing);
    auto* stream = gx_aurora::g_audio_stream;
    assert(gx_aurora::retry_audio_open(true) && stream == gx_aurora::g_audio_stream);
    SDL_DestroyAudioStream(stream); gx_aurora::g_audio_stream = nullptr;
    SDL_QuitSubSystem(SDL_INIT_AUDIO);
}
