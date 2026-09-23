// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef GXRUNTIME_AURORA_BACKEND_H
#define GXRUNTIME_AURORA_BACKEND_H

#include "gxruntime/platform.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AuroraBackendConfig {
    const char* app_name;
    unsigned window_width;
    unsigned window_height;
    bool vsync;
    bool allow_texture_dumps;
    bool info_logging;
    bool graphics_logging;
    bool force_untextured;
} AuroraBackendConfig;

bool dol_aurora_initialize(int argc, char** argv,
                           const AuroraBackendConfig* config);
void dol_aurora_shutdown(void);

// Host overlay hooks. The overlay callback runs on the main thread inside the
// open Aurora frame, just before it is submitted, so it may issue ImGui draw
// calls that composite over the game. The event observer sees every SDL event
// Aurora forwards (touch, keyboard, controller) as a const SDL_Event*.
typedef void (*DolAuroraOverlayFn)(void* user);
typedef void (*DolAuroraEventObserverFn)(const void* sdl_event, void* user);
void dol_aurora_set_overlay(DolAuroraOverlayFn draw, void* user);
void dol_aurora_set_event_observer(DolAuroraEventObserverFn observe, void* user);
// While no frame can be opened, the backend asks this predicate whether the
// host wants the guest held (for example an iOS app that is not active). If so
// it pumps events and sleeps on the main thread until the predicate clears.
typedef bool (*DolAuroraHoldFn)(void* user);
void dol_aurora_set_hold(DolAuroraHoldFn should_hold, void* user);

#ifdef __cplusplus
}
#endif

#endif
